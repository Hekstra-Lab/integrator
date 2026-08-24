"""Generate 2D JUNGFRAU shoeboxes with ground-truth intensity, profile and background.

Emits the same shoebox twice, which is the point of the dataset:

  * `counts_real.npy`    -- what the detector actually gives you. Calibrated
    photon-equivalents `(adu - pedestal)/gain / E_gamma`. Real-valued, frequently
    negative on background pixels. Train a Normal likelihood on these.
  * `counts_poisson.npy` -- the same pixels rounded back to integers,
    `round(clamp(x, 0))`. Train a Poisson likelihood on these.

plus the latent truth needed to score either of them:

  * `counts_true.npy`  -- the actual Poisson draw, i.e. the photon count the detector
    would have reported if it could count. This is what `counts_poisson` is trying to
    recover, so `(counts_poisson == counts_true).mean()` measures the conversion
    directly rather than by proxy.
  * `intensity.npy`, `background.npy`, `profiles.npy`, `profile_params.npy`.
  * `raw_adu.npy` -- the packed 16-bit words, so the calibration can be redone with a
    different assumed pedestal without regenerating.

Generative model, matching `src/integrator/simulate/generate.py`'s conventions:

    I ~ Exponential(1/mean_intensity)      Wilson prior for an acentric reflection
    B ~ Exponential(1/mean_background)     flat per-shoebox background
    profile from local `profiles.py` (elliptical 2D Gaussian, h ~ N(0, I_5))
    N ~ Poisson(I * profile + B)
    then through the JUNGFRAU readout in `detector.py`.

Run:  uv run python scripts/jungfrau_sim/generate.py --n 20000 --out data/jf_sim
      uv run python scripts/jungfrau_sim/generate.py --n 20000 --sigma-read 0.5 \
          --pedestal-drift 100 --out data/jf_sim_hard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from detector import JungfrauConfig, calibrate, readout, to_counts

# Local profile sampler: elliptical 2D Gaussians from h ~ N(0, I_5) (self-contained).
from profiles import h_to_physical_params, h_to_profile


def generate(
    n: int,
    h: int,
    w: int,
    cfg: JungfrauConfig,
    mean_intensity: float,
    mean_background: float,
    pedestal_drift_adu: float | tuple[float, float, float],
    seed: int,
) -> dict[str, torch.Tensor]:
    """Simulate `n` shoeboxes of `h` x `w` pixels through a JUNGFRAU readout."""
    g = torch.Generator().manual_seed(seed)

    hvec = torch.randn(n, 5, generator=g)
    center = (h - 1) / 2.0
    prof = h_to_profile(hvec, H=h, W=w, center_base=center)
    cx, cy, s1, s2, theta = h_to_physical_params(hvec, center_base=center)

    # Wilson prior: I ~ Exponential(mean). Inverse-CDF so the draw honours `generator`.
    u = torch.rand(n, generator=g).clamp_min(1e-12)
    intensity = -mean_intensity * torch.log(u)
    u = torch.rand(n, generator=g).clamp_min(1e-12)
    background = -mean_background * torch.log(u)

    lam = intensity.unsqueeze(1) * prof + background.unsqueeze(1)
    n_true = torch.poisson(lam, generator=g)

    raw = readout(n_true, cfg, generator=g)
    x, stage = calibrate(raw, cfg, pedestal_error_adu=pedestal_drift_adu)

    return {
        "counts_real": x,
        "counts_poisson": to_counts(x),
        "counts_true": n_true,
        "raw_adu": raw,
        "gain_stage": stage,
        "profiles": prof,
        "intensity": intensity,
        "background": background,
        "profile_params": torch.stack([cx, cy, s1, s2, theta], dim=1),
        "rate_true": lam,
    }


DTYPES = {
    "counts_real": np.float32,
    "counts_poisson": np.int32,
    "counts_true": np.int32,
    "raw_adu": np.uint16,
    "gain_stage": np.uint8,
    "profiles": np.float32,
    "intensity": np.float32,
    "background": np.float32,
    "profile_params": np.float32,
    "rate_true": np.float32,
}


def summarize(sim: dict[str, torch.Tensor], cfg: JungfrauConfig) -> dict:
    """Report what came out, and check the integer conversion against the truth.

    Broken down per gain stage, because the stages behave completely differently:
    read noise is 0.024 photons in G0 but ~0.72 in G1, so only G0 sits inside the
    +-0.5 photon rounding deadband. G1 misrounds routinely -- and it does not matter,
    because a pixel is only in G1 once it holds >25 photons, where Poisson noise
    (>=5 photons) dwarfs both the read noise and a +-1 rounding slip.
    """
    real, pois, true = sim["counts_real"], sim["counts_poisson"], sim["counts_true"]
    stage = sim["gain_stage"]
    err = (real - true).abs()

    per_stage = []
    for s in range(3):
        m = stage == s
        frac = float(m.double().mean())
        if not bool(m.any()):
            per_stage.append({"stage": f"G{s}", "frac_pixels": frac, "n": 0})
            continue
        counts_here = true[m]
        per_stage.append({
            "stage": f"G{s}",
            "frac_pixels": frac,
            "n": int(m.sum()),
            "sigma_read_photons": cfg.sigma_read_photons(s),
            "frac_exact": float((pois[m] == counts_here).double().mean()),
            "rms_err_photons": float(err[m].pow(2).mean().sqrt()),
            "mean_true_count": float(counts_here.mean()),
            # Read noise relative to the Poisson noise already present at this count.
            "read_vs_poisson_noise": cfg.sigma_read_photons(s)
            / max(float(counts_here.double().mean().sqrt()), 1e-9),
        })

    return {
        "pixels": int(real.numel()),
        "frac_counts_recovered_exactly": float((pois == true).double().mean()),
        "frac_real_negative": float((real < 0).double().mean()),
        "frac_true_zero": float((true == 0).double().mean()),
        "max_abs_calibration_error_photons": float(err.max()),
        "rms_calibration_error_photons": float(err.pow(2).mean().sqrt()),
        "per_gain_stage": per_stage,
        "intensity_mean": float(sim["intensity"].mean()),
        "intensity_p99": float(sim["intensity"].quantile(0.99)),
        "intensity_max": float(sim["intensity"].max()),
        "peak_pixel_max_photons": float(sim["rate_true"].max()),
        "sigma_read_photons_g0": cfg.sigma_read_photons(0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20000, help="number of shoeboxes")
    ap.add_argument("--h", type=int, default=20, help="shoebox height in pixels")
    ap.add_argument("--w", type=int, default=20, help="shoebox width in pixels")
    ap.add_argument("--mean-intensity", type=float, default=200.0)
    ap.add_argument("--mean-background", type=float, default=0.5)
    ap.add_argument(
        "--sigma-read",
        type=float,
        default=None,
        help="G0 read noise in photons; default is the real detector value (0.024)",
    )
    ap.add_argument(
        "--pedestal-drift",
        type=float,
        nargs="+",
        default=[0.0],
        help="assumed-pedestal error in ADU: one value for G0 only (G1/G2 left clean, "
        "the default reading of the published G0 figure), or three for all stages",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/jf_sim"))
    args = ap.parse_args()

    cfg = JungfrauConfig()
    if args.sigma_read is not None:
        cfg = cfg.with_sigma_read_photons_g0(args.sigma_read)

    drift = args.pedestal_drift
    if len(drift) == 1:
        drift = (drift[0], 0.0, 0.0)  # the published figure is a G0 measurement
    elif len(drift) != 3:
        ap.error("--pedestal-drift takes 1 value (G0) or 3 (G0 G1 G2)")
    drift = tuple(float(d) for d in drift)

    print(f"generating {args.n} shoeboxes of {args.h}x{args.w} = {args.h * args.w} px")
    print(f"  I ~ Exp(mean={args.mean_intensity}), B ~ Exp(mean={args.mean_background})")
    print(f"  G0 read noise {cfg.sigma_read_photons(0):.4f} photons"
          f"  ({cfg.noise_adu()[0]:.1f} ADU)")
    ph = tuple(d / abs(a) for d, a in zip(drift, cfg.adu_per_photon))
    print(f"  pedestal drift {drift} ADU"
          f"  = ({ph[0]:+.3f}, {ph[1]:+.3f}, {ph[2]:+.3f}) ph/px in G0/G1/G2")

    sim = generate(
        args.n, args.h, args.w, cfg,
        args.mean_intensity, args.mean_background, drift, args.seed,
    )
    stats = summarize(sim, cfg)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for key, tensor in sim.items():
        np.save(out / f"{key}.npy", tensor.numpy().astype(DTYPES[key]))

    manifest = {
        "n_reflections": args.n,
        "geometry": {"h": args.h, "w": args.w, "data_dim": "2d"},
        "generative_model": {
            "intensity": f"Exponential(mean={args.mean_intensity})",
            "background": f"Exponential(mean={args.mean_background})",
            "profile": "profiles.h_to_profile, h ~ N(0, I_5)",
            "counts": "Poisson(I * profile + B)",
        },
        "detector": {
            "photon_energy_kev": cfg.photon_energy_kev,
            "gain_adu_per_kev": list(cfg.gain_adu_per_kev),
            "pedestal_adu": list(cfg.pedestal_adu),
            "read_noise_adu": list(cfg.noise_adu()),
            "sigma_read_photons_g0": cfg.sigma_read_photons(0),
            "pedestal_drift_adu": list(drift),
            "switch_photons": list(cfg.switch_photons),
        },
        "seed": args.seed,
        "files": {k: f"{k}.npy" for k in sim},
        "stats": stats,
    }
    (out / "sim.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {len(sim)} arrays + sim.json to {out}/")
    print(f"  counts_real.npy     (N, {args.h * args.w}) float32   real line, can be < 0")
    print(f"  counts_poisson.npy  (N, {args.h * args.w}) int32     rounded + clamped")
    print(f"  counts_true.npy     (N, {args.h * args.w}) int32     latent truth")

    print("\nintegrity")
    print(f"  counts_poisson == counts_true : {100 * stats['frac_counts_recovered_exactly']:.4f}% of pixels")
    print(f"  counts_real < 0               : {100 * stats['frac_real_negative']:.2f}% of pixels")
    print(f"  true count == 0               : {100 * stats['frac_true_zero']:.2f}% of pixels")
    print(f"  |counts_real - truth| max/rms : {stats['max_abs_calibration_error_photons']:.4f}"
          f" / {stats['rms_calibration_error_photons']:.4f} photons")
    print(f"  intensity mean/p99/max        : {stats['intensity_mean']:.1f}"
          f" / {stats['intensity_p99']:.0f} / {stats['intensity_max']:.0f}")
    print(f"  brightest pixel               : {stats['peak_pixel_max_photons']:.1f} photons")

    print("\nper gain stage -- only G0 sits inside the +-0.5 photon rounding deadband")
    print(f"    {'':4} {'%px':>8} {'sig_read':>9} {'%exact':>9} {'rms err':>9}"
          f" {'<count>':>8} {'read/Poisson':>13}")
    for row in stats["per_gain_stage"]:
        if not row["n"]:
            print(f"    {row['stage']:4} {100 * row['frac_pixels']:>7.3f}%        (none)")
            continue
        print(f"    {row['stage']:4} {100 * row['frac_pixels']:>7.3f}%"
              f" {row['sigma_read_photons']:>9.3f} {100 * row['frac_exact']:>8.3f}%"
              f" {row['rms_err_photons']:>9.3f} {row['mean_true_count']:>8.1f}"
              f" {row['read_vs_poisson_noise']:>12.1%}")
    print("    read/Poisson = read noise as a fraction of the Poisson noise already there.")


if __name__ == "__main__":
    main()
