"""Unified SFX generator: per-image Wilson prior -> profile -> JUNGFRAU detector -> shoeboxes.

Composes the three pieces built earlier into one forward model, and emits the SAME
shoeboxes twice (integer + real-valued) so the Poisson-vs-Normal likelihood comparison
runs on identical ground truth.

Generative model (per shoebox = one observation of image i):

    per image i:   log G_i ~ Normal(log G_scale, sigma_logG^2)
                   B_i     = B_global            (or per-image if sigma_B > 0)
    per shoebox:   s^2                            resolution, uniform in reciprocal volume
                   mu   = G_i * exp(-2 B_i s^2)   Wilson mean of the OBSERVED intensity
                   I    ~ Exponential(mu)         the true (acentric Wilson) intensity
                   bg   ~ Exponential(bg_mean)    flat background
                   p    ~ profile sampler         normalized 2D Gaussian spot
                   lam_rho = I * p_rho + bg       per-pixel rate
                   N_rho ~ Poisson(lam_rho)       true photon counts
                   ADU  = JUNGFRAU readout(N)     real-valued detector output
                   -> counts_real (calibrated), counts_poisson (rounded)

Everything the downstream training needs to be scored against is saved: the per-image
G_i and B_i, the per-shoebox true I and background, the profile, and image_id.

Run:  uv run python scripts/jungfrau_sim/sfx_generate.py --n-images 150 --n-refl 100 \
          --out data/sfx_sim
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from detector import JungfrauConfig, calibrate, readout, to_counts
from profiles import h_to_profile

D_MIN, D_MAX = 1.5, 20.0


def generate(
    n_images: int,
    n_refl: int,
    h: int,
    w: int,
    sigma_logg: float,
    b_global: float,
    sigma_b: float,
    g_scale: float,
    bg_mean: float,
    cfg: JungfrauConfig,
    seed: int,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    n = n_images * n_refl

    # ── per-image Wilson prior parameters (the things to recover) ──────────────
    log_g = torch.randn(n_images, generator=g) * sigma_logg
    g_true = g_scale * log_g.exp()
    b_true = b_global + torch.randn(n_images, generator=g) * sigma_b
    image_id = torch.arange(n_images).repeat_interleave(n_refl)

    # ── per-shoebox resolution, uniform in reciprocal-space volume ─────────────
    s_min, s_max = 1.0 / (2.0 * D_MAX), 1.0 / (2.0 * D_MIN)
    u = torch.rand(n, generator=g)
    s = (u * (s_max**3 - s_min**3) + s_min**3) ** (1.0 / 3.0)
    s_sq = s**2

    # ── true intensity from the per-image Wilson mean ──────────────────────────
    mu = g_true[image_id] * torch.exp(-2.0 * b_true[image_id] * s_sq)
    u2 = torch.rand(n, generator=g).clamp_min(1e-12)
    i_true = -mu * torch.log(u2)                      # Exp(mean=mu)
    u3 = torch.rand(n, generator=g).clamp_min(1e-12)
    bg_true = -bg_mean * torch.log(u3)                # Exp(mean=bg_mean)

    # ── profile, per-pixel rate, photons, detector ─────────────────────────────
    hvec = torch.randn(n, 5, generator=g)
    prof = h_to_profile(hvec, H=h, W=w, center_base=(h - 1) / 2.0).double()
    lam = i_true.unsqueeze(1) * prof + bg_true.unsqueeze(1)
    n_photons = torch.poisson(lam, generator=g)
    raw = readout(n_photons, cfg, generator=g)
    x, _ = calibrate(raw, cfg)

    return {
        "counts_real": x,
        "counts_poisson": to_counts(x),
        "counts_true": n_photons,
        "profiles": prof,
        "image_id": image_id,
        "s_sq": s_sq,
        "intensity_true": i_true,
        "background_true": bg_true,
        "g_true_per_image": g_true,
        "b_true_per_image": b_true,
    }


DTYPES = {
    "counts_real": np.float32, "counts_poisson": np.int32, "counts_true": np.int32,
    "profiles": np.float32, "image_id": np.int64, "s_sq": np.float32,
    "intensity_true": np.float32, "background_true": np.float32,
    "g_true_per_image": np.float32, "b_true_per_image": np.float32,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-images", type=int, default=150)
    ap.add_argument("--n-refl", type=int, default=100, help="reflections per image")
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--w", type=int, default=20)
    ap.add_argument("--sigma-logg", type=float, default=0.6)
    ap.add_argument("--b-global", type=float, default=20.0)
    ap.add_argument("--sigma-b", type=float, default=0.0,
                    help=">0 makes B per-image; 0 keeps it global (the easy case)")
    ap.add_argument("--g-scale", type=float, default=200.0)
    ap.add_argument("--bg-mean", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/sfx_sim"))
    args = ap.parse_args()

    cfg = JungfrauConfig()
    sim = generate(
        args.n_images, args.n_refl, args.h, args.w, args.sigma_logg, args.b_global,
        args.sigma_b, args.g_scale, args.bg_mean, cfg, args.seed,
    )

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for k, v in sim.items():
        np.save(out / f"{k}.npy", v.numpy().astype(DTYPES[k]))

    manifest = {
        "n_images": args.n_images, "n_refl_per_image": args.n_refl,
        "n_obs": args.n_images * args.n_refl, "geometry": {"h": args.h, "w": args.w},
        "sigma_logG": args.sigma_logg, "b_global": args.b_global, "sigma_B": args.sigma_b,
        "per_image_B": args.sigma_b > 0, "g_scale": args.g_scale, "bg_mean": args.bg_mean,
        "seed": args.seed,
        "detector": {"read_noise_photons_g0": cfg.sigma_read_photons(0)},
    }
    (out / "sim.json").write_text(json.dumps(manifest, indent=2))

    real, pois, true = sim["counts_real"], sim["counts_poisson"], sim["counts_true"]
    peak = float((sim["intensity_true"].unsqueeze(1) * sim["profiles"]).max())
    print(f"wrote {args.n_images} images x {args.n_refl} refl = {args.n_images * args.n_refl}"
          f" shoeboxes ({args.h}x{args.w}) to {out}/")
    print(f"  per-image G: {float(sim['g_true_per_image'].min()):.0f}"
          f"-{float(sim['g_true_per_image'].max()):.0f} photons, B={args.b_global}"
          + (f"+/-{args.sigma_b}" if args.sigma_b else " (global)"))
    print(f"  intensity mean {float(sim['intensity_true'].mean()):.1f}, "
          f"brightest pixel {peak:.1f} ph, bg mean {args.bg_mean}")
    print(f"  counts_poisson == counts_true : "
          f"{100 * float((pois == true).double().mean()):.2f}% of pixels")
    print(f"  counts_real range [{float(real.min()):.2f}, {float(real.max()):.1f}], "
          f"{100 * float((real < 0).double().mean()):.1f}% negative")


if __name__ == "__main__":
    main()
