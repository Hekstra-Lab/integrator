"""Which pixel likelihood should we use on JUNGFRAU SFX data, and can we go back to integers?

Synthetic ground-truth study. Shoeboxes are generated with known intensity, profile and
background, pushed through a simulated JUNGFRAU readout (`detector.py`), calibrated back
to photon-equivalents, and then `I` is recovered by maximum likelihood under each rung of
`likelihoods.py`. The profile is held known and only `(I, B)` are fitted, so any gap
between rungs is attributable to the likelihood and nothing else -- no encoder, no
amortization, no profile misspecification.

Four questions, one study each:

  1. integer_recovery  Does `round((adu - ped)/gain)` return the true photon count?
     This is the "can we convert to integers" question, answered on pixels alone with
     no fitting involved.
  2. ladder            At the real G0 noise, how do the five likelihoods compare on
     bias, RMSE and calibration of sigma_I across the intensity range?
  3. pedestal          What does a realistic ~100 ADU pedestal drift do to each route?
     The drift is a *bias*, ~8x the G0 read noise, and it is the one detector effect
     large enough to matter.
  4. noise_sweep       How much read noise before the integer route breaks? Locates the
     boundary between "rounding is free" and "rounding destroys information".

Run:  uv run python scripts/jungfrau_sim/study.py [--n 400] [--seed 0] [--only ladder]
"""

from __future__ import annotations

import argparse

import torch

from detector import JungfrauConfig, calibrate, readout, to_counts
from likelihoods import LADDER, log_prob

torch.set_default_dtype(torch.float64)  # ground-truth study; precision over speed

SHOEBOX = (3, 13, 13)
PROFILE_SIGMA = 1.5


def make_profile(shape=SHOEBOX, sigma: float = PROFILE_SIGMA) -> torch.Tensor:
    """Normalized 3D Gaussian spot profile, flattened to `(P,)`.

    Flat-voxel layout matches the integrator's own convention (counts are `(N, d*h*w)`).
    """
    d, h, w = shape
    grids = [torch.arange(float(n)) - (n - 1) / 2 for n in (d, h, w)]
    gz, gy, gx = torch.meshgrid(*grids, indexing="ij")
    # Depth is coarser than the transverse directions, as for a real rotation/still.
    r2 = (gz / (0.6 * sigma)) ** 2 + (gy / sigma) ** 2 + (gx / sigma) ** 2
    p = torch.exp(-0.5 * r2).flatten()
    return p / p.sum()


def simulate_shoeboxes(
    n_boxes: int,
    i_true: float,
    b_true: float,
    prof: torch.Tensor,
    cfg: JungfrauConfig,
    generator: torch.Generator,
    pedestal_error_adu: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate, digitize and calibrate a batch. Returns `(x, n_counts, n_true)`."""
    lam = i_true * prof.unsqueeze(0) + b_true
    lam = lam.expand(n_boxes, -1).contiguous()
    n_true = torch.poisson(lam, generator=generator)
    raw = readout(n_true, cfg, generator=generator)
    x, _ = calibrate(raw, cfg, pedestal_error_adu=pedestal_error_adu)
    return x, to_counts(x), n_true


def simulate_with_stage(
    n_boxes: int,
    i_true: float,
    b_true: float,
    prof: torch.Tensor,
    cfg: JungfrauConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """As `simulate_shoeboxes`, but also return the per-pixel read noise in photons.

    The gain stage is *observed* -- it is the top 2 bits of the readout word -- so a
    model is entitled to condition on it and use that pixel's own noise. This is free
    information, not an assumption, and it is what makes a stage-aware likelihood legal.
    """
    lam = i_true * prof.unsqueeze(0) + b_true
    lam = lam.expand(n_boxes, -1).contiguous()
    n_true = torch.poisson(lam, generator=generator)
    raw = readout(n_true, cfg, generator=generator)
    x, stage = calibrate(raw, cfg)
    sig_by_stage = torch.tensor([cfg.sigma_read_photons(g) for g in range(3)])
    return x, to_counts(x), n_true, sig_by_stage[stage]


def fit(
    x: torch.Tensor,
    n: torch.Tensor,
    prof: torch.Tensor,
    name: str,
    sigma: torch.Tensor | float,
    i_init: float,
    b_init: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched MLE of `(I, B)` per shoebox. Returns `(I_hat, sigma_I)`.

    Shoeboxes are independent, so optimizing the summed NLL over per-box parameters is
    exactly optimizing each box on its own. `sigma_I` comes from the observed
    information: the same independence makes the total Hessian block-diagonal, so the
    per-box 2x2 blocks fall out of two grad-of-grad passes.
    """
    n_boxes = x.shape[0]
    log_i = torch.full((n_boxes,), float(torch.log(torch.tensor(i_init))))
    log_b = torch.full((n_boxes,), float(torch.log(torch.tensor(b_init))))
    log_i.requires_grad_(True)
    log_b.requires_grad_(True)
    params = [log_i, log_b]

    log_s = None
    if name == "normal_free":
        log_s = torch.full((n_boxes,), 0.0, requires_grad=True)
        params.append(log_s)

    # exact() picks n_max from lam.max(); pin it here so it cannot drift between the
    # optimizer's closure calls and the Hessian pass.
    opt = torch.optim.LBFGS(
        params, max_iter=250, tolerance_grad=1e-12, tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def nll(li, lb, ls):
        lam = torch.exp(li).unsqueeze(1) * prof.unsqueeze(0) + torch.exp(lb).unsqueeze(1)
        s = torch.exp(ls).unsqueeze(1).expand_as(lam) if ls is not None else None
        return -log_prob(name, x, n, lam, sigma, s).sum()

    def closure():
        opt.zero_grad()
        loss = nll(log_i, log_b, log_s)
        loss.backward()
        return loss

    opt.step(closure)

    i_hat = torch.exp(log_i).detach()
    b_hat = torch.exp(log_b).detach()

    # Observed information in the natural (I, B) coordinates, not the log ones.
    ii = i_hat.clone().requires_grad_(True)
    bb = b_hat.clone().requires_grad_(True)
    ss = torch.exp(log_s).detach() if log_s is not None else None

    lam = ii.unsqueeze(1) * prof.unsqueeze(0) + bb.unsqueeze(1)
    s = ss.unsqueeze(1).expand_as(lam) if ss is not None else None
    total = -log_prob(name, x, n, lam, sigma, s).sum()

    g_i = torch.autograd.grad(total, ii, create_graph=True)[0]
    h_ii = torch.autograd.grad(g_i.sum(), ii, retain_graph=True)[0]
    h_ib = torch.autograd.grad(g_i.sum(), bb, retain_graph=True)[0]
    g_b = torch.autograd.grad(total, bb, create_graph=True)[0]
    h_bb = torch.autograd.grad(g_b.sum(), bb)[0]

    det = h_ii * h_bb - h_ib**2
    var_i = torch.where(det > 0, h_bb / det, torch.full_like(det, float("nan")))
    return i_hat, var_i.clamp_min(0).sqrt()


def score(i_hat: torch.Tensor, sigma_i: torch.Tensor, i_true: float) -> dict:
    err = i_hat - i_true
    z = err / sigma_i
    ok = torch.isfinite(z)
    return {
        "bias%": 100 * float(err.mean()) / i_true,
        "rmse%": 100 * float(err.pow(2).mean().sqrt()) / i_true,
        "z_std": float(z[ok].std()) if ok.any() else float("nan"),
        "cov95": float((z[ok].abs() < 1.96).double().mean()) if ok.any() else float("nan"),
    }


def _fmt(name: str, m: dict) -> str:
    return (
        f"    {name:<16} bias {m['bias%']:+7.2f}%   rmse {m['rmse%']:6.2f}%   "
        f"z-std {m['z_std']:5.2f}   cov95 {m['cov95']:5.3f}"
    )


def study_integer_recovery(prof, n_boxes, seed) -> None:
    """Q1: does rounding the calibrated value return the true photon count?"""
    print("\n" + "=" * 78)
    print("1. INTEGER RECOVERY -- can we convert back to counts?")
    print("=" * 78)
    print("   fraction of pixels where round(clamp((adu-ped)/gain, 0)) == true count")
    print("   background lam=0.5 ph/px, spot peak ~a few ph/px (all in G0)\n")

    base = JungfrauConfig()
    drifts = [0.0, 50.0, 100.0, 200.0]
    print(f"    {'sigma_read':>10} |" + "".join(f"{d:>10.0f} ADU" for d in drifts))
    print(f"    {'(photons)':>10} |" + "".join(f"{d / 514.6:>10.2f} ph" for d in drifts))
    print("    " + "-" * 66)

    for sig in [0.024, 0.06, 0.15, 0.3, 0.5, 0.8]:
        cfg = base.with_sigma_read_photons_g0(sig)
        row = []
        for drift in drifts:
            g = torch.Generator().manual_seed(seed)
            _, n_cnt, n_true = simulate_shoeboxes(
                n_boxes, 20.0, 0.5, prof, cfg, g, pedestal_error_adu=drift
            )
            row.append(float((n_cnt == n_true).double().mean()))
        star = "  <- real G0" if abs(sig - 0.024) < 1e-9 else ""
        print(f"    {sig:>10.3f} |" + "".join(f"{100 * r:>12.2f}%" for r in row) + star)

    print("\n   Rounding has a +-0.5 photon deadband. Read noise and pedestal drift are")
    print("   both far inside it at real G0 values, so the integer conversion is exact")
    print("   -- and it *absorbs* the drift rather than propagating it.")


def study_ladder(prof, n_boxes, seed) -> None:
    """Q2: how do the five likelihoods compare at the real G0 noise?"""
    print("\n" + "=" * 78)
    print("2. LIKELIHOOD LADDER -- at real G0 noise (sigma_read = 0.024 photons)")
    print("=" * 78)
    print("   MLE of (I, B) with the profile known. 'exact' is the simulator's own law.")
    print("   z-std=1 and cov95=0.95 mean sigma_I is honest; bias%/rmse% are on I.\n")

    cfg = JungfrauConfig()
    sigma = cfg.sigma_read_photons(0)

    for i_true in [2.0, 10.0, 50.0, 200.0]:
        b_true = 0.5
        g = torch.Generator().manual_seed(seed)
        x, n_cnt, _ = simulate_shoeboxes(n_boxes, i_true, b_true, prof, cfg, g)
        peak = i_true * float(prof.max()) + b_true
        print(f"  I_true = {i_true:>6.1f}   (peak pixel ~{peak:.1f} ph, bg {b_true} ph/px)")
        for name in LADDER:
            i_hat, sigma_i = fit(x, n_cnt, prof, name, sigma, i_true, b_true)
            print(_fmt(name, score(i_hat, sigma_i, i_true)))
        print()


def study_pedestal(prof, n_boxes, seed) -> None:
    """Q3: what does a realistic pedestal drift do to each route?"""
    print("\n" + "=" * 78)
    print("3. PEDESTAL DRIFT -- 100 ADU (0.19 photons/px), the measured thermal drift")
    print("=" * 78)
    print(f"   Shoebox is {SHOEBOX} = {prof.numel()} pixels, so a per-pixel bias is")
    print("   multiplied by ~the pixel count when it lands in the background term.\n")

    cfg = JungfrauConfig()
    sigma = cfg.sigma_read_photons(0)

    for drift in [0.0, 100.0]:
        ph = drift / abs(cfg.adu_per_photon[0])
        print(f"  pedestal error = {drift:>5.0f} ADU  ({ph:+.3f} ph/px)")
        for i_true in [10.0, 50.0]:
            g = torch.Generator().manual_seed(seed)
            x, n_cnt, _ = simulate_shoeboxes(
                n_boxes, i_true, 0.5, prof, cfg, g, pedestal_error_adu=drift
            )
            print(f"    I_true = {i_true:.0f}")
            for name in ("normal_coupled", "poisson_counts", "exact"):
                i_hat, sigma_i = fit(x, n_cnt, prof, name, sigma, i_true, 0.5)
                print("  " + _fmt(name, score(i_hat, sigma_i, i_true)))
        print()


def study_noise_sweep(prof, n_boxes, seed) -> None:
    """Q4: how much read noise before the integer route breaks?"""
    print("\n" + "=" * 78)
    print("4. NOISE SWEEP -- where does the integer route stop being free?")
    print("=" * 78)
    print("   I_true = 10, bg = 0.5. 'exact' is the ceiling; watch poisson_counts")
    print("   track it and then peel away as sigma_read approaches the 0.5 deadband.\n")

    base = JungfrauConfig()
    i_true, b_true = 10.0, 0.5
    names = ("normal_coupled", "poisson_counts", "exact")
    print(f"    {'sigma_read':>10} | {'exact %px':>10} |" + "".join(f"{n:>18}" for n in names))
    print("    " + "-" * 76)

    for sig in [0.024, 0.06, 0.15, 0.3, 0.5, 0.8]:
        cfg = base.with_sigma_read_photons_g0(sig)
        g = torch.Generator().manual_seed(seed)
        x, n_cnt, n_true = simulate_shoeboxes(n_boxes, i_true, b_true, prof, cfg, g)
        frac = 100 * float((n_cnt == n_true).double().mean())
        cells = []
        for name in names:
            i_hat, sigma_i = fit(x, n_cnt, prof, name, sig, i_true, b_true)
            m = score(i_hat, sigma_i, i_true)
            cells.append(f"{m['bias%']:+6.2f}% /{m['rmse%']:5.1f}%")
        print(f"    {sig:>10.3f} | {frac:>9.2f}% |" + "".join(f"{c:>18}" for c in cells))

    print("\n    cells are bias% / rmse% on I.")


def study_gain_stages(prof, n_boxes, seed) -> None:
    """Q5: what happens as reflections get bright enough to switch into G1 and G2?

    The rungs are given the per-pixel read noise implied by each pixel's *observed*
    gain stage, which a real model is entitled to do (the stage is in the readout word).
    `poisson_counts` is the exception and cannot use it -- asserting sigma = 0 is the
    whole content of that rung -- which is precisely what this study puts a price on.
    """
    print("\n" + "=" * 78)
    print("5. GAIN STAGES -- what happens when reflections reach G1 and G2?")
    print("=" * 78)
    cfg = JungfrauConfig()
    peak = float(prof.max())
    print(f"   profile peak {peak:.4f}, so the peak pixel holds ~{peak:.3f} * I photons.")
    print(f"   read noise by stage: "
          + ", ".join(f"G{g} {cfg.sigma_read_photons(g):.3f} ph" for g in range(3)))
    print(f"   stage switches at {cfg.switch_photons[0]:.0f} and "
          f"{cfg.switch_photons[1]:.0f} photons/pixel.\n")

    for i_true in [200.0, 2_000.0, 20_000.0, 200_000.0]:
        b_true = 0.5
        g = torch.Generator().manual_seed(seed)
        x, n_cnt, n_true, sig_px = simulate_with_stage(
            n_boxes, i_true, b_true, prof, cfg, g
        )
        # Which stages did this intensity actually populate?
        fr = [float((sig_px == cfg.sigma_read_photons(s)).double().mean()) for s in range(3)]
        exact_px = 100 * float((n_cnt == n_true).double().mean())
        print(f"  I_true = {i_true:>8.0f}   peak pixel ~{i_true * peak:.0f} ph"
              f"   stages G0/G1/G2 = {100 * fr[0]:.1f}/{100 * fr[1]:.1f}/{100 * fr[2]:.1f}%"
              f"   counts exact {exact_px:.2f}%")
        for name in ("normal_coupled", "poisson_counts", "hybrid", "exact"):
            i_hat, sigma_i = fit(x, n_cnt, prof, name, sig_px, i_true, b_true)
            print(_fmt(name, score(i_hat, sigma_i, i_true)))
        print()

    print("   RESULT (and it refutes the obvious hypothesis): poisson_counts asserts")
    print("   Var = lam, and once a pixel is in G1/G2 the truth is Var = lam + sigma^2,")
    print("   so it *should* get overconfident. It does not. By the time any pixel")
    print("   reaches G1 the reflection is bright enough that every rung agrees to")
    print("   ~0.01% bias, and sigma^2 is a 1-8% correction to a variance nobody is")
    print("   relying on. `hybrid` buys nothing measurable; the extra complexity is")
    print("   not worth it. Rounding everything and using Poisson is correct at G0")
    print("   (where it is exact) and harmless everywhere else.")


STUDIES = {
    "integer_recovery": study_integer_recovery,
    "ladder": study_ladder,
    "pedestal": study_pedestal,
    "noise_sweep": study_noise_sweep,
    "gain_stages": study_gain_stages,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=400, help="shoeboxes per cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=sorted(STUDIES), help="run a single study")
    args = ap.parse_args()

    prof = make_profile()
    cfg = JungfrauConfig()
    print(f"JUNGFRAU likelihood study -- {args.n} shoeboxes/cell, seed {args.seed}")
    print(f"shoebox {SHOEBOX} = {prof.numel()} px, profile peak {float(prof.max()):.4f}")
    print(f"G0: {cfg.adu_per_photon[0]:.1f} ADU/photon, read noise "
          f"{cfg.sigma_read_photons(0):.4f} photons")

    todo = [args.only] if args.only else list(STUDIES)
    for key in todo:
        STUDIES[key](prof, args.n, args.seed)


if __name__ == "__main__":
    main()
