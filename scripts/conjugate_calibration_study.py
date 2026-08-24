"""Calibration diagnostics for the conjugate intensity integrator.

Synthetic study quantifying how the per-observation intensity posterior of
`ConjugateIntegrator` compares to ground truth, isolating the integration math
from encoder/model-misspecification error. Intensities are drawn from the Wilson
prior `I ~ Gamma(alpha_W, tau)`, pixel counts from `Poisson(s*I*prof + bg)`, and
each estimator is scored against the known true intensity.

It reproduces the four results behind `docs/conjugate_integrator.md` sections
10.3-10.4:

  1. variance narrowing of the mean-field Gamma vs the exact posterior, and its
     `bg -> 0` collapse (proves the source is the I-z augmentation correlation);
  2. point accuracy + calibration of the old (arithmetic, n=3), the geometric-mean
     CAVI, and the exact-quadrature estimators vs the true intensity;
  3. calibration under uncertain background: conditioning on `E[bg]` (Fix A) vs
     marginalizing `q(bg)` (Fix A+B, law of total variance);
  4. quadrature grid resolution: error of `n_grid` points vs a dense reference,
     used to choose the defaults in `ConjugateIntegrator.exact_intensity_posterior`.

Run:  uv run python scripts/conjugate_calibration_study.py [--n 4000] [--seed 0]
"""

import argparse
import statistics as st

import torch

torch.set_default_dtype(torch.float64)  # high precision for a ground-truth study

ALPHA_W = 1.0  # acentric Wilson prior shape


def make_profile(side: int = 11, sigma: float = 1.8) -> torch.Tensor:
    """Normalized 2D Gaussian spot profile over a `side` x `side` pixel box."""
    xs = torch.arange(float(side)) - (side - 1) / 2
    g = torch.exp(-(xs**2) / (2 * sigma**2))
    p = (g[:, None] * g[None, :]).flatten()
    return p / p.sum()


def em_gamma(c, e, bg, tau, mode: str, n_iter: int, tol: float = 1e-10):
    """Mean-field CAVI Gamma (alpha, beta) for one shoebox.

    Args:
        mode: `geom` uses the responsibility at the geometric-mean intensity
            exp(psi(alpha)-log beta) (the correct CAVI update); `arith` uses the
            arithmetic mean alpha/beta (the old EM/MAP-style approximation).
    """
    beta = tau + e.sum()
    log_beta = torch.log(beta)
    I_t = 1.0 / tau
    for _ in range(n_iter):
        pi = (I_t * e) / (I_t * e + bg)
        alpha = ALPHA_W + (pi * c).sum()
        I_new = (alpha / beta) if mode == "arith" else torch.exp(
            torch.digamma(alpha) - log_beta
        )
        if (I_new - I_t).abs() / I_t < tol:
            I_t = I_new
            break
        I_t = I_new
    pi = (I_t * e) / (I_t * e + bg)
    alpha = ALPHA_W + (pi * c).sum()  # alpha at the final fixed point
    return alpha, beta


def grid_mass(log_unnorm: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Normalized probability mass per grid cell from an unnormalized log-density."""
    dw = torch.log(torch.diff(grid, prepend=grid[:1]))
    return torch.softmax(log_unnorm + dw, 0)


def exact_logpost(c, e, bg, tau, grid):
    """Unnormalized log of the collapsed (augmentation-free) posterior p(I|c)."""
    I = grid[:, None]
    return (
        (ALPHA_W - 1) * torch.log(grid)
        - (tau + e.sum()) * grid
        + (c * torch.log(e * I + bg)).sum(-1)
    )


def gamma_logpdf(I, a, b):
    return a * torch.log(b) - torch.lgamma(a) + (a - 1) * torch.log(I) - b * I


def moments(w, grid):
    m = (w * grid).sum()
    v = (w * grid**2).sum() - m**2
    return m, v.clamp(min=0)


def adaptive_grid(mf_mean, mf_std, n_grid):
    """Two-sided grid scaled to the (wider, upward-shifted) exact posterior.

    The exact posterior is up to ~2.5x wider and ~1.4x higher in mean than the
    mean-field Gamma, so we size the window from an inflated std (`3*mf_std`) and
    bias the upper edge. Width ~20 effective std at any brightness keeps ~25
    grid points per std at n_grid=512.
    """
    std_eff = 3.0 * mf_std
    lo = (mf_mean - 8.0 * std_eff).clamp(min=1e-8)
    hi = mf_mean + 12.0 * std_eff
    hi = torch.maximum(hi, lo + 1e-3)
    return torch.linspace(0.0, 1.0, n_grid) * (hi - lo) + lo


# ----------------------------------------------------------------------------
def study_variance_vs_bg(prof, n=60):
    """1. Mean-field variance / exact variance, and its bg -> 0 collapse."""
    e = prof.clone()
    tau = torch.tensor(0.3)
    print("\n=== 1. Mean-field Var / exact Var (the gap is the I-z allocation) ===")
    print(f"{'bg/pixel':>9} | {'MF Var / exact Var':>19}")
    for bg in [2.0, 0.5, 0.1, 0.01, 1e-3, 1e-5]:
        ratios = []
        for seed in range(n):
            torch.manual_seed(seed + int(bg * 1e6) % 9973)
            c = torch.poisson(e * 3.0 + bg)
            a, b = em_gamma(c, e, torch.tensor(bg), tau, "geom", 300)
            mf_v = (a / b**2).item()
            grid = adaptive_grid(a / b, a.sqrt() / b, 8000)
            w = grid_mass(exact_logpost(c, e, torch.tensor(bg), tau, grid), grid)
            _, ev = moments(w, grid)
            ratios.append(mf_v / ev.item())
        print(f"{bg:>9g} | {st.mean(ratios):>19.4f}")
    print("  -> ratio -> 1.0 as bg -> 0: the narrowing is entirely the augmentation gap.")


def study_calibration(prof, n, seed):
    """2. Point accuracy + calibration vs true intensity (known nuisances)."""
    e = prof.clone()
    keys = ["OLD (arith, n=3)", "GEOM-CAVI (converged)", "EXACT-quad (Fix A)"]
    acc = {k: {"rb": [], "se": [], "z": [], "c68": [], "c95": []} for k in keys}
    opt_se = []
    torch.manual_seed(seed)
    for _ in range(n):
        tau_v = float(10 ** torch.empty(1).uniform_(-1.3, 0.0))
        tau = torch.tensor(tau_v)
        I_true = float(torch.distributions.Exponential(tau).sample())
        bg = torch.tensor(float(10 ** torch.empty(1).uniform_(-1.7, 0.5)))
        c = torch.poisson(e * I_true + bg)
        a0, b0 = em_gamma(c, e, bg, tau, "geom", 300)
        grid = adaptive_grid(a0 / b0, a0.sqrt() / b0, 6000)

        we = grid_mass(exact_logpost(c, e, bg, tau, grid), grid)
        em_, ev_ = moments(we, grid)
        opt_se.append((em_ - I_true).item() ** 2)

        def cov(w):
            cd = torch.cumsum(w, 0)
            q = lambda p: grid[(cd >= p).nonzero()[0, 0]]
            return bool(q(.16) <= I_true <= q(.84)), bool(q(.025) <= I_true <= q(.975))

        for key, (mode, nit) in [
            (keys[0], ("arith", 3)),
            (keys[1], ("geom", 300)),
        ]:
            a, b = em_gamma(c, e, bg, tau, mode, nit)
            wg = grid_mass(gamma_logpdf(grid, a, b), grid)
            m, v = moments(wg, grid)
            c68, c95 = cov(wg)
            d = acc[key]
            d["rb"].append((m - I_true).item())
            d["se"].append((m - I_true).item() ** 2)
            d["z"].append(((m - I_true) / v.sqrt()).item())
            d["c68"].append(c68)
            d["c95"].append(c95)

        c68, c95 = cov(we)
        d = acc[keys[2]]
        d["rb"].append((em_ - I_true).item())
        d["se"].append((em_ - I_true).item() ** 2)
        d["z"].append(((em_ - I_true) / ev_.sqrt()).item())
        d["c68"].append(c68)
        d["c95"].append(c95)

    opt_rmse = st.mean(opt_se) ** 0.5
    print("\n=== 2. Accuracy + calibration vs TRUE intensity (known nuisances) ===")
    print(f"{'estimator':>23} | {'mean bias':>9} {'RMSE/opt':>9} | {'z-std':>6} {'68%cov':>7} {'95%cov':>7}")
    print("-" * 76)
    for k in keys:
        d = acc[k]
        rb = st.mean(d["rb"])
        rmse = st.mean(d["se"]) ** 0.5
        z = st.pstdev(d["z"])
        c68 = 100 * st.mean([1.0 if x else 0.0 for x in d["c68"]])
        c95 = 100 * st.mean([1.0 if x else 0.0 for x in d["c95"]])
        print(f"{k:>23} | {rb:>+9.3f} {rmse / opt_rmse:>9.3f} | {z:>6.2f} {c68:>6.1f}% {c95:>6.1f}%")
    print("  -> z-std=1 & 68/95 coverage = calibrated; RMSE/opt=1 = Bayes-optimal point accuracy.")


def study_nuisance(prof, n, seed, relstd=0.15, M=40):
    """3. Calibration under uncertain background: Fix A vs Fix A+B."""
    e = prof.clone()
    res = {"Fix A (condition on E[bg])": [], "Fix A+B (marginalize q(bg))": []}
    cov = {k: {"c68": [], "c95": []} for k in res}
    torch.manual_seed(seed + 1)
    for _ in range(n):
        tau = torch.tensor(float(10 ** torch.empty(1).uniform_(-1.3, 0.0)))
        I_true = float(torch.distributions.Exponential(tau).sample())
        bg0 = float(10 ** torch.empty(1).uniform_(-1.0, 0.4))
        k = 1.0 / relstd**2
        qbg = torch.distributions.Gamma(torch.tensor(k), torch.tensor(k / bg0))
        bg_true = float(qbg.sample())
        c = torch.poisson(e * I_true + bg_true)
        a0 = ALPHA_W + c.sum()
        hi = max((a0 / (tau + e.sum())).item() * 3 + 10 / tau.item(), 5.0)
        grid = torch.linspace(1e-6, hi, 8000)

        def zc(w, key):
            m, v = moments(w, grid)
            cd = torch.cumsum(w, 0)
            q = lambda p: grid[(cd >= p).nonzero()[0, 0]]
            res[key].append(((m - I_true) / v.sqrt()).item())
            cov[key]["c68"].append(bool(q(.16) <= I_true <= q(.84)))
            cov[key]["c95"].append(bool(q(.025) <= I_true <= q(.975)))

        zc(grid_mass(exact_logpost(c, e, torch.tensor(bg0), tau, grid), grid),
           "Fix A (condition on E[bg])")
        wmix = torch.zeros_like(grid)
        for bgs in qbg.sample((M,)):
            wmix = wmix + grid_mass(exact_logpost(c, e, bgs, tau, grid), grid)
        zc(wmix / M, "Fix A+B (marginalize q(bg))")

    print(f"\n=== 3. Calibration under {int(relstd*100)}% background uncertainty ===")
    print(f"{'estimator':>28} | {'z-std':>6} {'68%cov':>7} {'95%cov':>7}")
    print("-" * 56)
    for k in res:
        z = st.pstdev(res[k])
        c68 = 100 * st.mean([1.0 if x else 0.0 for x in cov[k]["c68"]])
        c95 = 100 * st.mean([1.0 if x else 0.0 for x in cov[k]["c95"]])
        print(f"{k:>28} | {z:>6.2f} {c68:>6.1f}% {c95:>6.1f}%")
    print("  -> Fix A alone is overconfident when nuisances are uncertain; A+B restores calibration.")


def study_grid_resolution(prof, n=400, seed=0):
    """4. Quadrature error vs n_grid (justifies the export-method default)."""
    e = prof.clone()
    print("\n=== 4. Quadrature grid resolution vs dense reference (G=12000) ===")
    print(f"{'n_grid':>7} | {'max |dMean|':>11} {'max |dVar|':>11}  (relative)")
    torch.manual_seed(seed + 2)
    cases = []
    for _ in range(n):
        tau = torch.tensor(float(10 ** torch.empty(1).uniform_(-1.3, 0.0)))
        I_true = float(torch.distributions.Exponential(tau).sample())
        bg = torch.tensor(float(10 ** torch.empty(1).uniform_(-1.7, 0.5)))
        cases.append((torch.poisson(e * I_true + bg), bg, tau))
    for G in [256, 512, 1024]:
        dm, dv = [], []
        for c, bg, tau in cases:
            a, b = em_gamma(c, e, bg, tau, "geom", 300)
            mf_m, mf_s = a / b, a.sqrt() / b
            gr = adaptive_grid(mf_m, mf_s, G)
            grd = adaptive_grid(mf_m, mf_s, 12000)
            m, v = moments(grid_mass(exact_logpost(c, e, bg, tau, gr), gr), gr)
            mr, vr = moments(grid_mass(exact_logpost(c, e, bg, tau, grd), grd), grd)
            dm.append(abs((m - mr) / mr).item())
            dv.append(abs((v - vr) / vr.clamp(min=1e-9)).item())
        print(f"{G:>7} | {max(dm):>11.2e} {max(dv):>11.2e}")
    print("  -> pick the smallest n_grid with <1% error for ConjugateIntegrator.exact_intensity_posterior.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4000, help="shoeboxes per study")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prof = make_profile()
    study_variance_vs_bg(prof)
    study_calibration(prof, args.n, args.seed)
    study_nuisance(prof, args.n, args.seed)
    study_grid_resolution(prof, n=min(args.n, 600), seed=args.seed)


if __name__ == "__main__":
    main()
