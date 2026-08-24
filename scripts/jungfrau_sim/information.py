"""Information budget: what does each modelling choice actually cost, in bits and in Fisher?

Two currencies, answering two different questions.

`fisher_*` -- Fisher information about `lam` per pixel. This is the estimation currency:
`Var(lam_hat) >= 1/J` by Cramer-Rao, so a route with 90% of the Fisher information gives
error bars ~5% wider, no matter how clever the estimator. Pure Poisson (`J = 1/lam`) is
the ceiling: it is what a hypothetical noiseless photon-counting detector would give.

`entropy_*` -- H(N | observation) in bits. This is the *recovery* currency: how much
uncertainty about the latent count survives. H(N|x) ~ 0 means x pins N down, so rounding
to an integer throws nothing away and pre-converting is free.

The two disagree in an instructive way, which is the point of computing both:
in G2 the count is NOT recoverable (H is large) yet rounding costs almost no Fisher
information, because at lam ~ 800 nobody cares about +-1.

All quantities are computed by direct numerical integration/summation rather than by
asymptotic argument, so they are checked against the closed forms in `selftest.py`.
"""

from __future__ import annotations

import math

import torch

import likelihoods as lk


def _grid(lam: float, sigma: float, n_sd: float = 12.0, n_pts: int = 20001):
    """Integration grid over x wide enough to hold the full Poisson+Gaussian mass."""
    sd = math.sqrt(lam + sigma**2)
    lo = lam - n_sd * sd - 5.0 * sigma
    hi = lam + n_sd * sd + 5.0 * sigma
    return torch.linspace(lo, hi, n_pts, dtype=torch.float64)


def fisher_exact(lam: float, sigma: float) -> float:
    """J(lam) for the true Poisson-Gaussian law, by numerical integration.

    J = E[(d/dlam log p(x|lam))^2], with the score taken by autograd on the exact
    log-density so there is no hand-derived derivative to get wrong.
    """
    x = _grid(lam, sigma)
    # One score per grid point: independent lam per point, so a single backward pass
    # through the sum gives the whole vector of d/dlam log p(x_i | lam).
    lam_v = torch.full_like(x, lam, requires_grad=True)
    logp = lk.exact(x, lam_v, sigma)
    (s,) = torch.autograd.grad(logp.sum(), lam_v)
    p = logp.detach().exp()
    return float(torch.trapezoid(p * s**2, x))


def fisher_poisson(lam: float) -> float:
    """J(lam) = 1/lam. The noiseless photon-counting ceiling."""
    return 1.0 / lam


def fisher_rounded_poisson(lam: float, sigma: float, n_max: int | None = None) -> float:
    """J(lam) for the *actual* distribution of round(clamp(x, 0)).

    Not `1/lam`: that is what the Poisson model *claims*. The rounded value is a
    genuinely different random variable, and this integrates its true pmf. Comparing
    this to `fisher_exact` isolates the cost of quantization alone.
    """
    if n_max is None:
        n_max = int(math.ceil(lam + 12.0 * math.sqrt(lam + 1.0) + 12.0 * sigma + 10.0))
    k = torch.arange(n_max + 1, dtype=torch.float64)

    lam_v = torch.tensor(lam, dtype=torch.float64, requires_grad=True)
    # P(round(clamp(x,0)) = k) = sum_n Poisson(n;lam) * [Phi(u_k) - Phi(l_k)] about n.
    n = torch.arange(n_max + 40, dtype=torch.float64)
    log_pois = n * torch.log(lam_v) - lam_v - torch.lgamma(n + 1.0)
    pois = log_pois.exp()

    def phi(z):
        return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

    # bin edges for k: [k-0.5, k+0.5), except k=0 which absorbs everything below 0.5
    hi = (k + 0.5).unsqueeze(1)
    lo = torch.where(k > 0, k - 0.5, torch.tensor(-float("inf"), dtype=torch.float64))
    lo = lo.unsqueeze(1)
    z_hi = (hi - n.unsqueeze(0)) / sigma
    z_lo = (lo - n.unsqueeze(0)) / sigma
    pk = ((phi(z_hi) - phi(z_lo)) * pois.unsqueeze(0)).sum(1)

    pk = pk / pk.sum()

    # J = sum_k (dp_k/dlam)^2 / p_k, the discrete form. One grad per k.
    grads = []
    for i in range(len(pk)):
        g = torch.autograd.grad(pk[i], lam_v, retain_graph=True)[0]
        grads.append(g)
    dp = torch.stack(grads)
    return float((dp**2 / pk.detach().clamp_min(1e-300)).sum())


def entropy_count_given_x(lam: float, sigma: float) -> float:
    """H(N | x) in bits: how much uncertainty about the latent count survives.

    Zero means x determines N, so rounding to an integer is a sufficient statistic and
    pre-converting the data loses nothing.
    """
    x = _grid(lam, sigma)
    n_max = int(math.ceil(lam + 12.0 * math.sqrt(lam + 1.0) + 12.0 * sigma + 10.0))
    n = torch.arange(n_max + 1, dtype=torch.float64)

    log_pois = n * math.log(lam) - lam - torch.lgamma(n + 1.0)
    # joint log p(x, n) on the grid
    lj = log_pois.unsqueeze(0) - 0.5 * (
        ((x.unsqueeze(1) - n.unsqueeze(0)) / sigma) ** 2 + math.log(2 * math.pi)
    ) - math.log(sigma)
    log_px = torch.logsumexp(lj, dim=1)
    log_post = lj - log_px.unsqueeze(1)  # p(n | x)
    post = log_post.exp()
    h_given_x = -(post * log_post / math.log(2.0)).nan_to_num(0.0).sum(1)
    px = log_px.exp()
    return float(torch.trapezoid(px * h_given_x, x))


def entropy_count(lam: float) -> float:
    """H(N) in bits for N ~ Poisson(lam), the information there is to lose."""
    n_max = int(math.ceil(lam + 14.0 * math.sqrt(lam + 1.0) + 20.0))
    n = torch.arange(n_max + 1, dtype=torch.float64)
    logp = n * math.log(lam) - lam - torch.lgamma(n + 1.0)
    p = logp.exp()
    p = p / p.sum()
    return float(-(p * torch.log2(p.clamp_min(1e-300))).sum())


def main() -> None:
    """Print the per-pixel information budget for a representative pixel in each stage."""
    from detector import JungfrauConfig

    torch.set_default_dtype(torch.float64)
    cfg = JungfrauConfig()
    cases = [
        ("G0", 0, 0.5), ("G0", 0, 5.0), ("G0", 0, 20.0),
        ("G1", 1, 40.0), ("G1", 1, 300.0),
        ("G2", 2, 1000.0), ("G2", 2, 5000.0),
    ]
    print("INFORMATION BUDGET per pixel\n")
    print(f"{'stage':>5} {'lam':>7} {'sigma':>7} | {'J_exact/J_pois':>15}"
          f" {'J_round/J_exact':>16} | {'H(N)':>7} {'H(N|x)':>8} {'count?':>8}")
    print("-" * 88)
    for name, g, lam in cases:
        s = cfg.sigma_read_photons(g)
        je, jp, jr = fisher_exact(lam, s), fisher_poisson(lam), fisher_rounded_poisson(lam, s)
        hn, hnx = entropy_count(lam), entropy_count_given_x(lam, s)
        rec = "yes" if hnx < 0.01 else ("partly" if hnx < 1.0 else "no")
        print(f"{name:>5} {lam:7.1f} {s:7.3f} | {je / jp:14.4f}  {jr / je:15.4f} "
              f" | {hn:7.3f} {hnx:8.3f} {rec:>8}")
    print()
    print("J_exact/J_pois  what the read noise costs vs a noiseless photon counter")
    print("J_round/J_exact what rounding costs on top of that -- ~1.0 everywhere")
    print("H(N|x)          bits of the latent count that are unrecoverable")
    print()
    print("The two currencies disagree on purpose. Rounding is ~free in Fisher terms in")
    print("every stage, yet the count is only *recoverable* in G0. Those are different")
    print("questions: pre-converting is safe everywhere, but 'the integers are the true")
    print("counts' is only true in G0, and only G0 may then assume Var = lam.")


if __name__ == "__main__":
    main()
