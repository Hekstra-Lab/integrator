"""The exact per-reflection marginal of the joint model, from pixels to the merged intensity J_h.

For pixel k of observation i of reflection h,

    c_ik ~ Poisson(g_i J_h p_ik(eta_i) + b_i),   b_i ~ Gamma(a_b, r_b),   eta_i ~ N(mu_i, diag sd_i^2),
    J_h ~ prior (Wilson acentric Gamma(1, 1/Sigma), centric Gamma(1/2, 1/(2 Sigma)), or flat),

with g_i the observation's scale. Background and shape are local to an observation, so

    p(pixels_h | g) = int p(J) prod_i l_i(g_i J) dJ,
    l_i(I) = int int p(c_i | I, b, eta) p(b) p(eta) db deta,

and l_i, the observation's pixel likelihood of its intensity with b and eta integrated
out, is everything the pixels say about J_h. Three pieces:

    `obs_loglik`     log int p(c | I, b, p) p(b) db at given intensities, b by Laplace
    `obs_loglik_is`  log l(I) with eta integrated by importance sampling from a Gaussian
                     q(eta) (normally the Laplace fit at the observation's own mode):
                     logmeanexp_s [log l(I | eta_s) + log p(eta_s) - log q(eta_s)]. The weights
                     p/q are what make the estimate hold at every I; equal weights under q
                     estimate int l(I | eta) q(eta) deta instead, which is the right mixture for
                     the observation's own posterior of I (`posterior.collapsed.collapsed_mixture`)
                     but not for l(I) away from that observation's mode
    `reflection_marginal`  the J grid of each reflection, sum_i log l_i(g_i J) by index_add,
                     the prior, and the log marginal with the normalised posterior weights of J

The J grid of a reflection is set from the inverse-variance combination of its
observations' least-squares intensities divided by their scales, +- `width` standard
deviations (floored at 0); its bounds are detached, so the gradient of the log marginal in
g, p and the priors is that of the quadrature at fixed nodes. The Laplace mode b* is
found without gradient and then passed through one Newton step with gradient, so it
carries db*/dI: the envelope theorem covers the mode term, but the Laplace curvature term
is not stationary in b, and dropping db*/dI put the gradient in g off by ~40% (tested).
All arrays are float64.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

D = torch.float64


def ls_intensity(c: Tensor, m: Tensor, p: Tensor, iters: int = 3) -> tuple[Tensor, Tensor, Tensor]:
    """Least-squares (I, b) for one profile per row and the Schur sd of I there; (B,) each."""
    n = m.sum(-1).clamp_min(1)
    b = (c * m).sum(-1) / n
    s2 = (p * p * m).sum(-1).clamp_min(1e-12)
    I = torch.zeros_like(b)
    for _ in range(iters):
        I = ((p * (c - b[:, None]) * m).sum(-1) / s2).clamp_min(0)
        b = (((c - I[:, None] * p) * m).sum(-1) / n).clamp_min(0.05)
    lam = I[:, None] * p + b[:, None]
    fii, fib, fbb = (m * p * p / lam).sum(-1), (m * p / lam).sum(-1), (m / lam).sum(-1)
    return I, b, (1 / (fii - fib**2 / fbb).clamp_min(1e-12)).sqrt()


def obs_loglik(c: Tensor, m: Tensor, p: Tensor, I: Tensor, a_b: Tensor, r_b: Tensor, newton_b: int = 8) -> Tensor:
    """log int p(c | I, b, p) Gamma(b; a_b, r_b) db, b by Laplace; c, m, p (B, P), I (B, G); returns (B, G).

    The Poisson term omits log c!, a constant of the data. Differentiable in I, p and the
    b prior, including through b*(I) (one Newton step with gradient after the search).
    """
    c, m, p, I = c.to(D), m.to(D), p.to(D), I.to(D)
    a_b = torch.as_tensor(a_b, dtype=D, device=c.device).expand(c.shape[0])[:, None]
    r_b = torch.as_tensor(r_b, dtype=D, device=c.device).expand(c.shape[0])[:, None]
    with torch.no_grad():
        n = m.sum(-1, keepdim=True).clamp_min(1)
        b = ((c * m).sum(-1, keepdim=True) / n - I.detach() * (p * m).sum(-1, keepdim=True) / n).clamp_min(0.05)
        for _ in range(newton_b):
            lam = I.detach()[:, :, None] * p[:, None, :] + b[:, :, None]
            g1 = (m[:, None] * (c[:, None] / lam - 1)).sum(-1) + (a_b - 1) / b - r_b
            g2 = -(m[:, None] * c[:, None] / lam**2).sum(-1) - (a_b - 1) / b**2
            b = (b - g1 / g2.clamp_max(-1e-12)).clamp_min(1e-4)
    # one more Newton step WITH gradient: at convergence it leaves b* unchanged and gives
    # db*/dI = -(d g1/dI) / g2, which the Laplace curvature term needs (it is not stationary in b)
    lam = I[:, :, None] * p[:, None, :] + b[:, :, None]
    g1 = (m[:, None] * (c[:, None] / lam - 1)).sum(-1) + (a_b - 1) / b - r_b
    g2 = -(m[:, None] * c[:, None] / lam**2).sum(-1) - (a_b - 1) / b**2
    b = (b - g1 / g2.clamp_max(-1e-12)).clamp_min(1e-4)
    lam = I[:, :, None] * p[:, None, :] + b[:, :, None]
    g2 = -(m[:, None] * c[:, None] / lam**2).sum(-1) - (a_b - 1) / b**2
    return ((m[:, None] * (c[:, None] * torch.log(lam.clamp_min(1e-300)) - lam)).sum(-1)
            + (a_b - 1) * torch.log(b) - r_b * b + a_b * torch.log(r_b) - torch.lgamma(a_b)
            + 0.5 * math.log(2 * math.pi) - 0.5 * torch.log((-g2).clamp_min(1e-300)))


def obs_loglik_is(c: Tensor, m: Tensor, profile: Callable[[Tensor], Tensor], eta_loc: Tensor, eta_cov: Tensor,
                  prior_mean: Tensor, prior_sd: Tensor, I: Tensor, a_b: Tensor, r_b: Tensor, n_eta: int = 16,
                  generator: torch.Generator | None = None, newton_b: int = 8) -> Tensor:
    """log l(I) = log int l(I | eta) N(eta; prior_mean, prior_sd^2) deta by importance sampling from N(eta_loc, eta_cov).

    `profile` maps eta (B, E) to the profile (B, P). Returns (B, G).
    """
    B, E = eta_loc.shape
    L = torch.linalg.cholesky(eta_cov.to(D) + 1e-10 * torch.eye(E, dtype=D, device=eta_loc.device))
    z = torch.randn((n_eta, B, E), dtype=D, device=eta_loc.device, generator=generator)
    etas = eta_loc.to(D)[None] + torch.einsum("bij,sbj->sbi", L, z)
    log_q = (-0.5 * (z**2).sum(-1) - torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1)[None]
             - 0.5 * E * math.log(2 * math.pi))
    sd = prior_sd.to(D).expand(B, E)
    log_p = (-0.5 * (((etas - prior_mean.to(D)[None]) / sd[None]) ** 2).sum(-1) - torch.log(sd).sum(-1)[None]
             - 0.5 * E * math.log(2 * math.pi))
    terms = torch.stack([obs_loglik(c, m, profile(etas[s]), I, a_b, r_b, newton_b) for s in range(n_eta)])
    terms = terms + (log_p - log_q)[:, :, None]
    return torch.logsumexp(terms, 0) - math.log(n_eta)


def log_prior_J(xs: Tensor, kind: str, sigma: Tensor | None = None, centric: Tensor | None = None) -> Tensor:
    """log density of J at the nodes xs (H, G): 'wilson' (acentric Exp(1/Sigma), centric Gamma(1/2, 1/(2 Sigma))) or 'flat'."""
    if kind == "flat":
        return torch.zeros_like(xs)
    if kind != "wilson":
        raise ValueError(f"prior kind must be wilson or flat, got {kind}")
    if sigma is None:
        raise ValueError("the wilson prior needs sigma")
    sig = sigma.to(D)[:, None]
    cen = (centric if centric is not None else torch.zeros(xs.shape[0], dtype=torch.bool, device=xs.device))[:, None]
    k = torch.where(cen, torch.full_like(sig, 0.5), torch.ones_like(sig))
    r = k / sig
    return k * torch.log(r) - torch.lgamma(k) + (k - 1) * torch.log(xs.clamp_min(1e-300)) - r * xs


def j_grid(i_hat: Tensor, sd_i: Tensor, g: Tensor, refl: Tensor, n_refl: int, grid: int = 128,
           width: float = 9.0) -> tuple[Tensor, Tensor]:
    """Cell midpoints (H, G) and widths (H,) of each reflection's J grid, from its observations' i_hat / g; detached."""
    with torch.no_grad():
        j = i_hat.to(D) / g.to(D)
        w = (g.to(D) / sd_i.to(D)) ** 2
        num = torch.zeros(n_refl, dtype=D, device=j.device).index_add_(0, refl, w * j)
        den = torch.zeros(n_refl, dtype=D, device=j.device).index_add_(0, refl, w)
        jh, sd = num / den.clamp_min(1e-300), den.clamp_min(1e-300).rsqrt()
        gmin = torch.full((n_refl,), float("inf"), dtype=D, device=j.device).scatter_reduce_(0, refl, g.to(D), "amin")
        lo = (jh - width * sd).clamp_min(0)
        hi = jh + width * sd + 3 / gmin
        t = (torch.arange(grid, dtype=D, device=j.device) + 0.5) / grid
        return lo[:, None] + (hi - lo)[:, None] * t, (hi - lo) / grid


def reflection_marginal(obs_ll: Tensor, refl: Tensor, n_refl: int, log_prior: Tensor, dx: Tensor) -> tuple[Tensor, Tensor]:
    """log int p(J) prod_i l_i(g_i J) dJ per reflection and the normalised posterior weights of J on its nodes.

    obs_ll (N, G) is log l_i at I = g_i * xs[refl[i]]; log_prior (H, G); dx (H,). Midpoint rule.
    """
    tot = torch.zeros(n_refl, obs_ll.shape[1], dtype=D, device=obs_ll.device).index_add(0, refl, obs_ll.to(D)) + log_prior
    log_m = torch.logsumexp(tot, 1) + torch.log(dx)
    return log_m, torch.softmax(tot.detach(), 1)
