"""The posterior of (I, b) given a profile, computed rather than approximated, and a Laplace q(eta) around it.

For one shoebox with valid-pixel counts c_k, mask m_k and profile p_k,

    c_k ~ Poisson(I p_k + b),  I ~ Gamma(a_I, r_I),  b ~ Gamma(a_b, r_b)

`collapsed` evaluates the joint posterior of (I, b) given p by putting I on a
grid of G cell midpoints over +-9 Schur standard deviations of a least-squares
start (floored at 0) and integrating b out by Laplace at each grid point
(`newton_b` steps for the mode b*(I)). It returns the normalised cell weights
of I, b*(I) and the Laplace sd of b | I, and the log evidence

    log Z(p) = log integral integral p(c | I, b, p) p(I) p(b) dI db

up to the Laplace step in b and the grid. Simulation-based calibration on the
mos010 prior SBC set found this posterior uniform in I and b (12 var(u) 1.03 and
1.00 with the amortized q(eta); 0.99 for the joint log-likelihood given the true
profile), where the conjugate q(I) of `svae` is 2.15: that update conditions on
the gate's split of the counts at its mean, so its variance is about I against
the Schur complement I + b n_eff (median ratio 7.1).

`laplace_eta` finds the joint mode of (eta, I, b) by damped Newton on the full
Hessian, from a start (normally the amortized q(eta) location and the conjugate
means), and returns the mode and its inverse Hessian; a full-covariance Gaussian
on eta from that block, with (I, b) | eta from `collapsed`, calibrated I, b,
the joint log-likelihood, n_eff and every eta component (12 var(u) 0.98-1.02).

All arrays are float64 inside.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Collapsed:
    xs: Tensor  # (B, G) cell midpoints of I
    w: Tensor  # (B, G) normalised posterior mass of each cell
    dx: Tensor  # (B,) cell width
    b_hat: Tensor  # (B, G) mode of b given I = xs
    sd_b: Tensor  # (B, G) Laplace sd of b given I = xs
    log_z: Tensor  # (B,) log evidence

    def mean_var_I(self) -> tuple[Tensor, Tensor]:
        m = (self.w * self.xs).sum(1)
        v = (self.w * self.xs**2).sum(1) - m**2 + self.dx**2 / 12
        return m, v

    def mean_var_b(self) -> tuple[Tensor, Tensor]:
        m = (self.w * self.b_hat).sum(1)
        v = (self.w * (self.sd_b**2 + self.b_hat**2)).sum(1) - m**2
        return m, v

    def cdf_I(self, x0: Tensor) -> Tensor:
        """CDF at x0 of the piecewise-constant density on the cells."""
        frac = ((x0[:, None] - (self.xs - self.dx[:, None] / 2)) / self.dx[:, None]).clamp(0, 1)
        return (self.w * frac).sum(1)

    def cdf_b(self, b0: Tensor) -> Tensor:
        z = (b0[:, None] - self.b_hat) / self.sd_b
        return (self.w * 0.5 * (1 + torch.erf(z / math.sqrt(2)))).sum(1)

    def sample(self, n: int, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        """n draws of (I, b) per row, (B, n) each."""
        idx = torch.multinomial(self.w, n, replacement=True, generator=generator)
        u = torch.rand(idx.shape, dtype=self.xs.dtype, device=self.xs.device, generator=generator)
        z = torch.randn(idx.shape, dtype=self.xs.dtype, device=self.xs.device, generator=generator)
        I = self.xs.gather(1, idx) + (u - 0.5) * self.dx[:, None]
        b = self.b_hat.gather(1, idx) + self.sd_b.gather(1, idx) * z
        return I.clamp_min(1e-9), b.clamp_min(1e-6)


def log_lik(c: Tensor, m: Tensor, I: Tensor, b: Tensor, p: Tensor) -> Tensor:
    """sum_k m_k (c_k log(I p_k + b) - (I p_k + b)), the Poisson log-likelihood without log c!; broadcasts I, b."""
    lam = I[..., None] * p + b[..., None]
    return (m * (c * torch.log(lam.clamp_min(1e-30)) - lam)).sum(-1)


def collapsed(c: Tensor, m: Tensor, p: Tensor, a_I: Tensor, r_I: Tensor, a_b: Tensor, r_b: Tensor,
              grid: int = 128, newton_b: int = 6, width: float = 9.0) -> Collapsed:
    """Grid posterior of I with b integrated by Laplace; c, m, p (B, P); priors (B,) each."""
    c, m, p = c.double(), m.double(), p.double()
    a_I, r_I, a_b, r_b = (torch.as_tensor(x, dtype=torch.float64, device=c.device).expand(c.shape[0])
                          for x in (a_I, r_I, a_b, r_b))
    n = m.sum(-1).clamp_min(1)
    b_s = (c * m).sum(-1) / n
    s2 = (p * p * m).sum(-1).clamp_min(1e-12)
    for _ in range(3):  # least-squares start of (I, b)
        i_s = ((p * (c - b_s[:, None]) * m).sum(-1) / s2).clamp_min(0)
        b_s = (((c - i_s[:, None] * p) * m).sum(-1) / n).clamp_min(0.05)
    lam = i_s[:, None] * p + b_s[:, None]
    fii, fib, fbb = (m * p * p / lam).sum(-1), (m * p / lam).sum(-1), (m / lam).sum(-1)
    sd = (1 / (fii - fib**2 / fbb).clamp_min(1e-12)).sqrt()
    lo = (i_s - width * sd).clamp_min(0)
    hi = i_s + width * sd + 3
    t = (torch.arange(grid, dtype=torch.float64, device=c.device) + 0.5) / grid
    xs = lo[:, None] + (hi - lo)[:, None] * t
    b = b_s[:, None].expand(-1, grid).clone()
    for _ in range(newton_b):
        lam = xs[:, :, None] * p[:, None, :] + b[:, :, None]
        g1 = (m[:, None] * (c[:, None] / lam - 1)).sum(-1) + (a_b[:, None] - 1) / b - r_b[:, None]
        g2 = -(m[:, None] * c[:, None] / lam**2).sum(-1) - (a_b[:, None] - 1) / b**2
        b = (b - g1 / g2).clamp_min(1e-3)
    lam = xs[:, :, None] * p[:, None, :] + b[:, :, None]
    g2 = -(m[:, None] * c[:, None] / lam**2).sum(-1) - (a_b[:, None] - 1) / b**2
    lp = ((m[:, None] * (c[:, None] * torch.log(lam) - lam)).sum(-1)
          + (a_b[:, None] - 1) * torch.log(b) - r_b[:, None] * b + a_b[:, None] * torch.log(r_b[:, None]) - torch.lgamma(a_b)[:, None]
          + (a_I[:, None] - 1) * torch.log(xs) - r_I[:, None] * xs + a_I[:, None] * torch.log(r_I[:, None]) - torch.lgamma(a_I)[:, None]
          + 0.5 * math.log(2 * math.pi) - 0.5 * torch.log(-g2))
    mx = lp.max(1, keepdim=True).values
    w = torch.exp(lp - mx)
    dx = (hi - lo) / grid
    log_z = mx[:, 0] + torch.log(w.sum(1) * dx)
    return Collapsed(xs, w / w.sum(1, keepdim=True), dx, b, (-1 / g2).sqrt(), log_z)


def laplace_eta(neg_log_post: Callable[[Tensor], Tensor], theta0: Tensor, steps: int = 40,
                positive: tuple[int, ...] = (), floor: float = 1e-3) -> tuple[Tensor, Tensor, Tensor]:
    """Damped-Newton mode of a batch of D-dim negative log posteriors and the inverse Hessian there.

    `neg_log_post` maps (B, D) to (B,). Components in `positive` are clamped at `floor`.
    Returns the mode (B, D), the covariance (B, D, D) from |eigenvalues| of the Hessian,
    and a flag (B,) for a non-positive-definite Hessian at the mode.
    """

    def grad_hess(th):
        with torch.enable_grad():
            th = th.detach().requires_grad_(True)
            f = neg_log_post(th)
            g, = torch.autograd.grad(f.sum(), th, create_graph=True)
            H = torch.stack([torch.autograd.grad(g[:, j].sum(), th, retain_graph=True)[0]
                             for j in range(th.shape[1])], 1)
        return f.detach(), g.detach(), H.detach()

    th = theta0.double()
    B, D = th.shape
    lam = torch.full((B,), 1e-2, dtype=torch.float64, device=th.device)
    f, g, H = grad_hess(th)
    for _ in range(steps):
        dH = torch.diagonal(H, dim1=1, dim2=2).abs().clamp_min(1e-8)
        A = H + lam[:, None, None] * torch.diag_embed(dH)
        ev, V = torch.linalg.eigh(A)
        A = (V * ev.abs().clamp_min(1e-8)[:, None]) @ V.transpose(1, 2)
        new = th - torch.linalg.solve(A, g[..., None])[..., 0]
        for j in positive:
            new[:, j] = new[:, j].clamp_min(floor)
        with torch.no_grad():
            fn = neg_log_post(new)
        ok = fn < f
        th = torch.where(ok[:, None], new, th)
        lam = torch.where(ok, lam / 3, lam * 5).clamp(1e-9, 1e8)
        f, g, H = grad_hess(th)
    ev, V = torch.linalg.eigh(H)
    cov = (V / ev.abs().clamp_min(1e-8)[:, None]) @ V.transpose(1, 2)
    return th, cov, ev[:, 0] <= 0


def laplace_collapsed_batch(qp, metadata: dict, counts: Tensor, mask: Tensor, raw_delta: Tensor, raw_scale: Tensor,
                            a_I: Tensor, r_I: Tensor, a_b: Tensor, r_b: Tensor, I_start: Tensor, b_start: Tensor,
                            n_eta: int = 16, grid: int = 128, steps: int = 40,
                            generator: torch.Generator | None = None, keep: bool = False,
                            eta_prior_scale: float = 1.0, newton_b: int = 6) -> dict:
    """Laplace q(eta) (full covariance, at the joint mode of (eta, I, b)) with (I, b) | eta collapsed.

    `qp` is a `streak_kmix_v2_profile` surrogate (float64 copy recommended) exposing
    `eta_prior`, `profile_from_eta`, `q_eta_from_raw` and `eta_sd`. The joint
    negative log posterior is -log p(c | I, b, eta) - log p(I) - log p(b) - log p(eta)
    with p(eta) = N(eta_mean, diag sigma^2); its mode is found from the amortized
    q(eta) location and (I_start, b_start). n_eta draws eta_s ~ N(mode_eta, cov_eta)
    each give a collapsed (I, b) posterior; the returned moments are those of the
    equal-weight mixture (law of total variance). With `keep`, the profiles and
    `Collapsed` objects of the draws are returned too. `eta_prior_scale` multiplies
    sigma (1 = the model's prior; 100 effectively removes the profile prior, so each
    reflection's shape comes from its own pixels). `newton_b` is the number of Newton
    steps to the conditional b-mode inside `collapsed` (6, the default, is unconverged
    on bright reflections: 0.8 % low in the mean at I ~ 770 on 49 pixels).
    """
    D = torch.float64
    c = counts.reshape(counts.shape[0], -1).to(D)
    m = mask.reshape(mask.shape[0], -1).to(D)
    B = c.shape[0]
    md = {k: (v.to(D) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in metadata.items()}
    phi, fxy, eta_mean = qp.eta_prior(md, m)
    loc, _ = qp.q_eta_from_raw(eta_mean, raw_delta.to(D), raw_scale.to(D))
    # eta_prior_scale > 1 widens the profile prior N(eta_mean, sigma^2); large values remove it
    sig = qp.eta_sd.detach().to(D) * float(eta_prior_scale)
    NE = eta_mean.shape[1]
    a_I, r_I, a_b, r_b = (torch.as_tensor(x, dtype=D, device=c.device).expand(B) for x in (a_I, r_I, a_b, r_b))

    def nlp(th):
        eta, I, b = th[:, :NE], th[:, NE], th[:, NE + 1]
        p = qp.profile_from_eta(phi, fxy, eta, m)
        return (-log_lik(c, m, I, b, p) + r_I * I - (a_I - 1) * torch.log(I) + r_b * b - (a_b - 1) * torch.log(b)
                + 0.5 * (((eta - eta_mean) / sig) ** 2).sum(-1))

    start = torch.cat([loc.detach(), I_start.to(D).reshape(B, 1).clamp_min(0.5), b_start.to(D).reshape(B, 1).clamp_min(0.05)], 1)
    mode, cov, npd = laplace_eta(nlp, start, steps=steps, positive=(NE, NE + 1))
    out = collapsed_mixture(qp, phi, fxy, c, m, mode[:, :NE], cov[:, :NE, :NE], a_I, r_I, a_b, r_b,
                            n_eta=n_eta, grid=grid, generator=generator, keep=keep, newton_b=newton_b)
    out["non_pd"] = npd
    return out


def collapsed_mixture(qp, phi: Tensor, fxy: Tensor, c: Tensor, m: Tensor, eta_loc: Tensor, eta_cov: Tensor,
                      a_I: Tensor, r_I: Tensor, a_b: Tensor, r_b: Tensor, n_eta: int = 16, grid: int = 128,
                      generator: torch.Generator | None = None, keep: bool = False, newton_b: int = 6) -> dict:
    """eta_s ~ N(eta_loc, eta_cov), s = 1..n_eta, each with its collapsed (I, b) posterior; mixture moments.

    `newton_b` is passed to `collapsed`: Newton steps to the b-mode at each grid point in I."""
    D = torch.float64
    c, m = c.to(D), m.to(D)
    B, NE = eta_loc.shape
    L = torch.linalg.cholesky(eta_cov.to(D) + 1e-10 * torch.eye(NE, dtype=D, device=c.device))
    z = torch.randn((n_eta, B, NE), dtype=D, device=c.device, generator=generator)
    etas = eta_loc.to(D)[None] + torch.einsum("bij,sbj->sbi", L, z)
    mI, vI, mb, vb, kept = [], [], [], [], []
    with torch.no_grad():
        for s in range(n_eta):
            p = qp.profile_from_eta(phi, fxy, etas[s], m)
            q = collapsed(c, m, p, a_I, r_I, a_b, r_b, grid=grid, newton_b=newton_b)
            a, v = q.mean_var_I()
            mI.append(a)
            vI.append(v)
            a, v = q.mean_var_b()
            mb.append(a)
            vb.append(v)
            if keep:
                kept.append((p, q))
    mI, vI, mb, vb = (torch.stack(x) for x in (mI, vI, mb, vb))
    out = {"i_mean": mI.mean(0), "i_var": vI.mean(0) + mI.var(0, unbiased=False),
           "b_mean": mb.mean(0), "b_var": vb.mean(0) + mb.var(0, unbiased=False),
           "eta_mode": eta_loc.to(D), "eta_cov": eta_cov.to(D), "eta_samples": etas, "phi": phi, "fxy": fxy}
    if keep:
        out["draws"] = kept
    return out


def collapsed_log_z(c: Tensor, m: Tensor, p: Tensor, a_I: Tensor, r_I: Tensor, a_b: Tensor, r_b: Tensor,
                    grid: int = 64, newton_b: int = 6, width: float = 9.0) -> tuple[Tensor, Collapsed]:
    """log Z(p), differentiable in p, and the (detached) `Collapsed` posterior.

    The grid range and the modes b*(I) are found without gradient; log Z is then
    evaluated with gradient at those fixed points. Since b* maximises the inner
    integrand, d log Z / d p through b* vanishes to first order (envelope theorem),
    so the gradient is correct to that order while the inner Newton steps keep no
    graph. Everything is float64.
    """
    with torch.no_grad():
        q = collapsed(c, m, p.detach(), a_I, r_I, a_b, r_b, grid=grid, newton_b=newton_b, width=width)
    c, m, p = c.double(), m.double(), p.double()
    B = c.shape[0]
    a_I, r_I, a_b, r_b = (torch.as_tensor(x, dtype=torch.float64, device=c.device).expand(B) for x in (a_I, r_I, a_b, r_b))
    xs, b = q.xs, q.b_hat
    lam = xs[:, :, None] * p[:, None, :] + b[:, :, None]
    g2 = -(m[:, None] * c[:, None] / lam**2).sum(-1) - (a_b[:, None] - 1) / b**2
    lp = ((m[:, None] * (c[:, None] * torch.log(lam) - lam)).sum(-1)
          + (a_b[:, None] - 1) * torch.log(b) - r_b[:, None] * b + a_b[:, None] * torch.log(r_b[:, None]) - torch.lgamma(a_b)[:, None]
          + (a_I[:, None] - 1) * torch.log(xs) - r_I[:, None] * xs + a_I[:, None] * torch.log(r_I[:, None]) - torch.lgamma(a_I)[:, None]
          + 0.5 * math.log(2 * math.pi) - 0.5 * torch.log(-g2))
    log_z = torch.logsumexp(lp, 1) + torch.log(q.dx)
    return log_z, q
