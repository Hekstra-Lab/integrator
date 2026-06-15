"""Generalized Inverse Gaussian (GIG) -- the conjugate merged posterior for the
hierarchical Gamma-random-effect model.

    q(I_h) = GIG(p, a, b),   density  proportional to  I^{p-1} exp(-(a*I + b/I)/2),
    a, b > 0,  p in R.

It arises as the *exact* CAVI posterior for the per-HKL merged intensity I_h when
each observation's de-scaled intensity is a Gamma random effect centred on I_h
(J_i | I_h ~ Gamma(nu, nu/I_h)) under the Gamma (Wilson) prior I_h ~ Gamma(alpha_W,
tau_h):

    p = alpha_W - nu * N_h,    a = 2 * tau_h,    b = 2 * nu * sum_i E[J_i].

The Wilson Gamma prior is the b -> 0 special case (GIG -> Gamma) sitting inside
the GIG family, so this is exact conjugate inference, not an approximation.

The ELBO needs

    E[I_h]   = sqrt(b/a) * K_{p+1}(w)/K_p(w)              (the merged estimate)
    E[1/I_h] = sqrt(a/b) * K_{p-1}(w)/K_p(w)              (feeds the per-obs update)
    ELBO_I   = log(2 * K_p(w)) - (p/2) * log(a/b)         (the per-HKL ELBO term)

with w = sqrt(a*b), all built from r = K_{p-1}(w)/K_p(w) and log K_p(w), evaluated
by a custom autograd Function over scipy's exponentially scaled Bessel `kve`.

The Function differentiates w.r.t. BOTH the argument w (analytic recurrences) and
the ORDER p. The order-derivative `D_q := d/dp log K_q(w)` is a genuine
transcendental primitive (no closed form); it is supplied by a central finite
difference over an overflow-safe `log K`, both legs forced onto one branch (the
`exp(E[log I_h])`-style entropy term still cancels out of the ELBO, so D_p enters
ONLY through the order-dependence of the log-partition / ratios). When nu is a
fixed hyperparameter the order p has no gradient (`p.requires_grad is False`) and
the p-derivative path is skipped, so the fixed-nu cost is unchanged; when nu is
learnable, p = alpha_W - nu*N_h carries a gradient and D_p is computed.
"""

import math

import numpy as np
import torch
from scipy.special import kve
from torch import Tensor

_DP_FD_STEP = 1e-3  # central-FD step in the order p; validated sweet spot


def _log_besselk_uniform(p, w):
    """log K_p(w) via the DLMF 10.41 uniform large-order asymptotic (2 terms).

    Even in p; used ONLY where `kve` overflows to +inf (large |p|, small w). Its
    O(1e-3) absolute error is smooth in p, so it cancels in the central
    difference that defines D_p -- PROVIDED both FD legs use the same branch
    (see `_d_p_log_besselk`).
    """
    v = np.abs(p)
    z = w / v
    s = np.sqrt(1.0 + z * z)
    eta = s + np.log(z / (1.0 + s))
    log_lead = (
        0.5 * np.log(np.pi / (2.0 * v)) - v * eta - 0.25 * np.log(1.0 + z * z)
    )
    t = 1.0 / s
    u1 = (3.0 * t - 5.0 * t**3) / 24.0
    u2 = (81.0 * t**2 - 462.0 * t**4 + 385.0 * t**6) / 1152.0
    return log_lead + np.log(1.0 + u1 / v + u2 / (v * v))


def _log_besselk_on(p, w, use_exact, kv):
    """log K_p(w) on a CHOSEN branch: exact `log kve - w` where `use_exact`,
    else the uniform asymptotic. `kv` is the precomputed `kve(p, w)`."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        exact = np.log(np.where(use_exact, kv, 1.0)) - w
        p_as = np.where(np.abs(p) < 1e-9, 1e-9, p)  # avoid v=0 in the asymptotic
        asymp = _log_besselk_uniform(p_as, w)
    return np.where(use_exact, exact, asymp)


def _log_besselk(p, w):
    """Overflow-safe log K_p(w), vectorized float64 (per-element best branch).

    `kve(p, w)` overflows to +inf (the value, not the e^w scaling) for very large
    |p| at small w; there we fall back to the uniform asymptotic.
    """
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        kv = kve(p, w)
    use_exact = np.isfinite(kv) & (kv > 0.0)
    return _log_besselk_on(p, w, use_exact, kv)


def _d_p_log_besselk(p, w, h=_DP_FD_STEP):
    """D_p := d/dp log K_p(w). Central FD with BOTH legs on ONE branch.

    The seam fix: if one leg lands on the exact `kve` branch and the other on the
    asymptotic branch, the ~1e-3 asymptotic error does NOT cancel and injects an
    O(1) error in D_p. So the branch is decided once per element (exact only if
    BOTH legs' `kve` are finite) and applied to both legs. Validated to max
    relerr ~1e-6 over p in [-300, 1] (and finite, ~1e-7, far past that), w in
    [1e-2, 1e3], vs an mpmath dps=50 oracle. Odd in p; D_0 = 0 exactly.
    """
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    plo, phi = p - h, p + h
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        kv_lo = kve(plo, w)
        kv_hi = kve(phi, w)
    use_exact = (
        np.isfinite(kv_lo) & (kv_lo > 0.0) & np.isfinite(kv_hi) & (kv_hi > 0.0)
    )
    log_lo = _log_besselk_on(plo, w, use_exact, kv_lo)
    log_hi = _log_besselk_on(phi, w, use_exact, kv_hi)
    return (log_hi - log_lo) / (2.0 * h)


class _KvRatioLogK(torch.autograd.Function):
    """Return (r, logKp) = (K_{p-1}(w)/K_p(w), log K_p(w)), differentiable in
    BOTH w and the order p.

    w-derivatives (modified-Bessel recurrences):
        d/dw r       = r^2 + ((2p - 1)/w) r - 1
        d/dw logK_p  = -r - p/w
    p-derivatives (order-derivative primitive D_q = d/dp log K_q(w)):
        d/dp logK_p  = D_p
        d/dp r       = r * (D_{p-1} - D_p)     (since log r = logK_{p-1} - logK_p)
    The p-gradient path is taken ONLY when `p.requires_grad` (learnable nu); else
    the fixed-nu path adds no extra Bessel calls.
    """

    @staticmethod
    def forward(ctx, omega: Tensor, p: Tensor):
        om = omega.detach().cpu().double().numpy()
        pp = p.detach().cpu().double().numpy()
        log_kpm1 = _log_besselk(pp - 1.0, om)
        log_kp = _log_besselk(pp, om)
        # Overflow-safe ratio: never the raw `kve(p-1)/kve(p)`, whose numerator
        # and denominator overflow at orders one apart (silent finite/inf -> 0).
        r = np.exp(log_kpm1 - log_kp)

        ctx.need_p_grad = bool(p.requires_grad)
        r_t = torch.as_tensor(r, dtype=omega.dtype, device=omega.device)
        log_kp_t = torch.as_tensor(
            log_kp, dtype=omega.dtype, device=omega.device
        )

        if ctx.need_p_grad:
            d_p = _d_p_log_besselk(pp, om)
            d_pm1 = _d_p_log_besselk(pp - 1.0, om)
            dr_dp = r * (d_pm1 - d_p)
            dlogk_dp = d_p
            dr_dp_t = torch.as_tensor(
                dr_dp, dtype=omega.dtype, device=omega.device
            )
            dlogk_dp_t = torch.as_tensor(
                dlogk_dp, dtype=omega.dtype, device=omega.device
            )
        else:
            dr_dp_t = torch.zeros_like(r_t)
            dlogk_dp_t = torch.zeros_like(r_t)

        ctx.save_for_backward(omega, p, r_t, dr_dp_t, dlogk_dp_t)
        return r_t, log_kp_t

    @staticmethod
    def backward(ctx, grad_r, grad_logk):
        omega, p, r, dr_dp, dlogk_dp = ctx.saved_tensors
        dr_dw = r * r + ((2.0 * p - 1.0) / omega) * r - 1.0
        dlogk_dw = -r - p / omega
        grad_omega = grad_r * dr_dw + grad_logk * dlogk_dw
        if ctx.need_p_grad:
            grad_p = grad_r * dr_dp + grad_logk * dlogk_dp
        else:
            grad_p = None
        return grad_omega, grad_p


def _kv_ratio_logk(omega: Tensor, p: Tensor) -> tuple[Tensor, Tensor]:
    return _KvRatioLogK.apply(omega, p)


def gig_moments(
    p: Tensor, a: Tensor, b: Tensor, eps: float = 1e-12
) -> tuple[Tensor, Tensor]:
    """E[I] and E[1/I] of GIG(p, a, b). Inputs broadcast; a, b > 0.

    Uses E[I^k] = (b/a)^{k/2} K_{p+k}(w)/K_p(w) with the recurrence
    K_{p+1}/K_p = K_{p-1}/K_p + 2p/w, so only the single ratio
    r = K_{p-1}(w)/K_p(w) is evaluated. Differentiable in p (via `_kv_ratio_logk`)
    when p carries a gradient, e.g. p = alpha_W - nu*N_h with learnable nu.
    """
    a = a.clamp(min=eps)
    b = b.clamp(min=eps)
    omega = (a * b).clamp(min=eps).sqrt()
    r, _ = _kv_ratio_logk(omega, p)
    r_plus = r + 2.0 * p / omega  # K_{p+1}/K_p
    sqrt_b_a = (b / a).sqrt()
    e_i = sqrt_b_a * r_plus
    e_inv_i = r / sqrt_b_a  # sqrt(a/b) * r
    return e_i, e_inv_i


def gig_intensity_elbo(
    p: Tensor, a: Tensor, b: Tensor, eps: float = 1e-12
) -> Tensor:
    """Per-HKL ELBO contribution of the I_h node: log(2 K_p(w)) - (p/2) log(a/b).

    This is the combined p(I_h) + sum_i p(J_i|I_h)_{I-part} - q(I_h) term after
    the exact conjugate cancellation, i.e. the log-partition
    log integral_0^inf I^{p-1} exp(-(a I + b/I)/2) dI -- no E[log I_h], no GIG
    entropy. Differentiable in p (d/dp = D_p - 0.5 log(a/b)); with learnable nu
    the order-derivative enters here through p = alpha_W - nu*N_h.
    """
    a = a.clamp(min=eps)
    b = b.clamp(min=eps)
    omega = (a * b).clamp(min=eps).sqrt()
    _, log_kp = _kv_ratio_logk(omega, p)
    return math.log(2.0) + log_kp - 0.5 * p * (a / b).log()


def gig_mean_var(
    p: Tensor, a: Tensor, b: Tensor, eps: float = 1e-12
) -> tuple[Tensor, Tensor]:
    """E[I] and Var[I] of GIG(p, a, b) -- for merged-intensity export.

    Var[I] = E[I^2] - E[I]^2 with E[I^2] = (b/a) K_{p+2}(w)/K_p(w), expanded via
    the recurrence K_{p+2}/K_p = 1 + 2(p+1)(K_{p+1}/K_p)/w. Reuses the single
    ratio r = K_{p-1}/K_p (and r_plus = K_{p+1}/K_p = r + 2p/w), so no extra
    Bessel evaluation beyond gig_moments.
    """
    a = a.clamp(min=eps)
    b = b.clamp(min=eps)
    omega = (a * b).clamp(min=eps).sqrt()
    r, _ = _kv_ratio_logk(omega, p)
    r_plus = r + 2.0 * p / omega  # K_{p+1}/K_p
    b_a = b / a
    e_i = b_a.sqrt() * r_plus
    e_i2 = b_a * (1.0 + 2.0 * (p + 1.0) * r_plus / omega)  # (b/a) K_{p+2}/K_p
    var_i = (e_i2 - e_i.pow(2)).clamp(min=0.0)
    return e_i, var_i
