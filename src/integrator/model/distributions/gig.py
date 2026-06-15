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

Training needs only three GIG quantities, and NONE needs the Bessel
order-derivative (which E[log I_h] / the GIG entropy would require) -- those
cancel out of the ELBO (see `intensity_elbo`):

    E[I_h]   = sqrt(b/a) * K_{p+1}(w)/K_p(w)              (the merged estimate)
    E[1/I_h] = sqrt(a/b) * K_{p-1}(w)/K_p(w)              (feeds the per-obs update)
    ELBO_I   = log(2 * K_p(w)) - (p/2) * log(a/b)         (the per-HKL ELBO term)

with w = sqrt(a*b). Everything is built from r = K_{p-1}(w)/K_p(w) and
log K_p(w), evaluated by a custom autograd Function over scipy's exponentially
scaled Bessel `kve`; the w-derivative is supplied analytically from the
modified-Bessel recurrences, so the order p -- fixed, because nu is a
hyperparameter -- never needs a gradient.
"""

import math

import numpy as np
import torch
from scipy.special import kve
from torch import Tensor


class _KvRatioLogK(torch.autograd.Function):
    """Return (r, logKp) = (K_{p-1}(w)/K_p(w),  log K_p(w)), differentiable in w.

    p is a constant (nu is a fixed hyperparameter, so the GIG order
    p = alpha_W - nu*N_h carries no gradient). The w-derivatives come from the
    modified-Bessel recurrences K_v'(w) = -K_{v-1}(w) - (v/w) K_v(w) and
    K_{v-1}(w) - K_{v+1}(w) = -(2v/w) K_v(w):

        d/dw  r       = r^2 + ((2p - 1)/w) r - 1
        d/dw  logK_p  = -r - p/w

    The exponential scaling of `kve` (kve(v, w) = K_v(w) e^w) cancels in the
    ratio r and contributes a -w shift to log K_p, keeping both finite at large
    w / large |p|.
    """

    @staticmethod
    def forward(ctx, omega: Tensor, p: Tensor):
        om = omega.detach().cpu().double().numpy()
        pp = p.detach().cpu().double().numpy()
        kv_pm1 = kve(pp - 1.0, om)
        kv_p = kve(pp, om)
        r = kv_pm1 / kv_p
        log_kp = np.log(kv_p) - om
        r_t = torch.as_tensor(r, dtype=omega.dtype, device=omega.device)
        log_kp_t = torch.as_tensor(
            log_kp, dtype=omega.dtype, device=omega.device
        )
        ctx.save_for_backward(omega, p, r_t)
        return r_t, log_kp_t

    @staticmethod
    def backward(ctx, grad_r, grad_logk):
        omega, p, r = ctx.saved_tensors
        dr_dw = r * r + ((2.0 * p - 1.0) / omega) * r - 1.0
        dlogk_dw = -r - p / omega
        grad_omega = grad_r * dr_dw + grad_logk * dlogk_dw
        return grad_omega, None


def _kv_ratio_logk(omega: Tensor, p: Tensor) -> tuple[Tensor, Tensor]:
    return _KvRatioLogK.apply(omega, p)


def gig_moments(
    p: Tensor, a: Tensor, b: Tensor, eps: float = 1e-12
) -> tuple[Tensor, Tensor]:
    """E[I] and E[1/I] of GIG(p, a, b). Inputs broadcast; a, b > 0.

    Uses E[I^k] = (b/a)^{k/2} K_{p+k}(w)/K_p(w) with the recurrence
    K_{p+1}/K_p = K_{p-1}/K_p + 2p/w, so only the single ratio
    r = K_{p-1}(w)/K_p(w) is evaluated.
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
    log integral_0^inf I^{p-1} exp(-(a I + b/I)/2) dI. No E[log I_h], no GIG
    entropy, no Bessel order-derivative.
    """
    a = a.clamp(min=eps)
    b = b.clamp(min=eps)
    omega = (a * b).clamp(min=eps).sqrt()
    _, log_kp = _kv_ratio_logk(omega, p)
    return math.log(2.0) + log_kp - 0.5 * p * (a / b).log()
