"""Direct profile fitting: closed-form intensities, four global shape parameters.

The model is unchanged -- `c_i ~ Poisson(I p_i + B)` over a shoebox -- but nothing
here is amortized by a network and nothing is optimized by SGD over the dataset.

Two facts measured on hewl_1118 motivate it. `PhysicalProfileSurrogate` already
ignores its encoder input: the profile is computed from `(d, lambda, x, y)` and
four global scalars, so the profile head needs no network. And the exported
posterior mean satisfies `corr(E_q[I], I_rb) = 1.0000` with median ratio
0.999--1.009 on all four arms, so the intensity head reproduces a closed form.
What is left for the encoder is the background, and for a flat background that is
one more column in the same linear solve.

So the fit is a two-level problem, and both levels are cheap:

  inner  for a fixed profile, `log p(c | I, B)` is concave in `(I, B)` jointly --
         the rate is linear in them -- so a handful of IRLS passes reach the
         unique optimum, in closed form, per reflection, vectorized.
  outer  four global shape parameters, fitted by maximizing the profiled
         likelihood. By the envelope theorem its gradient is the partial
         derivative at the inner optimum, so the inner solve does not need to be
         differentiated through.

The variance is the Schur complement of the same `(I, B)` information that the
inner solve already forms, which is the quantity `rao_blackwellized_intensity`
exports; see `docs/rb_export_2026-11-12.md`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

_logger = logging.getLogger(__name__)

__all__ = [
    "DirectFitResult",
    "background_basis",
    "fit_globals",
    "profiled_loglik",
    "solve_profile_fit",
]


@dataclass
class DirectFitResult:
    """Per-reflection outputs of one direct fit."""

    intensity: Tensor
    variance: Tensor
    background: Tensor
    background_variance: Tensor
    n_eff: Tensor
    extras: dict[str, Tensor] = field(default_factory=dict)


def background_basis(
    shape: tuple[int, int, int], kind: str, device, dtype
) -> Tensor:
    """Background design columns over the shoebox grid, shape `(K, P)`.

    `flat` is the one column the current model has, a scalar background per
    shoebox. `plane` adds the two centred pixel coordinates, which is the tilted
    plane the per-reflection ring was meant to approximate. Both are linear in
    their coefficients, so either keeps the inner problem concave and closed form.
    """
    d, h, w = (int(v) for v in shape)
    _, yy, xx = torch.meshgrid(
        torch.arange(d, device=device, dtype=dtype) - (d - 1) / 2,
        torch.arange(h, device=device, dtype=dtype) - (h - 1) / 2,
        torch.arange(w, device=device, dtype=dtype) - (w - 1) / 2,
        indexing="ij",
    )
    one = torch.ones(d * h * w, device=device, dtype=dtype)
    if kind == "flat":
        return one.unsqueeze(0)
    if kind == "plane":
        # scaled to O(1) so the normal equations stay well conditioned
        sy = max(h - 1, 1) / 2.0
        sx = max(w - 1, 1) / 2.0
        return torch.stack([one, yy.reshape(-1) / sy, xx.reshape(-1) / sx], 0)
    raise ValueError(f"unknown background basis {kind!r}; use 'flat' or 'plane'")


def _masked_median(counts: Tensor, m: Tensor) -> Tensor:
    """Median over the VALID pixels, shape `(B, 1)`.

    Multiplying by the mask first turns every masked pixel into a zero that then
    enters the median; on shoeboxes more than half masked that sends the seed to
    exactly zero. `tests/test_svae_responsibility.py` pins this.
    """
    big = counts.masked_fill(m <= 0, float("nan"))
    med = big.nanmedian(dim=-1, keepdim=True).values
    return torch.nan_to_num(med, nan=0.0)


def solve_profile_fit(
    counts: Tensor,
    mask: Tensor,
    profile: Tensor,
    bg_cols: Tensor | None = None,
    n_iter: int = 5,
    eps: float = 1e-6,
    ridge: float = 1e-6,
    shift_cols: Tensor | None = None,
) -> DirectFitResult:
    """Jointly fit `I` and the background coefficients by Poisson IRLS.

    For the identity link the IRLS weights are `1/lambda` and the working response
    is `c` itself, so each pass is a weighted least squares of the counts on the
    design `[p, bg_cols]`. The normal matrix it forms is the Fisher information,
    so the variance comes out of the same solve:

        Var(I) = [F^-1]_{00},

    which for a flat background is the Schur complement
    `(F_II - F_Ib^2 / F_bb)^-1`. Charging nothing for the background instead --
    `1/F_II` -- understates the variance by `1 - rho^2`.

    `I` is allowed to come out negative, which is the honest estimate for a null
    reflection, but the rate it is evaluated at is floored: a negative rate is
    impossible and lets the peak pixel carry weight `1/eps`.

    Args:
        counts: Observed counts, shape `(B, P)`.
        mask: Valid-pixel mask, shape `(B, P)`.
        profile: Normalized profile, shape `(B, P)`.
        bg_cols: Background design, shape `(K, P)`; flat if omitted.
        n_iter: IRLS passes. The weights need an intensity to exist, so pass one
            is a summation estimate.
        eps: Rate floor.
        ridge: Damping, as a fraction of the mean diagonal of the normal matrix.
            It is relative rather than absolute because the information scales
            with the background: at `f_II ~ 3e-4` an absolute `1e-8` is a 3e-5
            perturbation of the variance, large enough to show up against the
            Schur form it is supposed to reproduce. Only bites on shoeboxes
            whose valid support cannot separate the columns.
        shift_cols: Optional zero-sum columns, shape `(B, S, P)`, carried in the
            design between the profile and the background. With
            `PhysicalProfileSurrogate.profile_and_grad`'s two derivatives this
            solves for the reflection's centre as well as its intensity: the
            coefficient of column `s` is `alpha_s = I * delta_s`, so the shift
            is `alpha / I`. Each column must sum to zero over the valid pixels,
            which the derivative of a normalized profile does by construction;
            a column with a non-zero sum trades against `I` instead of against
            the centre, and the intensity stops being identified. Omitted, the
            solve is bit-identical to the two-column one.

    Returns:
        A `DirectFitResult` with `(B,)` intensity, variance, background level,
        background variance and `n_eff`. With `shift_cols`, `extras` carries
        `shift` and `shift_var` of shape `(B, S)`.
    """
    b, p_pix = counts.shape[0], counts.shape[-1]
    dtype = profile.dtype
    m = mask.reshape(b, -1).to(dtype)
    c = counts.reshape(b, -1).to(dtype) * m
    pi = profile.reshape(b, -1) * m
    pi = pi / pi.sum(-1, keepdim=True).clamp_min(eps)

    if bg_cols is None:
        bg_cols = torch.ones(1, p_pix, device=counts.device, dtype=dtype)
    k = bg_cols.shape[0]
    # Zero-sum columns, one per shift direction, carried between the profile and
    # the background so that index 0 stays the intensity. Their coefficient is
    # `alpha = I * delta` for a shift `delta`, because `I p(mu + delta)` expands
    # to `I p + I delta . dp/dmu`.
    n_shift = 0
    blocks = [pi.unsqueeze(1)]
    if shift_cols is not None:
        n_shift = int(shift_cols.shape[1])
        blocks.append(shift_cols.reshape(b, n_shift, -1) * m.unsqueeze(1))
    blocks.append(bg_cols.unsqueeze(0).expand(b, -1, -1) * m.unsqueeze(1))
    design = torch.cat(blocks, dim=1)

    b0 = _masked_median(counts.reshape(b, -1).to(dtype), m)
    i_hat = ((c - b0 * m) * m).sum(-1)
    coef = torch.zeros(b, k + n_shift, device=counts.device, dtype=dtype)
    coef[:, n_shift] = b0.squeeze(-1)

    eye = torch.eye(
        k + n_shift + 1, device=counts.device, dtype=dtype
    ).unsqueeze(0)
    info = eye.expand(b, -1, -1).clone()

    def _scaled(a: Tensor) -> tuple[Tensor, Tensor]:
        """Jacobi-scale the normal matrix to a unit diagonal, with its scaling.

        The columns differ by five orders of magnitude -- the profile sums to one
        over 625 pixels while the background column is all ones, giving
        `F_II ~ 3e-4` against `F_bb ~ 27` -- so a ridge expressed against the raw
        diagonal is either meaningless on one column or a 1e-3 perturbation on
        the other. On the unit diagonal `ridge` is a true relative damping, and
        the scaling also keeps the 2x2 solve honest in float32.
        """
        dg = a.diagonal(dim1=-2, dim2=-1)
        # A column can be exactly zero -- with the `plane` basis, a shoebox whose
        # valid pixels all lie on one row or column carries no information about
        # that coordinate -- and `rsqrt(0)` would scale the matrix to NaN rather
        # than leaving it merely singular. Flooring the diagonal relative to the
        # largest entry keeps the scaling finite, and the ridge then drives the
        # unidentified coefficient to zero instead of failing the solve.
        floor = dg.amax(-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
        s = dg.clamp_min(floor * 1e-10).rsqrt()
        return s.unsqueeze(-1) * a * s.unsqueeze(-2) + ridge * eye, s

    def _solve(a: Tensor, rhs: Tensor) -> Tensor:
        a_s, s = _scaled(a)
        y = torch.linalg.solve(a_s, (s * rhs).unsqueeze(-1)).squeeze(-1)
        return s * y

    def _inverse(a: Tensor) -> Tensor:
        a_s, s = _scaled(a)
        return s.unsqueeze(-1) * torch.linalg.inv(a_s) * s.unsqueeze(-2)

    for _ in range(max(1, n_iter)):
        bg = torch.einsum("bk,bkp->bp", coef, design[:, 1:])
        rate = (i_hat.clamp_min(0.0).unsqueeze(-1) * pi + bg).clamp_min(eps)
        w = m / rate
        # A = X diag(w) X^T, rhs = X diag(w) c
        xw = design * w.unsqueeze(1)
        info = torch.einsum("bip,bjp->bij", xw, design)
        rhs = torch.einsum("bip,bp->bi", xw, c)
        sol = _solve(info, rhs)
        i_hat = sol[:, 0]
        coef = sol[:, 1:]

    # The information must be evaluated at the CONVERGED fit, not at the rate the
    # last IRLS pass was built from: that lag leaves Var(I) about 10% high on a
    # background-dominated shoebox, which is the size of the effect being exported.
    bg = torch.einsum("bk,bkp->bp", coef, design[:, 1:])
    rate = (i_hat.clamp_min(0.0).unsqueeze(-1) * pi + bg).clamp_min(eps)
    xw = design * (m / rate).unsqueeze(1)
    info = torch.einsum("bip,bjp->bij", xw, design)

    cov = _inverse(info)
    var_i = cov[:, 0, 0].clamp_min(eps)
    var_b = cov[:, 1 + n_shift, 1 + n_shift].clamp_min(eps)
    n_eff = 1.0 / pi.pow(2).sum(-1).clamp_min(torch.finfo(dtype).tiny)

    extras = {"profile_sum": pi.sum(-1), "valid_pixels": m.sum(-1)}
    if n_shift:
        # alpha = I * delta, so the shift and its variance both divide by I.
        # A near-zero intensity carries no information about where its spot is,
        # and the division would invent a large shift from a small alpha, so the
        # denominator is floored and the variance is left large for the shrink
        # to act on rather than being clamped quietly.
        alpha = coef[:, :n_shift]
        var_alpha = torch.stack(
            [cov[:, 1 + s, 1 + s].clamp_min(eps) for s in range(n_shift)], -1
        )
        denom = i_hat.abs().clamp_min(eps).unsqueeze(-1)
        extras["shift"] = alpha / denom
        extras["shift_var"] = var_alpha / denom.pow(2)
        extras["shift_alpha"] = alpha
        extras["shift_alpha_var"] = var_alpha

    return DirectFitResult(
        intensity=i_hat,
        variance=var_i,
        background=coef[:, n_shift],
        background_variance=var_b,
        n_eff=n_eff,
        extras=extras,
    )


def shrink_shift(
    delta: Tensor,
    var_delta: Tensor,
    bin_index: Tensor | None = None,
    n_bins: int = 1,
    tau_scale: float | None = None,
    tau_prob: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Shrink a per-reflection centre shift toward the predicted centre.

    Each direction gets its own scale, because the two are different physics:
    a radial error comes from the wavelength assignment and the detector
    distance, a tangential one from the crystal orientation, and on Laue data
    there is no reason for them to be the same size.

    The estimate `delta = alpha / I` is noisy exactly where it is least
    informative -- a weak reflection's first moment is dominated by background
    fluctuation -- so letting it move the profile unshrunk would scatter weak
    spots off their predicted positions and bias their intensities low, which is
    the failure the recentring is supposed to fix. Empirical Bayes on
    `delta ~ N(0, tau^2)` does the right thing automatically: a bright spot keeps
    its measured shift, a weak one is held at the prediction.

    Reuses `integrator.fit.merge.update_tau` and `shrink_difference`, which fit
    the same marginal and carry the two-pass warning that matters here as well
    -- the scale must be read off the UNSHRUNK estimate.

    Args:
        delta: Shift estimate, shape `(B, S)`.
        var_delta: Its variance, same shape.
        bin_index: Group label per reflection for the scale, shape `(B,)`;
            a single group when omitted. Binning by image is the motivated
            choice, since centroid error drifts with the crystal.
        n_bins: Number of groups.
        tau_scale: Penalised-complexity scale in pixels; `None` is plain
            empirical Bayes.
        tau_prob: Tail probability for `tau_scale`.

    Returns:
        `(delta_shrunk, weight)`, both shape `(B, S)`. The posterior variance of
        the shift is `weight * var_delta`; `shrink_difference` returns that
        rather than the weight itself, so the weight is formed here.
    """
    from integrator.fit.merge import update_tau

    n_refl, n_dir = delta.shape
    if bin_index is None:
        bin_index = torch.zeros(n_refl, dtype=torch.long, device=delta.device)
        n_bins = 1
    out = torch.zeros_like(delta)
    wt = torch.zeros_like(delta)
    for j in range(n_dir):
        d_j = delta[:, j].contiguous()
        v_j = var_delta[:, j].contiguous()
        tau = update_tau(
            d_j, v_j, bin_index, n_bins, tau_scale=tau_scale, tau_prob=tau_prob
        )
        t2 = tau[bin_index].pow(2)
        w = t2 / (t2 + v_j).clamp_min(1e-12)
        w = torch.where(torch.isfinite(v_j), w, torch.zeros_like(w))
        out[:, j] = w * torch.nan_to_num(d_j, nan=0.0, posinf=0.0, neginf=0.0)
        wt[:, j] = w
    return out, wt


def solve_profile_fit_recentered(
    counts: Tensor,
    mask: Tensor,
    profile_fn,
    bg_cols: Tensor | None = None,
    n_iter: int = 5,
    n_outer: int = 3,
    shrink: bool = True,
    max_shift: float | Sequence[float] = 3.0,
    bin_index: Tensor | None = None,
    n_bins: int = 1,
    tau_scale: float | None = None,
    eps: float = 1e-6,
    ridge: float = 1e-6,
) -> tuple[DirectFitResult, Tensor]:
    """Fit intensity, background and the reflection's centre together.

    The profile family is unchanged -- the same oriented Gaussian with the same
    four global widths -- but its centre is no longer pinned to the middle of
    the shoebox. Expanding `I p(mu + delta)` to first order in `delta` makes the
    two centre derivatives ordinary columns of the design, so each outer pass is
    the same weighted least squares and the centre falls out of it rather than
    being searched for. The expansion is local, so the step is iterated: three
    passes move a spot several sigma if the data ask for it.

    Why this is not the encoder's per-reflection freedom: the derivative columns
    sum to zero over the valid pixels, so they span the two translation
    directions and nothing else. They cannot move total mass between the profile
    and the uniform direction, which is the degeneracy
    `p' = (f/f')p + (1-f/f')u` that leaves `I` unidentified when a network is
    free to reshape the profile per reflection.

    Args:
        counts: Observed counts, shape `(B, P)`.
        mask: Valid-pixel mask, shape `(B, P)`.
        profile_fn: Callable taking an offset of shape `(B, S)` and returning
            `(profile, grad)` of shapes `(B, P)` and `(B, S, P)`, which is
            `PhysicalProfileSurrogate.profile_and_grad` bound to a batch.
        bg_cols: Background design, shape `(K, P)`; flat if omitted.
        n_iter: IRLS passes inside each outer pass.
        n_outer: Outer passes. One is a single linearized step.
        shrink: Shrink the shift toward the predicted centre by empirical
            Bayes between outer passes. Off, every weak reflection is free to
            chase its own background fluctuation.
        max_shift: Clamp on the accumulated shift, in pixels. A spot that wants
            to move further than this is more likely a neighbour in the box
            than a mis-predicted centroid. A sequence gives one clamp per
            direction, which is what a rotation series needs: a 3-frame
            shoebox has half a frame of room along the rocking axis against ten
            pixels in the detector plane, so a single scalar is either useless
            in one direction or lets the profile leave the box in the other.
        bin_index: Group label for the shrinkage scale, shape `(B,)`.
        n_bins: Number of groups.
        tau_scale: Penalised-complexity scale in pixels, or `None` for plain
            empirical Bayes.
        eps: Rate floor.
        ridge: Relative damping.

    Returns:
        `(result, mu)` where `mu` is the fitted centre, shape `(B, S)`, in
        `(radial, tangential)` pixels. The returned `result` is the solve at
        that centre with the derivative columns still in the design, so
        `Var(I)` is charged for having located the spot rather than
        conditioning on the centre being right. That charge is conservative
        when `shrink` is on, which costs less than two parameters.
    """
    mu = None
    cap: Tensor | None = None
    for _ in range(max(1, n_outer)):
        profile, grad = profile_fn(mu)
        res = solve_profile_fit(
            counts, mask, profile, bg_cols, n_iter=n_iter, eps=eps,
            ridge=ridge, shift_cols=grad,
        )
        delta = res.extras["shift"]
        var_delta = res.extras["shift_var"]
        if shrink:
            delta, _ = shrink_shift(
                delta, var_delta, bin_index, n_bins, tau_scale=tau_scale
            )
        if mu is None:
            mu = torch.zeros_like(delta)
        if cap is None:
            if isinstance(max_shift, (int, float)):
                cap = torch.full_like(delta[:1], float(max_shift))
            else:
                cap = torch.tensor(
                    list(max_shift), dtype=delta.dtype, device=delta.device
                ).reshape(1, -1)
                if cap.shape[1] != delta.shape[1]:
                    raise ValueError(
                        f"max_shift has {cap.shape[1]} entries but the profile "
                        f"exposes {delta.shape[1]} shift directions"
                    )
        mu = (mu + delta).clamp(-cap, cap)

    profile, grad = profile_fn(mu)
    res = solve_profile_fit(
        counts, mask, profile, bg_cols, n_iter=n_iter, eps=eps, ridge=ridge,
        shift_cols=grad,
    )
    res.extras["mu"] = mu
    return res, mu


def profiled_loglik(
    counts: Tensor,
    mask: Tensor,
    profile: Tensor,
    intensity: Tensor,
    background: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Poisson log-likelihood at a given `(I, B)`, dropping `log c!`, shape `(B,)`.

    Only the profile carries gradient here. By the envelope theorem the derivative
    of the profiled likelihood with respect to the shape parameters equals this
    partial derivative evaluated at the inner optimum, so `intensity` and
    `background` enter as constants and the IRLS solve is never differentiated.
    """
    b = counts.shape[0]
    m = mask.reshape(b, -1).to(profile.dtype)
    c = counts.reshape(b, -1).to(profile.dtype) * m
    pi = profile.reshape(b, -1) * m
    pi = pi / pi.sum(-1, keepdim=True).clamp_min(eps)
    rate = (
        intensity.clamp_min(0.0).unsqueeze(-1) * pi + background.unsqueeze(-1)
    ).clamp_min(eps)
    return (m * (c * rate.log() - rate)).sum(-1)


def fit_globals(
    surrogate,
    batches: list[tuple[Tensor, Tensor, dict]],
    bg_kind: str = "flat",
    max_iter: int = 60,
    n_iter_inner: int = 5,
    log_every: int = 10,
) -> dict[str, float]:
    """Fit the profile's global shape parameters by maximum likelihood.

    `batches` is a held list of `(counts, mask, metadata)` already on the target
    device, so the objective is a fixed function rather than a stochastic one and
    L-BFGS can be used. The parameters are the four logs the surrogate carries, so
    the fit is over four numbers however large the dataset is.

    Returns the fitted values in px^2 and the per-reflection log-likelihood before
    and after, which is the only evidence that the fit did anything.
    """
    params = [p for p in surrogate.parameters() if p.requires_grad]
    if not params:
        raise ValueError(
            "the profile surrogate exposes no fittable parameters; "
            "set learn_globals=True"
        )
    shape = tuple(surrogate.profile_shape)
    dev = params[0].device
    bg_cols = background_basis(shape, bg_kind, dev, torch.float32)
    n_refl = sum(cnt.shape[0] for cnt, _, _ in batches)

    def closure_value() -> Tensor:
        total = 0.0
        for cnt, msk, md in batches:
            prof = surrogate._profile(md, msk)
            with torch.no_grad():
                res = solve_profile_fit(
                    cnt, msk, prof.detach(), bg_cols, n_iter=n_iter_inner
                )
            total = total + profiled_loglik(
                cnt, msk, prof, res.intensity, res.background
            ).sum()
        return -total / n_refl

    with torch.no_grad():
        before = float(-closure_value())

    opt = torch.optim.LBFGS(
        params,
        max_iter=max_iter,
        history_size=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )
    state = {"n": 0}

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = closure_value()
        loss.backward()
        state["n"] += 1
        if log_every and state["n"] % log_every == 0:
            _logger.info(
                "  lbfgs eval %3d  mean loglik %.4f  a=%.1f b=%.2f c=%.2f z=%.2f",
                state["n"],
                float(-loss),
                float(surrogate.log_a.exp()),
                float(surrogate.log_b.exp()),
                float(surrogate.log_c.exp()),
                float(surrogate.log_z.exp()),
            )
        return loss

    opt.step(closure)

    with torch.no_grad():
        after = float(-closure_value())

    out = {
        "a": float(surrogate.log_a.exp()),
        "b": float(surrogate.log_b.exp()),
        "c": float(surrogate.log_c.exp()),
        "z": float(surrogate.log_z.exp()),
        "loglik_before": before,
        "loglik_after": after,
        "n_refl": n_refl,
        "n_eval": state["n"],
    }
    if not math.isfinite(after) or after < before:
        _logger.warning(
            "global fit did not improve the likelihood: %.4f -> %.4f", before, after
        )
    return out
