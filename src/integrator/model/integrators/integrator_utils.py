from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from integrator.configs.integrator import IntegratorCfg
from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)


@dataclass
class IntegratorBaseOutputs:
    rates: Tensor
    counts: Tensor
    mask: Tensor
    qbg: Any
    qp: Any
    qi: Any
    zp: Tensor
    zbg: Tensor
    metadata: dict[str, torch.Tensor]


@dataclass
class IntegratorModelArgs:
    cfg: IntegratorCfg
    loss: nn.Module
    surrogates: dict[str, nn.Module]
    encoders: dict[str, nn.Module]


def _effective_pixels(profile: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Effective pixel count `n_eff = 1/sum(pi^2)` over the valid support, and that support's size.

    `n_eff` is the Kish effective sample size of the profile: how many pixels the
    signal effectively occupies. A top-hat over `m` pixels gives exactly `m`. It sets
    the background-limited variance `Var(I) = n_eff * b`, and `n_eff / P` is the
    squared intensity/background correlation, which is why a profile that broadens as
    signal-to-noise falls degrades both the variance and the separability of `I` from `b`.
    """
    tiny = torch.finfo(profile.dtype).tiny
    m = mask.reshape(mask.shape[0], -1).to(profile.dtype)
    p = profile.reshape(profile.shape[0], -1) * m
    p = p / p.sum(-1, keepdim=True).clamp_min(tiny)
    n_eff = 1.0 / p.pow(2).sum(-1).clamp_min(tiny)
    return n_eff, m.sum(-1)



def _samples_first(x: Tensor, batch: int) -> Tensor:
    """Reorder a sampled tensor to (S, B, ...).

    The hierarchical path permutes samples into the middle axis before forming the
    rate, so the layout differs between integrators; locate the batch axis rather
    than assuming which end the samples sit on.
    """
    if x.dim() < 3:
        return x.unsqueeze(0)
    return x.transpose(0, 1) if x.shape[0] == batch else x

def rao_blackwellized_intensity(
    counts: Tensor,
    mask: Tensor,
    profile: Tensor,
    background: Tensor,
    n_iter: int = 2,
    gate_half_signal: float | None = None,
) -> dict[str, Tensor]:
    """Profile-fit intensity with the profile's own uncertainty propagated into sigma.

    The exported `sqrt(Var_q[I])` is a plug-in: it conditions on one profile and one
    background and so reports `Var(I | h)`, dropping the contribution of not knowing
    `h`. The law of total variance supplies the missing piece,

        Var(I) = E_h[Var(I | h)] + Var_h[E(I | h)]

    so this fits `I` once per latent sample and combines the mean Fisher variance
    with the spread of the fits. The conditional term is the Schur complement of the
    `(I, b)` information, not `1/F_II`: the background is estimated, and charging
    nothing for that understates the variance by `1 - rho^2` with
    `rho^2 = F_Ib^2/(F_II F_bb)`.

    The second term is only as good as `q(h)`. Where the data cannot constrain the
    profile, the spread of the fits reflects an over-broad posterior rather than real
    uncertainty, and the combined sigma then exceeds the scatter actually seen
    between symmetry mates. `gate_half_signal` damps that term by `s/(s+k)` in signal
    photons; it is a pragmatic guard, and the underlying problem is the calibration of
    `q(h)` at low signal, not this formula.

    Args:
        counts: Observed counts, shape (B, P).
        mask: Valid-pixel mask, shape (B, P).
        profile: Profile samples on the simplex, shape (S, B, P).
        background: Per-pixel background samples, shape (S, B, P) or (S, B, 1).
        n_iter: Reweighting passes. The weights depend on the fitted intensity, so
            one pass is a summation estimate and two is enough in practice.
        gate_half_signal: Signal photons at which the profile-variance term is
            half-weighted. `None` applies it in full.

    Returns:
        `i_rb` and `var_rb`, plus the two variance components and the gate, each
        shape (B,).
    """
    eps = 1e-6
    m = mask.reshape(mask.shape[0], -1).to(counts.dtype)
    c = counts.reshape(counts.shape[0], -1).to(counts.dtype) * m
    pi = profile.reshape(profile.shape[0], profile.shape[1], -1) * m
    bg = background.reshape(background.shape[0], background.shape[1], -1)
    if bg.shape[-1] == 1:
        bg = bg.expand_as(pi)
    bg = bg * m

    resid = c.unsqueeze(0) - bg
    # a summation start, then reweight; the weights need an intensity to exist
    i_hat = resid.sum(-1).clamp_min(0.0)
    for _ in range(n_iter):
        rate = (i_hat.unsqueeze(-1) * pi + bg).clamp_min(eps)
        w = m / rate
        denom = (w * pi * pi).sum(-1).clamp_min(eps)
        i_hat = (w * pi * resid).sum(-1) / denom

    rate = (i_hat.unsqueeze(-1) * pi + bg).clamp_min(eps)
    inv = m / rate
    f_ii = (inv * pi * pi).sum(-1)
    f_ib = (inv * pi).sum(-1)
    f_bb = (inv).sum(-1).clamp_min(eps)
    schur = (f_ii - f_ib * f_ib / f_bb).clamp_min(eps)

    var_cond = (1.0 / schur).mean(0)
    var_profile = i_hat.var(0, unbiased=True) if i_hat.shape[0] > 1 else torch.zeros_like(var_cond)
    i_rb = i_hat.mean(0)

    if gate_half_signal is None:
        gate = torch.ones_like(var_cond)
    else:
        signal = (resid.mean(0) * m).sum(-1).clamp_min(0.0)
        gate = signal / (signal + max(float(gate_half_signal), eps))

    return {
        "i_rb": i_rb,
        "var_rb": var_cond + gate * var_profile,
        "var_rb_cond": var_cond,
        "var_rb_profile": var_profile,
        "rb_gate": gate,
    }

def _assemble_outputs(
    out: IntegratorBaseOutputs,
) -> dict[str, Any]:
    is_profile_output = isinstance(out.qp, ProfileSurrogateOutput)

    qp_mean = out.qp.mean_profile if is_profile_output else out.qp.mean

    base = {
        "rates": out.rates,
        "counts": out.counts,
        "mask": out.mask,
        "zp": out.zp,
        "zbg": out.zbg,
        "qbg_mean": out.qbg.mean,
        "qbg_var": out.qbg.variance,
        "qp_mean": qp_mean,
        "qi_mean": out.qi.mean,
        "qi_var": out.qi.variance,
        "profile": qp_mean,
    }

    n_eff, n_valid_px = _effective_pixels(qp_mean, out.mask)
    base["n_eff"] = n_eff
    base["n_valid_px"] = n_valid_px

    # the plug-in sigma conditions on one profile; this adds the variance of not
    # knowing it, which is the term the downstream error model has been inflating
    # away. Opt-in via predict_keys, so nothing changes for a run that does not ask.
    # the likelihood averages softmax over latent samples, which is broader than
    # the softmax at the posterior mean, so report the breadth of both
    mc_profile = out.zp
    if mc_profile.dim() == 3:
        # the hierarchical path permutes samples into the middle axis, so locate
        # the batch axis rather than assuming which end the samples sit on
        batch = qp_mean.shape[0]
        sample_axis = 1 if mc_profile.shape[0] == batch else 0
        mc_profile = mc_profile.mean(sample_axis)
    base["n_eff_mc"] = _effective_pixels(mc_profile, out.mask)[0]

    if is_profile_output:
        base["qp_loc"] = out.qp.loc
        base["qp_scale"] = out.qp.scale

    if out.metadata is None:
        return base

    distribution_params = {
        "qbg_params": {
            name: getattr(out.qbg, name) for name in out.qbg.arg_constraints
        },
        "qi_params": {
            name: getattr(out.qi, name) for name in out.qi.arg_constraints
        },
    }
    # qp is a Distribution
    if not is_profile_output:
        distribution_params["qp_params"] = {
            name: getattr(out.qp, name) for name in out.qp.arg_constraints
        }

    base.update(out.metadata)
    base.update(distribution_params)

    return base


def coordinate_channels(
    metadata: dict,
    shoebox_shape: tuple[int, ...],
    beam_center: Sequence[float] | None,
    *,
    device=None,
    dtype=None,
) -> Tensor:
    """Per-pixel coordinates in the beam-centred radial/tangential frame.

    A convolutional encoder is translation-equivariant, so it cannot tell where on
    the detector a shoebox came from -- but the streak direction is fixed by the
    azimuth about the beam centre, which is precisely that information. Supplying
    each pixel's offset resolved along the local radial and tangential axes puts
    every reflection in a common frame, so one kernel can express an elongated
    spot at any azimuth instead of the network relearning a rotated copy per
    detector position.

    This is the encoder-side counterpart to `GeometricProfileSurrogate`'s `steer`,
    which applies the same rotation analytically to the decoded profile.

    Returns `(B, 2, *shoebox_shape)`: the radial channel first, then the
    tangential, each in units of the box half-width so both are order 1.
    """
    xc = metadata.get("xyzcal.px.0")
    yc = metadata.get("xyzcal.px.1")
    if xc is None or yc is None or beam_center is None:
        raise KeyError(
            "coordinate_channels needs xyzcal.px.0/1 in metadata and a beam "
            "centre; pass coord_channels=False to run without them"
        )
    xc = torch.as_tensor(xc, device=device, dtype=dtype).reshape(-1)
    yc = torch.as_tensor(yc, device=device, dtype=dtype).reshape(-1)
    b = xc.shape[0]

    phi = torch.atan2(yc - float(beam_center[1]), xc - float(beam_center[0]))
    cos, sin = torch.cos(phi), torch.sin(phi)

    *lead, h, w = shoebox_shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    # centre the grid and scale by the half-width, so the channels are order 1
    half = max((h - 1) / 2.0, (w - 1) / 2.0, 1.0)
    du = (xx - (w - 1) / 2.0) / half
    dv = (yy - (h - 1) / 2.0) / half

    c = cos.view(b, 1, 1)
    s = sin.view(b, 1, 1)
    radial = du * c + dv * s
    tangential = -du * s + dv * c

    out = torch.stack((radial, tangential), dim=1)          # (B, 2, H, W)
    for _ in lead:                                           # broadcast over depth
        out = out.unsqueeze(2)
    if lead:
        out = out.expand(b, 2, *lead, h, w)
    return out.contiguous()
