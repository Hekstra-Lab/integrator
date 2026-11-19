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
    noise_a: float = 1.0,
    noise_b: float = 0.0,
    pedestal: str = "none",
    pedestal_frac: float = 0.5,
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
        noise_a, noise_b: The per-pixel variance function, `Var(c) = a * rate + b`.
            `(1, 0)` is Poisson and is the default, so a run that does not set them
            exports exactly what it exported before. They are for a detector whose
            stored value is not a photon count: with `g` units per photon and read
            noise `sigma_r`, `Var = g * rate + sigma_r^2`, and `a < 1` is what a
            sub-Poisson residual means. On hewl_1118 the held-out chi-square per
            pixel is 0.24 at high resolution, so the Poisson weights there are
            about four times too cautious.
        pedestal: `"per_reflection"` estimates an additive offset carrying no
            photons from each shoebox's OWN background pixels and subtracts it
            from the rate before the variance is formed, so
            `Var = a * (rate - c0) + b`. `"none"` is the default and leaves
            `c0 = 0`. For background pixels `mean = c0 + N` and `var = N`, so
            `c0 = mean - var` with nothing else needed -- no detector constant,
            no bias frame, no fit. Being local to one shoebox it is already the
            right panel, the right image and the right local regime, which
            matters because a multi-panel detector reads out each module with
            its own offset and the offset drifts with temperature.

            On hewl_1118 `c0` is 9.06 ADU against an apparent background of
            10.5, so a background pixel holds about 1.6 photons and its Poisson
            variance is overstated sevenfold. Measured across shoeboxes `c0` has
            spread 0.46 against a counting floor of 0.17, so it is not constant
            and a single global value leaves about 29% error where this leaves
            about 12%.
        pedestal_frac: Fraction of the valid pixels, taken from the lowest mean
            profile, used as the background sample for `c0`. Profile rather than
            geometry, because this function sees flat arrays and not the
            shoebox's shape.

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

    # An additive offset carrying no photons, from the pixels this shoebox
    # itself says are background. It enters the VARIANCE only: the mean is
    # unaffected because `bg` already absorbs it.
    if pedestal == "per_reflection":
        pbar = pi.mean(0)
        big = torch.finfo(pbar.dtype).max
        pb = torch.where(m > 0, pbar, torch.full_like(pbar, big))
        thr = pb.quantile(pedestal_frac, dim=-1, keepdim=True)
        sel = ((pb <= thr) & (m > 0)).to(c.dtype)
        nb = sel.sum(-1, keepdim=True).clamp_min(2.0)
        mu_b = (c * sel).sum(-1, keepdim=True) / nb
        var_b = (((c - mu_b) ** 2) * sel).sum(-1, keepdim=True) / (nb - 1.0)
        # `c0 <= mu_b` holds by construction since `var_b >= 0`. What it does
        # NOT survive is a profile flat enough that the low-profile half is
        # not background at all, which is where the measured spread of 45 ADU
        # around a median of 9.04 comes from. Bounding by the smallest count
        # in the box was tried and is WRONG: about 4% of background pixels
        # read below the offset, so there is Gaussian read noise and a stored
        # value is not bounded below by the pedestal. On hewl_1118 the global
        # value is flat across the detector -- 8.77 to 9.25 over six bins in x
        # and y -- so `rb_noise_b` with a fitted constant is the safer choice
        # until the tail is understood.
        c0 = (mu_b - var_b).clamp_min(0.0)
    elif pedestal == "none":
        c0 = torch.zeros_like(c[:, :1])
    else:
        raise ValueError(
            f"pedestal must be 'none' or 'per_reflection'; got {pedestal!r}"
        )

    def pixel_var(rate_):
        return (noise_a * (rate_ - c0) + noise_b).clamp_min(eps)

    resid = c.unsqueeze(0) - bg
    # a summation start, then reweight; the weights need an intensity to exist
    i_hat = resid.sum(-1).clamp_min(0.0)
    for _ in range(n_iter):
        # The variance model is floored at the background (Kabsch 2010): a
        # NEGATIVE i_hat is a legitimate estimate but an impossible rate, and
        # feeding it through to `rate` lets the peak pixel -- where pi is
        # largest -- clamp to `eps` and carry weight 1/eps. Flooring the
        # intensity instead of the rate keeps the weights finite, and for a
        # genuinely null reflection w = 1/bg is the Fisher-optimal choice.
        rate = (i_hat.clamp_min(0.0).unsqueeze(-1) * pi + bg).clamp_min(eps)
        # the optimal weight is the inverse VARIANCE of the pixel, which is
        # the rate only when the counts are Poisson
        w = m / pixel_var(rate)
        denom = (w * pi * pi).sum(-1).clamp_min(eps)
        i_hat = (w * pi * resid).sum(-1) / denom

    # Same floor for the information: evaluated at a negative i_hat the Fisher
    # terms OVERSTATE the information for exactly the negative tail, and with
    # the rate clamped they overstate it by ~1/eps, which would hand careless a
    # weak observation with sigma ~0.01 against an honest ~3.
    rate = (i_hat.clamp_min(0.0).unsqueeze(-1) * pi + bg).clamp_min(eps)
    inv = m / pixel_var(rate)
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
        "rb_pedestal": c0.reshape(-1),
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
class ScatterLoggerMixin:
    """End-of-epoch log-log scatters of model vs DIALS quantities (merge-quality
    readouts), reusable by any integrator.

    Each train batch stashes a subsample of (model, DIALS) pairs for every enabled
    scatter; at epoch end each is plotted (with its log-CC), logged to wandb as
    `<name>_vs_dials` / `<name>_logCC`, and saved to `<run>/<name>_scatter/`.
    Built-in scatters (each gated on its `log_*_scatter` cfg flag AND the needed
    keys being present, so they no-op safely):

      - "intensity":  model `scale*I_h` (or `qi.mean` when there is no scale) vs
                      DIALS `intensity.sum.value`.
      - "background": model `qbg.mean * n_valid_pixels` vs DIALS
                      `background.sum.value`.

    Host integrators call `self._init_scatter_logger(cfg)` in `__init__` and, in
    the train branch of `_step`, `self._collect_scatters(outputs, metadata, mask)`.
    `on_train_epoch_end` chains to `super()`, so list the mixin BEFORE the
    integrator base. Add custom scatters with `self._stash_scatter(name, model,
    ref, xlabel, ylabel)`.
    """

    def _init_scatter_logger(self, cfg) -> None:
        self.log_intensity_scatter = bool(
            getattr(cfg, "log_intensity_scatter", False)
        )
        self.log_background_scatter = bool(
            getattr(cfg, "log_background_scatter", False)
        )
        # name -> [model_chunks, ref_chunks, xlabel, ylabel]
        self._scatter_buffers: dict[str, list] = {}

    def _stash_scatter(
        self,
        name: str,
        model_vals: Tensor,
        ref_vals: Tensor,
        xlabel: str,
        ylabel: str,
        max_per_batch: int = 64,
    ) -> None:
        """Stash a per-batch subsample of (model, reference) for scatter `name`."""
        with torch.no_grad():
            mi = model_vals.detach().float().reshape(-1).cpu()
            ri = ref_vals.detach().float().reshape(-1).cpu()
            n = min(mi.shape[0], ri.shape[0])
            k = min(max_per_batch, n)
            if k <= 0:
                return
            idx = torch.randperm(n)[:k]
            buf = self._scatter_buffers.setdefault(
                name, [[], [], xlabel, ylabel]
            )
            buf[0].append(mi[:n][idx])
            buf[1].append(ri[:n][idx])

    def _scaled_model_intensity(self, outputs: dict) -> Tensor:
        """Model observed intensity = scale * qi.mean (qi.mean if no scale)."""
        model_i = outputs["qi"].mean
        scale = outputs.get("scale")
        return scale * model_i if scale is not None else model_i

    def _collect_intensity_scatter(
        self, outputs: dict, metadata: dict, max_per_batch: int = 64
    ) -> None:
        """Stash (model scale*I, DIALS intensity.sum.value). Model on x."""
        if "intensity.sum.value" not in metadata or "qi" not in outputs:
            return
        self._stash_scatter(
            "intensity",
            self._scaled_model_intensity(outputs),
            metadata["intensity.sum.value"],
            r"model  scale $\cdot$ $I$",
            "DIALS intensity.sum.value",
            max_per_batch,
        )

    def _collect_intensity_prf_scatter(
        self, outputs: dict, metadata: dict, max_per_batch: int = 64
    ) -> None:
        """Stash (model scale*I, DIALS intensity.prf.value) -- vs the profile-fit
        intensity (DIALS' primary value), model on x."""
        if "intensity.prf.value" not in metadata or "qi" not in outputs:
            return
        self._stash_scatter(
            "intensity_prf",
            self._scaled_model_intensity(outputs),
            metadata["intensity.prf.value"],
            r"model  scale $\cdot$ $I$",
            "DIALS intensity.prf.value",
            max_per_batch,
        )

    def _collect_variance_scatter(
        self, outputs: dict, metadata: dict, max_per_batch: int = 64
    ) -> None:
        """Stash (model scale^2 * Var[I], DIALS intensity.prf.variance) -- a
        sigma(I) calibration check (Var[s*I] = s^2 Var[I]), model on x."""
        if "intensity.prf.variance" not in metadata or "qi" not in outputs:
            return
        model_v = outputs["qi"].variance
        scale = outputs.get("scale")
        if scale is not None:
            model_v = scale.pow(2) * model_v
        self._stash_scatter(
            "intensity_var",
            model_v,
            metadata["intensity.prf.variance"],
            r"model  $s^2\,\mathrm{Var}[I]$",
            "DIALS intensity.prf.variance",
            max_per_batch,
        )

    def _collect_background_scatter(
        self,
        outputs: dict,
        metadata: dict,
        mask: Tensor,
        counts: Tensor,
        max_per_batch: int = 64,
    ) -> None:
        """Stash (model per-pixel bg, data per-pixel bg) -- a per-pixel comparison.

        The model `qbg.mean` is a per-pixel background rate, so the reference is a
        per-pixel background from the data: the robust per-shoebox median of the
        masked counts (the small Bragg peak does not move the median in a mostly-
        background box) -- the same estimator `prepare_bg_prior` uses. DIALS
        `background.sum.value` is NOT used: it is the background integrated over
        only the foreground/peak pixels, so it is region-mismatched to the model's
        whole-box per-pixel rate (and the foreground pixel count is not available).
        """
        if "qbg" not in outputs or counts is None:
            return
        bg = outputs["qbg"].mean.reshape(-1)
        cm = counts.reshape(counts.shape[0], -1).clamp(min=0).float()
        mk = mask.reshape(mask.shape[0], -1).float().to(cm.device)
        valid = mk > 0
        filled = torch.where(valid, cm, torch.full_like(cm, float("inf")))
        n_valid = valid.sum(-1).clamp(min=1).long()
        sorted_c, _ = filled.sort(dim=-1)
        med_idx = ((n_valid - 1) // 2).unsqueeze(1)
        data_bg = sorted_c.gather(1, med_idx).squeeze(1)
        self._stash_scatter(
            "background",
            bg,
            data_bg,
            "model bg/pixel (qbg.mean)",
            "data bg/pixel (shoebox median)",
            max_per_batch,
        )

    def _collect_scatters(
        self,
        outputs: dict,
        metadata: dict,
        mask: Tensor | None = None,
        counts: Tensor | None = None,
    ) -> None:
        """Collect every enabled+available scatter. Call in the train `_step`."""
        if getattr(self, "log_intensity_scatter", False):
            self._collect_intensity_scatter(outputs, metadata)
            self._collect_intensity_prf_scatter(outputs, metadata)
            self._collect_variance_scatter(outputs, metadata)
        if (
            getattr(self, "log_background_scatter", False)
            and mask is not None
            and counts is not None
        ):
            self._collect_background_scatter(outputs, metadata, mask, counts)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        import numpy as np

        for name, buf in list(getattr(self, "_scatter_buffers", {}).items()):
            model_list, ref_list, xlabel, ylabel = buf
            if not model_list:
                continue
            model_i = torch.cat(model_list).numpy()
            ref_i = torch.cat(ref_list).numpy()
            if len(model_i) > 5000:
                sel = np.random.choice(len(model_i), 5000, replace=False)
                model_i, ref_i = model_i[sel], ref_i[sel]
            self._plot_scatter(name, model_i, ref_i, xlabel, ylabel)
        if hasattr(self, "_scatter_buffers"):
            self._scatter_buffers.clear()

    def _plot_scatter(
        self, name: str, model_i, ref_i, xlabel: str, ylabel: str
    ) -> None:
        """Log-log scatter of a model quantity (x) vs its DIALS reference (y)."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            return

        keep = (
            (model_i > 0)
            & (ref_i > 0)
            & np.isfinite(model_i)
            & np.isfinite(ref_i)
        )
        mi, di = model_i[keep], ref_i[keep]
        if len(mi) < 10:
            return
        log_cc = float(np.corrcoef(np.log(mi), np.log(di))[0, 1])

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(mi, di, s=4, alpha=0.3, edgecolors="none")  # model x, DIALS y
        lo, hi = float(min(di.min(), mi.min())), float(max(di.max(), mi.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"epoch {self.current_epoch}  log-CC={log_cc:.3f}  n={len(mi)}"
        )
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()

        try:
            import wandb

            if self.logger is not None and hasattr(self.logger, "experiment"):
                self.logger.experiment.log(
                    {
                        f"{name}_vs_dials": wandb.Image(fig),
                        f"{name}_logCC": log_cc,
                        "epoch": self.current_epoch,
                    }
                )
        except Exception:
            pass
        try:
            from pathlib import Path

            save_dir = getattr(self.logger, "save_dir", None)
            out = Path(save_dir or ".") / f"{name}_scatter"
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"epoch{self.current_epoch:04d}.png", dpi=110)
        except Exception:
            pass
        plt.close(fig)
