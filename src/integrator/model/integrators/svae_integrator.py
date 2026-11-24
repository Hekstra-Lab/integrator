import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integrator import configs
from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.integrators.base_integrator import BaseIntegrator
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
)

logger = logging.getLogger(__name__)


def _sample_profile(qp, mc_samples: int) -> Tensor:
    if isinstance(qp, ProfileSurrogateOutput):
        return qp.zp.permute(1, 0, 2)
    return qp.rsample([mc_samples]).permute(1, 0, 2)


def _shape_signal_border_bg(counts: Tensor, mask: Tensor, side: int, width: int = 4) -> Tensor:
    """Signal photons per shoebox, for `gate_extent`, before any posterior exists.

    `hierarchical` subtracts its q(B) mean, but here q(B) is formed after the
    profile, from the gate. The background per pixel is instead the mean over a
    frame `width` pixels wide at the box edge, which a centred spot does not reach.
    The mean over a symmetric frame is unbiased for a flat or linearly tilted
    background; a median over the box is not, since skewed counts put the median
    below the mean and that bias, summed over the box, opened the gate on noise
    (498 photons for a reflection with E[I] = 13 on simulated DHFR). Detached, for
    the reason `hierarchical_integrator._shape_signal` gives.
    """
    with torch.no_grad():
        m = mask.reshape(counts.shape).to(counts.dtype)
        frame = torch.ones(side, side, dtype=torch.bool, device=counts.device)
        frame[width:-width, width:-width] = False
        fm = m * frame.reshape(-1).to(counts.dtype)
        bg = (counts * fm).sum(dim=-1) / fm.sum(dim=-1).clamp_min(1.0)
        total = (counts * m).sum(dim=-1)
        return (total - bg * m.sum(dim=-1)).clamp(min=0.0)


def _add_group_outputs(out: dict, metadata: dict, loss) -> None:
    group_labels = metadata["group_label"].long()
    out["group_label"] = group_labels
    if hasattr(loss, "log_alpha_per_group"):
        out["alpha_per_refl"] = F.softplus(
            loss.log_alpha_per_group[group_labels]
        )


class SVAEIntegrator(BaseIntegrator):
    """Per-observation SVAE: a learned responsibility gate over a conjugate Poisson-Gamma update."""

    REQUIRED_ENCODERS = {
        "profile": ("profile_encoder", configs.ProfileEncoderArgs),
        "signal": ("intensity_encoder", configs.IntensityEncoderArgs),
    }

    def __init__(self, cfg, loss, encoders, surrogates, optimizer=None):
        super().__init__(cfg, loss, encoders, surrogates, optimizer)
        self.gate_head = nn.Linear(
            self.cfg.encoder_out, math.prod(self.shoebox_shape)
        )

    def _intensity_prior(self, metadata, device):
        """`(concentration, rate)` of the Wilson prior on I, centric-aware.

        This integrator derives q(I) from conjugacy instead of routing the
        intensity through `WilsonLoss.forward`, so it needs the prior itself.
        It previously hardcoded `alpha = 1`, the acentric form, for every
        reflection. A centric `|F|^2` is `Sigma` times a chi-squared with ONE
        degree of freedom, `Gamma(1/2, tau/2)`: same mean, twice the variance,
        far more mass near zero. Using the acentric shape on a centric biases
        weak ones UP by about half a count and over-shrinks strong ones in
        weak shells by up to 17%.

        Delegating to the loss keeps q's prior and the KL's prior identical --
        they were otherwise built from different formulas -- and picks up the
        `refit_residual` correction as well. Datasets with no `centric`
        column fall back to the acentric form, as before.
        """
        return self.loss.intensity_prior_params(metadata, device)

    def _background_prior(self, metadata, device):
        # Delegating to the loss keeps q(b)'s conjugate prior and the KL's
        # prior identical, as for the intensity prior above. For a loss whose
        # prior is the fixed Gamma(bg_concentration, bg_rate) it returns the
        # same tensors as the inline lookup below; a loss that learns the
        # prior (`polychromatic_wilson_bgmap`) then reaches q(b) as well as
        # the KL. Losses without the method keep the inline lookup.
        loss_prior = getattr(self.loss, "_background_prior", None)
        if loss_prior is not None:
            g = metadata.get("group_label")
            return loss_prior(g, metadata, device)
        conc = self.loss.bg_concentration  # () or (n_bins,)
        rate = self.loss.bg_rate
        if conc.ndim == 1:
            g = metadata["group_label"].to(device).long()  # (B,)
            conc, rate = conc[g], rate[g]  # (B,)
        return conc, rate

    def _get_pixel_probs(self, qp, x_signal, raw_counts=None, mask=None):
        prf = (
            qp.mean_profile
        )  # signal shape from the profile surrogate, (B, P)
        if self.cfg.gate == "learned":
            signal_prob = torch.sigmoid(self.gate_head(x_signal))  # (B, P)
            return signal_prob, 1.0 - signal_prob, prf

        # `responsibility` and `joint`: compute the gate instead of
        # learning it. They differ only in how (I, B) are estimated.
        #
        # The gate's job is the Poisson split. For c_i = s_i + b_i with
        # s_i ~ Poisson(I p_i) and b_i ~ Poisson(B), the exact posterior share
        # of a count belonging to signal is r_i = I p_i / (I p_i + B), and with
        # that share E[sum_i r_i c_i] = sum_i I p_i = I identically: the
        # estimator is unbiased for free, with nothing fitted.
        #
        # The learned head cannot give that guarantee. It is a free output over
        # every pixel with no structural tie to the profile, so the model holds
        # two independent accounts of where the spot is, and nothing makes them
        # agree or makes the fixed point hold. Measured on the simulation, the
        # intensity came out as b*sum(g) -- the gate's total mass times the
        # background -- so at the weak end it was whatever the network said the
        # mass was, with no direct data term holding it in place.
        #
        # r depends on I, which is what we are estimating, so this is one
        # coordinate-ascent step from a starting estimate rather than a closed
        # form. The background starts from the box median, which is robust
        # because a Laue shoebox is 96.6% background, and the intensity from
        # the profile-weighted least squares given that background.
        if raw_counts is None:
            raise ValueError(
                f"gate={self.cfg.gate!r} needs raw_counts: the gate is "
                "derived from the data, not emitted by a head"
            )
        m = mask.squeeze(-1) if mask is not None else torch.ones_like(prf)
        c = raw_counts * m
        # The median must run over the VALID pixels only. Taking it over
        # `raw_counts * m` instead counts every masked pixel as a zero, which
        # drags the median down in proportion to the masked fraction: measured
        # on the simulation, -1 ADU at 10-30% masked, -3 at 30-50%, and exactly
        # 0 for the 7.6% of shoeboxes that are more than half masked. b0 = 0
        # sends r = I p / (I p + b0) to 1 wherever the profile is non-zero, so
        # the gate saturates and the intensity collapses to the box total.
        # Cast to the PROFILE's dtype, which is always floating point. The
        # loader hands counts over as int32, and a NaN cannot be written into
        # an integer tensor; the mask is bool, so its dtype is no good either.
        b0 = (
            raw_counts.to(prf.dtype)
            .masked_fill(m <= 0, float("nan"))
            .nanmedian(dim=-1, keepdim=True)
            .values
        )
        # all-masked rows have no median; fall back to zero signal there, which
        # the clamp below turns into a background-only shoebox
        b0 = torch.nan_to_num(b0, nan=0.0)
        # The responsibility is an E-step quantity: evaluated at the current
        # profile and, under `gate_detach`, held fixed for the update. Leaving
        # it differentiable lets KL(q(I) || prior) pull the profile down where
        # the counts are large, because r increases with p; see `gate_detach`.
        prf_r = prf.detach() if getattr(self.cfg, "gate_detach", False) else prf
        if self.cfg.gate == "joint":
            # (I, B) together against the design [p, 1], on the valid pixels.
            # `b0` above is an order statistic and is not used here: the
            # median quantises to zero on a faint background, which saturates
            # the gate, and swapping it for another robust statistic moves the
            # bias the other way. `c = I p + B` is linear in both and `p` is
            # known, so the 2x2 normal equation is the estimator that does
            # neither. See scripts/gate_study.py.
            n = m.sum(-1, keepdim=True).clamp_min(1.0)
            spp = (prf_r.pow(2) * m).sum(-1, keepdim=True)
            sp = (prf_r * m).sum(-1, keepdim=True)
            spc = (prf_r * c).sum(-1, keepdim=True)
            sc = c.sum(-1, keepdim=True)
            det = (spp * n - sp * sp).clamp_min(1e-12)
            i0 = ((n * spc - sp * sc) / det).clamp_min(0.0)
            b0 = ((spp * sc - sp * spc) / det).clamp_min(0.0)
        else:
            s2 = (prf_r.pow(2) * m).sum(-1, keepdim=True).clamp_min(1e-12)
            i0 = ((prf_r * (c - b0) * m).sum(-1, keepdim=True)
                  / s2).clamp_min(0.0)
        lam_s = i0 * prf_r
        signal_prob = lam_s / (lam_s + b0.clamp_min(1e-6) + 1e-12)
        signal_prob = signal_prob * m
        return signal_prob, (1.0 - signal_prob) * m, prf

    def _get_qbg(self, bg_prob, raw_counts, metadata, mask):
        alpha_bg, beta_bg = self._background_prior(metadata, raw_counts.device)
        m = mask.squeeze(-1)
        alpha_ = alpha_bg + (bg_prob * raw_counts * m).sum(-1)
        beta_ = beta_bg + m.sum(-1)
        return torch.distributions.Gamma(alpha_, beta_)

    def _get_qi(self, signal_prob, raw_counts, metadata, mean_prf, mask,
                x_signal=None):
        if self.cfg.intensity_posterior == "amortized":
            # Two free parameters, chosen by the ELBO: the family every other
            # integrator in this package uses, and the only one of the three
            # repairs that is strictly variational.
            return self.surrogates["qi"](x_signal, x_signal)

        alpha_I, beta_I = self._intensity_prior(metadata, raw_counts.device)
        m = mask.squeeze(-1)
        alpha_ = alpha_I + (signal_prob * raw_counts * m).sum(-1)
        beta_ = beta_I + (mean_prf * m).sum(-1)
        if (
            self.cfg.intensity_variance == "conjugate"
            and self.cfg.intensity_mean == "conjugate"
        ):
            return torch.distributions.Gamma(alpha_, beta_)

        # The conjugate rate is tau + sum(profile * mask), and the profile is
        # normalized, so it is tau + 1 for every reflection: the Fano factor
        # 1/(1+tau) belongs to the PRIOR and only the shape carries data. That
        # update is exact for `c ~ Poisson(I p)` with the signal counts
        # observed directly -- it is the augmented model with the split known.
        # Here the split is not known, it is estimated by the gate, and the
        # uncertainty discarded with it is the dominant term when the box is
        # 96.6% background.
        s1 = (mean_prf * m).sum(-1)
        s2 = (mean_prf.pow(2) * m).sum(-1)
        n = m.sum(-1)
        det = (n * s2 - s1 * s1).clamp_min(1e-12)

        if self.cfg.intensity_mean == "gls":
            # the (I, background) weighted least squares the Normal likelihood
            # implies, which under a constant pixel variance is the exact
            # conditional posterior mean given the profile
            sc = (raw_counts * m).sum(-1)
            spc = (mean_prf * raw_counts * m).sum(-1)
            mu = ((n * spc - s1 * sc) / det).clamp_min(
                self.cfg.intensity_mean_floor
            )
        else:
            mu = (alpha_ / beta_).clamp_min(1e-6)

        # [J^-1]_II for J = (1/s0^2) [[s2, s1], [s1, n]]: the background is a
        # nuisance parameter, not a known constant, so it is profiled out by
        # Schur complement rather than conditioned on.
        #
        # Optionally detached: the variance falls as the profile sharpens and
        # -KL rewards a wide q, so leaving the gradient attached lets the model
        # buy entropy by spreading the profile instead of by fitting. The
        # reported variance is identical either way.
        if self.cfg.variance_detach_profile:
            pv = mean_prf.detach()
            s1v = (pv * m).sum(-1)
            s2v = (pv.pow(2) * m).sum(-1)
            detv = (n * s2v - s1v * s1v).clamp_min(1e-12)
        else:
            detv = det
        var = self.cfg.pixel_variance * n / detv
        if self.cfg.profile_error_frac > 0:
            var = var + (self.cfg.profile_error_frac * mu).pow(2)
        # rate as k/mu, not mu/var: identical unless k hits the floor, and
        # then this keeps the MEAN exact and lets the variance inflate, which
        # is the right way round
        k = (mu * mu / var).clamp_min(self.cfg.intensity_k_min)
        return torch.distributions.Gamma(k, k / mu)

    def _background_posterior(
        self, *, shoebox, x_signal, bg_prob, raw_counts, mask, metadata
    ):
        """q(B). Base SVAE: conjugate Poisson-Gamma update from the gate."""
        return self._get_qbg(bg_prob, raw_counts, metadata, mask)

    def _forward_impl(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        # counts
        counts = torch.clamp(counts, min=0)

        # batch size
        b = shoebox.shape[0]
        shoebox_masked = shoebox * mask
        shoebox_reshaped = shoebox_masked.reshape(b, 1, *self.shoebox_shape)

        # `profile_coord_channels` widens conv1 in the factory, so the forward
        # has to widen the tensor to match; without this the config builds a
        # model that cannot run
        profile_in = self._profile_encoder_input(shoebox_reshaped, metadata)

        # profile representation
        x_profile = self.encoders["profile"](profile_in)
        # `mask` as every other integrator passes it: without it the profile
        # normalizes over the whole box, so mass landing on pixels the
        # likelihood never sees is never paid for and the intensity grows by
        # 1 / (mass on valid pixels) to compensate. It also makes the
        # conjugate exposure term `sum_valid pi` equal 1 for a full box.
        # only a surrogate with `gate_extent` reads it; others never receive it,
        # so their call is unchanged
        extra = {}
        if getattr(self.surrogates["qp"], "gate_extent", False):
            extra["shape_signal"] = _shape_signal_border_bg(
                counts, mask, self.shoebox_shape[-1]
            )
        qp = self.surrogates["qp"](
            x_profile,
            mc_samples=self.mc_samples,
            metadata=metadata,
            mask=mask,
            # Added, not substituted, exactly as in the hierarchical
            # integrator: surrogates that do not want the shoebox absorb it
            # through **kwargs and decode bit-identically. A basis-tied
            # encoder needs it, because its sufficient statistic is W^T c.
            counts=counts,
            **extra,
        )

        # get signal probs
        # both encoders are handed the same tensor, so `shared_encoder` just
        # reuses the profile representation rather than learning a second one
        x_signal = (
            x_profile
            if self.cfg.shared_encoder
            else self.encoders["signal"](shoebox_reshaped)
        )
        i_prob, bg_prob, mean_prf = self._get_pixel_probs(
            qp, x_signal, raw_counts=counts, mask=mask
        )

        # intensity: conjugate Poisson-Gamma update.
        # background: dispatched -- conjugate here, variational in the hybrid.
        qi = self._get_qi(
            i_prob, counts, metadata, mean_prf, mask, x_signal=x_signal
        )
        qbg = self._background_posterior(
            shoebox=shoebox_reshaped,
            x_signal=x_signal,
            bg_prob=bg_prob,
            raw_counts=counts,
            mask=mask,
            metadata=metadata,
        )

        # samples
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        # pixel poisson rates
        rate = zI * zp + zbg

        if "is_coset" in metadata:
            coset = metadata["is_coset"].bool().view(-1, 1, 1)
            rate = torch.where(coset, zbg, rate)

        out = IntegratorBaseOutputs(
            rates=rate,
            counts=counts,
            mask=mask,
            qbg=qbg,
            qp=qp,
            qi=qi,
            zp=zp,
            zbg=zbg,
            metadata=metadata,
        )
        out = _assemble_outputs(out)
        # the gate is this integrator's own quantity and nothing downstream can
        # recompute it: the conjugate variance understates the real one by
        # sum(c g^2) / sum(c g), which needs g per pixel
        out["signal_prob"] = i_prob
        _add_group_outputs(out, metadata, self.loss)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }


class SVAEHybridIntegrator(SVAEIntegrator):
    """Hybrid SVAE: conjugate intensity, amortized (variational) background.

    Keeps the conjugate q(I) update but replaces the conjugate q(B) -- whose
    sufficient statistic sum_p (1 - g_p) c_p is dominated by the gate's behavior
    at the bright peak -- with a learned Gamma surrogate. The background gets its
    own dedicated k_bg/r_bg intensity encoders (as in HierarchicalIntegrator), so
    it is fully decoupled from the responsibility gate's signal encoder: gate
    error at the peak no longer biases B (and through it, the Bijvoet difference).
    """

    REQUIRED_ENCODERS = {
        "profile": ("profile_encoder", configs.ProfileEncoderArgs),
        "signal": ("intensity_encoder", configs.IntensityEncoderArgs),
        "k_bg": ("intensity_encoder", configs.IntensityEncoderArgs),
        "r_bg": ("intensity_encoder", configs.IntensityEncoderArgs),
    }

    def _background_posterior(
        self, *, shoebox, x_signal, bg_prob, raw_counts, mask, metadata
    ):
        x_k_bg = self.encoders["k_bg"](shoebox)
        x_r_bg = self.encoders["r_bg"](shoebox)
        return self.surrogates["qbg"](x_k_bg, x_r_bg)
