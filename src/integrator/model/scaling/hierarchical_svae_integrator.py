"""Hierarchical Structured-VAE integrator (Option 3 + 3a, GIG-exact).

The full two-level generative model: a per-HKL merged intensity I_h with a Wilson
(Gamma) prior, a per-observation de-scaled intensity J_i as a Gamma random effect
centred on I_h, and Poisson pixels under the signal/background augmentation:

    I_h        ~ Gamma(alpha_W, tau_h)                     # Wilson prior  (alpha_W = 1)
    J_i | I_h  ~ Gamma(nu, nu / I_h)                       # random effect; E[J_i]=I_h
    c_{i,p}    ~ Poisson(s_i * J_i * prof_{i,p} + bg_i)    # pixels (+ augmentation)

It unifies the per-observation integrator (bottom half: pixels -> q(J_i)) and the
merging model (top half: {q(J_i)} -> q(I_h)) into one graph; the random effect's
dispersion nu is the merging error model written as a prior.

Inference (mean-field q(I_h) prod_i q(J_i)):

  - Bottom (amortized SVAE): a head emits a per-pixel signal attribution g_{i,p};
    the per-observation potential is summed and combined with the random-effect
    prior,
        a_i = nu      + sum_p g_{i,p} c_{i,p}              # shape: prior + signal counts
        b_i = nu E[1/I_h] + sum_p s_i prof_{i,p}           # rate:  prior + scaled exposure
        q(J_i) = Gamma(a_i, b_i).

  - Top (GIG-exact): the CAVI optimum for I_h under the Gamma prior + Gamma random
    effect is a Generalized Inverse Gaussian (NOT a Gamma; the Wilson Gamma is its
    b->0 special case),
        q(I_h) = GIG(p = alpha_W - nu N_h,  a = 2 tau_h,  b = 2 nu sum_i E[J_i]).

The two levels are coupled (E[1/I_h] feeds b_i; E[J_i] feeds b) and solved by a
short CAVI loop. The intensity ELBO needs only the GIG moments and log-partition,
NOT E[log I_h] / the GIG entropy (the Bessel order-derivative): those cancel
exactly (see `integrator.model.distributions.gig.gig_intensity_elbo`). With nu a
fixed hyperparameter the GIG order is constant, so no order-gradient is needed.

Pair with `GroupedAsuIdBatchSampler` (`group_by_asu_id: true`) so each HKL's
observations are complete in the batch; the loss must set `pi_weight: 0` so the
ELBO's intensity terms are not double-counted (this model supplies them).
"""

import math
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gamma

from integrator import configs
from integrator.model.distributions.gig import (
    gig_intensity_elbo,
    gig_moments,
)
from integrator.model.integrators.base_integrator import (
    BaseIntegrator,
    _log_loss,
)
from integrator.model.integrators.hierarchical_integrator import (
    _add_group_outputs,
    _sample_profile,
)
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
)
from integrator.model.scaling.chebyshev_scale import ChebyshevScale, MLPScale
from integrator.model.scaling.conjugate_merging import _scatter_sum_compact


class HierarchicalSVAEIntegrator(BaseIntegrator):
    """Two-level hierarchical SVAE: per-obs J_i (Gamma) + per-HKL I_h (GIG)."""

    REQUIRED_ENCODERS = {
        "profile": configs.ProfileEncoderArgs,
        "k_bg": configs.IntensityEncoderArgs,
        "r_bg": configs.IntensityEncoderArgs,
    }

    def __init__(
        self,
        cfg: configs.IntegratorCfg,
        loss: nn.Module,
        encoders: dict[str, nn.Module],
        surrogates: dict[str, nn.Module],
    ):
        super().__init__(cfg, loss, encoders, surrogates)

        self.alpha_W = float(cfg.wilson_alpha)
        self.nu = float(cfg.link_nu)
        self.n_cavi_iters = int(cfg.n_cavi_iters)
        self.amortize_merge = bool(getattr(cfg, "amortize_merge", True))
        self.merge_kl_weight = float(getattr(cfg, "merge_kl_weight", 1.0))
        self.sample_I = bool(cfg.sample_I_h)

        if self.nu <= 0:
            raise ValueError(f"link_nu (nu) must be > 0, got {self.nu}")
        if abs(self.alpha_W - 1.0) > 1e-8:
            raise ValueError(
                "HierarchicalSVAEIntegrator requires wilson_alpha == 1.0 (the "
                f"Wilson prior shape); got {self.alpha_W}."
            )
        # The intensity ELBO (J random-effect KL + GIG I-block) is supplied here,
        # so the loss must NOT also add its Gamma-Gamma KL_i.
        pi_weight = float(getattr(self.loss, "pi_weight", 1.0))
        if abs(pi_weight) > 1e-8:
            raise ValueError(
                "HierarchicalSVAEIntegrator supplies the intensity ELBO itself; "
                "set loss pi_weight == 0 so KL_i is not double-counted; got "
                f"{pi_weight}."
            )
        if getattr(self.loss, "_apply_lp", False):
            raise ValueError(
                "LP is applied through the scale here, so enabling the loss's "
                "lp_correction would double-count it. Set lp_correction: false."
            )

        # Per-pixel signal attribution head (SVAE), shared with the profile
        # encoder; zero-init -> g = 0.5 at start (neutral), learns from the ELBO.
        self.n_pixels = int(math.prod(self.shoebox_shape))
        self.resp_head = nn.Linear(cfg.encoder_out, self.n_pixels)
        nn.init.zeros_(self.resp_head.weight)
        nn.init.zeros_(self.resp_head.bias)

        # Amortized merge head: a learned multiplicative correction to the
        # prior-free MLE that forms the GIG natural parameter b in one pass (no
        # CAVI fixed point). Inputs are per-observation [log a_i, log e_i, log
        # s_i, d]; the final layer is zero-init so the correction starts at 1 and
        # b begins exactly at the MLE-based merge `2 nu * sum_i (a_i/e_i)`.
        if self.amortize_merge:
            self.delta_head = nn.Sequential(
                nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1)
            )
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)

        # Per-observation scale s_i (LP lives here). Minimal selection; extend as
        # the other merging integrators do if richer scales are needed.
        if getattr(cfg, "scale_none", False):
            self.scale_fn = None
        elif cfg.scale_mlp:
            # `frame` here is the ROTATION frame (xyzcal.px.2), a lab-frame
            # input. Optionally also feed the CRYSTAL-frame spherical-harmonic
            # absorption (`absorption_sh`) -- the direction the anomalous-safe
            # absorption surface needs and which lab-frame inputs cannot build;
            # `even_only` keeps only Friedel-symmetric (even-l) harmonics.
            n_abs_sh = 0
            if getattr(cfg, "scale_mlp_absorption", False):
                n_abs_sh = (int(cfg.scale_sh_lmax) + 1) ** 2 - 1
            self.scale_fn = MLPScale(
                hidden_dim=cfg.scale_mlp_hidden,
                n_layers=cfg.scale_mlp_layers,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                beam_center=cfg.scale_beam_center,
                r_max=cfg.scale_r_max,
                d_min=getattr(cfg, "dmin", 1.0),
                d_max=60.0,
                head_init_std=getattr(cfg, "scale_head_init_std", 0.0),
                n_abs_sh=n_abs_sh,
                absorption_even_only=getattr(
                    cfg, "scale_mlp_absorption_even_only", True
                ),
            )
        else:
            self.scale_fn = ChebyshevScale(
                degree=cfg.scale_degree,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
            )

    def _get_scale(self, metadata: dict, device: torch.device) -> Tensor:
        b = metadata["d"].shape[0]
        if self.scale_fn is None:
            return torch.ones(b, device=device)
        frame = metadata["xyzcal.px.2"].to(device).float()
        lp = metadata["lp"].to(device).float().clamp(min=1e-8)
        if isinstance(self.scale_fn, MLPScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            d = metadata["d"].to(device).float()
            a = None
            if self.scale_fn.n_abs_sh > 0:
                if "absorption_sh" not in metadata:
                    raise KeyError(
                        "MLPScale with scale_mlp_absorption needs 'absorption_sh' "
                        "in metadata; run scripts/extract_crystal_frame_sh.py "
                        "(--lmax = scale_sh_lmax) and point the data loader's "
                        "reference at the augmented metadata_sh.pt."
                    )
                a = metadata["absorption_sh"].to(device).float()
            return self.scale_fn(frame, x_det, y_det, lp, d, a)
        return self.scale_fn(frame) / lp

    def _wilson_tau(self, d: Tensor) -> Tensor:
        """Wilson prior rate tau from resolution d (lp lives in the scale)."""
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        return self.loss._get_tau({"d": d}, s_sq, d.device)

    def _amortized_merge(
        self,
        a_i: Tensor,
        e_i: Tensor,
        scale: Tensor,
        d_obs: Tensor,
        p_h: Tensor,
        a_h: Tensor,
        inverse: Tensor,
        asu_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Single-pass amortized merge. Returns (b_i, b_h, E_I_h).

        A learned per-observation correction forms the GIG natural parameter
        b = 2 nu sum_i delta_i directly, with delta_i anchored on the prior-free
        MLE a_i/e_i (which carries no I_h dependence, so the whole pass is a
        feed-forward DAG -- no fixed point). q(I_h)=GIG(p_h,a_h,b_h) is still
        exact; only how b is determined is amortized. Recovers the CAVI fixed
        point when the correction learns the prior shrinkage e_i/b_i.
        """
        mle = a_i / e_i.clamp(min=1e-12)  # prior-free per-obs intensity
        feat = torch.stack(
            [
                a_i.clamp(min=1e-6).log(),
                e_i.clamp(min=1e-6).log(),
                scale.clamp(min=1e-8).log(),
                d_obs,
            ],
            dim=-1,
        )  # (B, 4)
        corr = self.delta_head(feat).squeeze(-1).clamp(-10.0, 10.0)
        delta_i = mle * torch.exp(corr)  # ~ E[J_i] (shrunk MLE)
        ej_sum, _, _ = _scatter_sum_compact(delta_i, asu_ids)
        b_h = 2.0 * self.nu * ej_sum
        e_i_h, e_inv_Ih = gig_moments(p_h, a_h, b_h)
        b_i = self.nu * e_inv_Ih[inverse] + e_i
        i_block = self._i_block_elbo(p_h, a_h, b_h, a_i, b_i, e_inv_Ih, asu_ids)
        return b_i, b_h, e_i_h, i_block

    def _i_block_elbo(
        self,
        p_h: Tensor,
        a_h: Tensor,
        b_h: Tensor,
        a_i: Tensor,
        b_i: Tensor,
        e_inv_Ih: Tensor,
        asu_ids: Tensor,
    ) -> Tensor:
        """Per-HKL I_h ELBO contribution (exact for any b_h).

        The conjugate cancellation `log(2 K_p) - (p/2) log(a/b)` is the I-block
        ONLY when b_h equals the CAVI value 2*nu*sum_i E[J_i] (then the E[1/I_h]
        coefficient is zero). With an amortized b_h that differs, the E[1/I_h]
        term survives and must be added back:

            L_I = log-partition + (b_h/2 - nu * sum_i E[J_i]) * E[1/I_h],

        E[J_i] = a_i/b_i under q(J_i). The correction needs only E[1/I_h] (a
        Bessel ratio, no order-derivative) and is identically zero in CAVI mode.
        Without it `kl_intensity` is not a valid KL and can go negative.
        """
        ej_sum, _, _ = _scatter_sum_compact(a_i / b_i, asu_ids)
        correction = (0.5 * b_h - self.nu * ej_sum) * e_inv_Ih
        return gig_intensity_elbo(p_h, a_h, b_h) + correction

    def _cavi(
        self,
        a_i: Tensor,
        e_i: Tensor,
        tau_hkl: Tensor,
        p_h: Tensor,
        a_h: Tensor,
        inverse: Tensor,
        asu_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Two-level CAVI fixed point (derived/purist). Returns (b_i, b_h, E_I_h).

        Unrolled with gradients (a short loop; with nu fixed the GIG order p_h is
        constant). At the fixed point q(J_i)=Gamma(a_i,b_i) and q(I_h)=GIG(p_h,
        a_h,b_h) are mutually consistent; we return the pair from the final
        iteration (same E[1/I_h]).
        """
        e_inv_Ih = tau_hkl  # init E[1/I_h] at the Wilson scale (per HKL)
        b_i = b_h = e_i_h = None
        for _ in range(self.n_cavi_iters):
            b_i = self.nu * e_inv_Ih[inverse] + e_i  # q(J) rate
            e_j = a_i / b_i  # E[J_i]
            ej_sum, _, _ = _scatter_sum_compact(e_j, asu_ids)
            b_h = 2.0 * self.nu * ej_sum
            e_i_h, e_inv_Ih = gig_moments(p_h, a_h, b_h)
        # i_block correction is identically zero here (b_h = 2 nu sum E[J_i]),
        # but compute it the same way for a single code path.
        i_block = self._i_block_elbo(p_h, a_h, b_h, a_i, b_i, e_inv_Ih, asu_ids)
        return b_i, b_h, e_i_h, i_block

    def _j_block(self, a: Tensor, b: Tensor) -> Tensor:
        """Per-observation ELBO term E_q[log p(J|I)]_{J-only} - E_q[log q(J)].

        = nu log nu - lgamma(nu) + lgamma(a) - a log b + a + (nu - a)(psi(a) - log b)
        The I_h-coupling part of log p(J|I) is folded into the GIG log-partition
        (the exact conjugate cancellation), so only the J-only remainder is here.
        """
        nu = self.nu
        log_b = b.clamp(min=1e-12).log()
        psi_a = torch.digamma(a.clamp(min=1e-6))
        return (
            nu * math.log(nu)
            - math.lgamma(nu)
            + torch.lgamma(a.clamp(min=1e-6))
            - a * log_b
            + a
            + (nu - a) * (psi_a - log_b)
        )

    def _forward_impl(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        counts = counts.clamp(min=0)
        B = shoebox.shape[0]
        device = shoebox.device
        shoebox_reshaped = (shoebox * mask).reshape(B, 1, *self.shoebox_shape)

        x_profile = self.encoders["profile"](shoebox_reshaped)
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)

        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)
        qp = self.surrogates["qp"](x_profile, mc_samples=self.mc_samples)
        profile_mean = qp.mean_profile  # (B, P)

        scale = self._get_scale(metadata, device)  # (B,)
        asu_ids = metadata["asu_id"].long().to(device)
        d_obs = metadata["d"].to(device).float()

        # Per-observation potentials (SVAE): signal counts + scaled exposure.
        g = torch.sigmoid(self.resp_head(x_profile))  # (B, P)
        sigma_i = (g * counts * mask).sum(dim=-1)  # signal counts
        e_i = scale * (profile_mean * mask).sum(dim=-1)  # scaled exposure

        # Per-HKL Wilson rate and GIG order/scale (constant in the network).
        tau_obs = self._wilson_tau(d_obs)
        tau_sum, inverse, unique = _scatter_sum_compact(tau_obs, asu_ids)
        cnt, _, _ = _scatter_sum_compact(torch.ones_like(tau_obs), asu_ids)
        tau_hkl = tau_sum / cnt.clamp(min=1.0)  # (n_unique,)
        p_h = self.alpha_W - self.nu * cnt  # GIG order (fixed)
        a_h = 2.0 * tau_hkl  # GIG 'a'

        a_i = self.nu + sigma_i  # q(J) shape: random-effect prior + signal counts
        if self.amortize_merge:
            b_i, b_h, e_i_h, i_block = self._amortized_merge(
                a_i, e_i, scale, d_obs, p_h, a_h, inverse, asu_ids
            )
        else:
            b_i, b_h, e_i_h, i_block = self._cavi(
                a_i, e_i, tau_hkl, p_h, a_h, inverse, asu_ids
            )

        qJ = Gamma(a_i.clamp(min=1e-6), b_i.clamp(min=1e-12))

        # Reconstruction is driven by the per-observation latent J_i.
        if self.sample_I:
            zJ = qJ.rsample([self.mc_samples]).clamp(min=1e-10)  # (S, B)
        else:
            zJ = (a_i / b_i).unsqueeze(0).expand(self.mc_samples, B)
        zJ_scaled = (scale.unsqueeze(0) * zJ).unsqueeze(-1).permute(1, 0, 2)

        zp = _sample_profile(qp, self.mc_samples)
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        rate = zJ_scaled * zp + zbg

        if "is_coset" in metadata:
            coset = metadata["is_coset"].bool().view(-1, 1, 1)
            rate = torch.where(coset, zbg, rate)

        out = IntegratorBaseOutputs(
            rates=rate,
            counts=counts,
            mask=mask,
            qbg=qbg,
            qp=qp,
            qi=qJ,
            zp=zp,
            zbg=zbg,
            metadata=metadata,
        )
        out = _assemble_outputs(out)
        out["asu_id"] = asu_ids
        out["I_h_mean"] = e_i_h[inverse]  # merged estimate, broadcast to obs
        if "group_label" in metadata:
            _add_group_outputs(out, metadata, self.loss)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qJ,
            "qbg": qbg,
            "a_i": a_i,
            "b_i": b_i,
            "p_h": p_h,
            "a_h": a_h,
            "b_h": b_h,
            "E_I_h": e_i_h,
            "i_block": i_block,
            "scale": scale,
            "n_hkl": len(unique),
            "g_mean": g.mean().detach(),
        }

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        B = counts.shape[0]
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]
        group_labels = metadata["group_label"].long()

        # Loss module supplies recon NLL + KL_prf + KL_bg only (pi_weight == 0).
        loss_dict = self.loss(
            rate=forward_out["rates"],
            counts=forward_out["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=forward_out["mask"],
            group_labels=group_labels,
            metadata=metadata,
        )
        total_loss = loss_dict["loss"]

        # Intensity ELBO = sum_obs J-block + sum_HKL I-block; the loss is its
        # negation (a valid KL >= 0), summed and put on the per-observation scale
        # (matches the merging integrators' per-HKL KL normalization). The I-block
        # carries the amortized-b correction (_i_block_elbo), without which this
        # is not a true KL and can go negative.
        j_block = self._j_block(outputs["a_i"], outputs["b_i"]).sum()
        i_block = outputs["i_block"].sum()
        intensity_nelbo = -(j_block + i_block) / B
        total_loss = total_loss + self.merge_kl_weight * intensity_nelbo

        penalty, penalty_components = self._profile_basis_penalty()
        for name, value in penalty_components.items():
            self.log(f"{step} {name}", value, on_step=False, on_epoch=True)
        total_loss = total_loss + penalty

        _log_loss(
            self,
            kl=loss_dict["kl_mean"] + intensity_nelbo,
            nll=loss_dict["neg_ll_mean"],
            total_loss=total_loss,
            step=step,
            kl_components={
                "kl_prf": loss_dict["kl_prf_mean"],
                "kl_intensity": intensity_nelbo,
                "kl_bg": loss_dict["kl_bg_mean"],
            },
        )

        with torch.no_grad():
            self.log(f"{step} I_h_mean", outputs["E_I_h"].mean(), on_epoch=True)
            self.log(
                f"{step} J_mean",
                (outputs["a_i"] / outputs["b_i"]).mean(),
                on_epoch=True,
            )
            self.log(f"{step} g_mean", outputs["g_mean"], on_epoch=True)
            self.log(
                f"{step} n_hkl",
                torch.tensor(float(outputs["n_hkl"])),
                on_epoch=True,
            )

        # End-of-epoch model-vs-DIALS scatters (no-op unless log_*_scatter set).
        # The intensity scatter uses scale * E[J_i] (outputs["scale"] present)
        # against DIALS intensity.sum.value -- the predicted observed intensity.
        if step == "train":
            self._collect_scatters(outputs, metadata, mask, counts)

        return {
            "loss": total_loss,
            "forward_out": forward_out,
            "loss_components": {
                "loss": total_loss.detach(),
                "nll": loss_dict["neg_ll_mean"].detach(),
                "kl": (loss_dict["kl_mean"] + intensity_nelbo).detach(),
                "kl_prf": loss_dict["kl_prf_mean"].detach(),
                "kl_intensity": intensity_nelbo.detach(),
                "kl_bg": loss_dict["kl_bg_mean"].detach(),
            },
        }
