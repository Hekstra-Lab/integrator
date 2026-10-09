"""Hierarchical per-observation integrator + learned scale + merge.

The per-obs-integration scaling model (the 35+ path). It reuses the
`HierarchicalIntegrator` forward verbatim -- a *free* per-observation intensity
`q(I_i)` with `rate = I_i*prf + bg` (genuine integration) -- and adds, on top:

  - a per-observation scale `s_i = scale_fn(geometry)`;
  - a per-HKL merge of the descaled intensities (forward weighted-least-squares
    combine of `q(I_i)` and `s_i`, grouped by the anomalous asu_id);
  - a scaling-consistency coupling `(m_i - s_i*I_h)^2` that identifies `s_i`
    (the data-only term, now on the proper latent `m_i = E[q(I_i)]`); and
  - a Wilson prior on the merged `I_h`, plus an intensity-dependent
    overdispersion error model for the merged sigma.

Designed to **warm-start from a trained `HierarchicalIntegrator` checkpoint**
(`init_from_checkpoint`): the encoders + qp + qbg + qi transfer (identical
layout), and with `freeze_integration: true` only the scale + merge head trains
-- so the integration that earns the high peak heights is preserved and we test
whether a learned scale+merge matches DIALS scaling. Unfreeze to fine-tune.

Pair with `group_by_asu_id: true` (the merge needs complete HKL groups).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Gamma, kl_divergence

from integrator import configs
from integrator.model.integrators.base_integrator import (
    BaseIntegrator,
    _log_loss,
)
from integrator.model.integrators.hierarchical_integrator import (
    _sample_profile,
)
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
)
from integrator.model.scaling.chebyshev_scale import (
    ChebyshevScale,
    MLPScale,
    PhysicalScale,
    SpatialChebyshevScale,
)
from integrator.model.scaling.conjugate_merging import _scatter_sum_compact

logger = logging.getLogger(__name__)


class HierarchicalScalingIntegrator(BaseIntegrator):
    """Hierarchical per-obs integrator with a learned scale + merge head."""

    REQUIRED_ENCODERS = {
        "profile": configs.ProfileEncoderArgs,
        "k_i": configs.IntensityEncoderArgs,
        "r_i": configs.IntensityEncoderArgs,
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

        if cfg.n_hkl is None:
            raise ValueError("HierarchicalScalingIntegrator requires n_hkl.")
        if getattr(self.loss, "_apply_lp", False):
            raise ValueError(
                "Scale carries LP, so I_h is LP-corrected; lp_correction would "
                "double-count it. Set loss.args.lp_correction: false."
            )

        self.n_hkl = int(cfg.n_hkl)
        self.alpha_W = float(getattr(cfg, "wilson_alpha", 1.0))
        self.merge_kl_weight = float(getattr(cfg, "merge_kl_weight", 1.0))
        self.consistency_weight = float(getattr(cfg, "consistency_weight", 1.0))
        self.scaling_lr = getattr(cfg, "scaling_lr", None)
        self.merge_overdispersion = bool(
            getattr(cfg, "merge_overdispersion", True)
        )
        self.freeze_integration = bool(getattr(cfg, "freeze_integration", False))
        self.scale_restraint_weight = float(
            getattr(cfg, "scale_absorption_restraint", 0.0)
        )
        if self.freeze_integration and not getattr(
            cfg, "init_from_checkpoint", None
        ):
            logger.warning(
                "freeze_integration=True with no init_from_checkpoint: the "
                "encoders + qp/qbg/qi are FROZEN AT RANDOM INIT, so the merged "
                "output is noise. Set init_from_checkpoint to a trained "
                "hierarchical checkpoint."
            )

        # Scale function (same options as the other merging integrators).
        if getattr(cfg, "scale_physical", False):
            # DIALS-style: smooth scale(phi) x decay(phi,d) x SH absorption on the
            # crystal-frame direction (precomputed `absorption_sh`, see
            # scripts/extract_crystal_frame_sh.py). scale_sh_lmax MUST match the
            # extractor's --lmax.
            self.scale_fn = PhysicalScale(
                n_sh=(int(cfg.scale_sh_lmax) + 1) ** 2 - 1,
                degree_scale=cfg.scale_degree,
                degree_decay=cfg.scale_degree_decay,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                absorption_init_std=cfg.scale_absorption_init_std,
            )
            if self.scale_restraint_weight == 0.0:
                logger.warning(
                    "scale_physical with scale_absorption_restraint=0: the "
                    "absorption surface is bounded only by its low dimension. "
                    "Set scale_absorption_restraint (~1e-2) to oppose run-away "
                    "and protect the anomalous (odd-l) band."
                )
        elif cfg.scale_mlp:
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
            )
        elif cfg.scale_spatial:
            self.scale_fn = SpatialChebyshevScale(
                degree_frame=cfg.scale_degree,
                degree_radius=cfg.scale_degree_radius,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                beam_center=cfg.scale_beam_center,
                r_min=cfg.scale_r_min,
                r_max=cfg.scale_r_max,
            )
        else:
            self.scale_fn = ChebyshevScale(
                degree=cfg.scale_degree,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
            )

        # Intensity-dependent overdispersion (AIMLESS SdAdd-style): the per-obs
        # variance used in the merge/consistency is v_eff = v + phi*m^2, phi>=0
        # learned. Down-weights bright observations beyond counting noise and
        # calibrates the merged sigma -- central to anomalous peak height.
        if self.merge_overdispersion:
            self.log_phi = nn.Parameter(torch.tensor(-5.0))  # phi ~ 0 at init

        # Merged per-HKL posterior, populated by finalize_merge (clean pass).
        self.register_buffer(
            "alpha_buffer",
            torch.full((self.n_hkl,), self.alpha_W),
            persistent=False,
        )
        self.register_buffer(
            "beta_buffer", torch.ones(self.n_hkl), persistent=False
        )
        self.register_buffer(
            "buffer_seen",
            torch.zeros(self.n_hkl, dtype=torch.bool),
            persistent=False,
        )

        if self.freeze_integration:
            self._freeze_integration()

    # ------------------------------------------------------------------

    def _freeze_integration(self) -> None:
        """Freeze encoders + qp/qbg/qi so only scale + merge head trains."""
        frozen = 0
        for enc in self.encoders.values():
            for p in enc.parameters():
                p.requires_grad_(False)
                frozen += 1
        for key in ("qp", "qbg", "qi"):
            if key in self.surrogates:
                for p in self.surrogates[key].parameters():
                    p.requires_grad_(False)
                    frozen += 1
        logger.info(
            "freeze_integration: froze %d integration tensors; training scale "
            "+ merge head only.",
            frozen,
        )

    def _get_scale(self, metadata: dict, device: torch.device) -> Tensor:
        frame = metadata["xyzcal.px.2"].to(device).float()
        lp = metadata["lp"].to(device).float().clamp(min=1e-8)
        if isinstance(self.scale_fn, PhysicalScale):
            if "absorption_sh" not in metadata:
                raise KeyError(
                    "PhysicalScale needs 'absorption_sh' in metadata; run "
                    "scripts/extract_crystal_frame_sh.py and point the data "
                    "loader's reference at the augmented metadata file."
                )
            d = metadata["d"].to(device).float()
            a = metadata["absorption_sh"].to(device).float()
            return self.scale_fn(frame, d, a) / lp
        if isinstance(self.scale_fn, MLPScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            d = metadata["d"].to(device).float()
            return self.scale_fn(frame, x_det, y_det, lp, d)
        elif isinstance(self.scale_fn, SpatialChebyshevScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            return self.scale_fn(frame, x_det, y_det) / lp
        return self.scale_fn(frame) / lp

    def _wilson_tau(self, d: Tensor) -> Tensor:
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        return self.loss._get_tau({"d": d}, s_sq, d.device)

    def _v_eff(self, m: Tensor, v: Tensor) -> Tensor:
        """Per-obs variance with the intensity-dependent overdispersion."""
        if self.merge_overdispersion:
            return v + F.softplus(self.log_phi) * m.pow(2)
        return v

    def _merge_wls(
        self, m: Tensor, v_eff: Tensor, s: Tensor, inverse: Tensor, n_unique: int
    ) -> tuple[Tensor, Tensor]:
        """Forward weighted-least-squares merge: I_h = argmin Σ w (m - s I_h)^2.

        Solution I_h = Σ w m s / Σ w s^2 (w = 1/v_eff), with curvature variance
        Var(I_h) = 1 / Σ w s^2. Carries gradient through s (and m if unfrozen).
        """
        device = m.device
        w = 1.0 / v_eff.clamp(min=1e-6)

        def scatter(x: Tensor) -> Tensor:
            return torch.zeros(
                n_unique, device=device, dtype=x.dtype
            ).scatter_add_(0, inverse, x)

        den = scatter(w * s * s).clamp(min=1e-12)
        i_h_mean = scatter(w * m * s) / den
        i_h_var = (1.0 / den).clamp(min=1e-12)
        return i_h_mean, i_h_var

    def _consistency_loss(
        self,
        m: Tensor,
        v_eff: Tensor,
        s: Tensor,
        inverse: Tensor,
        i_h_target: Tensor,
    ) -> Tensor:
        """Gaussian NLL of the scaled residual (scaling + error-model objective).

            L = mean_i [ (m_i - s_i I_h)^2 / v_eff_i + log v_eff_i ]

        The quadratic identifies the scale `s_i` (`i_h_target` = the stop-grad
        WLS merge); `m_i = E[q(I_i)]` is the proper per-obs latent. The
        `log v_eff` normalization is what makes the overdispersion identifiable
        -- it pins `phi` at reduced-chi^2 = 1; minimizing the weighted SSR alone
        would drive `phi -> infinity` (down-weight everything). Only
        multiply-measured HKLs contribute (singletons carry no internal-
        consistency information and would bias `phi`). Gauge-invariant in
        `s_i*I_h`.
        """
        ve = v_eff.clamp(min=1e-6)
        n_unique = i_h_target.shape[0]
        gsize = torch.zeros(n_unique, device=m.device).scatter_add_(
            0, inverse, torch.ones_like(m)
        )
        mult = gsize[inverse] >= 2.0
        if not bool(mult.any()):
            return m.new_zeros(())
        resid = m - s * i_h_target[inverse]
        per = (resid.pow(2) / ve + torch.log(ve))[mult]
        return per.sum() / mult.sum()

    # ------------------------------------------------------------------

    def _forward_impl(
        self, counts: Tensor, shoebox: Tensor, mask: Tensor, metadata: dict
    ) -> dict[str, Any]:
        counts = torch.clamp(counts, min=0)
        b = shoebox.shape[0]
        device = shoebox.device
        sr = (shoebox * mask).reshape(b, 1, *self.shoebox_shape)

        # Hierarchical integration (free per-obs qi), identical to the integrator.
        x_profile = self.encoders["profile"](sr)
        x_k_i = self.encoders["k_i"](sr)
        x_r_i = self.encoders["r_i"](sr)
        x_k_bg = self.encoders["k_bg"](sr)
        x_r_bg = self.encoders["r_bg"](sr)

        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)
        qi = self.surrogates["qi"](x_k_i, x_r_i)
        prf_labels = metadata.get(
            "profile_group_label", metadata.get("group_label")
        )
        prf_labels = prf_labels.long() if prf_labels is not None else None
        qp = self.surrogates["qp"](
            x_profile,
            mc_samples=self.mc_samples,
            group_labels=prf_labels,
            metadata=metadata,
        )

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        rate = zI * zp + zbg  # I_i * prf + bg  (no scale in the likelihood)

        if (
            self.coset_mode in ("override", "override_no_kl", "aux")
            and "is_coset" in metadata
        ):
            coset = metadata["is_coset"].bool().view(-1, 1, 1)
            rate = torch.where(coset, zbg, rate)

        # Scale + per-HKL merge.
        scale = self._get_scale(metadata, device)
        asu = metadata["asu_id"].long().to(device)
        d_obs = metadata["d"].to(device).float()
        d_sum, inverse, unique_asu = _scatter_sum_compact(d_obs, asu)
        count_h, _, _ = _scatter_sum_compact(torch.ones_like(d_obs), asu)
        tau_h = self._wilson_tau(d_sum / count_h.clamp(min=1.0))

        m = qi.mean
        v_eff = self._v_eff(m, qi.variance)
        i_h_mean, i_h_var = self._merge_wls(
            m, v_eff, scale, inverse, unique_asu.shape[0]
        )

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
        # NOTE: no _add_group_outputs here -- this model replaces the hierarchical
        # group prior with the merge + consistency, so it needs no group-level
        # outputs, and (unlike training batches) the predict loader's metadata may
        # not carry "group_label". Keeping the forward independent of it lets
        # finalize_merge / predict run on the grouped predict loader.

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
            "scale": scale,
            "inverse": inverse,
            "unique_asu": unique_asu,
            "m": m,
            "v_eff": v_eff,
            "i_h_mean": i_h_mean,
            "i_h_var": i_h_var,
            "tau_h": tau_h,
        }

    # ------------------------------------------------------------------

    def _merged_gamma(
        self, i_h_mean: Tensor, i_h_var: Tensor
    ) -> Gamma:
        """Gamma moment-matched to the merged (mean, var)."""
        var = i_h_var.clamp(min=1e-12)
        mean = i_h_mean.clamp(min=1e-8)
        return Gamma((mean.pow(2) / var).clamp(min=1e-6), (mean / var).clamp(min=1e-12))

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]
        gl = metadata.get("group_label")
        group_labels = (
            gl.long()
            if gl is not None
            else torch.zeros(
                counts.shape[0], dtype=torch.long, device=forward_out["rates"].device
            )
        )

        # Standard hierarchical ELBO (NLL + KL_prf + KL_bg; set pi_weight=0 in
        # the loss so the per-obs group-prior KL is skipped -- the coupling below
        # is the intensity prior now). Frozen integration => these have no grad.
        loss_dict = self.loss(
            rate=forward_out["rates"],
            counts=forward_out["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=forward_out["mask"],
            group_labels=group_labels,
            metadata=metadata,
            coset_mode=self.coset_mode,
        )
        total_loss = loss_dict["loss"]

        # Scaling-consistency coupling (identifies the scale) + Wilson on I_h.
        consist = self._consistency_loss(
            outputs["m"],
            outputs["v_eff"],
            outputs["scale"],
            outputs["inverse"],
            outputs["i_h_mean"].detach(),
        )
        q_I_h = self._merged_gamma(outputs["i_h_mean"], outputs["i_h_var"])
        p_I_h = Gamma(
            self.alpha_W * torch.ones_like(outputs["tau_h"]),
            outputs["tau_h"].clamp(min=1e-12),
        )
        wilson_kl = (
            kl_divergence(q_I_h, p_I_h).sum()
            / counts.shape[0]
            * self.merge_kl_weight
        )
        total_loss = (
            total_loss + self.consistency_weight * consist + wilson_kl
        )

        _log_loss(
            self,
            kl=loss_dict["kl_mean"] + wilson_kl,
            nll=loss_dict["neg_ll_mean"],
            total_loss=total_loss,
            step=step,
            kl_components={
                "kl_prf": loss_dict["kl_prf_mean"],
                "kl_bg": loss_dict["kl_bg_mean"],
                "wilson_i_h": wilson_kl.detach(),
            },
        )

        penalty, penalty_components = self._profile_basis_penalty()
        for name, value in penalty_components.items():
            self.log(f"{step} {name}", value, on_step=False, on_epoch=True)
        total_loss = total_loss + penalty

        # PhysicalScale absorption/decay restraint: the term that opposes the
        # consistency objective pulling the surface to absorb the anomalous
        # signal (and running away, cf. the Beer-Lambert absorption).
        if (
            isinstance(self.scale_fn, PhysicalScale)
            and self.scale_restraint_weight > 0.0
        ):
            restraint = self.scale_restraint_weight * self.scale_fn.restraint_penalty()
            total_loss = total_loss + restraint
            self.log(
                f"{step} scale_restraint", restraint.detach(), on_epoch=True
            )

        with torch.no_grad():
            self.log(f"{step} consistency", consist.detach(), on_epoch=True)
            self.log(
                f"{step} scale_mean", outputs["scale"].mean(), on_epoch=True
            )
            self.log(
                f"{step} scale_std", outputs["scale"].std(), on_epoch=True
            )
            self.log(
                f"{step} i_h_mean", outputs["i_h_mean"].mean(), on_epoch=True
            )
            if self.merge_overdispersion:
                self.log(
                    f"{step} phi",
                    F.softplus(self.log_phi).detach(),
                    on_epoch=True,
                )
            if isinstance(self.scale_fn, PhysicalScale):
                # Watch for absorption run-away (climbing rms / scale_std).
                self.log(
                    f"{step} abs_c_rms",
                    self.scale_fn.absorption_c.detach().pow(2).mean().sqrt(),
                    on_epoch=True,
                )

        return {"loss": total_loss, "forward_out": forward_out}

    # ------------------------------------------------------------------

    def get_merged_qi(self) -> Gamma:
        """Per-HKL Gamma posterior from the merge buffers (for MTZ output)."""
        return Gamma(
            self.alpha_buffer.clamp(min=1e-6),
            self.beta_buffer.clamp(min=1e-12),
        )

    @torch.no_grad()
    def finalize_merge(self, dataloader) -> None:
        """Merge per anomalous HKL over the full dataset into the buffers.

        One clean pass: per batch, forward (free qi) -> scale -> WLS merge, and
        store the moment-matched Gamma per HKL. Requires complete HKL groups per
        batch (group_by_asu_id); raises if an HKL spans batches.
        """
        self.eval()
        device = self.alpha_buffer.device
        self.alpha_buffer.fill_(self.alpha_W)
        self.beta_buffer.fill_(1.0)
        seen = torch.zeros(self.n_hkl, dtype=torch.bool, device=device)

        for batch in dataloader:
            counts, shoebox, mask, metadata = batch
            out = self(
                counts.to(device), shoebox.to(device), mask.to(device), metadata
            )
            unique = out["unique_asu"]
            q = self._merged_gamma(out["i_h_mean"], out["i_h_var"])
            if bool(seen[unique].any()):
                raise RuntimeError(
                    "finalize_merge requires a grouped (group_by_asu_id) loader "
                    "so each HKL is complete in one batch; found an HKL spanning "
                    "batches."
                )
            self.alpha_buffer[unique] = q.concentration
            self.beta_buffer[unique] = q.rate
            seen[unique] = True
        self.buffer_seen.copy_(seen)
        logger.info(
            "finalize_merge: %d/%d HKLs populated", int(seen.sum()), self.n_hkl
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Adam; scale_fn gets its own group at scaling_lr. Frozen params are
        excluded. Defers to base when scaling_lr is None (and nothing frozen-
        specific is needed)."""
        if self.scaling_lr is None:
            params = [p for p in self.parameters() if p.requires_grad]
            return torch.optim.Adam(
                params, lr=self.lr, weight_decay=self.weight_decay
            )
        scale_params: list[nn.Parameter] = []
        decoder_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("scale_fn."):
                scale_params.append(param)
            elif (
                self.decoder_weight_decay is not None
                and name.endswith("surrogates.qp.decoder.weight")
            ):
                decoder_params.append(param)
            else:
                other_params.append(param)
        groups: list[dict] = []
        if other_params:
            groups.append(
                {"params": other_params, "weight_decay": self.weight_decay}
            )
        if decoder_params:
            groups.append(
                {
                    "params": decoder_params,
                    "weight_decay": self.decoder_weight_decay,
                }
            )
        if scale_params:
            groups.append(
                {
                    "params": scale_params,
                    "weight_decay": self.weight_decay,
                    "lr": self.scaling_lr,
                }
            )
        return torch.optim.Adam(groups, lr=self.lr)
