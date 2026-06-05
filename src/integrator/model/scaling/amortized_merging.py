"""Amortized merging: per-HKL variational intensity from an amortized encoder.

A sibling of `ConjugateMergingIntegrator`. Same generative model and the same
scale / profile / background structure, but the per-HKL intensity posterior
`q(I_h)` is *amortized* — produced by the standard Gamma intensity surrogate
applied to per-HKL-aggregated encoder features — instead of *derived* in closed
form via the Poisson-Gamma conjugate update. This gives a controlled
conjugate-vs-amortized A/B (only the `I_h` mechanism differs).

Architecture (5 encoders, matching the base / hierarchical convention):

    shoebox -> profile encoder           -> qp                 (per obs)
    shoebox -> k_bg, r_bg encoders       -> qbg                (per obs)
    shoebox -> k_i, r_i encoders         -> x_k_i, x_r_i       (per obs)
        scatter_mean by asu_id           -> z_k_h, z_r_h       (per HKL)
        surrogates["qi"](z_k_h, z_r_h)   -> q(I_h) = Gamma     (per HKL)

All observations of an HKL share `q(I_h)`; the intensity sample is drawn once
per HKL and broadcast. The Wilson KL is applied per-HKL (counted once, no
overcounting). Unlike `DeepSetsMergingIntegrator` (one `k_i` encoder + bespoke
MLP heads), this reuses the 5-encoder layout and the existing `GammaA/B` `qi`
surrogate, so the only difference from `conjugate_merging` is derived vs learned
`I_h`.

Pair with `GroupedAsuIdBatchSampler` (`group_by_asu_id: true`) so each batch
contains complete HKL groups for the aggregation to be meaningful.
"""

from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gamma, kl_divergence

from integrator import configs
from integrator.model.integrators.base_integrator import (
    BaseIntegrator,
    _log_loss,
)
from integrator.model.integrators.hierarchical_integrator import (
    _add_group_outputs,
    _get_normalized_position,
    _sample_profile,
)
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
)
from integrator.model.scaling.chebyshev_scale import (
    ChebyshevScale,
    MLPScale,
    SpatialChebyshevScale,
)
from integrator.model.scaling.deepsets_merging import _scatter_mean_compact


class AmortizedMergingIntegrator(BaseIntegrator):
    """Per-HKL amortized variational intensity (sibling of conjugate merging).

    `q(I_h)` is produced by the `qi` Gamma surrogate applied to per-HKL
    aggregated `k_i` / `r_i` encoder features. Best paired with
    `GroupedAsuIdBatchSampler` (`group_by_asu_id: true`).
    """

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
            raise ValueError("AmortizedMergingIntegrator requires n_hkl.")

        self.n_hkl = cfg.n_hkl
        d = cfg.encoder_out

        # EMA buffer of per-HKL aggregated (k, r) features, used to decode the
        # merged q(I_h) at inference time (when a batch may not hold every
        # observation of an HKL).
        self.register_buffer("feat_k_ema", torch.zeros(cfg.n_hkl, d))
        self.register_buffer("feat_r_ema", torch.zeros(cfg.n_hkl, d))
        self.register_buffer(
            "feat_seen", torch.zeros(cfg.n_hkl, dtype=torch.bool)
        )
        self.ema_momentum = float(getattr(cfg, "ema_momentum", 0.95))

        # Per-HKL Wilson KL weight. ELBO-consistent scaling is N_HKL/N_obs
        # (~0.04 for HEWL); raise above for stronger Wilson regularization.
        self.merge_kl_weight = float(getattr(cfg, "merge_kl_weight", 1.0))

        # Scale function (identical to the other merging integrators)
        if cfg.scale_mlp:
            self.scale_fn = MLPScale(
                hidden_dim=cfg.scale_mlp_hidden,
                n_layers=cfg.scale_mlp_layers,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                beam_center=cfg.scale_beam_center,
                r_max=cfg.scale_r_max,
                d_min=getattr(cfg, "dmin", 1.0),
                d_max=60.0,
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

    # ------------------------------------------------------------------

    def _get_scale(self, metadata: dict, device: torch.device) -> Tensor:
        frame = metadata["xyzcal.px.2"].to(device).float()
        lp = metadata["lp"].to(device).float().clamp(min=1e-8)
        if isinstance(self.scale_fn, MLPScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            d = metadata["d"].to(device).float()
            return self.scale_fn(frame, x_det, y_det, lp, d)
        elif isinstance(self.scale_fn, SpatialChebyshevScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            return self.scale_fn(frame, x_det, y_det) / lp
        else:
            return self.scale_fn(frame) / lp

    def _update_ema(
        self, z_k_h: Tensor, z_r_h: Tensor, unique_hkls: Tensor
    ) -> None:
        """EMA update of the per-HKL aggregated features for HKLs in this batch."""
        was_seen = self.feat_seen[unique_hkls].unsqueeze(1)
        m = self.ema_momentum
        old_k = self.feat_k_ema[unique_hkls]
        old_r = self.feat_r_ema[unique_hkls]
        self.feat_k_ema[unique_hkls] = torch.where(
            was_seen, m * old_k + (1 - m) * z_k_h, z_k_h
        )
        self.feat_r_ema[unique_hkls] = torch.where(
            was_seen, m * old_r + (1 - m) * z_r_h, z_r_h
        )
        self.feat_seen[unique_hkls] = True

    def get_merged_qi(self) -> Gamma:
        """Per-HKL Gamma posterior from the EMA features (for MTZ output)."""
        return self.surrogates["qi"](self.feat_k_ema, self.feat_r_ema)

    # ------------------------------------------------------------------

    def _forward_impl(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        counts = torch.clamp(counts, min=0)

        b = shoebox.shape[0]
        device = shoebox.device
        shoebox_masked = shoebox * mask
        shoebox_reshaped = shoebox_masked.reshape(b, 1, *self.shoebox_shape)

        # Encoders (5): profile, k_i, r_i, k_bg, r_bg
        position = _get_normalized_position(metadata, device)
        x_profile = self.encoders["profile"](
            shoebox_reshaped, position=position
        )
        x_k_i = self.encoders["k_i"](shoebox_reshaped)
        x_r_i = self.encoders["r_i"](shoebox_reshaped)
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)

        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)
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

        # Aggregate intensity features per HKL, then amortize q(I_h).
        asu_ids = metadata["asu_id"].long().to(device)
        z_k_h, inverse, unique_hkls = _scatter_mean_compact(x_k_i, asu_ids)
        z_r_h, _, _ = _scatter_mean_compact(x_r_i, asu_ids)

        if self.training:
            with torch.no_grad():
                self._update_ema(z_k_h.detach(), z_r_h.detach(), unique_hkls)

        qi_h = self.surrogates["qi"](z_k_h, z_r_h)  # batch shape (n_unique,)

        # Sample intensity once per HKL, broadcast to observations.
        zI_h = qi_h.rsample([self.mc_samples]).clamp(min=1e-10)  # (S, n_unique)
        zI = zI_h[:, inverse]  # (S, B) — shared across obs of same HKL

        scale = self._get_scale(metadata, device)  # (B,)
        zI_scaled = (scale.unsqueeze(0) * zI).unsqueeze(-1).permute(1, 0, 2)
        # (B, S, 1)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)

        rate = zI_scaled * zp + zbg

        if "is_coset" in metadata:
            coset = metadata["is_coset"].bool().view(-1, 1, 1)
            rate = torch.where(coset, zbg, rate)

        # Per-observation qi (broadcast from qi_h) for the loss/output interface.
        qi = Gamma(
            qi_h.concentration[inverse].clamp(min=1e-6),
            qi_h.rate[inverse].clamp(min=1e-12),
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
        out["asu_id"] = asu_ids
        if "group_label" in metadata:
            _add_group_outputs(out, metadata, self.loss)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
            "qi_h": qi_h,
            "inverse": inverse,
            "unique_hkls": unique_hkls,
        }

    # ------------------------------------------------------------------

    def _wilson_kl_per_hkl(
        self, qi_h: Gamma, inverse: Tensor, metadata: dict
    ) -> Tensor:
        """Wilson KL on per-HKL q(I_h). Counted once per HKL (no overcounting)."""
        device = qi_h.concentration.device
        # d is identical for all obs of a HKL; average to per-HKL.
        d_per_obs = metadata["d"].to(device).float().unsqueeze(-1)
        d_per_hkl, _, _ = _scatter_mean_compact(d_per_obs, inverse)
        d_per_hkl = d_per_hkl.squeeze(-1)

        s_sq = 1.0 / (4.0 * d_per_hkl.clamp(min=1e-6).pow(2))
        tau = self.loss._get_tau({"d": d_per_hkl}, s_sq, device)
        p_i = Gamma(torch.ones_like(tau), tau)
        return kl_divergence(qi_h, p_i)

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]
        qi_h = outputs["qi_h"]
        inverse = outputs["inverse"]

        group_labels = metadata["group_label"].long()

        # Standard loss handles Poisson NLL + profile KL + bg KL.
        # Set pi_weight=0 in the YAML — the per-obs intensity KL is overcounted;
        # we apply the per-HKL Wilson KL below instead.
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

        kl_i_per_hkl = self._wilson_kl_per_hkl(qi_h, inverse, metadata)
        kl_i = kl_i_per_hkl.mean() * self.merge_kl_weight
        total_loss = total_loss + kl_i

        _log_loss(
            self,
            kl=loss_dict["kl_mean"] + kl_i,
            nll=loss_dict["neg_ll_mean"],
            total_loss=total_loss,
            step=step,
            kl_components={
                "kl_prf": loss_dict["kl_prf_mean"],
                "kl_bg": loss_dict["kl_bg_mean"],
                "kl_i_hkl": kl_i.detach(),
            },
        )

        penalty, penalty_components = self._profile_basis_penalty()
        for name, value in penalty_components.items():
            self.log(f"{step} {name}", value, on_step=False, on_epoch=True)
        total_loss = total_loss + penalty

        with torch.no_grad():
            self.log(
                f"{step} qi_h_mean", qi_h.mean.mean(),
                on_step=False, on_epoch=True,
            )
            self.log(
                f"{step} qi_h_var", qi_h.variance.mean(),
                on_step=False, on_epoch=True,
            )
            self.log(
                f"{step} qi_h_k", qi_h.concentration.mean(),
                on_step=False, on_epoch=True,
            )
            self.log(
                f"{step} qi_h_rate", qi_h.rate.mean(),
                on_step=False, on_epoch=True,
            )
            self.log(
                f"{step} buffer_coverage", self.feat_seen.float().mean(),
                on_step=False, on_epoch=True,
            )
            self.log(
                f"{step} n_unique_hkl",
                torch.tensor(len(outputs["unique_hkls"]), dtype=torch.float),
                on_step=False, on_epoch=True,
            )
            self.log(
                f"{step} obs_per_hkl",
                torch.tensor(
                    counts.shape[0] / len(outputs["unique_hkls"]),
                    dtype=torch.float,
                ),
                on_step=False, on_epoch=True,
            )

        return {
            "loss": total_loss,
            "forward_out": forward_out,
            "loss_components": {
                "loss": total_loss.detach(),
                "nll": loss_dict["neg_ll_mean"].detach(),
                "kl": (loss_dict["kl_mean"] + kl_i).detach(),
                "kl_prf": loss_dict["kl_prf_mean"].detach(),
                "kl_i": kl_i.detach(),
                "kl_bg": loss_dict["kl_bg_mean"].detach(),
            },
        }
