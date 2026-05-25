"""Merging integrator: integrator + careless-like merging in one model.

The pixel-level loss trains integration (profile, bg, encoder).
The merge loss trains merging (F²_h, scale).
F² never enters the pixel-level rate — clean separation.

    pixels → encoder → I_obs_i ─→ pixel_loss (Poisson NLL)
                          │
                          └──→ merge_loss (Normal NLL on I_obs vs scale × F²_h)
                                  ↑
                             F²_h embedding
"""

import math
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch import Tensor
from torch.distributions import Gamma, Normal

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


class MergingIntegrator(BaseIntegrator):
    """Joint integration + merging model.

    Integration layer (pixel-level):
        Encoder predicts per-observation qi (Gamma intensity).
        rate = qi_sample × profile + bg
        Poisson NLL on pixel counts.

    Merging layer (intensity-level):
        Per-HKL F² embedding (free parameter per unique reflection).
        Scale function converts F² to observed intensity scale.
        Normal NLL: I_obs ~ Normal(scale × F², sigma_obs)
        Wilson KL prior on F².

    The encoder bridges both: pixel_loss teaches it to extract intensity
    from pixels; merge_loss teaches it to produce I values consistent
    with a single F² per HKL.
    """

    _MANUAL_OPTIMIZATION = True

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
        self.automatic_optimization = False

        if cfg.n_hkl is None:
            raise ValueError("MergingIntegrator requires n_hkl in config.")

        # Per-HKL F² embedding (softplus for positivity)
        self.raw_F_sq = nn.Embedding(cfg.n_hkl, 1, sparse=True)
        nn.init.constant_(
            self.raw_F_sq.weight,
            math.log(math.expm1(max(cfg.scaling_init_mu, 1e-6))),
        )

        # Scale function
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

        self.scaling_lr = (
            cfg.scaling_lr if cfg.scaling_lr is not None else cfg.lr
        )
        self.merge_weight = getattr(cfg, "merge_weight", 1.0)
        self._clip_val = getattr(cfg, "gradient_clip_val", 1.0)
        self._clip_algo = getattr(cfg, "gradient_clip_algorithm", "norm")

    def _get_F_sq(self, asu_ids: Tensor) -> Tensor:
        return Fn.softplus(self.raw_F_sq(asu_ids).squeeze(-1))

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

        # Encoders
        position = _get_normalized_position(metadata, device)
        x_profile = self.encoders["profile"](
            shoebox_reshaped, position=position
        )
        x_k_i = self.encoders["k_i"](shoebox_reshaped)
        x_r_i = self.encoders["r_i"](shoebox_reshaped)
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)

        # Surrogates
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

        # Integration: encoder-predicted intensity → pixel rate
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

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
        out["asu_id"] = metadata["asu_id"].long().to(device)
        if "group_label" in metadata:
            _add_group_outputs(out, metadata, self.loss)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }

    def _merge_loss(
        self, qi: Gamma, metadata: dict, device: torch.device
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Careless-like merge loss: Normal NLL on I_obs vs scale × F²_h."""
        asu_ids = metadata["asu_id"].long().to(device)
        F_sq_h = self._get_F_sq(asu_ids)
        scale = self._get_scale(metadata, device)

        # Predicted intensity for this observation
        I_pred = scale * F_sq_h

        # Observed intensity from encoder (posterior mean)
        I_obs = qi.mean

        # Uncertainty from encoder (posterior std)
        sigma_obs = qi.variance.sqrt().clamp(min=1e-6)

        # Normal NLL: how well does scale × F²_h explain the encoder's I_obs?
        merge_nll = -Normal(I_pred, sigma_obs).log_prob(I_obs)

        components = {
            "merge_nll": merge_nll.mean().detach(),
            "F_sq_mean": F_sq_h.mean().detach(),
            "scale_mean": scale.mean().detach(),
        }

        return merge_nll.mean() * self.merge_weight, components

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]

        group_labels = metadata["group_label"].long()

        # Pixel-level loss (integration quality)
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

        _log_loss(
            self,
            kl=loss_dict["kl_mean"],
            nll=loss_dict["neg_ll_mean"],
            total_loss=total_loss,
            step=step,
            kl_components={
                k.removesuffix("_mean"): v
                for k, v in loss_dict.items()
                if k in ("kl_prf_mean", "kl_i_mean", "kl_bg_mean")
            },
        )

        # Profile penalty
        penalty, penalty_components = self._profile_basis_penalty()
        for name, value in penalty_components.items():
            self.log(f"{step} {name}", value, on_step=False, on_epoch=True)
        total_loss = total_loss + penalty

        # Merge loss (merging quality — separate gradient pathway)
        merge_loss, merge_components = self._merge_loss(
            outputs["qi"], metadata, shoebox.device
        )
        for name, value in merge_components.items():
            self.log(f"{step} {name}", value, on_step=False, on_epoch=True)
        total_loss = total_loss + merge_loss

        # Log qi stats
        with torch.no_grad():
            qi = outputs["qi"]
            self.log(
                f"{step} qi_var_mean",
                qi.variance.mean(),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} qi_mean_mean",
                qi.mean.mean(),
                on_step=False,
                on_epoch=True,
            )

        return {
            "loss": total_loss,
            "forward_out": forward_out,
            "loss_components": {
                "loss": total_loss.detach(),
                "nll": loss_dict["neg_ll_mean"].detach(),
                "kl": loss_dict["kl_mean"].detach(),
                "kl_prf": loss_dict["kl_prf_mean"].detach(),
                "kl_i": loss_dict["kl_i_mean"].detach(),
                "kl_bg": loss_dict["kl_bg_mean"].detach(),
            },
        }

    def training_step(self, batch, _batch_idx):
        main_opt, sparse_opt = self.optimizers()
        result = self._step(batch, step="train")
        loss = result["loss"]

        main_opt.zero_grad()
        sparse_opt.zero_grad()
        self.manual_backward(loss)
        if self._clip_val is not None and self._clip_val > 0:
            self.clip_gradients(
                main_opt,
                gradient_clip_val=self._clip_val,
                gradient_clip_algorithm=self._clip_algo,
            )
        main_opt.step()
        sparse_opt.step()

        schedulers = self.lr_schedulers()
        if schedulers is not None:
            if isinstance(schedulers, list):
                for s in schedulers:
                    s.step()
            else:
                schedulers.step()

        return result

    def validation_step(self, batch, _batch_idx):
        return self._step(batch, step="val")

    def configure_optimizers(self):
        sparse_params = list(self.raw_F_sq.parameters())
        sparse_ids = {id(p) for p in sparse_params}
        other_params = [
            p
            for p in self.parameters()
            if p.requires_grad and id(p) not in sparse_ids
        ]

        main_opt = torch.optim.Adam(
            other_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sparse_opt = torch.optim.SparseAdam(
            sparse_params,
            lr=self.scaling_lr,
        )

        if self.lr_schedule is None:
            return [main_opt, sparse_opt]

        if self.lr_schedule == "cosine_warmup":
            max_epochs = self.trainer.max_epochs
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                main_opt,
                lr_lambda=self._cosine_warmup_lambda(max_epochs),
            )
            return [main_opt, sparse_opt], [
                {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
            ]

        return [main_opt, sparse_opt]
