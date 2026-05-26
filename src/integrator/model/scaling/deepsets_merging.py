"""Deep Sets merging: pixel-level integration + merging in one forward pass.

Encoder produces per-observation features from shoeboxes. Features
are aggregated by HKL via scatter_mean (a permutation-invariant set
function). A decoder maps the aggregated features to F per unique HKL.
The rate uses the merged F — single loss, no dual optimization.

    shoebox → encoder → z_i → scatter_mean by HKL → z_h → decoder → F_h
    rate = scale × F_h² × profile + bg
    loss = Poisson NLL + Wilson KL on F + profile KL + bg KL
"""

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch import Tensor
from torch.distributions import Gamma

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


def _scatter_mean(
    src: Tensor, index: Tensor, dim_size: int
) -> Tensor:
    """Differentiable scatter mean: average src rows by index."""
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    count.scatter_add_(0, index.unsqueeze(1), torch.ones(len(index), 1, device=src.device))
    return out / count.clamp(min=1)


class DeepSetsMergingIntegrator(BaseIntegrator):
    """Integration + merging via Deep Sets aggregation.

    Single forward pass, single loss. The encoder produces per-observation
    features. scatter_mean aggregates by HKL. The decoder maps aggregated
    features to F. Merging is architectural, not a loss term.

    For anomalous data: F(+) and F(-) have separate asu_ids, so they
    aggregate independently and get different F values.

    A running buffer accumulates encoder features across the full dataset
    for merged F extraction at inference time.
    """

    REQUIRED_ENCODERS = {
        "profile": configs.ProfileEncoderArgs,
        "k_i": configs.IntensityEncoderArgs,
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
            raise ValueError("DeepSetsMergingIntegrator requires n_hkl.")

        self.n_hkl = cfg.n_hkl
        d = cfg.encoder_out

        # Decoder: aggregated features → F
        self.F_head = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )
        nn.init.zeros_(self.F_head[-1].weight)
        nn.init.zeros_(self.F_head[-1].bias)

        # Running buffer for merged F at inference
        self.register_buffer("feat_sum", torch.zeros(cfg.n_hkl, d))
        self.register_buffer("feat_count", torch.zeros(cfg.n_hkl, 1))

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
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)

        # Background surrogate
        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)

        # Profile surrogate
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

        # Deep Sets aggregation: scatter_mean by HKL
        asu_ids = metadata["asu_id"].long().to(device)
        z_h = _scatter_mean(x_k_i, asu_ids, self.n_hkl)

        # Update running buffer (detached, for inference)
        if self.training:
            with torch.no_grad():
                self.feat_sum.scatter_add_(
                    0,
                    asu_ids.unsqueeze(1).expand_as(x_k_i),
                    x_k_i.detach(),
                )
                self.feat_count.scatter_add_(
                    0,
                    asu_ids.unsqueeze(1),
                    torch.ones(b, 1, device=device),
                )

        # Decode F from aggregated features (gradient flows through scatter_mean)
        z_h_per_obs = z_h[asu_ids]  # (B, d) — same for all obs of same HKL
        F_h = Fn.softplus(self.F_head(z_h_per_obs).squeeze(-1))  # (B,)

        # Scale
        scale = self._get_scale(metadata, device)

        # MC samples for profile and background
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)

        # Rate: scale × F² × profile + bg (F is MERGED, shared per HKL)
        F_sq = (scale * F_h.pow(2)).view(b, 1, 1)
        rate = F_sq * zp + zbg

        if "is_coset" in metadata:
            coset = metadata["is_coset"].bool().view(-1, 1, 1)
            rate = torch.where(coset, zbg, rate)

        # Dummy qi for _assemble_outputs compatibility
        # Use F_h stats to create a representative Gamma
        qi = Gamma(
            concentration=F_h.detach().clamp(min=0.1),
            rate=torch.ones_like(F_h),
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
            "F_h": F_h,
        }

    def get_merged_F(self) -> Tensor:
        """Get fully merged F from the running buffer (for MTZ output)."""
        z_h = self.feat_sum / self.feat_count.clamp(min=1)
        return Fn.softplus(self.F_head(z_h).squeeze(-1)).detach()

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]

        group_labels = metadata["group_label"].long()

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

        penalty, penalty_components = self._profile_basis_penalty()
        for name, value in penalty_components.items():
            self.log(f"{step} {name}", value, on_step=False, on_epoch=True)
        total_loss = total_loss + penalty

        with torch.no_grad():
            F_h = outputs["F_h"]
            self.log(
                f"{step} F_mean", F_h.mean(), on_step=False, on_epoch=True
            )
            self.log(
                f"{step} F_std", F_h.std(), on_step=False, on_epoch=True
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
