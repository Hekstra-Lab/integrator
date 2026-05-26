"""Deep Sets merging: per-HKL intensity from aggregated encoder features.

Encoder produces per-observation features. With group-batched sampling
(GroupedAsuIdSampler), each batch contains complete HKL groups. A
permutation-invariant aggregation (scatter_mean) reduces the per-obs
features to per-HKL features. A decoder maps these to a per-HKL Gamma
intensity distribution qi_h.

All observations of the same HKL share qi_h. The rate equation uses a
shared intensity sample, drawn once per HKL. The Wilson KL is on qi_h
(per-HKL), eliminating overcounting.

Architecture:
    shoebox → encoder → z_i
    scatter_mean by asu_id → z_h
    z_h → I_head → (k_h, rate_h) → qi_h = Gamma
    broadcast qi_h sample to obs → scale × I × profile + bg
    loss = Poisson NLL + Wilson KL on qi_h (per-HKL) + KL_prf + KL_bg
"""

import math
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as Fn
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


def _scatter_mean_compact(
    src: Tensor, index: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Scatter mean over unique indices.

    Returns:
        out: (n_unique, d) — mean of src rows grouped by index.
        inverse: (B,) — maps each row in src to its position in out.
        unique_idx: (n_unique,) — the unique values of index.
    """
    unique_idx, inverse = torch.unique(index, return_inverse=True)
    n_groups = len(unique_idx)
    d = src.shape[1]
    out = torch.zeros(n_groups, d, device=src.device, dtype=src.dtype)
    count = torch.zeros(n_groups, 1, device=src.device, dtype=src.dtype)
    out.scatter_add_(0, inverse.unsqueeze(1).expand_as(src), src)
    count.scatter_add_(
        0, inverse.unsqueeze(1), torch.ones(len(index), 1, device=src.device)
    )
    return out / count.clamp(min=1), inverse, unique_idx


class DeepSetsMergingIntegrator(BaseIntegrator):
    """Pixel-level integration + per-HKL merging via Deep Sets.

    Best results when paired with the GroupedAsuIdSampler so each batch
    contains complete HKL groups. Without it, scatter_mean is mostly
    identity (one obs per HKL per batch) and the model behaves like the
    integrator with extra structure.
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

        # I_head: aggregated features → (k, rate) for Gamma(k, rate)
        self.I_head_k = nn.Linear(d, 1)
        self.I_head_r = nn.Linear(d, 1)

        self.k_min = cfg.scaling_k_min
        self.rate_min = getattr(cfg, "scaling_rate_min", 0.001)

        # Init: small weights (encoder gets gradient signal immediately) +
        # bias tuned to desired initial moments via softplus inverse.
        init_k = max(getattr(cfg, "scaling_init_k", 1.0), 1e-3)
        init_rate = max(getattr(cfg, "scaling_init_rate", 1.0), 1e-3)
        with torch.no_grad():
            nn.init.xavier_normal_(self.I_head_k.weight, gain=0.01)
            nn.init.xavier_normal_(self.I_head_r.weight, gain=0.01)
            self.I_head_k.bias.fill_(
                math.log(math.expm1(max(init_k - self.k_min, 1e-6)))
            )
            self.I_head_r.bias.fill_(
                math.log(math.expm1(max(init_rate - self.rate_min, 1e-6)))
            )

        # EMA buffer for inference (per-HKL aggregated features)
        self.register_buffer("feat_ema", torch.zeros(cfg.n_hkl, d))
        self.register_buffer(
            "feat_seen", torch.zeros(cfg.n_hkl, dtype=torch.bool)
        )
        self.ema_momentum = getattr(cfg, "ema_momentum", 0.95)

        # Merge KL weight (Wilson prior on qi_h)
        self.merge_kl_weight = getattr(cfg, "merge_kl_weight", 1.0)

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

    def _decode_qi_h(self, z_h: Tensor) -> Gamma:
        """Map aggregated features to per-HKL Gamma intensity."""
        k = Fn.softplus(self.I_head_k(z_h).squeeze(-1)) + self.k_min
        rate = Fn.softplus(self.I_head_r(z_h).squeeze(-1)) + self.rate_min
        return Gamma(k, rate)

    def _update_ema(
        self, z_h_detached: Tensor, unique_hkls: Tensor
    ) -> None:
        """EMA update for HKLs in this batch."""
        old = self.feat_ema[unique_hkls]
        was_seen = self.feat_seen[unique_hkls].unsqueeze(1)
        new = torch.where(
            was_seen,
            self.ema_momentum * old + (1 - self.ema_momentum) * z_h_detached,
            z_h_detached,
        )
        self.feat_ema[unique_hkls] = new
        self.feat_seen[unique_hkls] = True

    def get_merged_qi(self) -> Gamma:
        """Per-HKL Gamma from EMA buffer (for MTZ output)."""
        z_h = self.feat_ema
        return self._decode_qi_h(z_h)

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

        # Encoders
        position = _get_normalized_position(metadata, device)
        x_profile = self.encoders["profile"](
            shoebox_reshaped, position=position
        )
        x_k_i = self.encoders["k_i"](shoebox_reshaped)
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

        # Deep Sets: aggregate encoder features by HKL
        asu_ids = metadata["asu_id"].long().to(device)
        z_h, inverse, unique_hkls = _scatter_mean_compact(x_k_i, asu_ids)

        # Update EMA buffer (for inference output)
        if self.training:
            with torch.no_grad():
                self._update_ema(z_h.detach(), unique_hkls)

        # Decode per-HKL Gamma intensity
        qi_h = self._decode_qi_h(z_h)  # batch shape: (n_unique,)

        # Sample intensity per HKL, broadcast to per-observation
        zI_h = qi_h.rsample([self.mc_samples])  # (S, n_unique)
        zI = zI_h[:, inverse]  # (S, B) — shared across obs of same HKL

        # Scale per observation
        scale = self._get_scale(metadata, device)  # (B,)
        zI_scaled = (scale.unsqueeze(0) * zI).unsqueeze(-1).permute(1, 0, 2)
        # zI_scaled shape: (B, S, 1)

        # Profile and background
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)

        rate = zI_scaled * zp + zbg

        if "is_coset" in metadata:
            coset = metadata["is_coset"].bool().view(-1, 1, 1)
            rate = torch.where(coset, zbg, rate)

        # Per-observation qi (broadcast from qi_h) for _assemble_outputs
        k_per_obs = qi_h.concentration[inverse]
        rate_per_obs = qi_h.rate[inverse]
        qi = Gamma(k_per_obs, rate_per_obs)

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
        """Wilson KL on per-HKL Gamma. Counted once per HKL (no overcounting)."""
        device = qi_h.concentration.device
        # Average d-spacing per HKL (d is identical for all obs of same HKL)
        d_per_obs = metadata["d"].to(device).float().unsqueeze(-1)
        d_per_hkl, _, _ = _scatter_mean_compact(d_per_obs, inverse)
        d_per_hkl = d_per_hkl.squeeze(-1)

        s_sq = 1.0 / (4.0 * d_per_hkl.clamp(min=1e-6).pow(2))
        # Wilson tau from the loss module's G, B
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

        # Standard loss handles Poisson NLL, profile KL, bg KL
        # Set pi_weight=0 in YAML so it skips the (overcounted) per-obs intensity KL
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

        # Add per-HKL Wilson KL (counted once per unique HKL)
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
                f"{step} qi_h_mean",
                qi_h.mean.mean(),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} qi_h_var",
                qi_h.variance.mean(),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} qi_h_k",
                qi_h.concentration.mean(),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} qi_h_rate",
                qi_h.rate.mean(),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} n_unique_hkl",
                torch.tensor(len(outputs["unique_hkls"]), dtype=torch.float),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} obs_per_hkl",
                torch.tensor(
                    counts.shape[0] / len(outputs["unique_hkls"]),
                    dtype=torch.float,
                ),
                on_step=False,
                on_epoch=True,
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
