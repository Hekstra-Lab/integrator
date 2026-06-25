from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gamma

from integrator import configs
from integrator.model.scaling.base import ScalingLightningModule
from integrator.model.scaling.merge_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
    _sample_profile,
    _scatter_sum_compact,
)


class SVAEMergingIntegrator(ScalingLightningModule):
    """Per-HKL merging via a shared-logit per-pixel probability (structured VAE).

    Background is derived analytically, so only the profile + k_i encoders and
    the qp surrogate are needed.
    """

    REQUIRED_ENCODERS = {
        "profile": ("profile_encoder", configs.ProfileEncoderArgs),
        "k_i": ("intensity_encoder", configs.IntensityEncoderArgs),
    }

    DEFAULT_SURROGATES = {
        "qp": {
            "name": "learned_basis_profile",
            "args": {"latent_dim": 12, "init_std": 0.5, "prior_scale": 3.0},
        },
    }

    def __init__(self, cfg, loss, encoders, surrogates, optimizer=None):
        super().__init__(cfg, loss, encoders, surrogates, optimizer)
        # Signal photons come from the per-pixel signal_probability gate.
        d = cfg.encoder_out
        self.gate_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1)
        )
        nn.init.constant_(
            self.gate_head[-1].bias,
            float(getattr(cfg, "signal_probability_gate_init", -2.0)),
        )

    def _signal_probability_merge(
        self,
        logits: Tensor,
        gate: Tensor,
        counts: Tensor,
        scale: Tensor,
        profile_mean: Tensor,
        mask: Tensor,
        miller_idx: Tensor,
        d_per_obs: Tensor,
        group_labels: Tensor | None,
    ) -> tuple[
        Gamma,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Gamma,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Conjugate merge via a per-pixel signal probability.

        r_p = sigmoid(profile_logit_p + gate_i) splits each pixel's count into
        signal (r_p) and background (1-r_p). The same r feeds both conjugate
        distributions.

        Returns `(qi_h, alpha_h, beta_h, inverse, unique, tau_h, qbg, r,
        sig_counts, delta_beta)`.
        """

        device = counts.device
        b = counts.shape[0]
        m = mask.reshape(b, -1).float()
        c = counts.clamp(min=0).reshape(b, -1).float() * m

        # per pixel signal probability
        r = torch.sigmoid(logits + gate.unsqueeze(-1)) * m  # (B, P), masked
        total_counts = c.sum(dim=-1)
        sig_counts = (r * c).sum(dim=-1)  # sum_p r*c
        bg_counts = total_counts - sig_counts  # sum_p (1-r)*c
        n_pix = m.sum(dim=-1).clamp(min=1.0)  # background exposure P_i

        # per-HKL conjugate intensity posterior
        d_sum, inverse, unique = _scatter_sum_compact(d_per_obs, miller_idx)
        cnt, _, _ = _scatter_sum_compact(
            torch.ones_like(d_per_obs), miller_idx
        )
        tau_h = self._wilson_tau((d_sum / cnt.clamp(min=1.0)).clamp(min=1e-6))
        delta_beta = scale * (profile_mean * m).sum(dim=-1)
        alpha_sig, _, _ = _scatter_sum_compact(sig_counts, miller_idx)
        beta_sum, _, _ = _scatter_sum_compact(delta_beta, miller_idx)
        alpha_h = self.alpha_W + alpha_sig
        beta_h = tau_h + beta_sum
        qi_h = Gamma(alpha_h.clamp(min=1e-6), beta_h.clamp(min=1e-12))

        bg_a0, bg_b0 = self._bg_prior(group_labels, b, device)
        qbg = Gamma(
            (bg_a0 + bg_counts).clamp(min=1e-6),
            (bg_b0 + n_pix).clamp(min=1e-12),
        )

        return (
            qi_h,
            alpha_h,
            beta_h,
            inverse,
            unique,
            tau_h,
            qbg,
            r,
            sig_counts,
            delta_beta,
        )

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
        sr = (shoebox * mask).reshape(b, 1, *self.shoebox_shape)
        x_profile = self.encoders["profile"](sr)
        x_k_i = self.encoders["k_i"](sr)
        qp = self.surrogates["qp"](x_profile, mc_samples=self.mc_samples)
        profile_mean = qp.mean_profile
        scale = self._get_scale(metadata, device)

        miller_idx = metadata[self.merge_key].long().to(device)
        d_obs = metadata["d"].to(device).float()
        group_labels = (
            metadata["group_label"].long().to(device)
            if "group_label" in metadata
            else None
        )

        # shared profile logits + per-obs gate -> per-pixel signal
        gate = self.gate_head(x_k_i).squeeze(-1)
        (
            qi_h,
            alpha_h,
            beta_h,
            inverse,
            unique_hkls,
            tau_h,
            qbg,
            signal_probability,
            sig_counts,
            exposure,
        ) = self._signal_probability_merge(
            qp.mean_logits,
            gate,
            counts,
            scale,
            profile_mean,
            mask,
            miller_idx,
            d_obs,
            group_labels,
        )

        # sample I_h once per HKL (broadcast to obs) -> Poisson rate
        zI_h = qi_h.rsample([self.mc_samples]).clamp(min=1e-10)
        zI = zI_h[:, inverse]
        zI_scaled = (scale.unsqueeze(0) * zI).unsqueeze(-1).permute(1, 0, 2)
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)
        rate = zI_scaled * zp + zbg

        # If using coset data
        if "is_coset" in metadata:
            coset = metadata["is_coset"].bool().view(-1, 1, 1)
            rate = torch.where(coset, zbg, rate)

        qi = Gamma(
            alpha_h[inverse].clamp(min=1e-6),
            beta_h[inverse].clamp(min=1e-12),
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
        out[self.merge_key] = miller_idx
        # Model merged intensity on the observed scale
        out["scaled_intensity"] = scale * qi.mean

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
            "qi_h": qi_h,
            "alpha_h": alpha_h,
            "beta_h": beta_h,
            "tau_h": tau_h,
            "inverse": inverse,
            "unique_hkls": unique_hkls,
            "scale": scale,
            "signal_probability": signal_probability,
            "signal_counts": sig_counts,
            "exposure": exposure,
        }

    @torch.no_grad()
    def finalize_merge(self, dataloader) -> None:
        """Recompute the per-HKL posterior over the full dataset ."""
        self.eval()
        device = self.alpha_buffer.device
        seen = torch.zeros(self.n_hkl, dtype=torch.bool, device=device)
        self.alpha_buffer.fill_(self.alpha_W)
        self.beta_buffer.fill_(1.0)

        for batch in dataloader:
            counts, shoebox, mask, metadata = batch
            counts = counts.to(device)
            shoebox = shoebox.to(device)
            mask = mask.to(device)
            b = shoebox.shape[0]
            sr = (shoebox * mask).reshape(b, 1, *self.shoebox_shape)
            scale = self._get_scale(metadata, device)
            miller_idx = metadata[self.merge_key].long().to(device)
            d_obs = metadata["d"].to(device).float()
            qp = self.surrogates["qp"](
                self.encoders["profile"](sr), mc_samples=1
            )
            group_labels = (
                metadata["group_label"].long().to(device)
                if "group_label" in metadata
                else None
            )
            gate = self.gate_head(self.encoders["k_i"](sr)).squeeze(-1)
            _, alpha_h, beta_h, _, unique, _, _, _, _, _ = (
                self._signal_probability_merge(
                    qp.mean_logits,
                    gate,
                    counts,
                    scale,
                    qp.mean_profile,
                    mask,
                    miller_idx,
                    d_obs,
                    group_labels,
                )
            )
            if bool(seen[unique].any()):
                raise RuntimeError(
                    "finalize_merge requires a grouped (group_by_asu_id) loader "
                    "so each HKL is complete in one batch; found an HKL spanning "
                    "batches. Use predict_dataloader(grouped=True)."
                )
            self.alpha_buffer[unique] = alpha_h
            self.beta_buffer[unique] = beta_h
            seen[unique] = True
        self.buffer_seen.copy_(seen)
