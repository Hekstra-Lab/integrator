from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gamma, Normal

from integrator import configs
from integrator.model.scaling.difference_merging import (
    _DELTA_CLAMP,
    DifferenceMergingIntegrator,
)
from integrator.model.scaling.merge_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
    _sample_profile,
    _scatter_sum_compact,
)


class SVAEDifferenceMergingIntegrator(DifferenceMergingIntegrator):
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
        # Signal photons come from the responsibility, not a potential head.
        if hasattr(self, "alpha_head"):
            del self.alpha_head
        d = cfg.encoder_out
        self.gate_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1)
        )
        nn.init.constant_(
            self.gate_head[-1].bias,
            float(getattr(cfg, "responsibility_gate_init", -2.0)),
        )

    def _bg_prior(
        self, group_labels: Tensor | None, n: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Per-obs background Gamma prior (concentration, rate) from the loss."""
        conc = self.loss.bg_concentration.to(device)
        rate = self.loss.bg_rate.to(device)
        if conc.ndim == 1:  # per-resolution-bin priors
            g = (
                torch.zeros(n, dtype=torch.long, device=device)
                if group_labels is None
                else group_labels.to(device).long()
            )
            return conc[g], rate[g]
        return conc.expand(n), rate.expand(n)

    def _responsibility_common_mode(
        self,
        x_k_i: Tensor,
        logits: Tensor,
        counts: Tensor,
        scale: Tensor,
        profile_mean: Tensor,
        mask: Tensor,
        pooled_idx: Tensor,
        d_obs: Tensor,
        group_labels: Tensor | None,
    ) -> tuple:
        """Conjugate common mode q(I0) + background from the responsibility."""
        device = counts.device
        b = counts.shape[0]
        m = mask.reshape(b, -1).float()
        c = counts.clamp(min=0).reshape(b, -1).float() * m
        r = torch.sigmoid(logits + self.gate_head(x_k_i)) * m  # (B, P)

        total = c.sum(dim=-1)
        sig_counts = (r * c).sum(dim=-1)  # sum_p r*c   (signal)
        bg_counts = total - sig_counts  # sum_p (1-r)*c
        n_pix = m.sum(dim=-1).clamp(min=1.0)

        d_sum, inverse0, unique0 = _scatter_sum_compact(d_obs, pooled_idx)
        cnt, _, _ = _scatter_sum_compact(torch.ones_like(d_obs), pooled_idx)
        tau0 = self._wilson_tau((d_sum / cnt.clamp(min=1.0)).clamp(min=1e-6))
        delta_beta = scale * (profile_mean * m).sum(dim=-1)
        alpha_sig, _, _ = _scatter_sum_compact(sig_counts, pooled_idx)
        beta_sum, _, _ = _scatter_sum_compact(delta_beta, pooled_idx)
        alpha0 = self.alpha_W + alpha_sig
        beta0 = tau0 + beta_sum
        qi0 = Gamma(alpha0.clamp(min=1e-6), beta0.clamp(min=1e-12))

        bg_a0, bg_b0 = self._bg_prior(group_labels, b, device)
        qbg = Gamma(
            (bg_a0 + bg_counts).clamp(min=1e-6),
            (bg_b0 + n_pix).clamp(min=1e-12),
        )
        return (
            qi0, alpha0, beta0, inverse0, unique0, tau0, qbg, sig_counts,
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
        d_obs = metadata["d"].to(device).float()
        pooled_idx = metadata[self.friedel_key].long().to(device)
        group_labels = (
            metadata["group_label"].long().to(device)
            if "group_label" in metadata
            else None
        )

        (
            qi0_h,
            alpha0,
            beta0,
            inverse0,
            unique0,
            tau0,
            qbg,
            sig_counts,
            exposure,
        ) = self._responsibility_common_mode(
            x_k_i,
            qp.mean_logits,
            counts,
            scale,
            profile_mean,
            mask,
            pooled_idx,
            d_obs,
            group_labels,
        )
        n_pooled = unique0.shape[0]

        plus, centric_pooled, shell_pooled, s2_pooled = self._pooled_context(
            metadata, inverse0, n_pooled, d_obs, device
        )
        # The signed anomalous fraction reads the sign-split SIGNAL counts +
        # exposure (the same statistics that build the common mode).
        mu_delta, sd_delta, sigma_delta_shell = self._delta_posterior(
            sig_counts,
            exposure,
            tau0,
            inverse0,
            n_pooled,
            plus,
            centric_pooled,
            shell_pooled,
            s2_pooled,
        )

        zI0_h = qi0_h.rsample([self.mc_samples]).clamp(min=1e-10)
        z_delta_h = Normal(mu_delta, sd_delta).rsample([self.mc_samples])
        z_delta_h = torch.where(
            centric_pooled.unsqueeze(0),
            torch.zeros_like(z_delta_h),
            z_delta_h.clamp(-_DELTA_CLAMP, _DELTA_CLAMP),
        )
        sign = torch.where(plus, 1.0, -1.0)
        zI0 = zI0_h[:, inverse0]
        z_delta = z_delta_h[:, inverse0]
        zI_mate = (zI0 * (1.0 + sign.unsqueeze(0) * z_delta)).clamp(min=1e-10)

        zI_scaled = (
            (scale.unsqueeze(0) * zI_mate).unsqueeze(-1).permute(1, 0, 2)
        )
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)
        rate = zI_scaled * zp + zbg
        if "is_coset" in metadata:
            coset = metadata["is_coset"].bool().view(-1, 1, 1)
            rate = torch.where(coset, zbg, rate)

        mean_obs = (
            (alpha0 / beta0.clamp(min=1e-12))[inverse0]
            * (1.0 + sign * mu_delta[inverse0])
        ).clamp(min=1e-10)
        alpha_obs = alpha0[inverse0].clamp(min=1e-6)
        qi = Gamma(alpha_obs, (alpha_obs / mean_obs).clamp(min=1e-12))

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
        out[self.merge_key] = metadata[self.merge_key].long().to(device)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
            "qi_h": qi0_h,
            "alpha_h": alpha0,
            "beta_h": beta0,
            "tau_h": tau0,
            "inverse": inverse0,
            "unique_hkls": unique0,
            "scale": scale,
            "mu_delta": mu_delta,
            "sd_delta": sd_delta,
            "sigma_delta_shell": sigma_delta_shell,
            "centric_pooled": centric_pooled,
        }

    @torch.no_grad()
    def finalize_merge(self, dataloader) -> None:
        """Merge over the dataset (responsibility), expand (I0, delta) per mate."""
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
            qp = self.surrogates["qp"](
                self.encoders["profile"](sr), mc_samples=1
            )
            x_k_i = self.encoders["k_i"](sr)
            scale = self._get_scale(metadata, device)
            d_obs = metadata["d"].to(device).float()
            pooled_idx = metadata[self.friedel_key].long().to(device)
            group_labels = (
                metadata["group_label"].long().to(device)
                if "group_label" in metadata
                else None
            )

            (
                _,
                alpha0,
                beta0,
                inverse0,
                unique0,
                tau0,
                _,
                sig_counts,
                exposure,
            ) = self._responsibility_common_mode(
                x_k_i,
                qp.mean_logits,
                counts,
                scale,
                qp.mean_profile,
                mask,
                pooled_idx,
                d_obs,
                group_labels,
            )
            n_pooled = unique0.shape[0]
            plus, centric_pooled, shell_pooled, s2_pooled = (
                self._pooled_context(
                    metadata, inverse0, n_pooled, d_obs, device
                )
            )
            mu_delta, sd_delta, _ = self._delta_posterior(
                sig_counts,
                exposure,
                tau0,
                inverse0,
                n_pooled,
                plus,
                centric_pooled,
                shell_pooled,
                s2_pooled,
            )
            self._write_mate_buffers(
                alpha0,
                beta0,
                mu_delta,
                sd_delta,
                inverse0,
                n_pooled,
                plus,
                centric_pooled,
                metadata,
                seen,
                device,
            )
        self.buffer_seen.copy_(seen)
