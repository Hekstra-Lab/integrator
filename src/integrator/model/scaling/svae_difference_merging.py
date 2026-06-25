from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gamma, Normal

from integrator import configs
from integrator.model.scaling.base import ScalingLightningModule
from integrator.model.scaling.merge_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
    _sample_profile,
    _scatter_sum_compact,
)

_DELTA_CLAMP = 0.95  # keep 1 +/- delta strictly positive


class SVAEDifferenceMergingIntegrator(ScalingLightningModule):
    """Difference SVAE: per-pixel signal probability + empirical-Bayes anomalous delta."""

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
        if not self.anomalous:
            raise ValueError(
                "SVAEDifferenceMergingIntegrator needs anomalous: true (per-mate "
                "buffers keyed on miller_idx_unfriedelized)."
            )
        # sigma_delta^2 by detached method-of-moments (EMA buffer).
        sigma0 = float(getattr(cfg, "sigma_delta_init", 0.05))
        self.sigma_delta_ema = float(getattr(cfg, "sigma_delta_ema", 0.99))
        self.register_buffer(
            "sigma_delta_sq", torch.tensor(sigma0**2), persistent=True
        )
        # Signal photons come from the per-pixel signal_probability gate.
        d = cfg.encoder_out
        self.gate_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1)
        )
        nn.init.constant_(
            self.gate_head[-1].bias,
            float(getattr(cfg, "signal_probability_gate_init", -2.0)),
        )

    def _eb_delta(
        self,
        potential: Tensor,
        exposure: Tensor,
        tau_pooled: Tensor,
        inverse0: Tensor,
        n_pooled: int,
        plus: Tensor,
        centric_pooled: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Closed-form precision-weighted empirical-Bayes delta (no free head)."""
        device = potential.device
        eps = 1e-6

        def scatter(vals: Tensor, mask: Tensor) -> Tensor:
            out = torch.zeros(n_pooled, device=device, dtype=vals.dtype)
            return out.scatter_add(0, inverse0[mask], vals[mask])

        a_plus = 0.5 * self.alpha_W + scatter(potential, plus)
        a_minus = 0.5 * self.alpha_W + scatter(potential, ~plus)
        b_plus = 0.5 * tau_pooled + scatter(exposure, plus) + eps
        b_minus = 0.5 * tau_pooled + scatter(exposure, ~plus) + eps

        m_plus = a_plus / b_plus
        m_minus = a_minus / b_minus
        tot = (m_plus + m_minus).clamp(min=eps)
        delta_hat = (m_plus - m_minus) / tot

        # delta-method variance from the per-mate Gamma term 1/alpha
        v_h = (
            4.0
            * m_plus.pow(2)
            * m_minus.pow(2)
            / tot.pow(4)
            * (1.0 / a_plus.clamp(min=eps) + 1.0 / a_minus.clamp(min=eps))
        )

        acentric = ~centric_pooled
        with torch.no_grad():
            if self.training and bool(acentric.any()):
                s2 = (
                    (delta_hat[acentric].pow(2) - v_h[acentric])
                    .mean()
                    .clamp(min=1e-8)
                )
                self.sigma_delta_sq.mul_(self.sigma_delta_ema).add_(
                    (1.0 - self.sigma_delta_ema) * s2
                )
            sig2 = self.sigma_delta_sq.clamp(min=1e-8)

        w = sig2 / (sig2 + v_h.clamp(min=eps))
        mu_delta = (w * delta_hat).clamp(-_DELTA_CLAMP, _DELTA_CLAMP)
        sd_delta = (sig2 * v_h / (sig2 + v_h)).clamp(min=1e-12).sqrt() + eps

        mu_delta = torch.where(
            centric_pooled, torch.zeros_like(mu_delta), mu_delta
        )
        sd_delta = torch.where(
            centric_pooled, torch.full_like(sd_delta, eps), sd_delta
        )

        sigma_delta_shell = sig2.sqrt().expand(n_pooled)
        return mu_delta, sd_delta, sigma_delta_shell

    def _pooled_context(
        self,
        metadata: dict,
        inverse0: Tensor,
        n_pooled: int,
        device,
    ) -> tuple[Tensor, Tensor]:
        """Per-pooled-id helpers: (plus_obs, centric_pooled)."""
        plus = metadata["friedel_plus"].bool().to(device)
        centric_obs = (
            metadata["centric"].bool().to(device)
            if "centric" in metadata
            else torch.zeros_like(plus)
        )
        centric_pooled = torch.zeros(n_pooled, dtype=torch.bool, device=device)
        centric_pooled[inverse0[centric_obs]] = True
        return plus, centric_pooled

    def _signal_probability_common_mode(
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
        """Conjugate common mode q(I0) + background from the signal probability."""
        device = counts.device
        b = counts.shape[0]
        m = mask.reshape(b, -1).float()
        c = counts.clamp(min=0).reshape(b, -1).float() * m
        r = torch.sigmoid(logits + self.gate_head(x_k_i)) * m  # (B, P)

        total_counts = c.sum(dim=-1)
        sig_counts = (r * c).sum(dim=-1)  # sum_p r*c   (signal)
        bg_counts = total_counts - sig_counts  # sum_p (1-r)*c
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
            qi0,
            alpha0,
            beta0,
            inverse0,
            unique0,
            tau0,
            qbg,
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
        ) = self._signal_probability_common_mode(
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

        plus, centric_pooled = self._pooled_context(
            metadata, inverse0, n_pooled, device
        )
        # signed anomalous fraction from the sign-split signal counts + exposure
        mu_delta, sd_delta, sigma_delta_shell = self._eb_delta(
            sig_counts,
            exposure,
            tau0,
            inverse0,
            n_pooled,
            plus,
            centric_pooled,
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
        # Model merged intensity on the observed scale (DIALS cols pass through).
        out["scaled_intensity"] = scale * qi.mean

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
            # qi_h / tau_h / inverse / unique are the COMMON MODE, so the
            # base Wilson KL in _step acts on I0 only (the whole point).
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
            # per-obs sufficient statistics for validation merging stats.
            "signal_counts": sig_counts,
            "exposure": exposure,
        }

    def _extra_loss_terms(
        self, outputs: dict, metadata: dict
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """No delta KL: the EB shrink IS the posterior and sigma is MoM."""
        mu = outputs["mu_delta"]
        acentric = ~outputs["centric_pooled"]
        if not bool(acentric.any()):
            return mu.new_zeros(()), {}
        logs = {
            "abs_delta": mu[acentric].abs().mean().detach(),
            "sigma_delta": outputs["sigma_delta_shell"][acentric]
            .mean()
            .detach(),
        }
        return mu.new_zeros(()), logs

    def _write_mate_buffers(
        self,
        alpha0: Tensor,
        beta0: Tensor,
        mu_delta: Tensor,
        sd_delta: Tensor,
        inverse0: Tensor,
        n_pooled: int,
        plus: Tensor,
        centric_pooled: Tensor,
        metadata: dict,
        seen: Tensor,
        device,
    ) -> None:
        """Expand (I0, delta) per pooled id into per-mate Gamma buffers."""
        uf = metadata[self.merge_key].long().to(device)
        uf_plus = torch.full((n_pooled,), -1, dtype=torch.long, device=device)
        uf_minus = torch.full_like(uf_plus, -1)
        uf_plus[inverse0[plus]] = uf[plus]
        uf_minus[inverse0[~plus]] = uf[~plus]
        uf_centric = torch.maximum(uf_plus, uf_minus)  # the single shared id
        acentric = ~centric_pooled

        mean0 = (alpha0 / beta0.clamp(min=1e-12)).clamp(min=1e-10)
        var0 = (alpha0 / beta0.clamp(min=1e-12).pow(2)).clamp(min=1e-20)

        def write(uf_id: Tensor, factor: Tensor, rows: Tensor) -> None:
            sel = (uf_id >= 0) & rows
            ids = uf_id[sel]
            if bool(seen[ids].any()):
                raise RuntimeError(
                    "finalize_merge needs a grouped (group_by_asu_id) loader so "
                    "each Friedel pair is complete in one batch; found an id "
                    "spanning batches. Use predict_dataloader(grouped=True)."
                )
            mean = (mean0[sel] * factor[sel]).clamp(min=1e-10)
            var = (
                var0[sel] * factor[sel].pow(2)
                + mean0[sel].pow(2) * sd_delta[sel].pow(2)
            ).clamp(min=1e-20)
            self.alpha_buffer[ids] = (mean.pow(2) / var).clamp(min=1e-6)
            self.beta_buffer[ids] = (mean / var).clamp(min=1e-12)
            seen[ids] = True

        ones = torch.ones_like(mu_delta)
        write(uf_plus, 1.0 + mu_delta, acentric)
        write(uf_minus, 1.0 - mu_delta, acentric)
        write(uf_centric, ones, centric_pooled)  # delta = 0

    @torch.no_grad()
    def finalize_merge(self, dataloader) -> None:
        """Merge over the dataset (signal probability), expand (I0, delta) per mate."""
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
            ) = self._signal_probability_common_mode(
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
            plus, centric_pooled = self._pooled_context(
                metadata, inverse0, n_pooled, device
            )
            mu_delta, sd_delta, _ = self._eb_delta(
                sig_counts,
                exposure,
                tau0,
                inverse0,
                n_pooled,
                plus,
                centric_pooled,
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
