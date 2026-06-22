import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Gamma, Normal

from integrator.model.scaling.amortized_merging import (
    AmortizedMergingIntegrator,
)
from integrator.model.scaling.merge_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
    _sample_profile,
)
from integrator.model.scaling.mlp_scale import _chebyshev

_DELTA_CLAMP = 0.95  # keep 1 +/- delta strictly positive


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class DifferenceMergingIntegrator(AmortizedMergingIntegrator):
    """Per-HKL merge with the anomalous difference as an explicit latent.

    See the module docstring. Requires `anomalous: true` (buffers keyed on the
    Friedel-separate id) and a loader co-batching mates
    (`group_by_key: miller_idx_friedelized`).
    """

    def __init__(self, cfg, loss, encoders, surrogates, optimizer=None):
        super().__init__(cfg, loss, encoders, surrogates, optimizer)
        if not self.anomalous:
            raise ValueError(
                "DifferenceMergingIntegrator needs anomalous: true (per-mate "
                "buffers keyed on miller_idx_unfriedelized)."
            )

        self.delta_kl_weight = float(getattr(cfg, "delta_kl_weight", 1.0))
        self.n_shells = max(1, int(getattr(self.loss, "n_bins", 1) or 1))
        sigma0 = float(getattr(cfg, "sigma_delta_init", 0.05))
        raw0 = _inv_softplus(sigma0)
        # `eb` (default): closed-form precision-weighted empirical-Bayes delta --
        # the in-model version of the post-hoc EB shrink, robust to the
        # over-training decay of the free head. `head`: the legacy free
        # amortized head (kept for ablation; decays over training).
        self.delta_mode = str(getattr(cfg, "delta_mode", "eb"))

        if self.delta_mode == "eb":
            # sigma_delta^2 by DETACHED method-of-moments (EMA buffer), NOT
            # learned through the ELBO -- learning it collapses sigma -> 0.
            self.sigma_delta_ema = float(getattr(cfg, "sigma_delta_ema", 0.99))
            self.register_buffer(
                "sigma_delta_sq", torch.tensor(sigma0**2), persistent=True
            )
        else:
            self.sigma_delta_form = str(
                getattr(cfg, "sigma_delta_form", "loglinear")
            )
            self._build_sigma_delta(cfg, raw0)
            # zero-init so mu starts at the raw data estimate, sd at sigma_init.
            h = int(getattr(cfg, "delta_head_hidden", 16))
            self.delta_head = nn.Sequential(
                nn.Linear(4, h), nn.ReLU(), nn.Linear(h, 2)
            )
            nn.init.zeros_(self.delta_head[-1].weight)
            with torch.no_grad():
                self.delta_head[-1].bias.copy_(torch.tensor([0.0, raw0]))

    def _build_sigma_delta(self, cfg, raw0: float) -> None:
        """Parameterize the sigma_delta(d) prior; all init flat at sigma_delta_init.

        `loglinear` softplus(a + b s^2) (2 params), `binned` per shell, `cheby`
        low-order Chebyshev in s^2, `mlp` a tiny net. Continuous forms read s^2
        directly (no binning). s^2 is normalized to [-1, 1] over [dmin, d_max].
        """
        form = self.sigma_delta_form
        learn = bool(getattr(cfg, "sigma_delta_learn", True))
        d_max = 60.0
        s2_hi = 1.0 / (4.0 * float(cfg.dmin) ** 2)
        s2_lo = 1.0 / (4.0 * d_max**2)
        self.register_buffer("_s2_lo", torch.tensor(s2_lo))
        self.register_buffer("_s2_hi", torch.tensor(max(s2_hi, s2_lo + 1e-6)))

        if form == "binned":
            p = torch.full((self.n_shells,), raw0)
        elif form == "loglinear":
            p = torch.tensor([raw0, 0.0])  # [a, b]; b=0 -> flat start
        elif form == "cheby":
            deg = int(getattr(cfg, "sigma_delta_cheby_degree", 4))
            p = torch.zeros(deg + 1)
            p[0] = raw0
        elif form == "mlp":
            net = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
            nn.init.zeros_(net[-1].weight)
            nn.init.constant_(net[-1].bias, raw0)
            self.sigma_delta_mlp = net
            if not learn:
                for prm in net.parameters():
                    prm.requires_grad_(False)
            return
        else:
            raise ValueError(f"unknown sigma_delta_form {form!r}")

        if learn:
            self.sigma_delta_raw = nn.Parameter(p)
        else:
            self.register_buffer("sigma_delta_raw", p)

    def _s2_norm(self, s2: Tensor) -> Tensor:
        return (
            2.0 * (s2 - self._s2_lo) / (self._s2_hi - self._s2_lo) - 1.0
        ).clamp(-1.0, 1.0)

    def _sigma_delta(self, shell_pooled: Tensor, s2_pooled: Tensor) -> Tensor:
        """Prior sd of the anomalous fraction per pooled id (the chosen form)."""
        form = self.sigma_delta_form
        eps = 1e-6
        if form == "binned":
            return F.softplus(self.sigma_delta_raw)[shell_pooled] + eps
        if form == "loglinear":
            a, b = self.sigma_delta_raw[0], self.sigma_delta_raw[1]
            return F.softplus(a + b * s2_pooled) + eps
        if form == "cheby":
            basis = _chebyshev(
                self._s2_norm(s2_pooled), self.sigma_delta_raw.numel() - 1
            )
            return F.softplus(basis @ self.sigma_delta_raw) + eps
        # mlp
        x = self._s2_norm(s2_pooled).unsqueeze(-1)
        return F.softplus(self.sigma_delta_mlp(x).squeeze(-1)) + eps

    def _per_obs_potential(
        self,
        x_k_i: Tensor,
        x_r_i: Tensor,
        scale: Tensor,
        cond_mid: Tensor,
        d_per_obs: Tensor,
    ) -> Tensor:
        """Per-observation signal potential delta_alpha_i (mirrors `_merge`)."""
        cond = torch.stack(
            [
                scale.clamp(min=1e-8).log(),
                cond_mid.clamp(min=1e-8).log(),
                d_per_obs,
            ],
            dim=-1,
        )
        feat = torch.cat([x_k_i, x_r_i, cond], dim=-1)
        return F.softplus(self.alpha_head(feat)).squeeze(-1)

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
        """Closed-form precision-weighted empirical-Bayes delta (no free head).

        Sign-split the SAME per-obs signal potential and exposure that build the
        common mode into per-mate Gammas (splitting alpha_W and tau_h in half),
        form delta_hat = (m+ - m-)/(m+ + m-) and its precision-derived variance
        v_h, and shrink: mu = w*delta_hat, w = sigma^2/(sigma^2 + v_h),
        sd = sqrt(sigma^2 v_h/(sigma^2 + v_h)). sigma^2 is a DETACHED
        method-of-moments EMA over acentric pairs, so the estimator has no free
        capacity to drift onto per-batch noise. Centrics are pinned to 0.
        """
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
        # delta-method variance from the per-mate Gamma precisions (1/alpha).
        v_h = (
            4.0 * m_plus.pow(2) * m_minus.pow(2) / tot.pow(4)
            * (1.0 / a_plus.clamp(min=eps) + 1.0 / a_minus.clamp(min=eps))
        )

        acentric = ~centric_pooled
        with torch.no_grad():
            if self.training and bool(acentric.any()):
                s2 = (
                    delta_hat[acentric].pow(2) - v_h[acentric]
                ).mean().clamp(min=1e-8)
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

    def _delta_posterior(
        self,
        potential: Tensor,
        exposure: Tensor,
        tau_pooled: Tensor,
        inverse0: Tensor,
        n_pooled: int,
        plus: Tensor,
        centric_pooled: Tensor,
        shell_pooled: Tensor,
        s2_pooled: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """q(delta) per pooled id + its prior sd. `eb` mode = closed-form EB;
        `head` mode = the legacy free amortized head. Centrics get delta = 0."""
        if self.delta_mode == "eb":
            return self._eb_delta(
                potential, exposure, tau_pooled, inverse0, n_pooled, plus,
                centric_pooled,
            )

        device = potential.device
        eps = 1e-6

        def scatter(vals: Tensor, mask: Tensor) -> Tensor:
            out = torch.zeros(n_pooled, device=device, dtype=vals.dtype)
            return out.scatter_add(0, inverse0[mask], vals[mask])

        s_plus = scatter(potential, plus)
        s_minus = scatter(potential, ~plus)
        tot = s_plus + s_minus + eps
        raw_delta = (s_plus - s_minus) / tot

        feat = torch.stack(
            [
                (s_plus + eps).log(),
                (s_minus + eps).log(),
                tot.log(),
                raw_delta,
            ],
            dim=-1,
        )
        out = self.delta_head(feat)
        mu_delta = (raw_delta + out[:, 0]).clamp(-_DELTA_CLAMP, _DELTA_CLAMP)
        sd_delta = F.softplus(out[:, 1]) + eps

        # Centrics: delta = 0 exactly.
        mu_delta = torch.where(
            centric_pooled, torch.zeros_like(mu_delta), mu_delta
        )
        sd_delta = torch.where(
            centric_pooled, torch.full_like(sd_delta, eps), sd_delta
        )

        sigma_delta_shell = self._sigma_delta(shell_pooled, s2_pooled)
        return mu_delta, sd_delta, sigma_delta_shell

    def _pooled_context(
        self,
        metadata: dict,
        inverse0: Tensor,
        n_pooled: int,
        d_obs: Tensor,
        device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-pooled-id helpers: (plus_obs, centric_pooled, shell_pooled, s2_pooled)."""
        plus = metadata["friedel_plus"].bool().to(device)
        centric_obs = (
            metadata["centric"].bool().to(device)
            if "centric" in metadata
            else torch.zeros_like(plus)
        )
        centric_pooled = torch.zeros(n_pooled, dtype=torch.bool, device=device)
        centric_pooled[inverse0[centric_obs]] = True
        shell_pooled = torch.zeros(n_pooled, dtype=torch.long, device=device)
        if "group_label" in metadata:
            gl = (
                metadata["group_label"]
                .long()
                .to(device)
                .clamp(0, self.n_shells - 1)
            )
            shell_pooled[inverse0] = gl
        # Mean d per pooled id (constant within a pooled id) -> s^2 for the
        # continuous sigma_delta(s^2) forms.
        d_sum = torch.zeros(n_pooled, device=device).scatter_add(
            0, inverse0, d_obs
        )
        cnt = torch.zeros(n_pooled, device=device).scatter_add(
            0, inverse0, torch.ones_like(d_obs)
        )
        d_pooled = (d_sum / cnt.clamp(min=1.0)).clamp(min=1e-6)
        s2_pooled = 1.0 / (4.0 * d_pooled.pow(2))
        return plus, centric_pooled, shell_pooled, s2_pooled

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
        x_r_i = self.encoders["r_i"](sr)
        x_k_bg = self.encoders["k_bg"](sr)
        x_r_bg = self.encoders["r_bg"](sr)

        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)
        qp = self.surrogates["qp"](x_profile, mc_samples=self.mc_samples)
        profile_mean = qp.mean_profile
        scale = self._get_scale(metadata, device)
        d_obs = metadata["d"].to(device).float()
        cond_mid = self._cond_mid(metadata, device)

        # Common mode q(I0) on the Friedel-POOLED id (inherited conjugate merge).
        pooled_idx = metadata[self.friedel_key].long().to(device)
        qi0_h, alpha0, beta0, inverse0, unique0, tau0 = self._merge(
            x_k_i,
            x_r_i,
            scale,
            profile_mean,
            mask,
            pooled_idx,
            d_obs,
            cond_mid,
        )
        n_pooled = unique0.shape[0]

        # Signed anomalous fraction q(delta) per pooled id. The EB delta needs
        # the same per-obs signal potential + exposure that build the common
        # mode (delta_beta = scale * profile mass), split by sign.
        potential = self._per_obs_potential(
            x_k_i, x_r_i, scale, cond_mid, d_obs
        )
        exposure = scale * (profile_mean * mask).sum(dim=-1)
        plus, centric_pooled, shell_pooled, s2_pooled = self._pooled_context(
            metadata, inverse0, n_pooled, d_obs, device
        )
        mu_delta, sd_delta, sigma_delta_shell = self._delta_posterior(
            potential,
            exposure,
            tau0,
            inverse0,
            n_pooled,
            plus,
            centric_pooled,
            shell_pooled,
            s2_pooled,
        )

        # Sample I0 and delta (reparam), expand to observations, build the rate.
        zI0_h = qi0_h.rsample([self.mc_samples]).clamp(
            min=1e-10
        )  # (S, n_pooled)
        z_delta_h = Normal(mu_delta, sd_delta).rsample([self.mc_samples])
        z_delta_h = torch.where(
            centric_pooled.unsqueeze(0),
            torch.zeros_like(z_delta_h),
            z_delta_h.clamp(-_DELTA_CLAMP, _DELTA_CLAMP),
        )
        sign = torch.where(plus, 1.0, -1.0)  # (B,)
        zI0 = zI0_h[:, inverse0]  # (S, B)
        z_delta = z_delta_h[:, inverse0]  # (S, B)
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

        # Per-observation mate Gamma (mean = E[I0](1 +/- mu_delta); shape from I0).
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
            # qi_h / tau_h / inverse / unique are the COMMON MODE, so the
            # inherited Wilson KL in _step acts on I0 only (the whole point).
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

    def _extra_loss_terms(
        self, outputs: dict, metadata: dict
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """delta regularization. `eb`: none (the EB shrink IS the posterior and
        sigma is method-of-moments); `head`: per-pair KL[q || N(0, sigma^2)]."""
        mu = outputs["mu_delta"]
        sd = outputs["sd_delta"]
        sig = outputs["sigma_delta_shell"]
        acentric = ~outputs["centric_pooled"]
        if not bool(acentric.any()):
            return mu.new_zeros(()), {}

        logs = {
            "abs_delta": mu[acentric].abs().mean().detach(),
            "sigma_delta": sig[acentric].mean().detach(),
        }
        if self.delta_mode == "eb":
            return mu.new_zeros(()), logs

        kl = (
            (sig / sd).log()
            + (sd.pow(2) + mu.pow(2)) / (2.0 * sig.pow(2))
            - 0.5
        )
        # Per-pair normalization (once per acentric pooled id, on the same
        # footing as the likelihood), not /n_obs which is ~1/obs-per-HKL weaker.
        n_pairs = int(acentric.sum())
        kl_delta = kl[acentric].sum() / max(n_pairs, 1) * self.delta_kl_weight
        logs["kl_delta"] = kl_delta.detach()
        return kl_delta, logs

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
        """Merge over the dataset, then expand (I0, delta) into per-mate buffers.

        Common mode merged on the Friedel-pooled id; each pooled id's posterior
        is moment-matched into its +/- mate Gammas keyed on the Friedel-separate
        id, so the per-mate buffers / MTZ writer are unchanged. Requires a
        grouped loader (each pooled id complete in one batch).
        """
        self.eval()
        device = self.alpha_buffer.device
        seen = torch.zeros(self.n_hkl, dtype=torch.bool, device=device)
        self.alpha_buffer.fill_(self.alpha_W)
        self.beta_buffer.fill_(1.0)

        for batch in dataloader:
            _, shoebox, mask, metadata = batch
            shoebox = shoebox.to(device)
            mask = mask.to(device)
            b = shoebox.shape[0]
            sr = (shoebox * mask).reshape(b, 1, *self.shoebox_shape)
            x_k_i = self.encoders["k_i"](sr)
            x_r_i = self.encoders["r_i"](sr)
            scale = self._get_scale(metadata, device)
            d_obs = metadata["d"].to(device).float()
            cond_mid = self._cond_mid(metadata, device)
            profile_mean = self.surrogates["qp"](
                self.encoders["profile"](sr), mc_samples=1
            ).mean_profile

            pooled_idx = metadata[self.friedel_key].long().to(device)
            _, alpha0, beta0, inverse0, unique0, tau0 = self._merge(
                x_k_i,
                x_r_i,
                scale,
                profile_mean,
                mask,
                pooled_idx,
                d_obs,
                cond_mid,
            )
            n_pooled = unique0.shape[0]
            potential = self._per_obs_potential(
                x_k_i, x_r_i, scale, cond_mid, d_obs
            )
            exposure = scale * (profile_mean * mask).sum(dim=-1)
            plus, centric_pooled, shell_pooled, s2_pooled = (
                self._pooled_context(
                    metadata, inverse0, n_pooled, d_obs, device
                )
            )
            mu_delta, sd_delta, _ = self._delta_posterior(
                potential,
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
