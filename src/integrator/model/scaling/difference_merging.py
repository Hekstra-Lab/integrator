"""Difference-as-latent per-HKL merging (anomalous signal in its own coordinate).

Sibling of `AmortizedMergingIntegrator`. The leak that washes out the anomalous
signal is the coordinate the Wilson prior acts in: written on the two mates
(I+, I-), it shrinks their difference as a side effect of regularizing the
magnitude. This model reparameterizes each Friedel pair into a common mode and a
signed anomalous fraction,

    I(+/-) = I0 * (1 +/- delta),

merges the common mode q(I0) on the Friedel-POOLED id with the inherited
conjugate `_merge` (so the Wilson prior acts on I0 only), and gives delta its own
variational posterior q(delta) = Normal with prior N(0, sigma_delta(shell)^2).
sigma_delta is learned per resolution shell (in-model empirical Bayes), so the
shrinkage strength is fit from the data and is precision-weighted automatically;
centrics are pinned to delta = 0 (exact gauge anchors) and calibrate the noise
floor. Because the (I0, delta) axes are Fisher-orthogonal at delta = 0, the
common-mode prior cannot attenuate the difference.

The per-mate buffers stay keyed on `miller_idx_unfriedelized`, so `finalize_merge`
expands (I0, delta) back into per-mate Gammas by moment matching and the MTZ
writer / `get_merged_qi` are unchanged.
"""

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
        # One sigma_delta per resolution shell (reuse the loss's bin count).
        self.n_shells = max(1, int(getattr(self.loss, "n_bins", 1) or 1))
        raw0 = _inv_softplus(float(getattr(cfg, "sigma_delta_init", 0.05)))
        raw = torch.full((self.n_shells,), raw0)
        if bool(getattr(cfg, "sigma_delta_learn", True)):
            self.sigma_delta_raw = nn.Parameter(raw)
        else:
            self.register_buffer("sigma_delta_raw", raw)

        # delta head: per-pair sign-split signal stats -> (mu residual, log sd).
        # zero-init so mu starts at the raw data estimate and sd at sigma_delta_init.
        h = int(getattr(cfg, "delta_head_hidden", 16))
        self.delta_head = nn.Sequential(
            nn.Linear(4, h), nn.ReLU(), nn.Linear(h, 2)
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        with torch.no_grad():
            self.delta_head[-1].bias.copy_(torch.tensor([0.0, raw0]))

    def _sigma_delta(self) -> Tensor:
        return F.softplus(self.sigma_delta_raw) + 1e-6

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

    def _delta_posterior(
        self,
        potential: Tensor,
        inverse0: Tensor,
        n_pooled: int,
        plus: Tensor,
        centric_pooled: Tensor,
        shell_pooled: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Variational q(delta) per pooled id and its prior sd.

        Returns `(mu_delta, sd_delta, sigma_delta_shell)`. Centrics get
        mu = 0, sd -> 0 (delta pinned to 0).
        """
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
        mu_delta = torch.where(centric_pooled, torch.zeros_like(mu_delta), mu_delta)
        sd_delta = torch.where(
            centric_pooled, torch.full_like(sd_delta, eps), sd_delta
        )

        sigma_delta_shell = self._sigma_delta()[shell_pooled]
        return mu_delta, sd_delta, sigma_delta_shell

    def _pooled_context(
        self, metadata: dict, inverse0: Tensor, n_pooled: int, device
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Per-pooled-id sign mask helpers: (plus_obs, centric_pooled, shell_pooled)."""
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
            gl = metadata["group_label"].long().to(device).clamp(
                0, self.n_shells - 1
            )
            shell_pooled[inverse0] = gl
        return plus, centric_pooled, shell_pooled

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
            x_k_i, x_r_i, scale, profile_mean, mask, pooled_idx, d_obs, cond_mid
        )
        n_pooled = unique0.shape[0]

        # Signed anomalous fraction q(delta) per pooled id.
        potential = self._per_obs_potential(
            x_k_i, x_r_i, scale, cond_mid, d_obs
        )
        plus, centric_pooled, shell_pooled = self._pooled_context(
            metadata, inverse0, n_pooled, device
        )
        mu_delta, sd_delta, sigma_delta_shell = self._delta_posterior(
            potential, inverse0, n_pooled, plus, centric_pooled, shell_pooled
        )

        # Sample I0 and delta (reparam), expand to observations, build the rate.
        zI0_h = qi0_h.rsample([self.mc_samples]).clamp(min=1e-10)  # (S, n_pooled)
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

        zI_scaled = (scale.unsqueeze(0) * zI_mate).unsqueeze(-1).permute(1, 0, 2)
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
        """Per-pair KL[q(delta) || N(0, sigma_delta(shell)^2)] over acentric pairs."""
        mu = outputs["mu_delta"]
        sd = outputs["sd_delta"]
        sig = outputs["sigma_delta_shell"]
        acentric = ~outputs["centric_pooled"]
        if not bool(acentric.any()):
            return mu.new_zeros(()), {}

        kl = (
            (sig / sd).log()
            + (sd.pow(2) + mu.pow(2)) / (2.0 * sig.pow(2))
            - 0.5
        )
        n_obs = outputs["forward_out"]["counts"].shape[0]
        kl_delta = kl[acentric].sum() / n_obs * self.delta_kl_weight
        logs = {
            "kl_delta": kl_delta.detach(),
            "abs_delta": mu[acentric].abs().mean().detach(),
            "sigma_delta": sig[acentric].mean().detach(),
        }
        return kl_delta, logs

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
            _, alpha0, beta0, inverse0, unique0, _ = self._merge(
                x_k_i, x_r_i, scale, profile_mean, mask, pooled_idx, d_obs,
                cond_mid,
            )
            n_pooled = unique0.shape[0]
            potential = self._per_obs_potential(
                x_k_i, x_r_i, scale, cond_mid, d_obs
            )
            plus, centric_pooled, shell_pooled = self._pooled_context(
                metadata, inverse0, n_pooled, device
            )
            mu_delta, sd_delta, _ = self._delta_posterior(
                potential, inverse0, n_pooled, plus, centric_pooled, shell_pooled
            )

            # Map each pooled id to its +/- mate's Friedel-separate (buffer) id.
            uf = metadata[self.merge_key].long().to(device)
            uf_plus = torch.full((n_pooled,), -1, dtype=torch.long, device=device)
            uf_minus = torch.full_like(uf_plus, -1)
            uf_plus[inverse0[plus]] = uf[plus]
            uf_minus[inverse0[~plus]] = uf[~plus]

            mean0 = (alpha0 / beta0.clamp(min=1e-12)).clamp(min=1e-10)
            var0 = (alpha0 / beta0.clamp(min=1e-12).pow(2)).clamp(min=1e-20)

            def write(uf_id: Tensor, factor: Tensor) -> None:
                m = uf_id >= 0
                ids = uf_id[m]
                if bool(seen[ids].any()):
                    raise RuntimeError(
                        "finalize_merge needs a grouped (group_by_asu_id) loader "
                        "so each Friedel pair is complete in one batch; found an "
                        "id spanning batches. Use predict_dataloader(grouped=True)."
                    )
                mean = (mean0 * factor).clamp(min=1e-10)
                var = (
                    var0 * factor.pow(2) + mean0.pow(2) * sd_delta.pow(2)
                ).clamp(min=1e-20)
                self.alpha_buffer[ids] = (mean.pow(2) / var).clamp(min=1e-6)[m]
                self.beta_buffer[ids] = (mean / var).clamp(min=1e-12)[m]
                seen[ids] = True

            write(uf_plus, 1.0 + mu_delta)
            write(uf_minus, 1.0 - mu_delta)
        self.buffer_seen.copy_(seen)
