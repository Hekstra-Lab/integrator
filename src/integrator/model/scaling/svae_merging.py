"""Structured-VAE per-HKL merging via a per-pixel signal responsibility.

Sibling of `AmortizedMergingIntegrator`. Instead of a learned per-observation
potential head and a background surrogate, the profile logits are SHARED: a
per-observation gate forms a per-pixel signal responsibility

    r_p = sigmoid(profile_logit_p + gate_i)

(Cemgil's Poisson data augmentation, in one pass). The SAME r drives both
conjugate readouts:

    I_h ~ Gamma(alpha_W + sum_i sum_p r*c,   tau_h + sum_i s_i*sum_p prf)
    b_i ~ Gamma(a_bg    + sum_p (1-r)*c,      b_bg + P_i)

so signal + background = total count -- photons are conserved and the bg/signal
split is a partition, not two competing heads (robust to the bg-eats-signal
collapse). The background prior is the loss's per-resolution-bin Gamma; the gate
is a small head on the `k_i` features (initialized negative so r starts low and
signal emerges as the profile sharpens).

Subclasses `AmortizedMergingIntegrator` to reuse the scale field, the per-HKL
Wilson KL, the anomalous-preserving auxiliary losses and the train/finalize
plumbing; it overrides only the forward / merge and needs just the profile +
k_i encoders and the qp surrogate.
"""

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gamma

from integrator import configs
from integrator.model.scaling.amortized_merging import (
    AmortizedMergingIntegrator,
)
from integrator.model.scaling.merge_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
    _sample_profile,
    _scatter_sum_compact,
)


class SVAEMergingIntegrator(AmortizedMergingIntegrator):
    """Per-HKL merging via a shared-logit per-pixel responsibility (structured VAE).

    See the module docstring. Background is a conjugate readout (no surrogate),
    so only the profile + k_i encoders and the qp surrogate are needed.
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
        # The potential head is replaced by the responsibility gate; drop it so
        # it does not sit unused in the optimizer.
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
        """Per-obs background Gamma prior (concentration, rate) from the loss.

        Reuses the loss's per-resolution-bin background prior buffers (fit by
        prepare_per_bin_priors); indexes by `group_label` when per-bin, else
        broadcasts the shared scalar to all `n` observations.
        """
        conc = self.loss.bg_concentration.to(device)
        rate = self.loss.bg_rate.to(device)
        if conc.ndim == 1:  # per-resolution-bin priors
            if group_labels is None:
                g = torch.zeros(n, dtype=torch.long, device=device)
            else:
                g = group_labels.to(device).long()
            return conc[g], rate[g]
        return conc.expand(n), rate.expand(n)

    def _responsibility_merge(
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
    ) -> tuple[Gamma, Tensor, Tensor, Tensor, Tensor, Tensor, Gamma, Tensor]:
        """Conjugate merge via a per-pixel signal responsibility.

        r_p = sigmoid(profile_logit_p + gate_i) splits each pixel's count into
        signal (r_p) and background (1-r_p). The same r feeds both conjugate
        readouts (signal -> alpha_h, background -> per-obs Gamma); signal +
        background = total count, so photons are conserved.

        Returns `(qi_h, alpha_h, beta_h, inverse, unique, tau_h, qbg, r)`.
        """
        device = counts.device
        b = counts.shape[0]
        m = mask.reshape(b, -1).float()
        c = counts.clamp(min=0).reshape(b, -1).float() * m
        r = torch.sigmoid(logits + gate.unsqueeze(-1)) * m  # (B, P), masked

        total_counts = c.sum(dim=-1)
        sig_counts = (r * c).sum(dim=-1)            # sum_p r*c   (signal)
        bg_counts = total_counts - sig_counts       # sum_p (1-r)*c (conservation)
        n_pix = m.sum(dim=-1).clamp(min=1.0)        # background exposure P_i

        # Per-HKL intensity posterior (conjugate): signal counts -> alpha,
        # scale*profile-mass exposure -> beta.
        d_sum, inverse, unique = _scatter_sum_compact(d_per_obs, miller_idx)
        cnt, _, _ = _scatter_sum_compact(torch.ones_like(d_per_obs), miller_idx)
        tau_h = self._wilson_tau((d_sum / cnt.clamp(min=1.0)).clamp(min=1e-6))
        delta_beta = scale * (profile_mean * m).sum(dim=-1)
        alpha_sig, _, _ = _scatter_sum_compact(sig_counts, miller_idx)
        beta_sum, _, _ = _scatter_sum_compact(delta_beta, miller_idx)
        alpha_h = self.alpha_W + alpha_sig
        beta_h = tau_h + beta_sum
        qi_h = Gamma(alpha_h.clamp(min=1e-6), beta_h.clamp(min=1e-12))

        # Per-obs background posterior (conjugate) from the (1-r) counts.
        bg_a0, bg_b0 = self._bg_prior(group_labels, b, device)
        qbg = Gamma(
            (bg_a0 + bg_counts).clamp(min=1e-6),
            (bg_b0 + n_pix).clamp(min=1e-12),
        )
        return qi_h, alpha_h, beta_h, inverse, unique, tau_h, qbg, r

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

        # Shared profile logits + a per-obs gate -> per-pixel responsibility;
        # I_h and the per-obs background are both conjugate readouts of it.
        gate = self.gate_head(x_k_i).squeeze(-1)
        (
            qi_h,
            alpha_h,
            beta_h,
            inverse,
            unique_hkls,
            tau_h,
            qbg,
            responsibility,
        ) = self._responsibility_merge(
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

        # Sample I_h once per HKL (broadcast to obs) and form the Poisson rate.
        zI_h = qi_h.rsample([self.mc_samples]).clamp(min=1e-10)
        zI = zI_h[:, inverse]
        zI_scaled = (scale.unsqueeze(0) * zI).unsqueeze(-1).permute(1, 0, 2)
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)
        rate = zI_scaled * zp + zbg

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
            "responsibility": responsibility,
        }

    @torch.no_grad()
    def finalize_merge(self, dataloader) -> None:
        """Recompute the per-HKL posterior over the full dataset (responsibility).

        Like the parent, needs a loader that yields COMPLETE HKL groups per batch
        (`predict_dataloader(grouped=True)`); a guard raises otherwise.
        """
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
            _, alpha_h, beta_h, _, unique, _, _, _ = (
                self._responsibility_merge(
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
