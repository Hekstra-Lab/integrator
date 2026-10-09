from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import (
    Distribution,
    Gamma,
    Poisson,
)

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.kl_helpers import (
    _kl,
    _load_buffer,
    compute_bg_kl,
    compute_profile_kl,
)
from integrator.model.loss.wilson_loss import WilsonLoss


class PolyWilsonLoss(WilsonLoss):
    """ELBO loss for polychromatic Laue: per-wavelength-bin G_k, global B.

    G_k and B are point-estimate learnable parameters (not variational).
    """

    def __init__(
        self,
        *,
        wavelength_bin_edges: list[float] | str,
        init_log_K_per_bin: Sequence[float] | None = None,
        **kwargs,
    ):
        import inspect

        parent_params = set(
            inspect.signature(WilsonLoss.__init__).parameters
        ) - {"self"}
        parent_kwargs = {k: v for k, v in kwargs.items() if k in parent_params}
        super().__init__(
            **parent_kwargs,
        )

        edges = _load_buffer(wavelength_bin_edges).to(torch.float32)
        if edges.ndim != 1 or edges.numel() < 2:
            raise ValueError(
                "wavelength_bin_edges must be 1-D with >= 2 entries, got "
                f"shape={tuple(edges.shape)}"
            )
        if not torch.all(edges[1:] > edges[:-1]):
            raise ValueError(
                "wavelength_bin_edges must be strictly increasing"
            )

        n_bins = int(edges.numel() - 1)
        self.wavelength_bin_edges: Tensor
        self.register_buffer("wavelength_bin_edges", edges)
        self.n_lambda_bins = n_bins

        # Replace parent's scalar G with per-bin G vector
        if init_log_K_per_bin is not None:
            init_raw = torch.tensor(init_log_K_per_bin, dtype=torch.float32)
            if init_raw.shape != (n_bins,):
                raise ValueError(
                    f"init_log_K_per_bin must have shape ({n_bins},), got "
                    f"{tuple(init_raw.shape)}"
                )
        else:
            init_raw = torch.full((n_bins,), float(self.raw_G.detach()))
        del self.raw_G
        self.raw_G_per_bin = nn.Parameter(init_raw)

        self._oob_warned = False

    def _wavelength_to_bin(self, wavelength: Tensor) -> Tensor:
        """Bucketize per-refl wavelengths into [0, n_lambda_bins)."""
        edges = self.wavelength_bin_edges
        idx = torch.bucketize(wavelength, edges, right=True) - 1
        if not self._oob_warned:
            oob = (wavelength < edges[0]) | (wavelength > edges[-1])
            n_oob = int(oob.sum())
            if n_oob > 0:
                print(
                    f"[PolyWilsonLoss] {n_oob} reflection(s) have wavelength "
                    f"outside [{float(edges[0]):.4f}, {float(edges[-1]):.4f}] Å "
                    "and are being clamped to the nearest bin. "
                    "Widen wavelength_bin_edges to silence this warning."
                )
                self._oob_warned = True
        return idx.clamp(0, self.n_lambda_bins - 1)

    def posterior_means(self) -> dict[str, float]:
        G_per_bin = F.softplus(self.raw_G_per_bin).detach()
        out = {
            "G_mean": G_per_bin.mean().item(),
            "G_min": G_per_bin.min().item(),
            "G_max": G_per_bin.max().item(),
            "B": self.get_B().item(),
        }
        if self.learn_concentration:
            alphas = F.softplus(self.log_alpha_per_group).detach()
            out["alpha_mean"] = alphas.mean().item()
            out["alpha_min"] = alphas.min().item()
            out["alpha_max"] = alphas.max().item()
        return out

    def forward(
        self,
        rate: Tensor,
        counts: Tensor,
        qp: Distribution | ProfileSurrogateOutput,
        qi: Distribution,
        qbg: Distribution,
        mask: Tensor,
        group_labels: Tensor,
        **kwargs,
    ) -> dict[str, Tensor]:
        device = rate.device
        batch_size = rate.shape[0]
        counts = counts.to(device)
        mask = mask.to(device)
        groups = group_labels.long()

        kl = torch.zeros(batch_size, device=device)

        kl_prf = compute_profile_kl(
            qp,
            groups,
            self.profile_sigma_prior,
            None,
            None,
            self.pprf_weight,
            device,
            metadata=kwargs.get("metadata"),
        )
        kl = kl + kl_prf

        metadata = kwargs.get("metadata")
        if (
            metadata is None
            or "d" not in metadata
            or "wavelength" not in metadata
        ):
            raise ValueError(
                "PolyWilsonLoss requires metadata['d'] and metadata['wavelength']."
            )
        d = metadata["d"].to(device)
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        wavelength = metadata["wavelength"].to(device)
        lam_bin = self._wavelength_to_bin(wavelength)

        G_per_bin = F.softplus(self.raw_G_per_bin)
        G_per_refl = G_per_bin[lam_bin]
        B = self.get_B()
        tau = (1.0 / G_per_refl) * torch.exp(2.0 * B * s_sq)

        if self.learn_concentration:
            alpha_i = F.softplus(self.log_alpha_per_group[groups])
            p_i = Gamma(concentration=alpha_i, rate=alpha_i * tau)
        else:
            k_prior = self.k_prior
            p_i = Gamma(
                concentration=torch.full_like(tau, k_prior),
                rate=k_prior * tau,
            )

        kl_i = _kl(qi, p_i, self.mc_samples, eps=self.eps)
        kl_i = kl_i * self.pi_weight
        kl = kl + kl_i

        kl_bg = compute_bg_kl(
            qbg,
            groups,
            self.bg_rate_per_group,
            self.bg_concentration_per_group,
            self.bg_concentration,
            self.pbg_weight,
            self.mc_samples,
            self.eps,
        )
        kl = kl + kl_bg

        ll = Poisson(rate.clamp(min=1e-12)).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        neg_ll = (-ll_mean).sum(1)

        loss = (neg_ll + kl).mean()

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i.mean(),
            "kl_bg_mean": kl_bg.mean(),
        }
