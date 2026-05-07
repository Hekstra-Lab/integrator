import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution, Gamma, Poisson, kl_divergence

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.kl_helpers import (
    _load_buffer,
    compute_profile_kl,
)


class WilsonLoss(nn.Module):
    """Base ELBO loss with Wilson intensity prior.

    Subclasses implement `_get_tau` to define how the Wilson prior rate
    is computed (scalar G for monochromatic, G(λ) for polychromatic).
    """

    def __init__(
        self,
        *,
        mc_samples: int = 4,
        eps: float = 1e-6,
        # Background prior (global)
        bg_rate: float = 1.0,
        bg_concentration: float = 1.0,
        # Profile prior
        profile_basis: str | None = None,
        profile_prior_scale: float = 3.0,
        # B factor
        init_log_B: float = 3.0,
        b_min: float = 0.0,
        # Intensity prior shape
        learn_concentration: bool = False,
        init_alpha: float = 1.0,
        n_bins: int = 20,
        i_concentration_per_group: list[float] | str | None = None,
        # Prior configs from yaml
        pi_cfg=None,
        pbg_cfg=None,
        pprf_cfg=None,
        # KL weights
        pprf_weight: float = 1.0,
        pbg_weight: float = 1.0,
        pi_weight: float = 1.0,
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.eps = eps
        self.b_min = b_min
        self.bg_rate = bg_rate
        self.bg_concentration = bg_concentration
        self.profile_prior_scale = profile_prior_scale
        self.pprf_weight = (
            pprf_cfg.weight if pprf_cfg is not None else pprf_weight
        )
        self.pbg_weight = pbg_cfg.weight if pbg_cfg is not None else pbg_weight
        self.pi_weight = pi_cfg.weight if pi_cfg is not None else pi_weight

        # Profile prior scale from basis file
        if profile_basis is not None:
            basis = torch.load(profile_basis, weights_only=False)
            self.profile_prior_scale = float(
                basis.get("sigma_prior", profile_prior_scale)
            )

        # Auto-detect learn_concentration from pi_cfg
        if (
            pi_cfg is not None
            and hasattr(pi_cfg, "name")
            and pi_cfg.name == "gamma"
        ):
            learn_concentration = True
        self.learn_concentration = learn_concentration

        # Point-estimate B factor
        # used by polychromatic and monochromatic loss classes
        self.raw_B = nn.Parameter(torch.tensor(float(init_log_B)))

        # Per-bin learnable concentration (Gamma shape)
        if self.learn_concentration:
            if i_concentration_per_group is not None:
                alpha_init = _load_buffer(i_concentration_per_group).clamp(
                    min=0.1
                )
                init_raw = torch.log(torch.expm1(alpha_init))
            else:
                init_raw = torch.full(
                    (n_bins,), math.log(math.expm1(init_alpha))
                )
            self.log_alpha_per_group = nn.Parameter(init_raw)

    def get_B(self) -> Tensor:
        return F.softplus(self.raw_B) + self.b_min

    @abstractmethod
    def _get_tau(
        self, metadata: dict, s_sq: Tensor, device: torch.device
    ) -> Tensor:
        """Compute Wilson prior rate tau per reflection."""

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

        # Profile KL
        kl_prf = compute_profile_kl(
            qp, self.profile_prior_scale, self.pprf_weight, device
        )
        kl = kl + kl_prf

        # Wilson intensity KL
        metadata = kwargs.get("metadata")
        if metadata is None or "d" not in metadata:
            raise ValueError("Wilson loss requires metadata['d'].")
        d = metadata["d"].to(device)
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))

        tau = self._get_tau(metadata, s_sq, device)

        if self.learn_concentration:
            alpha_i = F.softplus(self.log_alpha_per_group[groups])
            p_i = Gamma(concentration=alpha_i, rate=alpha_i * tau)
        else:
            p_i = Gamma(concentration=torch.ones_like(tau), rate=tau)

        kl_i = kl_divergence(qi, p_i) * self.pi_weight
        kl = kl + kl_i

        # Background KL
        p_bg = Gamma(
            concentration=torch.tensor(self.bg_concentration, device=device),
            rate=torch.tensor(
                self.bg_concentration * self.bg_rate, device=device
            ),
        )
        kl_bg = kl_divergence(qbg, p_bg) * self.pbg_weight
        kl = kl + kl_bg

        # Poisson NLL
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
