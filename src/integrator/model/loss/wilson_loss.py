from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution, Gamma, Poisson, kl_divergence

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.kl_helpers import compute_profile_kl

_DEFAULT_PROFILE_PRIOR_SCALE = 3.0


class WilsonLoss(nn.Module):
    """Base ELBO loss with Wilson intensity prior.

    Subclasses implement `_get_tau` to define how the Wilson prior rate
    is computed (scalar G for monochromatic, G(lambda) for polychromatic).
    """

    def __init__(
        self,
        *,
        # Background Gamma prior: scalar, or per-resolution-bin prior
        bg_rate: float | list[float] = 1.0,
        bg_concentration: float | list[float] = 1.0,
        # B factor
        init_log_B: float = 3.0,
        b_min: float = 0.0,
        # Resolution bins for per-bin background prior
        n_bins: int = 1,
        # Prior configs from yaml
        pi_cfg=None,
        pbg_cfg=None,
        pprf_cfg=None,
        # KL weights
        profile_kl_weight: float = 1.0,
        background_kl_weight: float = 1.0,
        intensity_kl_weight: float = 1.0,
    ):
        super().__init__()
        self.b_min = b_min  # minimum B-factor
        self.n_bins = n_bins
        self.register_buffer(
            "bg_concentration",
            torch.as_tensor(bg_concentration, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "bg_rate",
            torch.as_tensor(bg_rate, dtype=torch.float32),
            persistent=False,
        )
        # keep the prior cfgs so run artifacts can record them
        self.pprf_cfg = pprf_cfg
        self.pbg_cfg = pbg_cfg
        self.pi_cfg = pi_cfg

        self.profile_kl_weight = (
            pprf_cfg.weight if pprf_cfg is not None else profile_kl_weight
        )
        self.background_kl_weight = pbg_cfg.weight if pbg_cfg is not None else background_kl_weight
        self.intensity_kl_weight = pi_cfg.weight if pi_cfg is not None else intensity_kl_weight

        # Point-estimate B factor
        # used by polychromatic and monochromatic loss classes
        self.raw_B = nn.Parameter(torch.tensor(float(init_log_B)))

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
        group_labels: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        device = rate.device
        batch_size = rate.shape[0]
        counts = counts.to(device)
        mask = mask.to(device)

        kl = torch.zeros(batch_size, device=device)

        # get metadata

        metadata = kwargs.get("metadata")
        if metadata is None or "d" not in metadata:
            raise ValueError("Wilson loss requires metadata['d'].")

        # profile kl-divergence
        prf_prior_scale = getattr(
            qp, "prior_scale", _DEFAULT_PROFILE_PRIOR_SCALE
        )
        kl_prf = compute_profile_kl(
            qp, prf_prior_scale, self.profile_kl_weight, device
        )
        kl = kl + kl_prf

        # Wilson intensity KL
        d = metadata["d"].to(device)
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))

        tau = self._get_tau(metadata, s_sq, device)

        p_i = Gamma(concentration=torch.ones_like(tau), rate=tau)

        kl_i = kl_divergence(qi, p_i) * self.intensity_kl_weight
        kl = kl + kl_i

        # background prior: shared Gamma, or per-resolution-bin
        if self.bg_concentration.ndim == 1:
            if group_labels is None:
                raise ValueError(
                    "per-bin background prior requires group_labels"
                )
            groups = group_labels.to(device).long()
            bg_conc = self.bg_concentration[groups]
            bg_rate = self.bg_rate[groups]
        else:
            bg_conc = self.bg_concentration
            bg_rate = self.bg_rate
        p_bg = Gamma(concentration=bg_conc, rate=bg_rate)
        kl_bg = kl_divergence(qbg, p_bg) * self.background_kl_weight
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
