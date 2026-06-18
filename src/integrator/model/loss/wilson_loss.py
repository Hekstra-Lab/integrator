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
from integrator.model.loss.kl_helpers import compute_profile_kl
from integrator.model.loss.learned_background import (
    ChebyshevBackgroundPrior,
    ChebyshevConcentration,
    ChebyshevProfilePriorScale,
)

_DEFAULT_PROFILE_PRIOR_SCALE = 3.0


class WilsonLoss(nn.Module):
    """Base ELBO loss with Wilson intensity prior.

    Subclasses implement `_get_tau` to define how the Wilson prior rate
    is computed (scalar G for monochromatic, G(lambda) for polychromatic).
    """

    def __init__(
        self,
        *,
        # Background prior
        bg_rate: float = 1.0,
        bg_concentration: float = 1.0,
        bg_prior_cfg: dict | None = None,
        # Profile prior
        profile_prior_cfg: dict | None = None,
        # B factor
        init_log_B: float = 3.0,
        b_min: float = 0.0,
        # Intensity prior shape
        learn_concentration: bool = False,
        init_alpha: float = 1.0,
        n_bins: int = 1,
        concentration_cfg: dict | None = None,
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
        self.b_min = b_min
        self.bg_rate = bg_rate
        self.bg_concentration = bg_concentration
        if bg_prior_cfg is not None:
            self.bg_prior = ChebyshevBackgroundPrior(**bg_prior_cfg)
        else:
            self.bg_prior = None
        if profile_prior_cfg is not None:
            self.profile_prior = ChebyshevProfilePriorScale(
                **profile_prior_cfg
            )
        else:
            self.profile_prior = None
        # keep the prior cfgs so run artifacts can record them
        self.pprf_cfg = pprf_cfg
        self.pbg_cfg = pbg_cfg
        self.pi_cfg = pi_cfg
        self.pprf_weight = (
            pprf_cfg.weight if pprf_cfg is not None else pprf_weight
        )
        self.pbg_weight = pbg_cfg.weight if pbg_cfg is not None else pbg_weight
        self.pi_weight = pi_cfg.weight if pi_cfg is not None else pi_weight

        self.learn_concentration = learn_concentration

        # Point-estimate B factor
        # used by polychromatic and monochromatic loss classes
        self.raw_B = nn.Parameter(torch.tensor(float(init_log_B)))

        # Continuous learnable concentration alpha(s^2)
        if concentration_cfg is not None:
            self.concentration_fn = ChebyshevConcentration(**concentration_cfg)
        else:
            self.concentration_fn = None

        # Per-bin learnable concentration (Gamma shape)
        if self.learn_concentration:
            init_raw = torch.full((n_bins,), math.log(math.expm1(init_alpha)))
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

        metadata = kwargs.get("metadata")
        if metadata is None or "d" not in metadata:
            raise ValueError("Wilson loss requires metadata['d'].")

        prf_prior_scale: float | Tensor
        if self.profile_prior is not None:
            x_px = metadata["xyzcal.px.0"].to(device)
            y_px = metadata["xyzcal.px.1"].to(device)
            prf_prior_scale = self.profile_prior(x_px, y_px)
        else:
            # owned by the surrogate; default only hit by Dirichlet (ignores it)
            prf_prior_scale = getattr(
                qp, "prior_scale", _DEFAULT_PROFILE_PRIOR_SCALE
            )
        kl_prf = compute_profile_kl(
            qp, prf_prior_scale, self.pprf_weight, device
        )
        kl = kl + kl_prf

        # Wilson intensity KL
        d = metadata["d"].to(device)
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))

        tau = self._get_tau(metadata, s_sq, device)

        if self.concentration_fn is not None:
            alpha_i = self.concentration_fn(s_sq)
            p_i = Gamma(concentration=alpha_i, rate=alpha_i * tau)
        elif self.learn_concentration:
            alpha_i = F.softplus(self.log_alpha_per_group[groups])
            p_i = Gamma(concentration=alpha_i, rate=alpha_i * tau)
        else:
            p_i = Gamma(concentration=torch.ones_like(tau), rate=tau)

        kl_i = kl_divergence(qi, p_i) * self.pi_weight
        kl = kl + kl_i

        if self.bg_prior is not None:
            x_px = metadata["xyzcal.px.0"].to(device)
            y_px = metadata["xyzcal.px.1"].to(device)
            bg_rate, bg_alpha = self.bg_prior(x_px, y_px)
            p_bg = Gamma(concentration=bg_alpha, rate=bg_alpha * bg_rate)
        else:
            p_bg = Gamma(
                concentration=torch.tensor(
                    self.bg_concentration, device=device
                ),
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
