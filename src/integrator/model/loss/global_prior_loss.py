import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, Gamma, Poisson, kl_divergence

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.kl_helpers import compute_profile_kl

_DEFAULT_PROFILE_PRIOR_SCALE = 3.0


class GlobalPriorLoss(nn.Module):
    """ELBO loss with a single global (resolution-independent) intensity prior.

    The pre-Wilson baseline. `q(I)` is regularized toward one global
    `Gamma(i_concentration, i_rate)` prior and `q(bg)` toward a global
    `Gamma(bg_concentration, bg_rate)`; both parameter pairs come from a
    dataset Gamma MLE (see `integrator.utils.prepare_priors.prepare_global_priors`).
    Unlike the Wilson prior, the intensity rate does not vary with resolution,
    so swapping this loss for `monochromatic_wilson` isolates exactly the effect
    of the resolution-dependent Wilson prior.

    The profile KL reuses `compute_profile_kl`, so this loss supports both a
    `Dirichlet` profile (KL against a uniform `Dirichlet(1)`) and the
    learned-basis profile (Gaussian KL on the latent).

    Args:
        i_concentration: Shape of the global intensity Gamma prior.
        i_rate: Rate of the global intensity Gamma prior.
        bg_concentration: Shape of the global background Gamma prior.
        bg_rate: Rate of the global background Gamma prior.
        profile_kl_weight: Weight on the profile KL term.
        background_kl_weight: Weight on the background KL term.
        intensity_kl_weight: Weight on the intensity KL term.
        pi_cfg: Optional intensity `PriorConfig`; only its `weight` is read.
        pbg_cfg: Optional background `PriorConfig`; only its `weight` is read.
        pprf_cfg: Optional profile `PriorConfig`; only its `weight` is read.
    """

    def __init__(
        self,
        *,
        i_concentration: float = 1.0,
        i_rate: float = 1.0,
        bg_concentration: float = 1.0,
        bg_rate: float = 1.0,
        profile_kl_weight: float = 1.0,
        background_kl_weight: float = 1.0,
        intensity_kl_weight: float = 1.0,
        pi_cfg=None,
        pbg_cfg=None,
        pprf_cfg=None,
    ):
        super().__init__()
        self.register_buffer(
            "i_concentration",
            torch.as_tensor(i_concentration, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "i_rate",
            torch.as_tensor(i_rate, dtype=torch.float32),
            persistent=False,
        )
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
        self.background_kl_weight = (
            pbg_cfg.weight if pbg_cfg is not None else background_kl_weight
        )
        self.intensity_kl_weight = (
            pi_cfg.weight if pi_cfg is not None else intensity_kl_weight
        )

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

        # profile kl-divergence (uniform Dirichlet, or Gaussian latent)
        prf_prior_scale = getattr(
            qp, "prior_scale", _DEFAULT_PROFILE_PRIOR_SCALE
        )
        kl_prf = compute_profile_kl(
            qp, prf_prior_scale, self.profile_kl_weight, device
        )
        kl = kl + kl_prf

        # global intensity prior: Gamma(i_concentration, i_rate)
        p_i = Gamma(concentration=self.i_concentration, rate=self.i_rate)
        kl_i = kl_divergence(qi, p_i) * self.intensity_kl_weight
        kl = kl + kl_i

        # global background prior: Gamma(bg_concentration, bg_rate)
        p_bg = Gamma(concentration=self.bg_concentration, rate=self.bg_rate)
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
