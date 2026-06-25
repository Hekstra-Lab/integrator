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


class MergingWilsonLoss(nn.Module):
    """Profile KL + background KL + Poisson NLL, with a learnable Wilson G/B.

    The Wilson prior rate is:
    tau(d) = (1/G) * exp(2 * B * s_sq),
    where s_sq = (sin(theta)/lambda)^2 = 1/(4 d^2)

    """

    def __init__(
        self,
        *,
        bg_rate: float | list[float] = 1.0,
        bg_concentration: float | list[float] = 1.0,
        init_log_B: float = 3.0,
        b_min: float = 0.0,
        init_log_G: float = 0.0,
        n_bins: int = 1,
        lp_correction: bool = False,
        profile_kl_weight: float = 1.0,
        background_kl_weight: float = 1.0,
        # Accepted for factory compatibility
        pprf_cfg=None,
        pbg_cfg=None,
        pi_cfg=None,
    ):
        super().__init__()
        self.b_min = b_min
        self.n_bins = n_bins
        self._apply_lp = lp_correction

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

        self.pprf_cfg = pprf_cfg
        self.pbg_cfg = pbg_cfg
        self.pi_cfg = pi_cfg
        self.profile_kl_weight = (
            pprf_cfg.weight if pprf_cfg is not None else profile_kl_weight
        )
        self.background_kl_weight = (
            pbg_cfg.weight if pbg_cfg is not None else background_kl_weight
        )

        # Learned point-estimate Wilson scale (G) and B-factor
        self.raw_B = nn.Parameter(torch.tensor(float(init_log_B)))
        self.raw_G = nn.Parameter(torch.tensor(float(init_log_G)))

    def get_B(self) -> Tensor:
        return F.softplus(self.raw_B) + self.b_min

    def get_G(self) -> Tensor:
        return F.softplus(self.raw_G)

    def _get_tau(
        self, metadata: dict, s_sq: Tensor, device: torch.device
    ) -> Tensor:
        """Wilson prior rate tau."""
        tau = (1.0 / self.get_G()) * torch.exp(2.0 * self.get_B() * s_sq)
        if self._apply_lp:
            lp = metadata["lp"].to(device).clamp(min=1e-8)
            tau = tau * lp
        return tau

    def forward(
        self,
        rate: Tensor,
        counts: Tensor,
        qp: Distribution | ProfileSurrogateOutput,
        qbg: Distribution,
        mask: Tensor,
        group_labels: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        device = rate.device
        counts = counts.to(device)
        mask = mask.to(device)

        # Profile KL
        prf_prior_scale = getattr(
            qp, "prior_scale", _DEFAULT_PROFILE_PRIOR_SCALE
        )
        kl_prf = compute_profile_kl(
            qp, prf_prior_scale, self.profile_kl_weight, device
        )

        # Background KL: shared Gamma, or per-resolution-bin Gamma.
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

        # Background & profile KL
        kl = kl_prf + kl_bg

        # Negative log-likelihood
        ll = Poisson(rate.clamp(min=1e-12)).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        neg_ll = (-ll_mean).sum(1)

        # full loss
        loss = (neg_ll + kl).mean()

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_bg_mean": kl_bg.mean(),
        }
