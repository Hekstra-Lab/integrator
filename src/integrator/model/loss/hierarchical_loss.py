"""
Hierarchical Loss — drop-in replacement for ``Loss`` that learns Gamma
prior parameters from data instead of using fixed hyperparameters.

Same ``forward()`` signature and return dict as ``Loss``, plus an extra
``hyperprior_log_prob`` entry.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, Poisson

from integrator.configs.priors import PriorConfig

from .hierarchical_gamma_prior import HierarchicalGammaPrior
from .loss import (
    _build_prior,
    _get_dirichlet_prior,
    _params_as_tensors,
    _prior_kl,
)


class HierarchicalLoss(nn.Module):
    """ELBO loss with learnable Gamma priors for intensity and/or background.

    When ``hierarchical_intensity=True``, the intensity prior Gamma(α, β)
    has *learnable* α and β (with a LogNormal hyperprior).  Otherwise the
    fixed values from ``pi_cfg`` are used — identical to :class:`Loss`.

    Same treatment for background via ``hierarchical_background``.

    Parameters
    ----------
    pprf_cfg, pi_cfg, pbg_cfg : PriorConfig | None
        Prior configs (same as ``Loss``).  For hierarchical targets the
        ``params.concentration`` and ``params.rate`` become the *initial*
        values of the learnable parameters.
    mc_samples, eps : int, float
        Monte-Carlo samples and epsilon (same as ``Loss``).
    hierarchical_intensity : bool
        Learn the intensity prior parameters.
    hierarchical_background : bool
        Learn the background prior parameters.
    hyperprior_scale : float
        σ of the LogNormal(0, σ) hyperprior on learned parameters.
    """

    def __init__(
        self,
        *,
        pprf_cfg: PriorConfig | None,
        pi_cfg: PriorConfig | None,
        pbg_cfg: PriorConfig | None,
        mc_samples: int = 100,
        eps: float = 1e-6,
        hierarchical_intensity: bool = True,
        hierarchical_background: bool = False,
        hyperprior_scale: float = 2.0,
    ):
        super().__init__()

        self.eps = eps
        self.mc_samples = mc_samples

        # -- profile prior (always fixed) ------------------------------------
        self.pprf_cfg = pprf_cfg
        self.pprf_params = (
            _get_dirichlet_prior(pprf_cfg) if pprf_cfg is not None else None
        )

        # -- intensity prior -------------------------------------------------
        self.pi_cfg = pi_cfg
        self.hierarchical_intensity = None
        self.pi_params = None
        if pi_cfg is not None:
            pi_tensors = _params_as_tensors(pi_cfg)
            assert pi_tensors is not None, "pi_cfg must have params"
            if hierarchical_intensity:
                self.hierarchical_intensity = HierarchicalGammaPrior(
                    init_concentration=pi_tensors["concentration"].item(),
                    init_rate=pi_tensors["rate"].item(),
                    hyperprior_scale=hyperprior_scale,
                )
            else:
                self.pi_params = pi_tensors

        # -- background prior ------------------------------------------------
        self.pbg_cfg = pbg_cfg
        self.hierarchical_background = None
        self.pbg_params = None
        if pbg_cfg is not None:
            pbg_tensors = _params_as_tensors(pbg_cfg)
            assert pbg_tensors is not None, "pbg_cfg must have params"
            if hierarchical_background:
                self.hierarchical_background = HierarchicalGammaPrior(
                    init_concentration=pbg_tensors["concentration"].item(),
                    init_rate=pbg_tensors["rate"].item(),
                    hyperprior_scale=hyperprior_scale,
                )
            else:
                self.pbg_params = pbg_tensors

    # --------------------------------------------------------------------- #

    def _intensity_prior_kl(
        self, qi: Distribution, device: torch.device
    ) -> torch.Tensor:
        """KL(qi || p_i) using learned or fixed intensity prior."""
        if self.hierarchical_intensity is not None and self.pi_cfg is not None:
            return (
                self.hierarchical_intensity.kl_divergence(
                    qi, mc_samples=self.mc_samples
                )
                * self.pi_cfg.weight
            )
        if self.pi_cfg is not None and self.pi_params is not None:
            return _prior_kl(
                prior_cfg=self.pi_cfg,
                q=qi,
                params=self.pi_params,
                weight=self.pi_cfg.weight,
                device=device,
                mc_samples=self.mc_samples,
            )
        return torch.zeros(1, device=device)

    def _background_prior_kl(
        self, qbg: Distribution, device: torch.device
    ) -> torch.Tensor:
        """KL(qbg || p_bg) using learned or fixed background prior."""
        if self.hierarchical_background is not None and self.pbg_cfg is not None:
            return (
                self.hierarchical_background.kl_divergence(
                    qbg, mc_samples=self.mc_samples
                )
                * self.pbg_cfg.weight
            )
        if self.pbg_cfg is not None and self.pbg_params is not None:
            return _prior_kl(
                prior_cfg=self.pbg_cfg,
                q=qbg,
                params=self.pbg_params,
                weight=self.pbg_cfg.weight,
                device=device,
                mc_samples=self.mc_samples,
            )
        return torch.zeros(1, device=device)

    def _joint_prior_kl(
        self, q_ib: Distribution, device: torch.device
    ) -> torch.Tensor:
        """KL for joint (I, B) posterior against (learned or fixed) priors."""
        p_i = (
            self.hierarchical_intensity.prior_distribution()
            if self.hierarchical_intensity is not None
            else (
                _build_prior(self.pi_cfg, self.pi_params, device)
                if self.pi_cfg and self.pi_params
                else None
            )
        )
        p_bg = (
            self.hierarchical_background.prior_distribution()
            if self.hierarchical_background is not None
            else (
                _build_prior(self.pbg_cfg, self.pbg_params, device)
                if self.pbg_cfg and self.pbg_params
                else None
            )
        )

        if p_i is None and p_bg is None:
            return torch.zeros(1, device=device)

        samples_ib = q_ib.rsample([self.mc_samples])  # [S, B, 2]
        log_q = q_ib.log_prob(samples_ib)  # [S, B]
        log_p = torch.zeros_like(log_q)
        if p_i is not None:
            log_p = log_p + p_i.log_prob(samples_ib[..., 0])
        if p_bg is not None:
            log_p = log_p + p_bg.log_prob(samples_ib[..., 1])

        weight = self.pi_cfg.weight if self.pi_cfg is not None else 1.0
        return (log_q - log_p).mean(dim=0) * weight

    def forward(
        self,
        rate: Tensor,
        counts: Tensor,
        qp: Distribution,
        mask: Tensor,
        qi: Distribution | None = None,
        qbg: Distribution | None = None,
        q_ib: Distribution | None = None,
    ) -> dict[str, Tensor]:
        device = rate.device
        batch_size = rate.shape[0]

        counts = counts.to(device)
        mask = mask.to(device)

        kl = torch.zeros(batch_size, device=device)
        kl_prf = torch.zeros(batch_size, device=device)
        kl_i = torch.zeros(batch_size, device=device)
        kl_bg = torch.zeros(batch_size, device=device)
        hyperprior_lp = torch.zeros(1, device=device)

        # -- profile KL (always fixed) --------------------------------------
        if self.pprf_cfg is not None and self.pprf_params is not None:
            kl_prf = _prior_kl(
                prior_cfg=self.pprf_cfg,
                q=qp,
                params=self.pprf_params,
                weight=self.pprf_cfg.weight,
                device=device,
                mc_samples=self.mc_samples,
            )
            kl += kl_prf

        # -- intensity / background KL --------------------------------------
        if q_ib is not None:
            kl_i = self._joint_prior_kl(q_ib, device)
            kl += kl_i
        else:
            if qi is not None and self.pi_cfg is not None:
                kl_i = self._intensity_prior_kl(qi, device)
                kl += kl_i
            if qbg is not None and self.pbg_cfg is not None:
                kl_bg = self._background_prior_kl(qbg, device)
                kl += kl_bg

        # -- hyperprior log prob (scalar, not per-sample) -------------------
        if self.hierarchical_intensity is not None:
            hyperprior_lp = (
                hyperprior_lp + self.hierarchical_intensity.hyperprior_log_prob()
            )
        if self.hierarchical_background is not None:
            hyperprior_lp = (
                hyperprior_lp + self.hierarchical_background.hyperprior_log_prob()
            )

        # -- NLL -------------------------------------------------------------
        ll = Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        neg_ll = (-ll_mean).sum(1)

        # -- total loss: ELBO - hyperprior log prob --------------------------
        # hyperprior_lp is log p(α, β); maximising it = minimising -hyperprior_lp
        loss = (neg_ll + kl).mean() - hyperprior_lp.squeeze()

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i.mean(),
            "kl_bg_mean": kl_bg.mean(),
            "hyperprior_log_prob": hyperprior_lp.squeeze(),
        }
