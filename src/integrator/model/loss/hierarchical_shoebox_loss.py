"""Hierarchical shoebox loss with fixed per-group intensity priors.

ELBO decomposition:
    L = E_q[ log p(x | I, prf, bg) ]           — Poisson NLL
      - KL( q(prf) || p(prf) )                  — profile prior
      - KL( q(I_i) || Exp(tau_{k(i)}) )         — per-group intensity prior
      - KL( q(bg_i) || Exp(lambda_bg) )          — background prior

tau_k are fixed (not learned) — loaded from reference data or YAML config.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, Gamma, Poisson

from integrator.configs.priors import PriorConfig
from integrator.model.distributions.logistic_normal import ProfilePosterior
from integrator.model.loss.loss import (
    _get_dirichlet_prior,
    _kl,
    _params_as_tensors,
    _prior_kl,
)


class HierarchicalShoeboxLoss(nn.Module):
    """ELBO loss with fixed per-group Exponential intensity priors.

    The intensity prior for reflection i in group k is Exp(tau_k), where
    tau_k values are fixed constants loaded from reference data.

    Parameters
    ----------
    pprf_cfg : PriorConfig or None
        Profile prior config (Dirichlet or LogisticNormal path).
    pbg_cfg : PriorConfig or None
        Background prior config (typically Exponential).
    pi_cfg : PriorConfig or None
        Not used — intensity prior comes from tau_per_group.
    mc_samples : int
        Monte Carlo samples for KL estimation.
    eps : float
        Numerical stability constant.
    tau_per_group : list[float] or str
        Fixed tau_k values per group. Either a list of floats or a path
        to a .pt file containing either a tensor or a dict with key 'tau_I'.
    dataset_size : int
        Kept for factory/setup compatibility.
    """

    def __init__(
        self,
        *,
        pprf_cfg: PriorConfig | None = None,
        pbg_cfg: PriorConfig | None = None,
        pi_cfg: PriorConfig | None = None,
        mc_samples: int = 4,
        eps: float = 1e-6,
        tau_per_group: list[float] | str | None = None,
        dataset_size: int = 1,
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.eps = eps
        self.dataset_size = dataset_size

        # Profile prior (Dirichlet path — ignored when qp is ProfilePosterior)
        self.pprf_cfg = pprf_cfg
        self.pprf_params = (
            _get_dirichlet_prior(pprf_cfg)
            if pprf_cfg is not None and pprf_cfg.name == "dirichlet"
            else None
        )

        # Background prior
        self.pbg_cfg = pbg_cfg
        self.pbg_params = (
            _params_as_tensors(pbg_cfg) if pbg_cfg is not None else None
        )

        # pi_cfg unused — intensity prior comes from tau_per_group
        self.pi_cfg = pi_cfg

        # Fixed per-group tau values
        if tau_per_group is None:
            raise ValueError("tau_per_group is required for HierarchicalShoeboxLoss")

        if isinstance(tau_per_group, str):
            loaded = torch.load(tau_per_group, weights_only=True)
            if isinstance(loaded, dict):
                tau_tensor = loaded["tau_I"].float()
            else:
                tau_tensor = loaded.float()
        else:
            tau_tensor = torch.tensor(tau_per_group, dtype=torch.float32)

        self.register_buffer("tau_per_group", tau_tensor)

    def forward(
        self,
        rate: Tensor,
        counts: Tensor,
        qp: Distribution | ProfilePosterior,
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

        kl = torch.zeros(batch_size, device=device)
        kl_prf = torch.zeros(batch_size, device=device)
        kl_i = torch.zeros(batch_size, device=device)
        kl_bg = torch.zeros(batch_size, device=device)

        # ── Profile KL ──────────────────────────────────────────────
        if isinstance(qp, ProfilePosterior):
            weight = self.pprf_cfg.weight if self.pprf_cfg is not None else 1.0
            kl_prf = qp.kl_divergence() * weight
            kl = kl + kl_prf
        elif self.pprf_cfg is not None and self.pprf_params is not None:
            kl_prf = _prior_kl(
                prior_cfg=self.pprf_cfg,
                q=qp,
                params=self.pprf_params,
                weight=self.pprf_cfg.weight,
                device=device,
                mc_samples=self.mc_samples,
                eps=self.eps,
            )
            kl = kl + kl_prf

        # ── Intensity KL: KL(q(I_i) || Exp(tau_{k(i)})) ──────────
        tau_per_refl = self.tau_per_group[group_labels.long()]  # [B]
        p_i = Gamma(
            concentration=torch.ones_like(tau_per_refl),
            rate=tau_per_refl,
        )  # Exp(tau) = Gamma(1, tau)
        kl_i = _kl(qi, p_i, self.mc_samples, eps=self.eps)
        kl = kl + kl_i

        # ── Background KL ───────────────────────────────────────────
        if self.pbg_cfg is not None and self.pbg_params is not None:
            kl_bg = _prior_kl(
                prior_cfg=self.pbg_cfg,
                q=qbg,
                params=self.pbg_params,
                weight=self.pbg_cfg.weight,
                device=device,
                mc_samples=self.mc_samples,
                eps=self.eps,
            )
            kl = kl + kl_bg

        # ── Poisson NLL ──────────────────────────────────────────────
        ll = Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        neg_ll = (-ll_mean).sum(1)

        # ── Total loss ───────────────────────────────────────────────
        loss = (neg_ll + kl).mean()

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i.mean(),
            "kl_bg_mean": kl_bg.mean(),
        }
