"""Hierarchical shoebox loss with per-group adaptive intensity priors.

ELBO decomposition:
    L = E_q[ log p(x | I, prf, bg) ]           — Poisson NLL
      - KL( q(prf) || p(prf) )                  — profile prior
      - KL( q(I_i) || Exp(τ_{k(i)}) )           — adaptive intensity prior
      - KL( q(bg_i) || Exp(λ_bg) )              — background prior
      - (1/N) Σ_k KL( q(log τ_k) || N(μ_0, σ_0²) )  — global hyperprior
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
    """ELBO loss with per-group learned Exponential intensity priors.

    The intensity prior for reflection i in group k is Exp(τ_k), where
    q(log τ_k) = Normal(μ_k, σ_k²) is learned by the GroupEncoder.
    The global hyperprior is p(log τ_k) = Normal(log_tau_mu, log_tau_sigma²).

    Parameters
    ----------
    pprf_cfg : PriorConfig or None
        Profile prior config (Dirichlet). None when using LogisticNormal.
    pbg_cfg : PriorConfig or None
        Background prior config (typically Exponential).
    pi_cfg : PriorConfig or None
        Not used (intensity prior is adaptive), kept for factory compat.
    mc_samples : int
        Monte Carlo samples for KL estimation.
    eps : float
        Numerical stability constant.
    log_tau_mu : float
        Prior mean for log τ_k (Normal hyperprior).
    log_tau_sigma : float
        Prior std for log τ_k (Normal hyperprior).
    dataset_size : int
        Total training set size N for global KL scaling.
    """

    def __init__(
        self,
        *,
        pprf_cfg: PriorConfig | None = None,
        pbg_cfg: PriorConfig | None = None,
        pi_cfg: PriorConfig | None = None,
        mc_samples: int = 4,
        eps: float = 1e-6,
        log_tau_mu: float = -6.9,
        log_tau_sigma: float = 1.0,
        dataset_size: int = 1,
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.eps = eps
        self.log_tau_mu = log_tau_mu
        self.log_tau_sigma = log_tau_sigma
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

        # pi_cfg unused — intensity prior comes from q(τ_k)
        self.pi_cfg = pi_cfg

    def forward(
        self,
        rate: Tensor,
        counts: Tensor,
        qp: Distribution | ProfilePosterior,
        qi: Distribution,
        qbg: Distribution,
        mask: Tensor,
        mu: Tensor,
        logvar: Tensor,
        tau_per_refl: Tensor,
        group_labels: Tensor,
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

        # ── Intensity KL: KL(q(I_i) || Exp(τ_{k(i)})) ──────────────
        tau_flat = tau_per_refl.squeeze(-1)
        p_i = Gamma(
            concentration=torch.ones_like(tau_flat),
            rate=tau_flat,
        )  # Exp(τ) = Gamma(1, τ)
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

        # ── Global KL: KL(N(μ_k, σ_k²) || N(μ_0, σ_0²)) / N ─────
        sigma_q_sq = logvar.exp()  # (n_groups,)
        sigma_p_sq = self.log_tau_sigma**2

        kl_global = (
            0.5
            * (
                sigma_q_sq / sigma_p_sq
                + (mu - self.log_tau_mu) ** 2 / sigma_p_sq
                - 1.0
                - torch.log(sigma_q_sq / sigma_p_sq)
            ).sum()
            / self.dataset_size
        )

        # ── Poisson NLL ──────────────────────────────────────────────
        ll = Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        neg_ll = (-ll_mean).sum(1)

        # ── Total loss ───────────────────────────────────────────────
        loss = (neg_ll + kl).mean() + kl_global

        # ── Diagnostics ─────────────────────────────────────────────
        tau_samples = tau_per_refl.squeeze(-1)

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i.mean(),
            "kl_bg_mean": kl_bg.mean(),
            "kl_global": kl_global,
            "tau_mean": tau_samples.mean().detach(),
            "tau_std": tau_samples.std().detach(),
        }
