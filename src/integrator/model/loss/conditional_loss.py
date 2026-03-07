"""
Conditional Loss — drop-in replacement for ``Loss`` / ``HierarchicalLoss``
that uses a small MLP to produce per-reflection Gamma prior parameters
conditioned on observable shoebox statistics.

Same ``forward()`` signature and return dict as ``HierarchicalLoss``, plus
``hyperprior_log_prob`` in the output dict.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, Poisson

from integrator.configs.priors import PriorConfig

from .conditional_gamma_prior import ConditionalGammaPrior
from .loss import (
    _build_prior,
    _get_dirichlet_prior,
    _kl,
    _params_as_tensors,
    _prior_kl,
)


def _compute_stats(counts: Tensor, mask: Tensor) -> Tensor:
    """Per-reflection summary statistics used to condition the prior MLP.

    Parameters
    ----------
    counts : Tensor[B, N]
        Raw pixel photon counts (flat).
    mask : Tensor[B, N, 1] or Tensor[B, N]
        Binary mask indicating valid pixels.

    Returns
    -------
    Tensor[B, 2]
        ``[log1p(masked_total), log1p(masked_max)]`` for each reflection.
    """
    m = mask.squeeze(-1)              # [B, N]
    masked = counts * m               # [B, N]
    total = masked.sum(-1)            # [B]
    max_val = masked.max(-1).values   # [B]
    return torch.stack([torch.log1p(total), torch.log1p(max_val)], dim=-1)


class ConditionalLoss(nn.Module):
    """ELBO loss with input-dependent (conditional) Gamma priors.

    When ``conditional_intensity=True`` the intensity prior Gamma(α_j, β_j)
    is produced by a small MLP conditioned on per-reflection shoebox
    statistics, so strong and weak reflections automatically receive different
    priors.  When ``False``, the fixed prior from ``pi_cfg`` is used,
    identical to :class:`Loss`.

    Same treatment for background via ``conditional_background``.

    Parameters
    ----------
    pprf_cfg, pi_cfg, pbg_cfg : PriorConfig | None
        Prior configs.  For conditional targets the ``params`` values become
        the *initial* baseline output of the MLP.
    mc_samples, eps : int, float
        Monte-Carlo samples and numerical epsilon (same as ``Loss``).
    conditional_intensity : bool
        Use the conditional MLP prior for intensity.
    conditional_background : bool
        Use the conditional MLP prior for background.
    hidden_dim : int
        Width of the hidden layer in the prior MLP(s).
    hyperprior_scale : float
        σ of the LogNormal(0, σ) hyperprior on the MLP baseline parameters.
    """

    def __init__(
        self,
        *,
        pprf_cfg: PriorConfig | None,
        pi_cfg: PriorConfig | None,
        pbg_cfg: PriorConfig | None,
        mc_samples: int = 100,
        eps: float = 1e-6,
        conditional_intensity: bool = True,
        conditional_background: bool = False,
        hidden_dim: int = 16,
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
        self.conditional_intensity_prior = None
        self.pi_params = None
        if pi_cfg is not None:
            pi_tensors = _params_as_tensors(pi_cfg)
            assert pi_tensors is not None, "pi_cfg must have params"
            if conditional_intensity:
                self.conditional_intensity_prior = ConditionalGammaPrior(
                    n_features=2,
                    hidden_dim=hidden_dim,
                    init_concentration=pi_tensors["concentration"].item(),
                    init_rate=pi_tensors["rate"].item(),
                    hyperprior_scale=hyperprior_scale,
                )
            else:
                self.pi_params = pi_tensors

        # -- background prior ------------------------------------------------
        self.pbg_cfg = pbg_cfg
        self.conditional_background_prior = None
        self.pbg_params = None
        if pbg_cfg is not None:
            pbg_tensors = _params_as_tensors(pbg_cfg)
            assert pbg_tensors is not None, "pbg_cfg must have params"
            if conditional_background:
                self.conditional_background_prior = ConditionalGammaPrior(
                    n_features=2,
                    hidden_dim=hidden_dim,
                    init_concentration=pbg_tensors["concentration"].item(),
                    init_rate=pbg_tensors["rate"].item(),
                    hyperprior_scale=hyperprior_scale,
                )
            else:
                self.pbg_params = pbg_tensors

    # --------------------------------------------------------------------- #

    def _intensity_prior_kl(
        self, qi: Distribution, stats: Tensor, device: torch.device
    ) -> Tensor:
        if self.conditional_intensity_prior is not None:
            p_i = self.conditional_intensity_prior(stats)
            weight = self.pi_cfg.weight if self.pi_cfg is not None else 1.0
            return _kl(qi, p_i, self.mc_samples) * weight
        if self.pi_cfg is not None and self.pi_params is not None:
            return _prior_kl(
                prior_cfg=self.pi_cfg,
                q=qi,
                params=self.pi_params,
                weight=self.pi_cfg.weight,
                device=device,
                mc_samples=self.mc_samples,
            )
        return torch.zeros(qi.batch_shape, device=device)

    def _background_prior_kl(
        self, qbg: Distribution, stats: Tensor, device: torch.device
    ) -> Tensor:
        if self.conditional_background_prior is not None:
            p_bg = self.conditional_background_prior(stats)
            weight = self.pbg_cfg.weight if self.pbg_cfg is not None else 1.0
            return _kl(qbg, p_bg, self.mc_samples) * weight
        if self.pbg_cfg is not None and self.pbg_params is not None:
            return _prior_kl(
                prior_cfg=self.pbg_cfg,
                q=qbg,
                params=self.pbg_params,
                weight=self.pbg_cfg.weight,
                device=device,
                mc_samples=self.mc_samples,
            )
        return torch.zeros(qbg.batch_shape, device=device)

    def _joint_prior_kl(
        self, q_ib: Distribution, stats: Tensor, device: torch.device
    ) -> Tensor:
        p_i = (
            self.conditional_intensity_prior(stats)
            if self.conditional_intensity_prior is not None
            else (
                _build_prior(self.pi_cfg, self.pi_params, device)
                if self.pi_cfg and self.pi_params
                else None
            )
        )
        p_bg = (
            self.conditional_background_prior(stats)
            if self.conditional_background_prior is not None
            else (
                _build_prior(self.pbg_cfg, self.pbg_params, device)
                if self.pbg_cfg and self.pbg_params
                else None
            )
        )

        if p_i is None and p_bg is None:
            return torch.zeros(q_ib.batch_shape, device=device)

        samples_ib = q_ib.rsample([self.mc_samples])  # [S, B, 2]
        log_q = q_ib.log_prob(samples_ib)             # [S, B]
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

        # Per-reflection summary statistics for the conditional prior MLP
        stats = _compute_stats(counts, mask)  # [B, 2]

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
            kl_i = self._joint_prior_kl(q_ib, stats, device)
            kl += kl_i
        else:
            if qi is not None and self.pi_cfg is not None:
                kl_i = self._intensity_prior_kl(qi, stats, device)
                kl += kl_i
            if qbg is not None and self.pbg_cfg is not None:
                kl_bg = self._background_prior_kl(qbg, stats, device)
                kl += kl_bg

        # -- hyperprior log prob (scalar, not per-sample) -------------------
        if self.conditional_intensity_prior is not None:
            hyperprior_lp = (
                hyperprior_lp
                + self.conditional_intensity_prior.hyperprior_log_prob()
            )
        if self.conditional_background_prior is not None:
            hyperprior_lp = (
                hyperprior_lp
                + self.conditional_background_prior.hyperprior_log_prob()
            )

        # -- NLL -------------------------------------------------------------
        ll = Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        neg_ll = (-ll_mean).sum(1)

        # -- total loss: ELBO - hyperprior log prob --------------------------
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
