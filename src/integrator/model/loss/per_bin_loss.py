"""Per-bin resolution loss with group-dependent priors for all components.

ELBO decomposition:
    L = E_q[ log p(x | I, prf, bg) ]             — Poisson NLL
      - KL( q(prf_i) || Dir(alpha_{k(i)}) )       — per-group profile prior
      - KL( q(I_i)   || Exp(tau_{k(i)}) )         — per-group intensity prior
      - KL( q(bg_i)  || Exp(lambda_{k(i)}) )      — per-group background prior

All prior parameters are fixed (not learned) — loaded from simulation output.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Dirichlet, Distribution, Gamma, Poisson

from integrator.model.distributions.logistic_normal import ProfilePosterior
from integrator.model.loss.loss import _kl


def _load_buffer(value: list[float] | str) -> Tensor:
    """Load a tensor from a list of floats or a .pt file path."""
    if isinstance(value, str):
        loaded = torch.load(value, weights_only=True)
        if isinstance(loaded, dict):
            # support dicts with a single key
            return next(iter(loaded.values())).float()
        return loaded.float()
    return torch.tensor(value, dtype=torch.float32)


class PerBinLoss(nn.Module):
    """ELBO loss with per-group priors for intensity, background, and profile.

    Parameters
    ----------
    mc_samples : int
        Monte Carlo samples for KL estimation.
    eps : float
        Numerical stability constant for Poisson rate.
    tau_per_group : list[float] or str
        Exponential rates for intensity prior, one per group.
    bg_rate_per_group : list[float] or str
        Exponential rates for background prior, one per group.
    concentration_per_group : str
        Path to .pt file with Dirichlet concentrations (n_groups, n_pixels).
    pprf_weight : float
        Scaling factor for profile KL term.
    pbg_weight : float
        Scaling factor for background KL term.
    pi_weight : float
        Scaling factor for intensity KL term.
    """

    def __init__(
        self,
        *,
        mc_samples: int = 4,
        eps: float = 1e-6,
        tau_per_group: list[float] | str,
        bg_rate_per_group: list[float] | str,
        concentration_per_group: str,
        pprf_weight: float = 1.0,
        pbg_weight: float = 1.0,
        pi_weight: float = 1.0,
        # accepted for factory compatibility but unused
        pprf_cfg=None,
        pbg_cfg=None,
        pi_cfg=None,
        dataset_size: int = 1,
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.eps = eps
        self.dataset_size = dataset_size
        self.pprf_weight = pprf_weight
        self.pbg_weight = pbg_weight
        self.pi_weight = pi_weight

        self.register_buffer("tau_per_group", _load_buffer(tau_per_group))
        self.register_buffer("bg_rate_per_group", _load_buffer(bg_rate_per_group))
        self.register_buffer(
            "concentration_per_group",
            _load_buffer(concentration_per_group).clamp(min=1e-6),
        )

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
        groups = group_labels.long()

        kl = torch.zeros(batch_size, device=device)
        kl_prf = torch.zeros(batch_size, device=device)
        kl_i = torch.zeros(batch_size, device=device)
        kl_bg = torch.zeros(batch_size, device=device)

        # ── Profile KL: KL(q(prf_i) || Dir(alpha_{k(i)})) ────────
        if isinstance(qp, ProfilePosterior):
            kl_prf = qp.kl_divergence() * self.pprf_weight
        else:
            alpha = self.concentration_per_group[groups]  # (B, n_pixels)
            p_prf = Dirichlet(alpha)
            kl_prf = _kl(qp, p_prf, self.mc_samples, eps=self.eps) * self.pprf_weight
        kl = kl + kl_prf

        # ── Intensity KL: KL(q(I_i) || Exp(tau_{k(i)})) ──────────
        tau_per_refl = self.tau_per_group[groups]  # (B,)
        p_i = Gamma(
            concentration=torch.ones_like(tau_per_refl),
            rate=tau_per_refl,
        )
        kl_i = _kl(qi, p_i, self.mc_samples, eps=self.eps) * self.pi_weight
        kl = kl + kl_i

        # ── Background KL: KL(q(bg_i) || Exp(lambda_{k(i)})) ─────
        bg_rate_per_refl = self.bg_rate_per_group[groups]  # (B,)
        p_bg = Gamma(
            concentration=torch.ones_like(bg_rate_per_refl),
            rate=bg_rate_per_refl,
        )
        kl_bg = _kl(qbg, p_bg, self.mc_samples, eps=self.eps) * self.pbg_weight
        kl = kl + kl_bg

        # ── Poisson NLL ───────────────────────────────────────────
        ll = Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        neg_ll = (-ll_mean).sum(1)

        # ── Total loss ────────────────────────────────────────────
        loss = (neg_ll + kl).mean()

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i.mean(),
            "kl_bg_mean": kl_bg.mean(),
        }
