"""Shared KL-divergence helpers for per-bin loss classes."""

import torch
from torch import Tensor
from torch.distributions import Dirichlet, Distribution, Gamma

from integrator.model.distributions.logistic_normal import ProfilePosterior
from integrator.model.loss.loss import _kl


def compute_profile_kl(
    qp: Distribution | ProfilePosterior,
    groups: Tensor,
    concentration_per_group: Tensor | None,
    pprf_weight: float,
    mc_samples: int,
    eps: float,
    device: torch.device,
    metadata: dict | None = None,
) -> Tensor:
    """Compute profile KL divergence (Dirichlet or latent Normal)."""
    meta = metadata or {}
    pgl = meta.get("profile_group_label") if isinstance(meta, dict) else None
    prf_groups = pgl.long().to(device) if pgl is not None else groups

    if isinstance(qp, ProfilePosterior):
        return qp.kl_divergence(prf_groups) * pprf_weight

    if concentration_per_group is None:
        raise RuntimeError(
            "concentration_per_group is required for Dirichlet profile surrogate"
        )
    alpha = concentration_per_group[prf_groups]
    p_prf = Dirichlet(alpha)
    return _kl(qp, p_prf, mc_samples, eps=eps) * pprf_weight


def compute_bg_kl(
    qbg: Distribution,
    groups: Tensor,
    bg_rate_per_group: Tensor,
    bg_concentration_per_group: Tensor | None,
    bg_concentration: float,
    pbg_weight: float,
    mc_samples: int,
    eps: float,
) -> Tensor:
    """Compute background KL divergence."""
    bg_rate_per_refl = bg_rate_per_group[groups]
    if bg_concentration_per_group is not None:
        alpha_bg = bg_concentration_per_group[groups]
    else:
        alpha_bg = torch.full_like(bg_rate_per_refl, bg_concentration)
    p_bg = Gamma(
        concentration=alpha_bg,
        rate=alpha_bg * bg_rate_per_refl,
    )
    return _kl(qbg, p_bg, mc_samples, eps=eps) * pbg_weight
