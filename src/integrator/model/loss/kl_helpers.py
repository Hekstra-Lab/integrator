"""Shared KL-divergence helpers for loss classes."""

import torch
from torch import Tensor
from torch.distributions import Dirichlet, Distribution, Gamma

from integrator.model.distributions.profile_surrogates import ProfileSurrogateOutput


def _kl(
    q: Distribution,
    p: Distribution,
    mc_samples: int,
    eps: float = 0.0,
) -> Tensor:
    """KL(q || p), using closed form if available, else MC estimate."""
    try:
        return torch.distributions.kl.kl_divergence(q, p)
    except NotImplementedError:
        samples = q.rsample(torch.Size([mc_samples]))
        if eps > 0:
            samples = samples.clamp(min=eps)
        log_q = q.log_prob(samples)
        log_p = p.log_prob(samples)
        return (log_q - log_p).mean(dim=0)


def compute_profile_kl_global(
    mu_h: Tensor,
    std_h: Tensor,
    sigma_prior: float,
) -> Tensor:
    """Closed-form KL(q(h) || p(h)) with global prior N(0, sigma_p^2 I).

    Args:
        mu_h: Posterior mean, shape (B, d).
        std_h: Posterior std, shape (B, d).
        sigma_prior: Prior standard deviation (scalar).

    Returns:
        KL per batch element, shape (B,).
    """
    sigma_p_sq = sigma_prior**2
    sigma_q_sq = std_h**2

    return 0.5 * (
        sigma_q_sq / sigma_p_sq
        + mu_h**2 / sigma_p_sq
        - 1.0
        - torch.log(sigma_q_sq / sigma_p_sq)
    ).sum(dim=-1)


def compute_profile_kl_per_bin(
    mu_h: Tensor,
    std_h: Tensor,
    mu_prior: Tensor,
    std_prior: Tensor,
    group_labels: Tensor,
) -> Tensor:
    """Closed-form KL(q(h) || p_k(h)) with per-bin prior.

    Args:
        mu_h: Posterior mean, shape (B, d).
        std_h: Posterior std, shape (B, d).
        mu_prior: Per-bin prior mean, shape (n_bins, d).
        std_prior: Per-bin prior std, shape (n_bins, d).
        group_labels: Bin index per reflection, shape (B,).

    Returns:
        KL per batch element, shape (B,).
    """
    mu_p = mu_prior[group_labels]  # (B, d)
    std_p = std_prior[group_labels]  # (B, d)

    var_q = std_h**2
    var_p = std_p**2

    return 0.5 * (
        var_q / var_p
        + (mu_h - mu_p) ** 2 / var_p
        - 1.0
        - torch.log(var_q / var_p)
    ).sum(dim=-1)


def _get_profile_groups(
    groups: Tensor,
    metadata: dict | None,
    device: torch.device,
) -> Tensor:
    """Resolve profile group labels from metadata or fallback to groups."""
    meta = metadata or {}
    pgl = meta.get("profile_group_label") if isinstance(meta, dict) else None
    if pgl is not None:
        return pgl.long().to(device)
    return groups


def compute_profile_kl(
    qp: Distribution | ProfileSurrogateOutput,
    groups: Tensor,
    sigma_prior: float,
    mu_prior_per_group: Tensor | None,
    std_prior_per_group: Tensor | None,
    concentration_per_group: Tensor | None,
    pprf_weight: float,
    mc_samples: int,
    eps: float,
    device: torch.device,
    metadata: dict | None = None,
) -> Tensor:
    """Compute profile KL divergence.

    For ProfileSurrogateOutput (latent-decoder surrogates): closed-form
    Normal-Normal KL using posterior params (mu_h, std_h) and prior params
    owned by the loss.

    For Dirichlet surrogates: MC-based KL using concentration_per_group.
    """
    prf_groups = _get_profile_groups(groups, metadata, device)

    if isinstance(qp, ProfileSurrogateOutput):
        if mu_prior_per_group is not None:
            kl = compute_profile_kl_per_bin(
                qp.mu_h, qp.std_h,
                mu_prior_per_group, std_prior_per_group,
                prf_groups,
            )
        else:
            kl = compute_profile_kl_global(qp.mu_h, qp.std_h, sigma_prior)
        return kl * pprf_weight

    # Dirichlet path
    if concentration_per_group is None:
        raise RuntimeError(
            "concentration_per_group is required for Dirichlet profile surrogate"
        )
    n_conc_groups = concentration_per_group.shape[0]
    if prf_groups.max() >= n_conc_groups:
        prf_groups = groups
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
