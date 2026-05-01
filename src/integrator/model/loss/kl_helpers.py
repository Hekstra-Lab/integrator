"""Shared KL-divergence helpers and utilities for loss classes."""

import torch
from torch import Tensor
from torch.distributions import Distribution, Gamma


def _load_buffer(value: list[float] | str) -> Tensor:
    """Load a tensor from a list of floats or a .pt file path."""
    if isinstance(value, str):
        loaded = torch.load(value, weights_only=True)
        if isinstance(loaded, dict):
            return next(iter(loaded.values())).float()
        return loaded.float()
    return torch.tensor(value, dtype=torch.float32)


from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)


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
    pprf_weight: float,
    device: torch.device,
    metadata: dict | None = None,
) -> Tensor:
    """Compute profile KL divergence.

    Only latent-decoder surrogates (`ProfileSurrogateOutput`) are supported;
    they use a closed-form Normal-Normal KL against either a global
    `sigma_prior` or a per-bin `(mu_prior_per_group, std_prior_per_group)`.
    The per-bin Dirichlet-concentration prior route was removed — pick a
    `ProfileSurrogateOutput`-emitting surrogate instead (e.g.
    `LearnedBasisProfileSurrogate`, `FixedBasisProfileSurrogate`,
    `LogisticNormalSurrogate`).
    """
    prf_groups = _get_profile_groups(groups, metadata, device)

    if not isinstance(qp, ProfileSurrogateOutput):
        raise NotImplementedError(
            f"Profile surrogate of type {type(qp).__name__} is not supported "
            "by compute_profile_kl. The per-bin Dirichlet concentration prior "
            "was removed; use a ProfileSurrogateOutput-emitting surrogate "
            "such as LearnedBasisProfileSurrogate, FixedBasisProfileSurrogate, "
            "or LogisticNormalSurrogate."
        )

    if mu_prior_per_group is not None:
        kl = compute_profile_kl_per_bin(
            qp.mu_h,
            qp.std_h,
            mu_prior_per_group,
            std_prior_per_group,
            prf_groups,
        )
    else:
        kl = compute_profile_kl_global(qp.mu_h, qp.std_h, sigma_prior)
    return kl * pprf_weight


def compute_zi_intensity_kl(
    qi,
    p_i: Distribution,
    pi0: float,
    mc_samples: int,
    eps: float = 0.0,
) -> Tensor:
    """KL for a zero-inflated Gamma posterior vs Gamma prior.

    KL = KL_bernoulli(π || π₀) + π · KL(Gamma(k,r) || p_i)

    The intensity KL is weighted by π — when π→0, no penalty for the
    Gamma parameters.
    """
    from integrator.model.distributions.zero_inflated_gamma import (
        ZeroInflatedGammaOutput,
    )

    if not isinstance(qi, ZeroInflatedGammaOutput):
        return _kl(qi, p_i, mc_samples, eps=eps)

    pi = qi.pi
    pi0_t = torch.tensor(pi0, device=pi.device, dtype=pi.dtype)

    kl_bernoulli = (
        pi * torch.log((pi + 1e-8) / pi0_t)
        + (1 - pi) * torch.log((1 - pi + 1e-8) / (1 - pi0_t))
    )

    kl_gamma = _kl(qi.gamma, p_i, mc_samples, eps=eps)

    return kl_bernoulli + pi * kl_gamma


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
