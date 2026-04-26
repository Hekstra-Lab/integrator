"""1D numerical quadrature for log p(X_i).

Since intensity I is 1D per reflection, we can compute the exact
log-evidence via trapezoidal quadrature. This gives us the KL gap:
    KL_gap = log p(X_i) - ELBO_i >= 0
"""

import math

import torch


def log_evidence_quadrature(
    counts_i: torch.Tensor,
    profile_i: torch.Tensor,
    B_i: float,
    alpha0: float,
    beta0: float,
    n_quad: int = 30000,
    I_max: float | None = None,
) -> float:
    """Compute log p(X_i) for a single reflection via quadrature.

    Args:
        counts_i: (441,) observed counts
        profile_i: (441,) normalized profile
        B_i: background rate (scalar, 0 for Case 1)
        alpha0: prior shape
        beta0: prior rate
        n_quad: number of quadrature points
        I_max: upper integration limit (auto if None)

    Returns:
        log p(X_i) as a float
    """
    S_i = counts_i.sum().item()
    mu0 = alpha0 / beta0

    if I_max is None:
        I_max = max(5.0 * S_i, 5.0 * mu0, 10000.0)

    I_grid = torch.linspace(1e-6, I_max, n_quad, dtype=torch.float64)
    dI = I_grid[1] - I_grid[0]

    counts_d = counts_i.to(torch.float64)
    profile_d = profile_i.to(torch.float64)

    # log integrand = log p(X_i | I) + log p(I)
    # = Σ_j [x_ij * log(I*p_ij + B_i) - (I*p_ij + B_i) - lgamma(x_ij+1)]
    #   + (α₀-1)*log(I) - β₀*I + α₀*log(β₀) - lgamma(α₀)
    log_integrand = torch.zeros(n_quad, dtype=torch.float64)

    for j in range(len(profile_d)):
        rate_j = I_grid * profile_d[j] + B_i  # (n_quad,)
        log_integrand += counts_d[j] * torch.log(rate_j) - rate_j
        log_integrand -= torch.lgamma(counts_d[j] + 1)

    # Prior
    log_integrand += (alpha0 - 1) * torch.log(I_grid) - beta0 * I_grid
    log_integrand += alpha0 * math.log(beta0) - math.lgamma(alpha0)

    # Trapezoidal rule in log space via log-sum-exp
    log_max = log_integrand.max()
    log_evidence = log_max + torch.log(
        (torch.exp(log_integrand - log_max) * dI).sum()
    )

    return log_evidence.item()


def batch_log_evidence(
    counts: torch.Tensor,
    profiles: torch.Tensor,
    B: torch.Tensor,
    alpha0: float,
    beta0: float,
    n_quad: int = 30000,
) -> torch.Tensor:
    """Compute log p(X_i) for all reflections.

    Args:
        counts: (N, 441) observed counts
        profiles: (N, 441) normalized profiles
        B: (N,) per-reflection background (0 for Case 1)
        alpha0: prior shape
        beta0: prior rate
        n_quad: quadrature points

    Returns:
        (N,) tensor of log p(X_i)
    """
    N = counts.shape[0]
    log_evi = torch.zeros(N)

    for i in range(N):
        log_evi[i] = log_evidence_quadrature(
            counts_i=counts[i],
            profile_i=profiles[i],
            B_i=B[i].item(),
            alpha0=alpha0,
            beta0=beta0,
            n_quad=n_quad,
        )
        if (i + 1) % 100 == 0:
            print(f"  Quadrature: {i + 1}/{N} reflections done")

    return log_evi
