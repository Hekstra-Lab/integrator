"""Case 1: Analytical conjugate posterior (B=0).
Under known profile with sum(p)=1 and zero background, the Poisson-Gamma
model is conjugate. The exact posterior is Gamma(α₀ + S_i, β₀ + 1) where
S_i = sum_j x_ij is total counts.
"""

import math

import torch


def run_case1(
    profiles: torch.Tensor,
    I_true: torch.Tensor,
    alpha0: float,
    beta0: float,
    seed: int = 42,
) -> dict:
    """Compute Case 1 analytical results.

    Args:
        profiles: (N, 441) normalized profiles
        I_true: (N,) true intensities
        alpha0: prior shape
        beta0: prior rate
        seed: random seed for count simulation

    Returns:
        dict with keys: counts, S, alpha_star, beta_star, mu, bias,
        predicted_bias, w, mu0, elbo
    """
    torch.manual_seed(seed)

    # Simulate B=0 counts
    rates = I_true.unsqueeze(1) * profiles  # (N, 441)
    counts = torch.poisson(rates)
    S = counts.sum(dim=1)  # (N,)

    # Analytical posterior: Gamma(α₀ + S, β₀ + 1)
    alpha_star = alpha0 + S
    beta_star = beta0 + 1.0
    mu = alpha_star / beta_star

    # Bias
    bias = mu - I_true
    w = beta0 / (beta0 + 1.0)
    mu0 = alpha0 / beta0
    predicted_bias = w * (mu0 - I_true)

    # Exact log-evidence (since KL=0, ELBO = log p(X))
    # log p(X_i) = Σ_j [x_ij*log(p_ij) - lgamma(x_ij+1)]
    #            + α₀*log(β₀) - lgamma(α₀)
    #            + lgamma(α₀+S_i) - (α₀+S_i)*log(β₀+1)
    log_pij = torch.log(profiles.clamp(min=1e-38))
    term1 = (counts * log_pij).sum(dim=1)  # (N,)
    term2 = -torch.lgamma(counts + 1).sum(dim=1)  # (N,)
    term3 = alpha0 * math.log(beta0) - math.lgamma(alpha0)
    term4 = torch.lgamma(alpha_star) - alpha_star * math.log(beta0 + 1.0)
    elbo = term1 + term2 + term3 + term4  # = log p(X_i) exactly

    return {
        "counts": counts,
        "S": S,
        "alpha_star": alpha_star,
        "beta_star": torch.full_like(S, beta_star),
        "mu": mu,
        "bias": bias,
        "predicted_bias": predicted_bias,
        "w": w,
        "mu0": mu0,
        "elbo": elbo,
    }
