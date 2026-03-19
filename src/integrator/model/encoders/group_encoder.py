"""
GroupEncoder: infer per-group variational parameters q(τ_k) = Gamma(α_k, β_k)
from pooled per-reflection encoder features.

Follows the DeepSets architecture (Zaheer et al. 2017) used by
Habermann et al. 2025 for amortized multilevel models:

    φ: per-element transform (before pooling)
    pool: permutation-invariant aggregation (scatter_mean)
    ρ: per-group transform → output variational parameters

The key insight from Habermann et al. is that the summary network
should transform features BEFORE pooling, not after. This lets
each reflection contribute a learned representation to the group
summary, rather than pooling raw features and hoping a post-hoc
MLP can extract group-level statistics.

Reference:
    Habermann et al. (2025). "Amortized Bayesian Multilevel Models."
    arXiv:2408.13230. Section 2.3, Figure 1 (right panel).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Gamma


class GroupEncoder(nn.Module):
    """Infer per-group rate posteriors q(τ_k) from local encoder features.

    Architecture (DeepSets):
        x_i → φ(x_i) → scatter_mean by group → ρ(·) → (α_k, β_k)

    Parameters
    ----------
    encoder_out : int
        Dimension of per-reflection encoder features.
    hidden_dim : int
        Width of φ and ρ hidden layers.
    """

    def __init__(self, encoder_out: int, hidden_dim: int = 64):
        super().__init__()

        # φ: per-element transform (applied before pooling)
        self.phi = nn.Sequential(
            nn.Linear(encoder_out, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # ρ: per-group transform (applied after pooling)
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Output heads for Gamma parameters
        self.head_alpha = nn.Linear(hidden_dim, 1)
        self.head_beta = nn.Linear(hidden_dim, 1)

        # Initialize heads so initial τ_k is in a reasonable range
        # softplus(0) = ln(2) ≈ 0.69, so α ≈ β ≈ 0.69 → mean τ ≈ 1.0
        nn.init.zeros_(self.head_alpha.weight)
        nn.init.zeros_(self.head_alpha.bias)
        nn.init.zeros_(self.head_beta.weight)
        nn.init.zeros_(self.head_beta.bias)

    def forward(
        self,
        x: Tensor,
        group_labels: Tensor,
    ) -> tuple[Gamma, Tensor]:
        """
        Parameters
        ----------
        x : (B, encoder_out)
            Per-reflection encoder features.
        group_labels : (B,)
            Integer group index for each reflection (0..K-1).

        Returns
        -------
        q_tau : Gamma distribution with batch shape (n_groups_in_batch,)
        tau_per_refl : (B, 1) sampled τ_k broadcast to each reflection.
        """
        # φ: transform each reflection's features
        z = self.phi(x)  # (B, hidden_dim)

        # Pool: scatter_mean by group
        unique_groups, inverse = torch.unique(
            group_labels, return_inverse=True
        )
        n_groups = unique_groups.shape[0]
        hidden_dim = z.shape[1]

        # scatter_mean via scatter_add + count
        inv = inverse.unsqueeze(1).expand_as(z)  # (B, hidden_dim)
        group_sum = torch.zeros(
            n_groups, hidden_dim, device=z.device, dtype=z.dtype
        )
        group_sum.scatter_add_(0, inv, z)

        group_count = torch.zeros(n_groups, device=z.device, dtype=z.dtype)
        group_count.scatter_add_(
            0, inverse, torch.ones_like(inverse, dtype=z.dtype)
        )

        group_mean = group_sum / group_count.unsqueeze(
            1
        )  # (n_groups, hidden_dim)

        # ρ: per-group transform
        h = self.rho(group_mean)  # (n_groups, hidden_dim)

        # Gamma parameters (both positive via softplus)
        alpha = (
            F.softplus(self.head_alpha(h)).squeeze(-1) + 1e-4
        )  # (n_groups,)
        beta = F.softplus(self.head_beta(h)).squeeze(-1) + 1e-4  # (n_groups,)

        q_tau = Gamma(concentration=alpha, rate=beta)

        # Sample one τ per group, broadcast to reflections
        tau_group = q_tau.rsample()  # (n_groups,)
        tau_per_refl = tau_group[inverse].unsqueeze(1)  # (B, 1)

        return q_tau, tau_per_refl
