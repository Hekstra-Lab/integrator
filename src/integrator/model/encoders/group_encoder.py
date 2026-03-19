import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Gamma


class GroupEncoder(nn.Module):
    """Infer per-group rate posteriors q(τ_k) from local encoder features.

    Architecture (DeepSets):
        x_i → φ(x_i) → mean-pool by group → ρ(·) → (α_k, β_k)

    Parameters
    ----------
    encoder_out : int
        Dimension of per-reflection encoder features.
    hidden_dim : int
        Width of φ and ρ hidden layers.
    """

    def __init__(self, encoder_out: int, hidden_dim: int = 64, alpha_min: float = 0.1):
        super().__init__()
        self.alpha_min = alpha_min

        # φ: per-element transform (before pooling)
        self.phi = nn.Sequential(
            nn.Linear(encoder_out, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # ρ: per-group transform (after pooling)
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.head_alpha = nn.Linear(hidden_dim, 1)
        self.head_beta = nn.Linear(hidden_dim, 1)

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
        q_tau : Gamma with batch shape (n_groups_in_batch,)
        tau_per_refl : (B, 1) sampled τ_k broadcast to each reflection.
        """
        # φ: transform each reflection
        z = self.phi(x)  # (B, hidden_dim)

        # Mean-pool by group (simple loop — K is small)
        unique_groups = torch.unique(group_labels)
        n_groups = unique_groups.shape[0]

        group_means = []
        for k in unique_groups:
            mask = group_labels == k
            group_means.append(z[mask].mean(dim=0))

        group_features = torch.stack(group_means)  # (n_groups, hidden_dim)

        # ρ: per-group transform → Gamma params
        h = self.rho(group_features)  # (n_groups, hidden_dim)

        alpha = (
            F.softplus(self.head_alpha(h)).squeeze(-1) + self.alpha_min
        )  # (n_groups,)
        beta = F.softplus(self.head_beta(h)).squeeze(-1) + self.alpha_min  # (n_groups,)

        q_tau = Gamma(concentration=alpha, rate=beta)

        # Sample one τ per group, broadcast to reflections
        tau_group = q_tau.rsample()  # (n_groups,)

        # Map back: unique_groups is sorted (from torch.unique), so use searchsorted
        indices = torch.searchsorted(unique_groups, group_labels)
        tau_per_refl = tau_group[indices].unsqueeze(1)  # (B, 1)

        return q_tau, tau_per_refl
