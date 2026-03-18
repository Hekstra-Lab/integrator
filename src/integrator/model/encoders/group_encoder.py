"""Permutation-invariant group encoder for hierarchical shoebox models.

Pools local encoder features by group label (scatter mean), then maps
through an MLP to produce per-group Gamma variational parameters
q(τ_k) = Gamma(α_q, β_q).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Gamma


class GroupEncoder(nn.Module):
    """Produce per-group rate priors q(τ_k) from pooled local features.

    Parameters
    ----------
    encoder_out : int
        Dimension of per-reflection local encoder features.
    hidden_dim : int
        Width of hidden layers in the MLP.
    eps : float
        Numerical stability constant.
    """

    def __init__(
        self,
        encoder_out: int,
        hidden_dim: int = 64,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.encoder_out = encoder_out

        self.mlp = nn.Sequential(
            nn.Linear(encoder_out, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head_alpha = nn.Linear(hidden_dim, 1)
        self.head_beta = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_intensity: Tensor,
        group_labels: Tensor,
    ) -> tuple[Gamma, Tensor]:
        """Pool features by group and produce q(τ_k).

        Parameters
        ----------
        x_intensity : Tensor [B, encoder_out]
            Per-reflection local encoder features.
        group_labels : Tensor [B]
            Integer group index for each reflection.

        Returns
        -------
        q_tau : Gamma
            Variational posterior over group rates, batch shape [n_groups_in_batch].
        tau_per_refl : Tensor [B, 1]
            Sampled τ_k broadcast back to per-reflection level.
        """
        unique_groups, inverse = torch.unique(group_labels, return_inverse=True)
        n_groups = unique_groups.shape[0]

        # Scatter-mean pooling (native PyTorch)
        inv_expanded = inverse.unsqueeze(1).expand_as(x_intensity)
        group_sum = torch.zeros(
            n_groups,
            self.encoder_out,
            device=x_intensity.device,
            dtype=x_intensity.dtype,
        )
        group_sum.scatter_add_(0, inv_expanded, x_intensity)

        group_count = torch.zeros(
            n_groups, device=x_intensity.device, dtype=x_intensity.dtype
        )
        group_count.scatter_add_(
            0, inverse, torch.ones_like(inverse, dtype=x_intensity.dtype)
        )

        group_features = group_sum / group_count.unsqueeze(1).clamp(min=1)

        # MLP → Gamma parameters
        h = self.mlp(group_features)
        alpha = F.softplus(self.head_alpha(h)).squeeze(-1) + self.eps
        beta = F.softplus(self.head_beta(h)).squeeze(-1) + self.eps

        q_tau = Gamma(concentration=alpha, rate=beta)

        # Sample and broadcast back to per-reflection
        tau_group = q_tau.rsample()
        tau_per_refl = tau_group[inverse].unsqueeze(1)

        return q_tau, tau_per_refl
