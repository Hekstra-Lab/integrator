import torch
import torch.nn as nn
from torch import Tensor


class GroupEncoder(nn.Module):
    """Infer per-group rate posteriors q(log τ_k) from local encoder features.

    Works in log-space: q(log τ_k) = Normal(μ_k, σ_k²), then τ_k = exp(log τ_k).
    This avoids Gamma rsample instabilities when τ → 0.

    Architecture (DeepSets):
        x_i → φ(x_i) → mean-pool by group → ρ(·) → (μ_k, logvar_k)

    Parameters
    ----------
    encoder_out : int
        Dimension of per-reflection encoder features.
    hidden_dim : int
        Width of φ and ρ hidden layers.
    log_tau_init : float
        Initial bias for head_mu (should match prior mean, e.g. -6.9).
    """

    def __init__(
        self,
        encoder_out: int,
        hidden_dim: int = 64,
        log_tau_init: float = -6.9,
    ):
        super().__init__()

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

        self.head_mu = nn.Linear(hidden_dim, 1)
        self.head_logvar = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.head_mu.weight)
        nn.init.constant_(self.head_mu.bias, log_tau_init)
        nn.init.zeros_(self.head_logvar.weight)
        nn.init.constant_(self.head_logvar.bias, -2.0)

    def forward(
        self,
        x: Tensor,
        group_labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x : (B, encoder_out)
            Per-reflection encoder features.
        group_labels : (B,)
            Integer group index for each reflection (0..K-1).

        Returns
        -------
        mu : (n_groups,)
            Posterior mean of log τ_k.
        logvar : (n_groups,)
            Posterior log-variance of log τ_k.
        tau_per_refl : (B, 1)
            Sampled τ_k = exp(log τ_k) broadcast to each reflection.
        """
        # φ: transform each reflection
        z = self.phi(x)  # (B, hidden_dim)

        # Mean-pool by group (simple loop — K is small)
        unique_groups = torch.unique(group_labels)

        group_means = []
        for k in unique_groups:
            mask = group_labels == k
            group_means.append(z[mask].mean(dim=0))

        group_features = torch.stack(group_means)  # (n_groups, hidden_dim)

        # ρ: per-group transform → Normal params in log-space
        h = self.rho(group_features)  # (n_groups, hidden_dim)

        mu = self.head_mu(h).squeeze(-1)  # (n_groups,)
        logvar = self.head_logvar(h).squeeze(-1).clamp(-10.0, 4.0)  # (n_groups,)

        # Reparameterized sample: log τ_k = μ_k + σ_k * ε, ε ~ N(0,1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        log_tau = mu + std * eps  # (n_groups,)
        tau_group = torch.exp(log_tau)  # (n_groups,), always positive

        # Map back: unique_groups is sorted (from torch.unique), so use searchsorted
        indices = torch.searchsorted(unique_groups, group_labels)
        tau_per_refl = tau_group[indices].unsqueeze(1)  # (B, 1)

        return mu, logvar, tau_per_refl
