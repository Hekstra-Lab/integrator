"""
Conditional Gamma Prior — per-reflection prior parameters produced by a small
MLP conditioned on observable shoebox summary statistics.

Instead of a single global Gamma(α, β) shared across all reflections, this
module outputs a *different* (α_j, β_j) for each reflection j by feeding
two shoebox statistics through a small network:
    - log1p(masked total photons)   — brightness
    - log1p(masked max pixel)       — peak intensity

The network is initialised so that it outputs (init_concentration, init_rate)
for any input at the start of training, and learns to deviate from that
baseline as training proceeds.

A LogNormal hyperprior is placed on the *baseline* (α, β) encoded in the
last-layer bias, regularising the network in the same spirit as
``HierarchicalGammaPrior``.

Reference: empirical Bayes extended to input-dependent priors
  (cf. Agrawal & Domke, NeurIPS 2021, arXiv:2111.03144).
"""

import math

import torch
import torch.nn as nn
from torch.distributions import Gamma, LogNormal


class ConditionalGammaPrior(nn.Module):
    """Input-dependent Gamma prior: Gamma(alpha(x), beta(x)).

    A two-layer MLP maps per-reflection statistics to Gamma parameters.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input statistics (default 2).
    hidden_dim : int
        Width of the hidden layer.
    init_concentration : float
        Baseline alpha value at initialisation.
    init_rate : float
        Baseline beta value at initialisation.
    hyperprior_scale : float
        σ of the LogNormal(0, σ) hyperprior placed on the baseline (α, β)
        encoded in the last-layer bias.
    """

    def __init__(
        self,
        n_features: int = 2,
        hidden_dim: int = 16,
        init_concentration: float = 1.0,
        init_rate: float = 0.5,
        hyperprior_scale: float = 2.0,
    ):
        super().__init__()
        self.hyperprior_scale = hyperprior_scale

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 2),
        )

        # Initialise last layer so that the network outputs constant
        # (init_concentration, init_rate) for any input at the start.
        last: nn.Linear = self.net[-1]  # type: ignore[assignment]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias[0] = math.log(init_concentration)
            last.bias[1] = math.log(init_rate)

    def forward(self, stats: torch.Tensor) -> Gamma:
        """Return a per-reflection Gamma prior.

        Parameters
        ----------
        stats : Tensor[B, n_features]
            Per-reflection summary statistics.

        Returns
        -------
        Gamma distribution with ``batch_shape = (B,)``.
        """
        out = self.net(stats)       # [B, 2]
        alpha = out[:, 0].exp()     # [B]
        beta = out[:, 1].exp()      # [B]
        return Gamma(alpha, beta)

    def hyperprior_log_prob(self) -> torch.Tensor:
        """Log p(alpha_baseline, beta_baseline) under the LogNormal hyperprior.

        The hyperprior is placed on the *exponentiated* last-layer bias,
        i.e. the baseline (α, β) values the network produces when its
        weights are zero.
        """
        last: nn.Linear = self.net[-1]  # type: ignore[assignment]
        hp = LogNormal(0.0, self.hyperprior_scale)
        alpha_baseline = last.bias[0].exp()
        beta_baseline = last.bias[1].exp()
        return hp.log_prob(alpha_baseline) + hp.log_prob(beta_baseline)

    def extra_repr(self) -> str:
        last: nn.Linear = self.net[-1]  # type: ignore[assignment]
        return (
            f"alpha_baseline={last.bias[0].exp().item():.4f}, "
            f"beta_baseline={last.bias[1].exp().item():.4f}, "
            f"hyperprior_scale={self.hyperprior_scale}"
        )
