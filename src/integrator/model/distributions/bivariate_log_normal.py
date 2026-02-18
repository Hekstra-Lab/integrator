"""Bivariate Log-Normal distribution for joint (I, B) posterior.

Replaces the mean-field factorization q(I)q(B) with a joint distribution
q(I, B) that can represent the anticorrelation between intensity and
background that arises from the Poisson likelihood structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, LogNormal, MultivariateNormal, constraints


class BivariateLogNormal(Distribution):
    """Joint distribution over (I, B) where (log I, log B) is bivariate normal.

    If (log I, log B) ~ N(loc, L @ L^T), then (I, B) ~ BivariateLogNormal.
    The Cholesky parameterization keeps the covariance positive-definite by
    construction and provides a natural reparameterization path.

    Args:
        loc:        Mean in log-space, shape [..., 2].
        scale_tril: Lower-triangular Cholesky factor of the log-space
                    covariance, shape [..., 2, 2].  Diagonal must be > 0.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale_tril": constraints.lower_cholesky,
    }
    has_rsample = True

    def __init__(
        self,
        loc: torch.Tensor,
        scale_tril: torch.Tensor,
        validate_args=None,
    ):
        self.loc = loc
        self.scale_tril = scale_tril
        self._mvn = MultivariateNormal(loc=loc, scale_tril=scale_tril)
        batch_shape = self._mvn.batch_shape
        event_shape = torch.Size([2])
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """Reparameterized samples of (I, B) in the original positive space.

        Sampling path:
            eps ~ N(0, I)
            z   = loc + L @ eps          (log-space bivariate normal)
            I, B = exp(z[0]), exp(z[1])

        Returns:
            Tensor of shape (*sample_shape, *batch_shape, 2).
        """
        log_z = self._mvn.rsample(sample_shape)  # (*sample_shape, *batch, 2)
        return torch.exp(log_z)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log probability density at x = (I, B).

        Uses the change-of-variables formula:
            log q(I, B) = log N(log I, log B | loc, Sigma) - log I - log B

        Args:
            x: Positive tensor of shape (*sample_shape, *batch_shape, 2).

        Returns:
            Log density, shape (*sample_shape, *batch_shape).
        """
        log_x = torch.log(x.clamp(min=1e-10))
        return self._mvn.log_prob(log_x) - log_x.sum(-1)

    @property
    def mean(self) -> torch.Tensor:
        """E[(I, B)] = exp(loc + 0.5 * diag(L @ L^T)), shape [..., 2]."""
        sigma_diag = (self.scale_tril**2).sum(-1)  # [..., 2]
        return torch.exp(self.loc + 0.5 * sigma_diag)

    @property
    def variance(self) -> torch.Tensor:
        """Var[(I, B)] = (exp(diag(Sigma)) - 1) * exp(2*loc + diag(Sigma))."""
        sigma_diag = (self.scale_tril**2).sum(-1)  # [..., 2]
        return (torch.exp(sigma_diag) - 1.0) * torch.exp(2.0 * self.loc + sigma_diag)

    def marginal_i(self) -> LogNormal:
        """Marginal distribution of I: LogNormal(loc[...,0], L_11)."""
        return LogNormal(loc=self.loc[..., 0], scale=self.scale_tril[..., 0, 0])

    def marginal_bg(self) -> LogNormal:
        """Marginal distribution of B: LogNormal(loc[...,1], sqrt(L_21^2 + L_22^2))."""
        scale = (self.scale_tril[..., 1, :] ** 2).sum(-1).sqrt()
        return LogNormal(loc=self.loc[..., 1], scale=scale)

    @property
    def log_space_correlation(self) -> torch.Tensor:
        """Pearson correlation of (log I, log B) in [-1, 1].

        Sigma = L @ L^T, so:
            Cov(log I, log B) = L_11 * L_21
            Std(log I)        = L_11
            Std(log B)        = sqrt(L_21^2 + L_22^2)
        """
        L_11 = self.scale_tril[..., 0, 0]
        L_21 = self.scale_tril[..., 1, 0]
        L_22 = self.scale_tril[..., 1, 1]
        cov = L_11 * L_21
        std_i = L_11
        std_bg = (L_21**2 + L_22**2).sqrt()
        return cov / (std_i * std_bg + 1e-8)


class BivariateLogNormalSurrogate(nn.Module):
    """Inference network module that outputs a BivariateLogNormal over (I, B).

    Maps an encoder representation x ∈ R^{in_features} to the 5 parameters
    of the joint posterior:

        mu_1, mu_2  — log-space means (unconstrained linear outputs)
        raw_L11     — raw diagonal L_11, constrained positive via softplus
        L_21        — off-diagonal entry (unconstrained, controls correlation)
        raw_L22     — raw diagonal L_22, constrained positive via softplus

    The resulting Cholesky factor is:
        L = [[L_11,  0  ],
             [L_21, L_22]]
    so Sigma = L @ L^T.  The correlation in log-space is L_11*L_21 / (L_11 * ||L[1,:]||).

    Args:
        in_features: Dimensionality of the encoder output.
        eps:         Small constant added to Cholesky diagonals for stability.
    """

    def __init__(
        self,
        in_features: int,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, 5)
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        x_: torch.Tensor | None = None,
    ) -> BivariateLogNormal:
        """
        Args:
            x:  Encoder output, shape [B, in_features].
            x_: Optional second encoder output (unused; kept for API parity
                with other surrogate modules).

        Returns:
            BivariateLogNormal with batch_shape=[B], event_shape=[2].
        """
        params = self.fc(x)  # [B, 5]
        mu_1, mu_2, raw_L11, L_21, raw_L22 = params.unbind(dim=-1)

        L_11 = F.softplus(raw_L11) + self.eps  # [B], positive
        L_22 = F.softplus(raw_L22) + self.eps  # [B], positive

        loc = torch.stack([mu_1, mu_2], dim=-1)  # [B, 2]

        # Build lower-triangular Cholesky: [[L_11, 0], [L_21, L_22]]
        zeros = torch.zeros_like(L_11)
        scale_tril = torch.stack(
            [
                torch.stack([L_11, zeros], dim=-1),  # row 0: [L_11, 0]
                torch.stack([L_21, L_22], dim=-1),  # row 1: [L_21, L_22]
            ],
            dim=-2,
        )  # [B, 2, 2]

        return BivariateLogNormal(loc=loc, scale_tril=scale_tril)
