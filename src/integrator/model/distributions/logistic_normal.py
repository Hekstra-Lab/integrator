"""Low-rank logistic-normal profile surrogate.

Profile surrogate:
    q(h|x) = N(mu_h, diag(sigma_h^2))
    prf = softmax(W @ h + b)

W (K, d) and b (K,) are fixed Hermite-Gaussian basis loaded from
profile_basis.pt.  Only mu_head and logvar_head are trainable.

Prior: h ~ N(0, sigma_p^2 * I_d)
KL:   closed-form diagonal Gaussian KL (~7 nats vs ~250 nats for Dirichlet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ProfilePosterior:
    """Variational posterior q(h|x) = N(mu_h, diag(sigma_h^2)).

    Profiles are recovered as prf = softmax(W @ h + b) where W and b are fixed.

    This is NOT a torch.distributions.Distribution.  It exposes:
      - .rsample(sample_shape)  — reparameterized profile samples
      - .kl_divergence()        — closed-form KL(q||p) per batch element
      - .mean                   — profile at the posterior mean h  (B, K)
      - .concentration          — None (Dirichlet compatibility shim)
    """

    def __init__(
        self,
        mu_h: Tensor,
        logvar_h: Tensor,
        W: Tensor,
        b: Tensor,
        sigma_prior: float,
    ) -> None:
        self.mu_h = mu_h          # (B, d)
        self.logvar_h = logvar_h  # (B, d)
        self.W = W                # (K, d)  fixed
        self.b = b                # (K,)    fixed
        self.sigma_prior = sigma_prior  # scalar

        # Dirichlet compatibility: integrators store qp.concentration
        self.concentration: Tensor | None = None

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def rsample(self, sample_shape: torch.Size = torch.Size([])) -> Tensor:
        """Reparameterized profile sample.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading sample dimensions (e.g. [S] for S MC samples).

        Returns
        -------
        Tensor, shape (*sample_shape, B, K)
            Probability vectors on the simplex.
        """
        std_h = torch.exp(0.5 * self.logvar_h)  # (B, d)
        eps = torch.randn(
            *sample_shape, *self.mu_h.shape, device=self.mu_h.device
        )  # (*sample_shape, B, d)
        h = self.mu_h + std_h * eps          # (*sample_shape, B, d)
        logits = h @ self.W.T + self.b       # (*sample_shape, B, K)
        return F.softmax(logits, dim=-1)

    def rsample_h(self, sample_shape: torch.Size = torch.Size([])) -> Tensor:
        """Reparameterized sample of the latent h (not the profile)."""
        std_h = torch.exp(0.5 * self.logvar_h)
        eps = torch.randn(
            *sample_shape, *self.mu_h.shape, device=self.mu_h.device
        )
        return self.mu_h + std_h * eps

    def h_to_profile(self, h: Tensor) -> Tensor:
        """Deterministic map from latent h to profile vector."""
        logits = h @ self.W.T + self.b
        return F.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mean(self) -> Tensor:
        """Profile evaluated at the posterior mean h.

        This is a point estimate, not the true mean of the profile
        distribution (which has no closed form under the logistic-normal).

        Returns: (B, K)
        """
        logits = self.mu_h @ self.W.T + self.b
        return F.softmax(logits, dim=-1)

    @property
    def mean_profile(self) -> Tensor:
        """Alias for .mean — profile at posterior mean h."""
        return self.mean

    @property
    def mean_h(self) -> Tensor:
        """Posterior mean of h.  Shape: (B, d)"""
        return self.mu_h

    # ------------------------------------------------------------------
    # KL divergence
    # ------------------------------------------------------------------

    def kl_divergence(self) -> Tensor:
        """Closed-form KL(q(h) || p(h)).

        q(h) = N(mu_h, diag(sigma_h^2))
        p(h) = N(0,    sigma_p^2 * I)

        Returns: (B,) — KL per batch element.
        """
        sigma_p_sq = self.sigma_prior ** 2
        sigma_q_sq = self.logvar_h.exp()

        kl = 0.5 * (
            sigma_q_sq / sigma_p_sq
            + self.mu_h ** 2 / sigma_p_sq
            - 1.0
            - torch.log(sigma_q_sq / sigma_p_sq)
        ).sum(dim=-1)

        return kl  # (B,)


# ---------------------------------------------------------------------------
# Surrogate module
# ---------------------------------------------------------------------------


class LogisticNormalSurrogate(nn.Module):
    """Profile surrogate using a low-rank logistic-normal parameterization.

    The profile is:
        prf = softmax(W @ h + b)

    where W (K, d) and b (K,) are a fixed Hermite-Gaussian basis loaded from
    ``basis_path``, and h is a d-dimensional latent vector.

    The variational posterior is:
        q(h | x) = N(mu_h(x), diag(sigma_h(x)^2))

    Only the two linear heads (mu_head, logvar_head) are trainable.

    Parameters
    ----------
    input_dim : int
        Dimension of the encoder output fed to this surrogate.
    basis_path : str
        Path to ``profile_basis.pt`` containing keys:
        ``W`` (K, d), ``b`` (K,), ``d`` (int), ``sigma_prior`` (float).
    """

    def __init__(self, input_dim: int, basis_path: str) -> None:
        super().__init__()

        basis = torch.load(basis_path, weights_only=False)
        self.W: Tensor
        self.b: Tensor
        self.register_buffer("W", basis["W"])  # (K, d)
        self.register_buffer("b", basis["b"])  # (K,)
        self.d: int = int(basis["d"])
        self.sigma_prior: float = float(basis.get("sigma_prior", 3.0))

        self.mu_head = nn.Linear(input_dim, self.d)
        self.logvar_head = nn.Linear(input_dim, self.d)

        # Initialise logvar_head so initial sigma^2 ≈ exp(-2) ≈ 0.14
        # (variance smaller than the prior sigma_p^2 = 9 → informative start)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -2.0)

    def forward(self, x: Tensor) -> ProfilePosterior:
        """Map encoder output to a ProfilePosterior.

        Parameters
        ----------
        x : Tensor, shape (B, input_dim)
            Output of the profile encoder.

        Returns
        -------
        ProfilePosterior
        """
        mu_h = self.mu_head(x)                              # (B, d)
        logvar_h = self.logvar_head(x).clamp(-10.0, 10.0)  # (B, d)

        return ProfilePosterior(
            mu_h=mu_h,
            logvar_h=logvar_h,
            W=self.W,
            b=self.b,
            sigma_prior=self.sigma_prior,
        )
