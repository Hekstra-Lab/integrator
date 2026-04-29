import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, kl_divergence


class LearnedSpectrum(nn.Module):
    """Continuous log G(λ) via Gaussian RBF basis expansion.

    log G(λ) = Σ_j c_j · φ_j(λ)

    where φ_j are Gaussian RBFs with fixed centers and widths.
    The coefficients c_j have a variational posterior q(c) = Normal(μ, σ).
    """

    def __init__(
        self,
        n_basis: int = 16,
        lambda_min: float = 0.9,
        lambda_max: float = 1.1,
        overlap_factor: float = 1.5,
        init_log_K: float = 0.0,
        hp_loc: float = 0.0,
        hp_scale: float = 3.0,
    ):
        super().__init__()
        self.n_basis = n_basis

        spacing = (lambda_max - lambda_min) / max(n_basis - 1, 1)
        pad = spacing * 0.5
        centers = torch.linspace(lambda_min - pad, lambda_max + pad, n_basis)
        width = spacing * overlap_factor

        self.register_buffer("centers", centers)
        self.register_buffer("width", torch.tensor(width))
        self.register_buffer("hp_loc", torch.tensor(hp_loc))
        self.register_buffer("hp_scale", torch.tensor(hp_scale))

        self.coeff_loc = nn.Parameter(torch.full((n_basis,), init_log_K))
        self.coeff_log_scale = nn.Parameter(torch.full((n_basis,), -2.0))

    def design_matrix(self, wavelength: Tensor) -> Tensor:
        """(B,) -> (B, n_basis)"""
        return torch.exp(
            -0.5 * ((wavelength.unsqueeze(-1) - self.centers) / self.width) ** 2
        )

    def q(self) -> Normal:
        return Normal(self.coeff_loc, F.softplus(self.coeff_log_scale))

    def p(self) -> Normal:
        return Normal(
            self.hp_loc.expand(self.n_basis),
            self.hp_scale.expand(self.n_basis),
        )

    def sample_log_G(self, wavelength: Tensor) -> Tensor:
        """Sample log G(λ) for each reflection. (B,) -> (B,)"""
        phi = self.design_matrix(wavelength)
        c = self.q().rsample()
        return phi @ c

    def mean_log_G(self, wavelength: Tensor) -> Tensor:
        """Posterior mean of log G(λ). (B,) -> (B,)"""
        phi = self.design_matrix(wavelength)
        return phi @ self.coeff_loc

    def kl(self) -> Tensor:
        """KL(q(c) || p(c)), summed over basis functions."""
        return kl_divergence(self.q(), self.p()).sum()
