import torch
import torch.nn as nn
from torch import Tensor


class ChebyshevSpectrum(nn.Module):
    """Continuous log G(λ) via Chebyshev polynomial expansion.

    log G(λ) = Σ_k c_k · T_k(x),  x = (λ - λ_mid) / scale ∈ [-1, 1]

    Chebyshev polynomials are bounded in [-1, 1] across the domain,
    avoiding the edge blow-up of monomial bases.
    """

    def __init__(
        self,
        degree: int = 4,
        lambda_min: float = 0.95,
        lambda_max: float = 1.25,
        # init log K value
        # Hyperprior defaults
    ):
        super().__init__()
        self.degree = degree
        self.n_basis = degree + 1

        lam_mid = (lambda_min + lambda_max) / 2.0
        lam_scale = (lambda_max - lambda_min) / 2.0

        self.register_buffer("lam_mid", torch.tensor(lam_mid))
        self.register_buffer("lam_scale", torch.tensor(lam_scale))

        init = torch.zeros(self.n_basis)
        self.c = nn.Parameter(init)

    @staticmethod
    def _chebyshev(x: Tensor, degree: int) -> list[Tensor]:
        """Evaluate Chebyshev polynomials T_0..T_degree via recurrence."""
        T = [torch.ones_like(x)]
        if degree >= 1:
            T.append(x)
        for k in range(2, degree + 1):
            T.append(2 * x * T[k - 1] - T[k - 2])
        return T

    def design_matrix(self, wavelength: Tensor) -> Tensor:
        """(B,) -> (B, degree+1)"""
        x = (wavelength - self.lam_mid) / self.lam_scale
        return torch.stack(self._chebyshev(x, self.degree), dim=-1)

    def get_log_G(self, wavelength: Tensor) -> Tensor:
        """(B,) -> (B,)"""
        phi = self.design_matrix(wavelength)
        return phi @ self.c
