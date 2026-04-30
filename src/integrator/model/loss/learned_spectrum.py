import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, kl_divergence


class ChebyshevSpectrum(nn.Module):
    """Continuous log G(λ) via Chebyshev polynomial expansion.

    log G(λ) = Σ_k c_k · T_k(x),  x = (λ - λ_mid) / scale ∈ [-1, 1]

    Chebyshev polynomials are bounded in [-1, 1] across the domain,
    avoiding the edge blow-up of monomial bases.
    """

    def __init__(
        self,
        degree: int = 4,
        lambda_min: float = 0.9,
        lambda_max: float = 1.1,
        init_log_K: float = 0.0,
        hp_loc: float = 0.0,
        hp_scale: float = 3.0,
    ):
        super().__init__()
        self.degree = degree
        self.n_basis = degree + 1

        lam_mid = (lambda_min + lambda_max) / 2.0
        lam_scale = (lambda_max - lambda_min) / 2.0

        self.register_buffer("lam_mid", torch.tensor(lam_mid))
        self.register_buffer("lam_scale", torch.tensor(lam_scale))
        self.register_buffer("hp_loc", torch.tensor(hp_loc))
        self.register_buffer("hp_scale", torch.tensor(hp_scale))

        init = torch.zeros(self.n_basis)
        init[0] = init_log_K
        self.coeff_loc = nn.Parameter(init)
        self.coeff_log_scale = nn.Parameter(torch.full((self.n_basis,), -2.0))

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

    def q(self) -> Normal:
        return Normal(self.coeff_loc, F.softplus(self.coeff_log_scale))

    def p(self) -> Normal:
        return Normal(
            self.hp_loc.expand(self.n_basis),
            self.hp_scale.expand(self.n_basis),
        )

    def sample_log_G(self, wavelength: Tensor) -> Tensor:
        """(B,) -> (B,)"""
        phi = self.design_matrix(wavelength)
        c = self.q().rsample()
        return phi @ c

    def mean_log_G(self, wavelength: Tensor) -> Tensor:
        """(B,) -> (B,)"""
        phi = self.design_matrix(wavelength)
        return phi @ self.coeff_loc

    def kl(self) -> Tensor:
        return kl_divergence(self.q(), self.p()).sum()
