import torch
import torch.nn as nn
from torch import Tensor


# NOTE: this is not a variational implementation; uses learnable parameters
class ChebyshevSpectrum(nn.Module):
    """Continuous log G(λ) via Chebyshev polynomial expansion."""

    def __init__(
        self,
        degree: int = 4,
        lambda_min: float = 0.95,
        lambda_max: float = 1.25,
        init_from: str | None = None,
    ):
        super().__init__()
        self.degree = degree
        self.n_basis = degree + 1

        lam_mid = (lambda_min + lambda_max) / 2.0
        lam_scale = (lambda_max - lambda_min) / 2.0

        self.register_buffer("lam_mid", torch.tensor(lam_mid))
        self.register_buffer("lam_scale", torch.tensor(lam_scale))

        if init_from is not None:
            saved = torch.load(
                init_from, map_location="cpu", weights_only=False
            )
            c_saved = saved["c"]
            if c_saved.shape[0] == self.n_basis:
                init = c_saved.clone()
            elif c_saved.shape[0] < self.n_basis:
                init = torch.zeros(self.n_basis)
                init[: c_saved.shape[0]] = c_saved
            else:
                init = c_saved[: self.n_basis].clone()
        else:
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
