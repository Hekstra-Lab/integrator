import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integrator.model.loss.learned_spectrum import ChebyshevSpectrum


class ChebyshevBackgroundPrior(nn.Module):
    """Smooth background prior Gamma(conc(r), conc(r)·rate(r)) as a function of
    detector radius from beam center, parameterized via Chebyshev polynomials.
    """

    def __init__(
        self,
        degree: int = 4,
        beam_center: list[float] | None = None,
        r_min: float = 0.0,
        r_max: float = 1500.0,
        init_rate: float = 1.0,
        init_alpha: float = 1.0,
    ):
        super().__init__()
        self.degree = degree
        n_basis = degree + 1

        cx, cy = beam_center or [0.0, 0.0]
        r_mid = (r_min + r_max) / 2.0
        r_scale = (r_max - r_min) / 2.0

        self.register_buffer("beam_cx", torch.tensor(cx))
        self.register_buffer("beam_cy", torch.tensor(cy))
        self.register_buffer("r_mid", torch.tensor(r_mid))
        self.register_buffer("r_scale", torch.tensor(r_scale))

        c_rate = torch.zeros(n_basis)
        c_rate[0] = math.log(init_rate)

        c_alpha = torch.zeros(n_basis)
        c_alpha[0] = math.log(math.expm1(init_alpha))

        self.c_rate = nn.Parameter(c_rate)
        self.c_alpha = nn.Parameter(c_alpha)

    def _get_radius(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.sqrt(
            (x - self.beam_cx).pow(2) + (y - self.beam_cy).pow(2)
        )

    def _design_matrix(self, r: Tensor) -> Tensor:
        x = (r - self.r_mid) / self.r_scale
        x = x.clamp(-1.0, 1.0)
        return torch.stack(
            ChebyshevSpectrum._chebyshev(x, self.degree), dim=-1
        )

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (bg_rate, bg_alpha) per reflection, both > 0."""
        r = self._get_radius(x, y)
        phi = self._design_matrix(r)
        bg_rate = torch.exp(phi @ self.c_rate)
        bg_conc = F.softplus(phi @ self.c_alpha)
        return bg_rate, bg_conc
