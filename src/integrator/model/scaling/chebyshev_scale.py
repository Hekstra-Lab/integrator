import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integrator.model.loss.learned_spectrum import ChebyshevSpectrum


class ChebyshevScale(nn.Module):
    """Smooth learnable scale factor s(frame) via Chebyshev polynomials.

    Models per-observation scale as a smooth function of frame number
    (image index), capturing beam decay, absorption drift, and other
    slow-varying experimental effects.  The LP correction is applied
    separately in the integrator.

    s(t) = softplus(sum_k c_k T_k(t))

    where t = (frame - frame_mid) / frame_half is normalized to [-1, 1].
    """

    def __init__(
        self,
        degree: int = 5,
        frame_min: float = 0.0,
        frame_max: float = 1000.0,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.degree = degree

        frame_mid = (frame_min + frame_max) / 2.0
        frame_half = (frame_max - frame_min) / 2.0

        self.register_buffer("frame_mid", torch.tensor(frame_mid))
        self.register_buffer("frame_half", torch.tensor(frame_half))

        c = torch.zeros(degree + 1)
        c[0] = math.log(math.expm1(init_scale))
        self.c = nn.Parameter(c)

    def forward(self, frame: Tensor) -> Tensor:
        """Evaluate scale at given frame positions.

        Args:
            frame: (B,) frame numbers (e.g. xyzcal.px.2).

        Returns:
            scale: (B,) positive scale factors.
        """
        t = ((frame - self.frame_mid) / self.frame_half).clamp(-1.0, 1.0)
        phi = torch.stack(
            ChebyshevSpectrum._chebyshev(t, self.degree), dim=-1
        )
        return F.softplus(phi @ self.c)
