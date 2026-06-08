import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integrator.model.loss.learned_spectrum import ChebyshevSpectrum


class ChebyshevScale(nn.Module):
    """Smooth learnable scale factor s(frame) via Chebyshev polynomials."""

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
        phi = torch.stack(ChebyshevSpectrum._chebyshev(t, self.degree), dim=-1)
        return F.softplus(phi @ self.c)


class SpatialChebyshevScale(nn.Module):
    """Scale factor s(frame, r) as tensor product of Chebyshev polynomials.

    Extends ChebyshevScale with spatial dependence on detector radius
    to capture absorption and detector efficiency variations.

    s(frame, r) = softplus(sum_jk c_jk T_j(frame_norm) T_k(r_norm))

    where r = distance from beam center.  The tensor product basis
    captures frame-only effects (beam decay), radius-only effects
    (absorption), and cross-terms (e.g. radiation damage that varies
    with scattering angle).
    """

    def __init__(
        self,
        degree_frame: int = 5,
        degree_radius: int = 5,
        frame_min: float = 0.0,
        frame_max: float = 1000.0,
        beam_center: list[float] | None = None,
        r_min: float = 0.0,
        r_max: float = 1500.0,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.degree_frame = degree_frame
        self.degree_radius = degree_radius

        frame_mid = (frame_min + frame_max) / 2.0
        frame_half = (frame_max - frame_min) / 2.0
        self.register_buffer("frame_mid", torch.tensor(frame_mid))
        self.register_buffer("frame_half", torch.tensor(frame_half))

        cx, cy = beam_center or [0.0, 0.0]
        r_mid = (r_min + r_max) / 2.0
        r_half = (r_max - r_min) / 2.0
        self.register_buffer("beam_cx", torch.tensor(cx))
        self.register_buffer("beam_cy", torch.tensor(cy))
        self.register_buffer("r_mid", torch.tensor(r_mid))
        self.register_buffer("r_half", torch.tensor(r_half))

        c = torch.zeros(degree_frame + 1, degree_radius + 1)
        c[0, 0] = math.log(math.expm1(init_scale))
        self.c = nn.Parameter(c)

    def forward(self, frame: Tensor, x: Tensor, y: Tensor) -> Tensor:
        """Evaluate scale at given frame and detector positions.

        Args:
            frame: (B,) frame numbers.
            x: (B,) detector x positions (xyzcal.px.0).
            y: (B,) detector y positions (xyzcal.px.1).

        Returns:
            scale: (B,) positive scale factors.
        """
        t = ((frame - self.frame_mid) / self.frame_half).clamp(-1.0, 1.0)
        r = torch.sqrt((x - self.beam_cx).pow(2) + (y - self.beam_cy).pow(2))
        rn = ((r - self.r_mid) / self.r_half).clamp(-1.0, 1.0)

        phi_t = torch.stack(
            ChebyshevSpectrum._chebyshev(t, self.degree_frame), dim=-1
        )
        phi_r = torch.stack(
            ChebyshevSpectrum._chebyshev(rn, self.degree_radius), dim=-1
        )

        out = (phi_t @ self.c * phi_r).sum(-1)
        return F.softplus(out)


class MLPScale(nn.Module):
    """MLP scale that replaces s/lp with a single learned correction.

    Takes per-observation features (frame, detector x/y, LP, d-spacing)
    and outputs a positive scale factor.

    rate = scale_mlp(features) × F^2 × profile + bg

    Features are normalized to [-1, 1] or [0, 1] using registered
    buffers for stable training.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        frame_min: float = 0.0,
        frame_max: float = 1000.0,
        beam_center: list[float] | None = None,
        r_max: float = 1500.0,
        d_min: float = 1.0,
        d_max: float = 60.0,
        head_init_std: float = 0.0,
    ):
        super().__init__()

        frame_mid = (frame_min + frame_max) / 2.0
        frame_half = max((frame_max - frame_min) / 2.0, 1.0)
        self.register_buffer("frame_mid", torch.tensor(frame_mid))
        self.register_buffer("frame_half", torch.tensor(frame_half))

        cx, cy = beam_center or [0.0, 0.0]
        self.register_buffer("beam_cx", torch.tensor(cx))
        self.register_buffer("beam_cy", torch.tensor(cy))
        self.register_buffer("r_max", torch.tensor(max(r_max, 1.0)))

        self.register_buffer("d_min", torch.tensor(d_min))
        self.register_buffer("d_max", torch.tensor(max(d_max, d_min + 1.0)))

        # Input: [frame_norm, radius_norm, d_norm, lp] = 4 features
        n_input = 4
        layers = []
        in_dim = n_input
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        # Bias 0 so softplus(0) ~ 0.69 (flat constant scale) at init. The output
        # weight is zero by default (legacy: hidden layers get zero gradient on
        # step 0); a small head_init_std seeds the spatial scale structure so it
        # develops from the first step without changing the init scale level.
        nn.init.zeros_(self.net[-1].bias)
        if head_init_std > 0.0:
            nn.init.normal_(self.net[-1].weight, std=head_init_std)
        else:
            nn.init.zeros_(self.net[-1].weight)

    def forward(
        self, frame: Tensor, x: Tensor, y: Tensor, lp: Tensor, d: Tensor
    ) -> Tensor:
        frame_norm = (frame - self.frame_mid) / self.frame_half
        r = torch.sqrt((x - self.beam_cx).pow(2) + (y - self.beam_cy).pow(2))
        r_norm = r / self.r_max
        d_norm = (d - self.d_min) / (self.d_max - self.d_min)

        features = torch.stack([frame_norm, r_norm, d_norm, lp], dim=-1)
        return F.softplus(self.net(features).squeeze(-1))
