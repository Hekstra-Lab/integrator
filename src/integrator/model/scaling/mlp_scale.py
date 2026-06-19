"""Per-observation scale fields for the merging models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _chebyshev(x: Tensor, degree: int) -> Tensor:
    """Chebyshev-T basis `T_0..T_degree` evaluated at `x` in `[-1, 1]`.

    Returns shape `(..., degree + 1)`.
    """
    x = x.clamp(-1.0, 1.0)
    terms = [torch.ones_like(x), x]
    for _ in range(2, degree + 1):
        terms.append(2.0 * x * terms[-1] - terms[-2])
    return torch.stack(terms[: degree + 1], dim=-1)


class ChebyshevScale(nn.Module):
    """Smooth positive scale as a Chebyshev polynomial of the rotation frame."""

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
        frame_half = max((frame_max - frame_min) / 2.0, 1.0)
        self.register_buffer("frame_mid", torch.tensor(frame_mid))
        self.register_buffer("frame_half", torch.tensor(frame_half))
        c = torch.zeros(degree + 1)
        # T_0 coefficient seeds a flat scale at `init_scale` (softplus output).
        c[0] = math.log(math.exp(init_scale) - 1.0)
        self.coeffs = nn.Parameter(c)

    def forward(self, frame: Tensor) -> Tensor:
        x = (frame - self.frame_mid) / self.frame_half
        basis = _chebyshev(x, self.degree)
        return F.softplus(basis @ self.coeffs)


class MLPScale(nn.Module):
    """MLP scale that replaces s/lp with a single learned correction."""

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
        n_abs_sh: int = 0,
        absorption_even_only: bool = True,
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

        # Optional crystal-frame SH absorption features as extra inputs
        self.n_abs_sh = int(n_abs_sh)
        n_abs_in = 0
        if self.n_abs_sh > 0:
            lmax = int(round(math.sqrt(self.n_abs_sh + 1))) - 1
            if (lmax + 1) ** 2 - 1 != self.n_abs_sh:
                raise ValueError(
                    f"n_abs_sh={self.n_abs_sh} is not (lmax+1)^2-1 for integer "
                    "lmax; pass (scale_sh_lmax+1)^2-1."
                )
            l_of_col = torch.tensor(
                [l for l in range(1, lmax + 1) for _ in range(2 * l + 1)]
            )
            keep = (
                (l_of_col % 2 == 0)
                if absorption_even_only
                else torch.ones_like(l_of_col, dtype=torch.bool)
            )
            self.register_buffer(
                "abs_cols",
                keep.nonzero(as_tuple=False).squeeze(-1),
                persistent=False,
            )
            n_abs_in = int(keep.sum())

        # Input: [frame_norm, radius_norm, d_norm, lp] + selected SH features.
        n_input = 4 + n_abs_in
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
        self,
        frame: Tensor,
        x: Tensor,
        y: Tensor,
        lp: Tensor,
        d: Tensor,
        absorption_sh: Tensor | None = None,
    ) -> Tensor:
        frame_norm = (frame - self.frame_mid) / self.frame_half
        r = torch.sqrt((x - self.beam_cx).pow(2) + (y - self.beam_cy).pow(2))
        r_norm = r / self.r_max
        d_norm = (d - self.d_min) / (self.d_max - self.d_min)

        features = torch.stack([frame_norm, r_norm, d_norm, lp], dim=-1)
        if self.n_abs_sh > 0:
            if absorption_sh is None:
                raise ValueError(
                    "MLPScale was built with crystal-frame SH inputs but "
                    "absorption_sh was not provided; point the data loader's "
                    "metadata reference at metadata_sh.pt."
                )
            if absorption_sh.shape[-1] != self.n_abs_sh:
                raise ValueError(
                    f"absorption_sh has {absorption_sh.shape[-1]} harmonics, "
                    f"expected {self.n_abs_sh} (scale_sh_lmax mismatch)."
                )
            features = torch.cat(
                [features, absorption_sh[:, self.abs_cols]], dim=-1
            )
        return F.softplus(self.net(features).squeeze(-1))
