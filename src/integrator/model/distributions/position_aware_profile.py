"""Position-aware profile surrogate.

Extends LearnedBasisProfileSurrogate with a detector-position-dependent
bias b(r, θ) that orients the default profile radially based on the
reflection's position on the detector.

    profile = softmax(W @ h + b(r, θ))

When the encoder is uncertain (h ≈ 0), the profile defaults to an
anisotropic Gaussian oriented radially — correctly capturing the
elongation direction without needing encoder signal.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .profile_surrogates import (
    LearnedBasisProfileSurrogate,
    ProfileSurrogateOutput,
    _sample_and_decode,
    _softplus_inverse,
)


class PositionAwareProfileSurrogate(LearnedBasisProfileSurrogate):
    """Learned basis profile with position-dependent anisotropic Gaussian bias.

    The bias b for each reflection is computed from its detector position
    (xcal, ycal) as a 2D anisotropic Gaussian whose major axis points
    radially away from the beam center.

    Args:
        beam_center: (cx, cy) in pixels. If None, uses (H/2, W/2) of shoebox.
        H, W: shoebox spatial dimensions.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int | None = None,
        output_dim: int = 625,
        init_std: float = 0.5,
        warmstart_basis_path: str | None = None,
        freeze_bias: bool = False,
        beam_center: list[float] | None = None,
        H: int = 25,
        W: int = 25,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            init_std=init_std,
            warmstart_basis_path=warmstart_basis_path,
            freeze_bias=False,
        )
        self.H = H
        self.W = W

        if beam_center is not None:
            self.register_buffer("beam_cx", torch.tensor(beam_center[0]))
            self.register_buffer("beam_cy", torch.tensor(beam_center[1]))
        else:
            self.register_buffer("beam_cx", torch.tensor(H / 2.0))
            self.register_buffer("beam_cy", torch.tensor(W / 2.0))

        # Learnable radial/tangential widths: sigma = softplus(raw) + eps
        # sigma_radial(r) = a_r + b_r * r  (linear in detector radius)
        # sigma_tangential(r) = a_t + b_t * r
        self.raw_sigma_radial = nn.Parameter(torch.tensor([1.5, 0.0]))  # [a, b]
        self.raw_sigma_tangential = nn.Parameter(torch.tensor([1.5, 0.0]))  # [a, b]

        # Precompute local pixel grid (relative to shoebox center)
        cy_local = (H - 1) / 2.0
        cx_local = (W - 1) / 2.0
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32) - cy_local,
            torch.arange(W, dtype=torch.float32) - cx_local,
            indexing="ij",
        )
        self.register_buffer("pixel_y", yy.reshape(-1))  # (K,)
        self.register_buffer("pixel_x", xx.reshape(-1))  # (K,)

    def _compute_position_bias(self, xcal: Tensor, ycal: Tensor) -> Tensor:
        """Compute per-reflection anisotropic Gaussian bias.

        Args:
            xcal: (B,) detector x position
            ycal: (B,) detector y position

        Returns:
            bias: (B, K) log-profile bias
        """
        # Direction from beam center to reflection (radial direction)
        dx = xcal - self.beam_cx
        dy = ycal - self.beam_cy
        r = torch.sqrt(dx**2 + dy**2).clamp(min=1.0)

        # Unit radial vector
        ux = dx / r  # (B,)
        uy = dy / r  # (B,)

        # Radial and tangential sigmas (linear in r, softplus for positivity)
        sigma_r = F.softplus(self.raw_sigma_radial[0] + self.raw_sigma_radial[1] * r / 1000.0) + 0.5
        sigma_t = F.softplus(self.raw_sigma_tangential[0] + self.raw_sigma_tangential[1] * r / 1000.0) + 0.5

        # Project each pixel onto radial and tangential axes
        # pixel_x, pixel_y: (K,) local coords
        # ux, uy: (B,)
        # radial component: pixel dot radial_unit
        px = self.pixel_x.unsqueeze(0)  # (1, K)
        py = self.pixel_y.unsqueeze(0)  # (1, K)
        proj_r = px * ux.unsqueeze(1) + py * uy.unsqueeze(1)  # (B, K)
        proj_t = -px * uy.unsqueeze(1) + py * ux.unsqueeze(1)  # (B, K)

        # Anisotropic Gaussian in log space
        log_profile = -0.5 * (
            (proj_r / sigma_r.unsqueeze(1)) ** 2
            + (proj_t / sigma_t.unsqueeze(1)) ** 2
        )

        return log_profile

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        group_labels: Tensor | None = None,
        metadata: dict | None = None,
    ) -> ProfileSurrogateOutput:
        mu_h = self.mu_head(x)  # (B, d)
        std_h = F.softplus(self.std_head(x))  # (B, d)

        # Position-dependent bias
        if metadata is not None and "xyzcal.px.0" in metadata:
            xcal = metadata["xyzcal.px.0"]
            ycal = metadata["xyzcal.px.1"]
            b = self._compute_position_bias(xcal, ycal) + self.decoder.bias
        else:
            b = self.decoder.bias

        zp, mean_profile = _sample_and_decode(
            mu_h, std_h, self.decoder.weight, b, mc_samples
        )
        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            mu_h=mu_h,
            std_h=std_h,
        )
