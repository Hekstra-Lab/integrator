"""Self-contained 2D Gaussian spot-profile sampler for the jungfrau_sim studies.

A local copy of the elliptical-Gaussian profile model so this directory does not depend on
`integrator.simulate`. `h ~ N(0, I_5)` maps to physical spot parameters (centre jitter, two
widths, a rotation), then to a normalized profile over an H x W grid. Kept behaviourally
identical to the original so previously generated datasets reproduce under the same seed.
"""

from __future__ import annotations

import torch
from torch import Tensor


def h_to_physical_params(
    h: Tensor,
    center_base: float = 10.0,
    center_scale: float = 1.5,
    log_sigma_base: float = 0.7,
    width_scale: float = 0.4,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Map latent `h` (..., 5) to `(cx, cy, sigma1, sigma2, theta)`.

    Centres jitter about `center_base`; widths are log-normal about `exp(log_sigma_base)`
    (~2 px); theta is in `(0, pi)`.
    """
    cx = center_base + h[..., 0] * center_scale
    cy = center_base + h[..., 1] * center_scale
    sigma1 = (log_sigma_base + h[..., 2] * width_scale).exp()
    sigma2 = (log_sigma_base + h[..., 3] * width_scale).exp()
    theta = torch.pi * torch.sigmoid(h[..., 4])
    return cx, cy, sigma1, sigma2, theta


def physical_params_to_profile(
    cx: Tensor,
    cy: Tensor,
    sigma1: Tensor,
    sigma2: Tensor,
    theta: Tensor,
    H: int = 21,
    W: int = 21,
) -> Tensor:
    """Render normalized 2D elliptical-Gaussian profiles, shape `(..., H*W)` summing to 1."""
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=cx.dtype),
        torch.arange(W, dtype=cx.dtype),
        indexing="ij",
    )
    batch_dims = cx.shape
    for _ in range(len(batch_dims)):
        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)

    cx = cx[..., None, None]
    cy = cy[..., None, None]
    sigma1 = sigma1[..., None, None]
    sigma2 = sigma2[..., None, None]
    theta = theta[..., None, None]

    dx = xx - cx
    dy = yy - cy
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    profile = torch.exp(-0.5 * (x_rot**2 / sigma1**2 + y_rot**2 / sigma2**2))
    profile = profile / profile.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-10)
    return profile.reshape(*batch_dims, H * W)


def h_to_profile(h: Tensor, H: int = 21, W: int = 21, **param_kwargs) -> Tensor:
    """Full pipeline: `h` (..., 5) -> physical params -> normalized profile (..., H*W)."""
    cx, cy, sigma1, sigma2, theta = h_to_physical_params(h, **param_kwargs)
    return physical_params_to_profile(cx, cy, sigma1, sigma2, theta, H, W)


def sample_profiles(
    N: int,
    H: int = 21,
    W: int = 21,
    *,
    center_base: float | None = None,
    center_scale: float = 1.5,
    log_sigma_base: float = 0.7,
    width_scale: float = 0.4,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample `N` normalized profiles from `h ~ N(0, I_5)`, shape `(N, H*W)`."""
    if center_base is None:
        center_base = (H - 1) / 2.0
    h = torch.randn(N, 5, generator=generator)
    return h_to_profile(
        h, H=H, W=W, center_base=center_base, center_scale=center_scale,
        log_sigma_base=log_sigma_base, width_scale=width_scale,
    )
