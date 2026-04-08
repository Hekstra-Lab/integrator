"""Sample 2D Gaussian profiles on a pixel grid.

Each profile is generated from a 5D latent h ~ N(0, I_5) mapped to
physical parameters (cx, cy, sigma1, sigma2, theta) via the same
transform used by PhysicalGaussianProfileSurrogate:

    cx     = center_base + h[0] * center_scale
    cy     = center_base + h[1] * center_scale
    sigma1 = exp(log_sigma_base + h[2] * width_scale)
    sigma2 = exp(log_sigma_base + h[3] * width_scale)
    theta  = pi * sigmoid(h[4])

This ensures the simulated profiles live in exactly the same family
as the surrogate posterior, which is required for valid SBC.
"""

import torch
from torch import Tensor


def sample_profiles(
    N: int,
    H: int = 21,
    W: int = 21,
    *,
    center_base: float | None = None,
    center_scale: float = 1.5,
    log_sigma_base: float = 0.7,
    width_scale: float = 0.4,
) -> Tensor:
    """Sample 2D Gaussian profiles from h ~ N(0, I_5).

    Parameters
    ----------
    N : int
        Number of profiles to generate.
    H, W : int
        Grid dimensions (default 21 x 21).
    center_base : float, optional
        Center of the grid in pixels. Defaults to (H-1)/2.
    center_scale : float
        Std of center jitter in pixels.
    log_sigma_base : float
        Base log-width. exp(0.7) ~ 2.0 pixels.
    width_scale : float
        Std of log-width variation.

    Returns
    -------
    profiles : Tensor, shape (N, H*W)
        Each row sums to 1.
    """
    if center_base is None:
        center_base = (H - 1) / 2.0

    h = torch.randn(N, 5)
    profiles = h_to_profile(
        h,
        H=H,
        W=W,
        center_base=center_base,
        center_scale=center_scale,
        log_sigma_base=log_sigma_base,
        width_scale=width_scale,
    )
    return profiles


def h_to_physical_params(
    h: Tensor,
    center_base: float = 10.0,
    center_scale: float = 1.5,
    log_sigma_base: float = 0.7,
    width_scale: float = 0.4,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Map latent h to physical profile parameters.

    Returns
    -------
    cx, cy : (...,) center coordinates in pixel space
    sigma1, sigma2 : (...,) widths in pixel space (positive)
    theta : (...,) rotation angle in (0, pi)
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
    """Render normalized 2D Gaussian profiles from physical parameters.

    Parameters
    ----------
    cx, cy, sigma1, sigma2, theta : (...,) batch of parameters

    Returns
    -------
    profiles : (..., H*W) normalized profiles
    """
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )

    # Expand grid for broadcasting: (H, W) -> (1..., H, W)
    batch_dims = cx.shape
    for _ in range(len(batch_dims)):
        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)

    # Expand params: (...,) -> (..., 1, 1)
    cx = cx[..., None, None]
    cy = cy[..., None, None]
    sigma1 = sigma1[..., None, None]
    sigma2 = sigma2[..., None, None]
    theta = theta[..., None, None]

    # Rotated coordinates
    dx = xx - cx
    dy = yy - cy
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    # Gaussian
    profile = torch.exp(-0.5 * (x_rot**2 / sigma1**2 + y_rot**2 / sigma2**2))

    # Normalize to sum to 1
    profile = profile / profile.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-10)

    # Flatten spatial dims
    return profile.reshape(*batch_dims, H * W)


def h_to_profile(
    h: Tensor,
    H: int = 21,
    W: int = 21,
    **param_kwargs,
) -> Tensor:
    """Full pipeline: h -> physical params -> normalized profile.

    Parameters
    ----------
    h : (..., 5) latent vector

    Returns
    -------
    profiles : (..., H*W) normalized profiles
    """
    cx, cy, sigma1, sigma2, theta = h_to_physical_params(h, **param_kwargs)
    return physical_params_to_profile(cx, cy, sigma1, sigma2, theta, H, W)
