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

    Args:
        N: Number of profiles to generate.
        H: Grid height (default 21).
        W: Grid width (default 21).
        center_base: Center of the grid in pixels. Defaults to (H-1)/2.
        center_scale: Std of center jitter in pixels.
        log_sigma_base: Base log-width. exp(0.7) ~ 2.0 pixels.
        width_scale: Std of log-width variation.

    Returns:
        Profiles tensor of shape (N, H*W). Each row sums to 1.
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

    Returns:
        Tuple of (cx, cy, sigma1, sigma2, theta) where cx, cy are center
        coordinates in pixel space, sigma1, sigma2 are widths (positive),
        and theta is the rotation angle in (0, pi). All have shape (...,).
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

    Args:
        cx: Center x-coordinates, shape (...,).
        cy: Center y-coordinates, shape (...,).
        sigma1: First width parameter, shape (...,).
        sigma2: Second width parameter, shape (...,).
        theta: Rotation angle, shape (...,).
        H: Grid height (default 21).
        W: Grid width (default 21).

    Returns:
        Normalized profiles tensor of shape (..., H*W).
    """
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
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

    profile = profile / profile.sum(dim=(-2, -1), keepdim=True).clamp(
        min=1e-10
    )

    return profile.reshape(*batch_dims, H * W)


def h_to_profile(
    h: Tensor,
    H: int = 21,
    W: int = 21,
    **param_kwargs,
) -> Tensor:
    """Full pipeline: h -> physical params -> normalized profile.

    Args:
        h: Latent vector of shape (..., 5).
        H: Grid height (default 21).
        W: Grid width (default 21).
        **param_kwargs: Extra kwargs passed to :func:`h_to_physical_params`.

    Returns:
        Normalized profiles tensor of shape (..., H*W).
    """
    cx, cy, sigma1, sigma2, theta = h_to_physical_params(h, **param_kwargs)
    return physical_params_to_profile(cx, cy, sigma1, sigma2, theta, H, W)
