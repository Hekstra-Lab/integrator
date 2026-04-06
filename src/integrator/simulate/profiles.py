"""Sample 2D Gaussian profiles on a pixel grid.

Each profile is a normalized probability map on an H x W grid, generated
by rendering a 2D Gaussian with random center, width, rotation, and
optional streak (elongation) and skew (asymmetry).
"""

import math

import torch
from torch import Tensor


def sample_profiles(
    N: int,
    H: int = 21,
    W: int = 21,
    *,
    center_std: float = 1.5,
    log_sigma_mean: float = 0.9,
    log_sigma_std: float = 0.3,
    skew_prob: float = 0.3,
    max_skew: float = 1.5,
    streak_prob: float = 0.2,
    max_elongation: float = 5.0,
) -> Tensor:
    """Sample diverse 2D Gaussian profiles on an H x W pixel grid.

    Each profile is rendered as a (possibly elongated / skewed) 2D
    Gaussian and normalized to sum to 1.

    Parameters
    ----------
    N : int
        Number of profiles to generate.
    H, W : int
        Grid dimensions (default 21 x 21).
    center_std : float
        Std of the center jitter around the grid midpoint.
    log_sigma_mean, log_sigma_std : float
        Mean and std of the log-normal width distribution.
        exp(0.9) ≈ 2.5 pixels.
    skew_prob : float
        Probability of adding asymmetric sigmoid skew.
    max_skew : float
        Maximum skew strength.
    streak_prob : float
        Probability of elongating one axis (streak / Laue-like).
    max_elongation : float
        Maximum sigma ratio for streaks.

    Returns
    -------
    profiles : Tensor, shape (N, H*W)
        Each row sums to 1.
    """
    cy_base = (H - 1) / 2.0
    cx_base = (W - 1) / 2.0

    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )

    profiles = torch.zeros(N, H * W)

    for i in range(N):
        # Random center
        cx = cx_base + torch.randn(1).item() * center_std
        cy = cy_base + torch.randn(1).item() * center_std

        # Random widths (log-normal)
        sigma_x = math.exp(
            torch.distributions.Normal(log_sigma_mean, log_sigma_std)
            .sample()
            .item()
        )
        sigma_y = math.exp(
            torch.distributions.Normal(log_sigma_mean, log_sigma_std)
            .sample()
            .item()
        )

        # Streak: occasionally elongate one axis
        if torch.rand(1).item() < streak_prob:
            elongation = 1.0 + torch.rand(1).item() * (max_elongation - 1.0)
            if torch.rand(1).item() < 0.5:
                sigma_x *= elongation
            else:
                sigma_y *= elongation

        # Random rotation
        theta = torch.rand(1).item() * math.pi
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        dx = xx - cx
        dy = yy - cy
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t

        # Base Gaussian
        profile = torch.exp(
            -0.5 * (x_rot**2 / sigma_x**2 + y_rot**2 / sigma_y**2)
        )

        # Asymmetric skew
        if torch.rand(1).item() < skew_prob:
            skew_strength = torch.rand(1).item() * max_skew
            skew_angle = torch.rand(1).item() * 2 * math.pi
            skew_coord = (
                dx * math.cos(skew_angle) + dy * math.sin(skew_angle)
            )
            profile = profile * torch.sigmoid(skew_strength * skew_coord)

        # Normalize
        profile = profile / profile.sum().clamp(min=1e-10)
        profiles[i] = profile.reshape(-1)

    return profiles
