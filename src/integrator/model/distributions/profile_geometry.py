"""Detector-geometry features for profile surrogates.

A profile surrogate that only sees a reflection's own pixels has to fall back to
one dataset-global profile when the counts stop constraining it. On Laue stills
that fallback is wrong by construction: spots streak along the radial direction,
so averaging over azimuth gives a round blob. Roughly half the variance in PDZ2
spot elongation and second moments is predictable from `(x, y, lambda)` alone,
and a third of the orientation, so it is worth conditioning on.

These helpers turn the metadata every data module already carries into a small,
scale-free feature vector, plus the analytic azimuth used to steer a profile
along the local streak direction.
"""

import math

import torch
from torch import Tensor

N_GEOM_FEATURES = 9


def detector_polar(
    metadata: dict,
    beam_center: tuple[float, float] | list[float],
) -> tuple[Tensor, Tensor, Tensor]:
    """Radius, normalized radius and azimuth of each reflection about the beam centre.

    The azimuth is the direction of increasing radius on the detector, which for
    a still is the direction a spot smears as its wavelength changes. Measured
    PDZ2 streaks sit 1.7 degrees from it, so it is taken as exact rather than learned.

    Returns:
        r: Radius in pixels, shape (B,).
        r_norm: Radius divided by the beam-centre radius, shape (B,).
        phi: Azimuth in radians, shape (B,).
    """
    x = metadata["xyzcal.px.0"].reshape(-1).float()
    y = metadata["xyzcal.px.1"].reshape(-1).float()
    x0, y0 = float(beam_center[0]), float(beam_center[1])
    dx, dy = x - x0, y - y0
    r = torch.hypot(dx, dy)
    r_scale = math.hypot(x0, y0) or 1.0
    return r, r / r_scale, torch.atan2(dy, dx)


def geometry_features(
    metadata: dict,
    beam_center: tuple[float, float] | list[float],
    lambda_min: float | None = None,
    lambda_max: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Scale-free geometry features and the analytic azimuth.

    Azimuth enters as `cos`/`sin` of `phi` and `2 phi` rather than as `phi`, so the
    features are continuous across the branch cut and can express a two-fold
    (axis-like) dependence without a discontinuity.

    Args:
        metadata: Batch metadata; needs `xyzcal.px.0`, `xyzcal.px.1` and, when a
            wavelength is present, `wavelength`.
        beam_center: Detector beam centre in pixels.
        lambda_min: Low end of the spectrum, used to map wavelength to [-1, 1].
        lambda_max: High end of the spectrum.

    Returns:
        features: Shape (B, `N_GEOM_FEATURES`).
        phi: Azimuth in radians, shape (B,).
    """
    _, r_norm, phi = detector_polar(metadata, beam_center)

    lam = metadata.get("wavelength")
    if lam is None:
        lam_n = torch.zeros_like(r_norm)
    else:
        lam = lam.reshape(-1).float()
        if lambda_min is not None and lambda_max is not None and lambda_max > lambda_min:
            mid = 0.5 * (lambda_max + lambda_min)
            half = 0.5 * (lambda_max - lambda_min)
            lam_n = (lam - mid) / half
        else:
            lam_n = lam - 1.0

    features = torch.stack(
        [
            r_norm,
            r_norm**2,
            torch.cos(phi),
            torch.sin(phi),
            torch.cos(2 * phi),
            torch.sin(2 * phi),
            lam_n,
            lam_n**2,
            r_norm * lam_n,
        ],
        dim=-1,
    )
    return features, phi
