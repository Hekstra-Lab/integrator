"""Predict Laue reflections from a known crystal orientation.

For a still in a polychromatic beam, every reciprocal-lattice point `q = A h`
within the resolution limit diffracts at the one wavelength that puts it on its
own Ewald sphere,

    lambda_h = -2 (q . s0_hat) / |q|^2,

and leaves along `s1 = q + s0_hat / lambda_h`. A reflection is recorded when
`lambda_h` falls inside the beam's band and the ray hits the panel.

Harmonics `n h` share the direction of `h` at `lambda_h / n`, so they land on the
same pixels. The predictor groups reflections by their primitive direction and
reports, for the lowest in-band order, how many orders are in band. A spot with
more than one is a sum of structure factors and does not factorize into a single
intensity times a profile.
"""

from dataclasses import dataclass
from math import gcd

import numpy as np


@dataclass
class LaueReflection:
    hkl: np.ndarray
    """Miller indices of the lowest in-band order, shape (N, 3), int."""
    wavelength: np.ndarray
    """Wavelength in Angstrom at which each reflection diffracts, shape (N,)."""
    d: np.ndarray
    """Resolution in Angstrom, shape (N,)."""
    s1: np.ndarray
    """Diffracted wave vector in inverse Angstrom, shape (N, 3)."""
    xy: np.ndarray
    """Predicted position in pixels (fast, slow), shape (N, 2)."""
    n_harmonics: np.ndarray
    """Number of orders of the same direction inside the band, shape (N,)."""


def _primitive(hkl: np.ndarray) -> np.ndarray:
    g = np.array([gcd(gcd(abs(int(h)), abs(int(k))), abs(int(l))) for h, k, l in hkl])
    return hkl // g[:, None]


def predict_laue(
    A: np.ndarray,
    s0_hat: np.ndarray,
    lambda_min: float,
    lambda_max: float,
    dmin: float,
    pix0: np.ndarray,
    fdet: np.ndarray,
    sdet: np.ndarray,
    pixel_size: float,
    n_fast: int,
    n_slow: int,
) -> LaueReflection:
    """All Laue reflections recorded on a flat panel.

    Args:
        A: Setting matrix with columns `a*, b*, c*` in the lab frame, inverse
            Angstrom; DIALS' `A = U B`.
        s0_hat: Unit direction of the incident beam's travel, shape (3,).
        lambda_min: Short-wavelength edge of the band, Angstrom.
        lambda_max: Long-wavelength edge of the band, Angstrom.
        dmin: Resolution limit, Angstrom.
        pix0: Lab vector from the crystal to the corner of pixel (0, 0).
        fdet: Unit fast-axis direction.
        sdet: Unit slow-axis direction.
        pixel_size: Pixel edge, same unit as `pix0`.
        n_fast: Panel width in pixels.
        n_slow: Panel height in pixels.
    """
    s0_hat = s0_hat / np.linalg.norm(s0_hat)
    hmax = np.floor(np.linalg.norm(np.linalg.inv(A), axis=1) / dmin).astype(int)
    grids = [np.arange(-m, m + 1) for m in hmax]
    hkl = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 3)
    hkl = hkl[np.abs(hkl).sum(1) > 0]
    q = hkl @ A.T
    q2 = (q**2).sum(1)
    keep = q2 <= 1.0 / dmin**2
    hkl, q, q2 = hkl[keep], q[keep], q2[keep]

    lam = -2.0 * (q @ s0_hat) / q2
    in_band = (lam >= lambda_min) & (lam <= lambda_max)
    hkl, q, q2, lam = hkl[in_band], q[in_band], q2[in_band], lam[in_band]

    s1 = q + s0_hat[None, :] / lam[:, None]
    normal = np.cross(fdet, sdet)
    dist = pix0 @ normal
    s1n = s1 @ normal
    hits = s1n * dist > 0
    t = dist / np.where(hits, s1n, 1.0)
    point = s1 * t[:, None] - pix0[None, :]
    fpx = (point @ fdet) / pixel_size
    spx = (point @ sdet) / pixel_size
    on_panel = hits & (fpx >= 0) & (fpx < n_fast) & (spx >= 0) & (spx < n_slow)

    hkl, lam, q2, s1 = hkl[on_panel], lam[on_panel], q2[on_panel], s1[on_panel]
    xy = np.stack([fpx[on_panel], spx[on_panel]], -1)

    prim = _primitive(hkl)
    _, group, counts = np.unique(prim, axis=0, return_inverse=True, return_counts=True)
    group = group.reshape(-1)
    order = np.abs(hkl).sum(1)
    lowest = np.zeros(len(hkl), dtype=bool)
    for g in np.unique(group):
        members = np.flatnonzero(group == g)
        lowest[members[np.argmin(order[members])]] = True

    return LaueReflection(
        hkl=hkl[lowest],
        wavelength=lam[lowest],
        d=1.0 / np.sqrt(q2[lowest]),
        s1=s1[lowest],
        xy=xy[lowest],
        n_harmonics=counts[group[lowest]],
    )
