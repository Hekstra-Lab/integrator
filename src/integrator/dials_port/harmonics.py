"""Real spherical harmonic basis for the DIALS absorption surface.

Ports `create_sph_harm_table` together with the crystal-frame geometry
that feeds it. The surface is evaluated on the average of the incident
and diffracted directions expressed in the crystal frame, after the
frame has been rotated to put the goniometer axis along `z`.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.special import sph_harm_y

from .io import Experiment
from .types import DTYPE, Reflections

__all__ = [
    "ALIGNMENT_AXIS",
    "SQRT2_DIALS",
    "align_axis_along_z",
    "calc_crystal_frame_vectors",
    "calc_theta_phi",
    "harmonic_table",
    "n_abs_params",
    "quantize_theta_phi",
    "scan_angles",
    "sph_harm_basis",
]

# DIALS truncates sqrt(2) to seven digits in scaling_helper.h:441. The
# exact value is used by default; the truncation shifts every m != 0
# column by 2.6e-10 relative, far below the tolerance we work to.
SQRT2_DIALS = 1.414213562

# scaler_factory.py hardcodes this rather than reading the goniometer,
# so the absorption frame is fixed regardless of the actual axis.
ALIGNMENT_AXIS = (1.0, 0.0, 0.0)


def n_abs_params(lmax: int) -> int:
    """Number of absorption surface parameters for a given `lmax`."""
    return lmax * (lmax + 2)


def scan_angles(expt: Experiment, image_index: np.ndarray) -> np.ndarray:
    """Rotation angle in radians at a fractional image index.

    Reproduces `Scan.get_angle_from_array_index`, which is affine in the
    zero-based array index rather than the image number.

    Args:
        expt: experiment carrying `osc_range` and `image_range`.
        image_index: `(N,)` zero-based array index, the z component of
            `xyzobs.px.value`.

    Returns:
        `(N,)` angles in radians.
    """
    first, last = expt.image_range
    start, end = expt.osc_range
    width = (end - start) / (last - first + 1)
    return np.deg2rad(
        start + np.asarray(image_index, dtype=np.float64) * width
    )


def calc_crystal_frame_vectors(
    s1: np.ndarray, angles: np.ndarray, expt: Experiment
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the beam directions into the crystal frame.

    Port of `dials.algorithms.scaling.scaling_utilities`.

    Args:
        s1: `(N, 3)` lab-frame diffracted beam vectors.
        angles: `(N,)` rotation angles in radians.
        expt: experiment carrying the goniometer and beam.

    Returns:
        `(s0c, s1c)`, each `(N, 3)`. Rows with a zero-length `s1` are
        left as zero, matching DIALS.
    """
    s1 = np.asarray(s1, dtype=np.float64)
    s0c = np.zeros_like(s1)
    s1c = np.zeros_like(s1)

    lengths = np.linalg.norm(s1, axis=1)
    non_zero = np.nonzero(lengths > 0.0)[0]
    s1n = s1[non_zero] / lengths[non_zero][:, np.newaxis]

    rot = Rotation.from_rotvec(
        angles[non_zero][:, np.newaxis] * expt.rotation_axis
    ).as_matrix()
    r = expt.setting_rotation @ rot @ expt.fixed_rotation
    r_inv = np.transpose(r, axes=(0, 2, 1))

    # expt.s0 is the sample-to-source direction, as stored in the .expt.
    s0c[non_zero] = r_inv @ expt.s0
    s1c[non_zero] = np.einsum("ijk,ik->ij", r_inv, s1n)
    return s0c, s1c


def align_axis_along_z(
    alignment_axis: tuple[float, float, float], vectors: np.ndarray
) -> np.ndarray:
    """Rotate `vectors` so that `alignment_axis` lies along `z`.

    Args:
        alignment_axis: axis to bring onto `z`.
        vectors: `(N, 3)` vectors to rotate.

    Returns:
        `(N, 3)` rotated vectors.
    """
    if alignment_axis == (0.0, 0.0, 1.0):
        return vectors
    ux, uy, uz = alignment_axis
    axis = np.array([uy, -ux, 0.0], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(uz / np.linalg.norm(alignment_axis))
    matrix = Rotation.from_rotvec(axis * angle).as_matrix()
    return vectors @ matrix.T


def calc_theta_phi(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polar coordinates in the physics convention.

    Args:
        xyz: `(N, 3)` vectors.

    Returns:
        `(theta, phi)`, with `theta` in `[0, pi]` measured from `z` and
        `phi` in `[0, 2pi)` measured from `x` in the `xy` plane.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    theta = np.arctan2(np.sqrt(y * y + x * x), z)
    phi = np.mod(np.arctan2(y, x) + 2.0 * np.pi, 2.0 * np.pi)
    return theta, phi


def quantize_theta_phi(
    theta: np.ndarray, phi: np.ndarray, points_per_degree: int
) -> tuple[np.ndarray, np.ndarray]:
    """Snap angles down to the DIALS harmonic lookup grid.

    For datasets over 100k reflections DIALS evaluates the surface from
    a precomputed table indexed by `calc_lookup_index`, which truncates
    both angles to the lower edge of a `1 / points_per_degree` degree
    bin. That is not a pure memory optimisation: it perturbs the
    absorption term by order `1e-4`, so it has to be reproduced to
    compare against a DIALS-written `inverse_scale_factor`.

    Args:
        theta: `(N,)` polar angles in radians.
        phi: `(N,)` azimuthal angles in radians.
        points_per_degree: lookup table resolution, 2 in DIALS.

    Returns:
        The truncated `(theta, phi)`.
    """
    # Bin indices are formed as in calc_lookup_index and the grid angle
    # is rebuilt as in create_sph_harm_lookup_table. Keeping that exact
    # order of operations matters: dividing by the step instead flips
    # the bin for the occasional angle that sits on a boundary.
    step = np.pi / (180.0 * points_per_degree)
    theta_index = np.floor(theta * points_per_degree * 180.0 / np.pi)
    phi_index = np.floor(phi * 180.0 * points_per_degree / np.pi)
    return theta_index * step, phi_index * step


def sph_harm_basis(
    theta_0: np.ndarray,
    phi_0: np.ndarray,
    theta_1: np.ndarray,
    phi_1: np.ndarray,
    lmax: int,
    sqrt2: float = float(np.sqrt(2.0)),
) -> np.ndarray:
    """Real spherical harmonic basis averaged over two directions.

    Columns run `l = 1..lmax` and, within each `l`, `m = -l..l`, which
    is the order DIALS stores absorption parameters in.

    Args:
        theta_0: `(N,)` polar angle of the first direction.
        phi_0: `(N,)` azimuthal angle of the first direction.
        theta_1: `(N,)` polar angle of the second direction.
        phi_1: `(N,)` azimuthal angle of the second direction.
        lmax: maximum degree.
        sqrt2: normalisation constant; pass `SQRT2_DIALS` to reproduce
            the truncated literal DIALS uses.

    Returns:
        `(N, lmax * (lmax + 2))` basis values.
    """
    columns = []
    for degree in range(1, lmax + 1):
        for order in range(-degree, degree + 1):
            y_0 = sph_harm_y(degree, abs(order), theta_0, phi_0)
            y_1 = sph_harm_y(degree, abs(order), theta_1, phi_1)
            if order == 0:
                columns.append(0.5 * (y_0.real + y_1.real))
                continue
            prefactor = sqrt2 * (-1.0) ** order / 2.0
            if order < 0:
                columns.append(prefactor * (y_0.imag + y_1.imag))
            else:
                columns.append(prefactor * (y_0.real + y_1.real))
    return np.stack(columns, axis=1)


def harmonic_table(
    refl: Reflections,
    expt: Experiment,
    lmax: int = 4,
    points_per_degree: int | None = None,
    sqrt2: float = float(np.sqrt(2.0)),
) -> torch.Tensor:
    """Absorption basis for a set of reflections.

    Args:
        refl: observations providing `s1` and `phi`.
        expt: experiment providing the goniometer, beam and scan.
        lmax: maximum spherical harmonic degree.
        points_per_degree: if given, quantize the angles onto the DIALS
            lookup grid before evaluating.
        sqrt2: normalisation constant, see `sph_harm_basis`.

    Returns:
        `(N, lmax * (lmax + 2))` basis tensor.
    """
    angles = scan_angles(expt, refl.phi.numpy())
    s0c, s1c = calc_crystal_frame_vectors(refl.s1.numpy(), angles, expt)
    s0c = align_axis_along_z(ALIGNMENT_AXIS, s0c)
    s1c = align_axis_along_z(ALIGNMENT_AXIS, s1c)

    theta_0, phi_0 = calc_theta_phi(s0c)
    theta_1, phi_1 = calc_theta_phi(s1c)
    if points_per_degree is not None:
        theta_0, phi_0 = quantize_theta_phi(theta_0, phi_0, points_per_degree)
        theta_1, phi_1 = quantize_theta_phi(theta_1, phi_1, points_per_degree)

    basis = sph_harm_basis(theta_0, phi_0, theta_1, phi_1, lmax, sqrt2)
    return torch.from_numpy(np.ascontiguousarray(basis)).to(DTYPE)
