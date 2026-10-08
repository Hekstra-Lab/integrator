"""Differentiable nanoBragg pixel kernel.

A PyTorch port of the per-pixel sum in James Holton's `nanoBragg.c`, restricted
to the pixels it is asked for, so a shoebox costs its own pixels and nothing else.
For each pixel the C program sums

    I_pix = r_e^2 * fluence / steps * polar * omega
            * sum_{subpixels, sources, mosaic domains} F_cell^2 * F_latt^2

over `steps = n_sub^2 * n_sources * n_domains` terms, with `F_latt` the Fourier
transform of the crystal shape evaluated at the fractional Miller index
`h = a . (s_hat_1 - s_hat_0) / lambda`. Here the sum is one tensor contraction,
and autograd replaces the hand-written derivatives diffBragg carries.

Conventions follow the C program, so that a port can be checked against it pixel
for pixel:

- `polar` and `omega` are evaluated once per pixel, at its first subpixel and first
  source, as `nanoBragg.c` does unless `-oversample_polar` / `-oversample_omega`.
- `F_latt` for the gauss, round and tophat shapes uses the distance of `h` from the
  nearest integer measured with the *unrotated* reciprocal vectors, as the C does.
- Source weights are normalized to a mean of one. `nanoBragg.c` ignores the weights
  in a `-sourcefile` entirely, which equals this when the weights are equal.

Lengths are in Angstrom for the crystal and wavelength. Detector lengths can be in
any unit as long as they are consistent with each other; only directions and the
ratio in the solid angle enter.

Not ported: phi rotation, detector thickness and absorption, the curved detector,
water background, structure-factor interpolation, noise. The kernel computes the
expected photons from Bragg scattering only.
"""

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

R_E_SQR = 7.94079248018965e-30
"""Thomson cross-section r_e^2 in m^2, the value hard-coded in nanoBragg.c."""

SHAPES = ("square", "round", "gauss", "tophat")

# Volume ratio between a cube and its inscribed sphere, used by nanoBragg's round shape.
_ROUND_NORM = 0.723601254558268


@dataclass
class Detector:
    """A flat detector panel in the lab frame.

    Attributes:
        pix0: Vector from the crystal to the corner of pixel (0, 0), shape (3,).
        fdet: Unit vector of increasing fast-axis pixel index, shape (3,).
        sdet: Unit vector of increasing slow-axis pixel index, shape (3,).
        odet: Unit vector normal to the panel, pointing away from the crystal, shape (3,).
        pixel_size: Pixel edge length, same unit as `pix0`.
    """

    pix0: Tensor
    fdet: Tensor
    sdet: Tensor
    odet: Tensor
    pixel_size: float

    @property
    def close_distance(self) -> Tensor:
        return torch.dot(self.pix0, self.odet)


@dataclass
class Beam:
    """A set of incident rays.

    Attributes:
        incident: Unit direction of travel of each source, shape (S, 3).
        wavelength: Wavelength of each source in Angstrom, shape (S,).
        weight: Relative weight of each source, shape (S,), or None for equal weights.
        fluence: Integrated fluence in photons per m^2.
        kahn: Kahn polarization factor; 0 is an unpolarized beam.
        polar_vector: Unit E-vector direction of a polarized beam, shape (3,).
        nopolar: Skip the polarization factor entirely.
    """

    incident: Tensor
    wavelength: Tensor
    weight: Tensor | None = None
    fluence: float = 1.259320152e29
    kahn: float = 0.0
    polar_vector: Tensor | None = None
    nopolar: bool = False


@dataclass
class Crystal:
    """A mosaic crystal.

    Attributes:
        cells: Real-space cell vectors of each mosaic domain as rows `a, b, c`, in
            Angstrom, shape (D, 3, 3).
        recip: Unrotated reciprocal cell vectors as rows `a*, b*, c*`, in inverse
            Angstrom, shape (3, 3). Only the gauss, round and tophat shapes use it.
        n_cells: Number of unit cells along `a, b, c`, shape (3,). Need not be integer.
        shape: One of `SHAPES`.
        fudge: nanoBragg's `-fudge` factor on the shape-transform radius.
    """

    cells: Tensor
    recip: Tensor
    n_cells: Tensor
    shape: str = "square"
    fudge: float = 1.0


def sincg(x: Tensor, n: Tensor) -> Tensor:
    """`sin(n x) / sin(x)`, the Fourier transform of a row of `n` unit cells.

    nanoBragg returns `n` only at `x == 0`. At other multiples of pi the ratio is
    0/0 in floating point, so it is replaced there by its limit `n cos(n x)/cos(x)`.
    """
    s = torch.sin(x)
    near = s.abs() < 1e-12
    safe = torch.where(near, torch.ones_like(s), s)
    regular = torch.sin(n * x) / safe
    limit = n * torch.cos(n * x) / torch.cos(x)
    return torch.where(near, limit, regular)


def sinc3(x: Tensor) -> Tensor:
    """Fourier transform of a sphere, `3 (sin x / x - cos x) / x^2`, equal to 1 at 0."""
    near = x.abs() < 1e-6
    safe = torch.where(near, torch.ones_like(x), x)
    regular = 3.0 * (torch.sin(safe) / safe - torch.cos(safe)) / safe**2
    return torch.where(near, torch.ones_like(x) - x**2 / 10.0, regular)


def _unitize(v: Tensor) -> Tensor:
    """Unit vector along the last axis; a zero vector stays zero, as in nanoBragg.c."""
    mag = v.norm(dim=-1, keepdim=True)
    return torch.where(mag > 0, v / torch.where(mag > 0, mag, torch.ones_like(mag)), torch.zeros_like(v))


def polarization_factor(
    kahn: float, incident: Tensor, diffracted: Tensor, axis: Tensor | None
) -> Tensor:
    """Kahn polarization factor, `0.5 (1 + cos^2 2theta - kahn cos(2 psi) sin^2 2theta)`.

    Args:
        kahn: Kahn factor; 0 gives the unpolarized `0.5 (1 + cos^2 2theta)`.
        incident: Unit incident directions, shape (..., 3).
        diffracted: Unit diffracted directions, shape (..., 3).
        axis: E-vector direction, shape (3,). Ignored when `kahn == 0`.
    """
    cos2t = (incident * diffracted).sum(-1)
    sin2t_sq = 1.0 - cos2t**2
    if kahn == 0.0:
        return 0.5 * (1.0 + cos2t**2)
    if axis is None:
        raise ValueError("a polarized beam (kahn != 0) needs `polar_vector`")
    axis = _unitize(axis)
    b_in = _unitize(torch.cross(axis.expand_as(incident), incident, dim=-1))
    e_in = _unitize(torch.cross(incident, b_in, dim=-1))
    psi = -torch.atan2((diffracted * b_in).sum(-1), (diffracted * e_in).sum(-1))
    return 0.5 * (1.0 + cos2t**2 - kahn * torch.cos(2.0 * psi) * sin2t_sq)


def subpixel_positions(
    fpix: Tensor, spix: Tensor, detector: Detector, oversample: int
) -> Tensor:
    """Lab-frame position of every subpixel centre, shape (P, oversample^2, 3).

    Subpixels are ordered slow-major, as nanoBragg's `subS` / `subF` loops are, so
    subpixel 0 is the one the C program uses for `polar` and `omega`.
    """
    sub = detector.pixel_size / oversample
    o = torch.arange(oversample, dtype=detector.pix0.dtype, device=detector.pix0.device)
    sub_s, sub_f = torch.meshgrid(o, o, indexing="ij")
    sub_s, sub_f = sub_s.reshape(-1), sub_f.reshape(-1)
    f_mm = sub * (fpix[:, None] * oversample + sub_f[None, :]) + 0.5 * sub
    s_mm = sub * (spix[:, None] * oversample + sub_s[None, :]) + 0.5 * sub
    return (
        f_mm[..., None] * detector.fdet
        + s_mm[..., None] * detector.sdet
        + detector.pix0
    )


def shape_transform(h: Tensor, crystal: Crystal) -> Tensor:
    """`F_latt` at fractional Miller indices `h`, shape (..., 3) -> (...)."""
    n = crystal.n_cells.to(h.dtype)
    if crystal.shape == "square":
        f = torch.ones_like(h[..., 0])
        for i in range(3):
            if float(n[i]) > 1.0:
                f = f * sincg(math.pi * h[..., i], n[i])
        return f

    dh = h - torch.ceil(h - 0.5)
    n_prod = n.prod()
    if crystal.shape == "round":
        hrad_sq = (dh**2 * n**2).sum(-1)
        return n_prod * _ROUND_NORM * sinc3(math.pi * torch.sqrt(hrad_sq * crystal.fudge))

    dq = dh @ crystal.recip.to(h.dtype)
    rad_star_sq = (dq**2).sum(-1) * n_prod**2
    if crystal.shape == "gauss":
        return n_prod * torch.exp(-(rad_star_sq / 0.63 * crystal.fudge))
    if crystal.shape == "tophat":
        return n_prod * (rad_star_sq * crystal.fudge < 0.3969).to(h.dtype)
    raise ValueError(f"unknown crystal shape {crystal.shape!r}; expected one of {SHAPES}")


def render_pixels(
    fpix: Tensor,
    spix: Tensor,
    detector: Detector,
    beam: Beam,
    crystal: Crystal,
    oversample: int = 1,
    f_cell: Callable[[Tensor], Tensor] | float = 1.0,
    point_pixel: bool = False,
    domain_chunk: int | None = None,
) -> Tensor:
    """Expected Bragg photons in each requested pixel.

    Args:
        fpix: Fast-axis pixel index of each pixel, shape (P,).
        spix: Slow-axis pixel index of each pixel, shape (P,).
        detector: Panel geometry.
        beam: Sources, fluence and polarization.
        crystal: Domains, shape transform and size.
        oversample: Subpixels per pixel edge.
        f_cell: Structure-factor amplitude, either a constant (1.0 gives the
            profile shape times partiality, independent of F) or a function of the
            nearest integer Miller indices, shape (..., 3) -> (...).
        point_pixel: Use `1 / airpath^2` for the solid angle, dropping obliquity.
        domain_chunk: Sum mosaic domains this many at a time to bound memory.

    Returns:
        Photons per pixel, shape (P,), on the scale of nanoBragg's float image.
    """
    dtype = detector.pix0.dtype
    pos = subpixel_positions(fpix.to(dtype), spix.to(dtype), detector, oversample)
    airpath = pos.norm(dim=-1)
    diffracted = pos / airpath[..., None]

    incident = beam.incident.to(dtype)
    lam = beam.wavelength.to(dtype)
    n_src = incident.shape[0]
    weight = (
        torch.ones(n_src, dtype=dtype, device=incident.device)
        if beam.weight is None
        else beam.weight.to(dtype) * n_src / beam.weight.to(dtype).sum()
    )

    # (P, O, S, 3) scattering vectors in inverse Angstrom.
    scatter = (diffracted[:, :, None, :] - incident[None, None, :, :]) / lam[None, None, :, None]

    cells = crystal.cells.to(dtype)
    n_dom = cells.shape[0]
    chunk = domain_chunk or n_dom
    total = torch.zeros(scatter.shape[:3], dtype=dtype, device=scatter.device)
    for start in range(0, n_dom, chunk):
        cells_c = cells[start : start + chunk]
        h = torch.einsum("posj,dij->posdi", scatter, cells_c)
        f_latt = shape_transform(h, crystal)
        if callable(f_cell):
            fc = f_cell(torch.ceil(h - 0.5))
            term = fc**2 * f_latt**2
        else:
            term = float(f_cell) ** 2 * f_latt**2
        total = total + term.sum(-1)

    intensity = (total * weight).sum(dim=(1, 2))
    steps = oversample**2 * n_src * n_dom

    if point_pixel:
        omega = 1.0 / airpath[:, 0] ** 2
    else:
        omega = detector.pixel_size**2 / airpath[:, 0] ** 2 * detector.close_distance / airpath[:, 0]
    if beam.nopolar:
        polar = torch.ones_like(omega)
    else:
        polar = polarization_factor(
            beam.kahn, incident[0].expand_as(diffracted[:, 0]), diffracted[:, 0], beam.polar_vector
        )
    return R_E_SQR * beam.fluence * intensity / steps * polar * omega


def mosaic_domains(
    n_domains: int, generator: torch.Generator | None = None, dtype=torch.float64
) -> tuple[Tensor, Tensor]:
    """Fixed random draws for a differentiable mosaic spread.

    Follows `mosaic_rotation_umat` in nanoBragg.c: a rotation about a uniformly
    random axis by `spread * (1 - r^2)^(1/3)` with `r` uniform on [-1, 1], and the
    first domain forced to the identity. Returning the axes and the angle fractions
    separately keeps the domains differentiable in the spread.

    Returns:
        axes: Unit rotation axes, shape (D, 3).
        fractions: Rotation angle as a fraction of the spread, shape (D,).
    """
    r = 2.0 * torch.rand(n_domains, 3, generator=generator, dtype=dtype) - 1.0
    xy = torch.sqrt(1.0 - r[:, 1] ** 2)
    axes = torch.stack(
        [xy * torch.sin(math.pi * r[:, 0]), xy * torch.cos(math.pi * r[:, 0]), r[:, 1]], dim=-1
    )
    fractions = (1.0 - r[:, 2] ** 2) ** (1.0 / 3.0)
    fractions[0] = 0.0
    return axes, fractions


def rotation_matrices(axes: Tensor, angles: Tensor) -> Tensor:
    """Rodrigues rotation matrices for unit `axes` (D, 3) and `angles` (D,), shape (D, 3, 3)."""
    x, y, z = axes.unbind(-1)
    c, s = torch.cos(angles), torch.sin(angles)
    t = 1.0 - c
    return torch.stack(
        [
            torch.stack([c + t * x * x, t * x * y - s * z, t * x * z + s * y], -1),
            torch.stack([t * x * y + s * z, c + t * y * y, t * y * z - s * x], -1),
            torch.stack([t * x * z - s * y, t * y * z + s * x, c + t * z * z], -1),
        ],
        -2,
    )


def mosaic_cells(cell: Tensor, axes: Tensor, fractions: Tensor, spread: Tensor) -> Tensor:
    """Rotate one cell (rows `a, b, c`, shape (3, 3)) into every mosaic domain, shape (D, 3, 3)."""
    rot = rotation_matrices(axes, fractions * spread)
    return torch.einsum("dij,kj->dki", rot, cell)
