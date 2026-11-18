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

`render_pixels` computes the expected photons from Bragg scattering alone.
`render_background` adds nonBragg's amorphous term, which is the same sum with the
lattice removed and a tabulated form factor in place of `F_cell F_latt`.

Not ported: phi rotation, detector thickness and absorption, the curved detector,
beam divergence, structure-factor interpolation, and noise. `noisify.c` applies
the noise to the float image, in this order (`noisify.c:579-617`): optional
multiplicative flicker and per-pixel calibration error, a Poisson draw per pixel,
then the point spread, then `gain`, an ADC offset and additive readout. The point
spread therefore smears photons that have already been drawn, which is why it
divides the per-pixel shot variance instead of leaving it Poisson. Nothing here
applies any of it.
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
    """Lab-frame position of every subpixel centre, shape (..., oversample^2, 3).

    `fpix` and `spix` share any leading shape, (P,) or (B, P), and may be
    fractional, which shifts the whole pixel. Subpixels are ordered slow-major, as
    nanoBragg's `subS` / `subF` loops are, so subpixel 0 is the one the C program
    uses for `polar` and `omega`.
    """
    sub = detector.pixel_size / oversample
    o = torch.arange(oversample, dtype=detector.pix0.dtype, device=detector.pix0.device)
    sub_s, sub_f = torch.meshgrid(o, o, indexing="ij")
    sub_s, sub_f = sub_s.reshape(-1), sub_f.reshape(-1)
    f_mm = sub * (fpix[..., None] * oversample + sub_f) + 0.5 * sub
    s_mm = sub * (spix[..., None] * oversample + sub_s) + 0.5 * sub
    return (
        f_mm[..., None] * detector.fdet
        + s_mm[..., None] * detector.sdet
        + detector.pix0
    )


def shape_transform(h: Tensor, crystal: Crystal) -> Tensor:
    """`F_latt` at fractional Miller indices `h`, shape (..., 3) -> (...)."""
    return lattice_transform(h, crystal.n_cells, crystal.recip, crystal.shape, crystal.fudge)


def lattice_transform(
    h: Tensor, n_cells: Tensor, recip: Tensor, shape: str, fudge: float = 1.0
) -> Tensor:
    """`F_latt` with crystal size and reciprocal cell broadcast against `h`.

    Args:
        h: Fractional Miller indices, shape (..., 3).
        n_cells: Cells along `a, b, c`, broadcastable to `h`.
        recip: Unrotated reciprocal vectors as rows, broadcastable to (..., 3, 3).
        shape: One of `SHAPES`.
        fudge: nanoBragg's `-fudge` factor on the shape-transform radius.
    """
    n = n_cells.to(h.dtype)
    if shape == "square":
        # nanoBragg skips an axis with N <= 1 rather than evaluating sincg there
        return torch.where(n > 1.0, sincg(math.pi * h, n), torch.ones_like(h)).prod(-1)

    dh = h - torch.ceil(h - 0.5)
    n_prod = n.prod(-1)
    if shape == "round":
        hrad_sq = (dh**2 * n**2).sum(-1)
        return n_prod * _ROUND_NORM * sinc3(math.pi * torch.sqrt(hrad_sq * fudge))

    dq = torch.einsum("...i,...ij->...j", dh, recip.to(h.dtype))
    rad_star_sq = (dq**2).sum(-1) * n_prod**2
    if shape == "gauss":
        return n_prod * torch.exp(-(rad_star_sq / 0.63 * fudge))
    if shape == "tophat":
        return n_prod * (rad_star_sq * fudge < 0.3969).to(h.dtype)
    raise ValueError(f"unknown crystal shape {shape!r}; expected one of {SHAPES}")


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


def render_shoeboxes(
    fpix: Tensor,
    spix: Tensor,
    detector: Detector,
    incident: Tensor,
    wavelength: Tensor,
    weight: Tensor,
    cells: Tensor,
    recip: Tensor,
    n_cells: Tensor,
    shape: str = "gauss",
    fudge: float = 1.0,
    target_hkl: Tensor | None = None,
    kahn: float = 0.0,
    polar_vector: Tensor | None = None,
    fluence: float = 1.259320152e29,
    oversample: int = 1,
    domain_chunk: int | None = None,
    checkpoint: bool = False,
) -> Tensor:
    """`render_pixels` for a batch of shoeboxes, each with its own crystal and sources.

    Every shoebox shares the incident direction but has its own wavelength grid
    (a spot only sees the band around the wavelength it diffracts at), its own
    crystal (the orientation of the image it came from), and optionally only its
    own reflection rendered, so that a neighbour's lattice point inside the box is
    not counted as this spot's profile.

    Args:
        fpix, spix: Fast and slow pixel indices, shape (B, P); fractional values
            shift the pixel.
        detector: Panel geometry, shared.
        incident: Unit direction of beam travel, shape (3,).
        wavelength: Wavelengths in Angstrom, shape (B, L).
        weight: Spectral weight of each wavelength, shape (B, L). Zero drops it.
        cells: Real-space cells of each mosaic domain, rows `a, b, c`, shape (B, D, 3, 3).
        recip: Unrotated reciprocal vectors as rows, shape (B, 3, 3).
        n_cells: Cells along `a, b, c`, shape (3,) or (B, 3).
        shape: One of `SHAPES`.
        fudge: Shape-transform radius factor.
        target_hkl: Miller indices to keep, shape (B, 3); None renders every lattice point.
        kahn: Kahn polarization factor.
        polar_vector: E-vector direction, shape (3,).
        fluence: Photons per m^2.
        oversample: Subpixels per pixel edge.
        domain_chunk: Sum this many mosaic domains at a time.
        checkpoint: Recompute each domain chunk in the backward pass instead of
            storing it, trading time for memory.

    Returns:
        Photons per pixel, shape (B, P). With equal weights and one shoebox this
        equals `render_pixels` on the same pixels.
    """
    dtype = detector.pix0.dtype
    pos = subpixel_positions(fpix.to(dtype), spix.to(dtype), detector, oversample)
    airpath = pos.norm(dim=-1)
    diffracted = pos / airpath[..., None]
    inc = incident.to(dtype)
    lam = wavelength.to(dtype)
    w = weight.to(dtype)

    # (B, P, O, L, 3) scattering vectors in inverse Angstrom
    scatter = (diffracted[:, :, :, None, :] - inc) / lam[:, None, None, :, None]
    n = n_cells.to(dtype)
    n = n.reshape(-1, 1, 1, 1, 1, 3) if n.dim() == 2 else n
    rcp = recip.to(dtype).reshape(-1, 1, 1, 1, 1, 3, 3)
    tgt = None if target_hkl is None else target_hkl.to(dtype).reshape(-1, 1, 1, 1, 1, 3)

    def chunk_sum(scatter_, cells_):
        h = torch.einsum("bpolj,bdij->bpoldi", scatter_, cells_)
        f_latt = lattice_transform(h, n, rcp, shape, fudge)
        term = f_latt**2
        if tgt is not None:
            term = term * (torch.ceil(h - 0.5) == tgt).all(-1).to(dtype)
        return term.sum(-1)

    n_dom = cells.shape[1]
    step = domain_chunk or n_dom
    total = 0.0
    for start in range(0, n_dom, step):
        cells_c = cells[:, start : start + step].to(dtype)
        if checkpoint and torch.is_grad_enabled():
            part = torch.utils.checkpoint.checkpoint(chunk_sum, scatter, cells_c, use_reentrant=False)
        else:
            part = chunk_sum(scatter, cells_c)
        total = total + part

    n_src = lam.shape[-1]
    w_norm = w * n_src / w.sum(-1, keepdim=True).clamp_min(1e-30)
    intensity = (total * w_norm[:, None, None, :]).sum(dim=(2, 3))
    steps = oversample**2 * n_src * n_dom

    omega = detector.pixel_size**2 / airpath[..., 0] ** 2 * detector.close_distance / airpath[..., 0]
    polar = polarization_factor(kahn, inc.expand_as(diffracted[..., 0, :]), diffracted[..., 0, :], polar_vector)
    return R_E_SQR * fluence * intensity / steps * polar * omega


def render_shoeboxes_continuous(
    fpix: Tensor,
    spix: Tensor,
    detector: Detector,
    incident: Tensor,
    spectrum: Callable[[Tensor], Tensor],
    domain_rot: Tensor,
    q_hkl: Tensor,
    n_cells: Tensor,
    kahn: float = 0.0,
    polar_vector: Tensor | None = None,
    fluence: float = 1.259320152e29,
    oversample: int = 1,
) -> Tensor:
    """Gauss-shape render of one reflection per shoebox, integrated over wavelength exactly.

    For the gauss crystal shape nanoBragg's lattice factor is
    `F_latt^2 = N^2 exp(-alpha |S - R q_hkl|^2)` with `N = Na Nb Nc` and
    `alpha = 2 N^2 / 0.63`; the distance is the same whether the domain rotation is
    applied to the cell or to the lattice point. On a pixel the scattering vector is
    `S = k u` with `k = 1 / lambda` and `u = s_hat_1 - s_hat_0` fixed, so `F_latt^2` is
    a Gaussian in `k` along a line, and

        int S(lambda) F_latt^2 dlambda
          = N^2 S(lambda*) / k*^2 * sqrt(pi / (alpha |u|^2)) * exp(-alpha d_perp^2),

    with `k* = u . q / |u|^2` the closest approach and `d_perp` the distance of the
    lattice point from the line. The spectrum is evaluated at `lambda* = 1 / k*`,
    which is exact when it is smooth over the lattice point's crossing, whose width
    in `k` is `1 / sqrt(2 alpha |u|^2)`. Against a 30001-point grid it agrees to
    1e-4 at 1000 cells under a 20%-wide spectrum, is 6% off when the spectrum is
    2% wide, and fails for a 200-cell crystal under that narrow spectrum; a flat
    band is exact except within a crossing-width of its edges. This replaces the
    wavelength grid of nanoBragg (and of `render_shoeboxes`) by its limit, so there
    is nothing to undersample.

    Args:
        fpix, spix: Pixel indices, shape (B, P).
        detector: Panel geometry.
        incident: Unit direction of beam travel, shape (3,).
        spectrum: Spectral density as a function of wavelength in Angstrom; zero
            outside the band.
        domain_rot: Mosaic domain rotations, shape (D, 3, 3).
        q_hkl: Each shoebox's reciprocal-lattice vector `A h` in the lab frame,
            inverse Angstrom, shape (B, 3).
        n_cells: Cells along `a, b, c`, shape (3,).

    Returns:
        `r_e^2 fluence polar omega / D * sum_domains int S F_latt^2 dlambda`, shape (B, P).
        Equal to `render_shoeboxes` in the limit of a fine uniform wavelength grid
        with weights `S`, times the integral of `S` over that grid.
    """
    dtype = detector.pix0.dtype
    pos = subpixel_positions(fpix.to(dtype), spix.to(dtype), detector, oversample)
    airpath = pos.norm(dim=-1)
    diffracted = pos / airpath[..., None]
    inc = incident.to(dtype)
    u = diffracted - inc  # (B, P, O, 3)
    u2 = (u**2).sum(-1)

    q_dom = torch.einsum("dij,bj->bdi", domain_rot.to(dtype), q_hkl.to(dtype))  # (B, D, 3)
    uq = torch.einsum("bpoi,bdi->bpod", u, q_dom)
    q2 = (q_dom**2).sum(-1)[:, None, None, :]
    k_star = uq / u2[..., None]
    d_perp2 = (q2 - uq**2 / u2[..., None]).clamp_min(0.0)

    n_prod = n_cells.to(dtype).prod()
    alpha = 2.0 * n_prod**2 / 0.63
    lam_star = 1.0 / k_star.clamp_min(1e-12)
    s_val = spectrum(lam_star) * (k_star > 0).to(dtype)
    integral = (
        n_prod**2 * s_val * lam_star**2
        * torch.sqrt(math.pi / (alpha * u2[..., None]))
        * torch.exp(-alpha * d_perp2)
    )
    intensity = integral.mean(dim=(-1, -2))  # mean over domains and subpixels

    omega = detector.pixel_size**2 / airpath[..., 0] ** 2 * detector.close_distance / airpath[..., 0]
    polar = polarization_factor(kahn, inc.expand_as(diffracted[..., 0, :]), diffracted[..., 0, :], polar_vector)
    return R_E_SQR * fluence * intensity * polar * omega


AVOGADRO = 6.02214179e23
"""Avogadro's number, the value hard-coded in `nanoBragg.c`."""


def amorphous_molecules(
    thickness_mm: float,
    beam_size_mm: float = 0.1,
    density_gcm3: float = 1.0,
    molecular_weight_da: float = 18.0,
) -> float:
    """Molecules of amorphous material the beam crosses.

    `nonBragg` takes the illuminated volume to be the sample thickness times a
    square beam cross-section, so a 0.5 mm path of water in a 0.1 mm beam holds
    1.673e17 molecules, which is the figure it prints for those settings.

    Args:
        thickness_mm: Path length through the material.
        beam_size_mm: Edge of the square beam profile.
        density_gcm3: Density of the material; 1.0 for water.
        molecular_weight_da: Mass per molecule; 18 for water.
    """
    volume_cm3 = thickness_mm * beam_size_mm**2 * 1e-3
    return volume_cm3 * density_gcm3 * AVOGADRO / molecular_weight_da


def interpolate_stol(stol: Tensor, curve_stol: Tensor, curve_f: Tensor) -> Tensor:
    """`F_bg` at each `sin(theta) / lambda`, linear between tabulated points.

    `nonBragg` passes the nearest four points to its `polint`, which is a
    four-point Lagrange polynomial, not the cubic spline its comment calls it. On
    the dense `water.stol`, whose points are 2e-4 inverse Angstrom apart, that
    polynomial and this line agree far more closely than either agrees with the
    data; on a sparse table, such as the fourteen-point default carried in
    `simtbx/nanoBragg/utils.py`, they would not, and that table should be
    resampled before it is used here.

    Values beyond either end are held flat, as `nonBragg` holds them.

    Args:
        stol: Where to evaluate, in inverse Angstrom, any shape.
        curve_stol: Tabulated `sin(theta) / lambda`, ascending, shape (N,).
        curve_f: Tabulated `F_bg` in electrons, shape (N,).
    """
    idx = torch.searchsorted(curve_stol, stol.reshape(-1).contiguous())
    idx = idx.clamp(1, curve_stol.numel() - 1)
    x0, x1 = curve_stol[idx - 1], curve_stol[idx]
    y0, y1 = curve_f[idx - 1], curve_f[idx]
    t = ((stol.reshape(-1) - x0) / (x1 - x0).clamp_min(1e-30)).clamp(0.0, 1.0)
    return (y0 + t * (y1 - y0)).reshape(stol.shape)


def render_background(
    fpix: Tensor,
    spix: Tensor,
    detector: Detector,
    beam: Beam,
    curve_stol: Tensor,
    curve_f: Tensor,
    molecules: float,
    oversample: int = 1,
    point_pixel: bool = False,
) -> Tensor:
    """Expected photons per pixel from amorphous scattering.

    This is `nonBragg`'s per-pixel sum, which is the Bragg sum with the lattice
    removed and a tabulated amorphous form factor in place of `F_cell F_latt`:

        I = r_e^2 * fluence * molecules / steps
            * sum_{subpixels, sources} F_bg(stol)^2 * polar * omega

    with `stol = |s - s_0| / 2` and `steps = oversample^2 * n_sources`. Every
    factor outside `F_bg` is one the Bragg path already computes, which is why
    this belongs in the kernel rather than in a second simulator.

    `polar` and `omega` are evaluated once per pixel, at its first subpixel and
    first source, matching `render_pixels` and the C program's default.

    Source weights are normalized to a mean of one, as `render_pixels` normalizes
    them, so that the Bragg and background arrays are on one scale. `nonBragg`
    instead uses the weights raw and still divides by `steps = oversample^2 *
    n_sources`, so for a spectrum whose weights do not average to one its absolute
    background differs from this by `sum(weight) / n_sources`. The two agree when
    the weights are equal, which is the case the C output was checked against.

    Not included, both of which `nonBragg` has: the detector-thickness capture
    fraction, which is 1 unless `-detector_thick` is set, and the sign flip that
    lets a negative tabulated `F_bg` produce negative intensity, which `water.stol`
    never triggers.

    The crystal's own scattering is not here: add this to `render_pixels` or
    `render_shoeboxes` to get the full image, and keep the two arrays apart when
    what is wanted is a known background.

    Args:
        fpix: Fast-axis pixel index of each pixel, shape (P,).
        spix: Slow-axis pixel index of each pixel, shape (P,).
        detector: Panel geometry.
        beam: Sources, fluence and polarization.
        curve_stol: Tabulated `sin(theta) / lambda`, ascending, shape (N,).
        curve_f: Tabulated `F_bg` in electrons, shape (N,).
        molecules: Molecules in the beam path, from `amorphous_molecules`.
        oversample: Subpixels per pixel edge.
        point_pixel: Use `1 / airpath^2` for the solid angle, dropping obliquity.

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

    # (P, O, S, 3) scattering vectors in inverse Angstrom; stol is half the length
    scatter = (
        diffracted[:, :, None, :] - incident[None, None, :, :]
    ) / lam[None, None, :, None]
    f_bg = interpolate_stol(0.5 * scatter.norm(dim=-1), curve_stol.to(dtype), curve_f.to(dtype))
    intensity = (f_bg**2 * weight).sum(dim=(1, 2))
    steps = oversample**2 * n_src

    if point_pixel:
        omega = 1.0 / airpath[:, 0] ** 2
    else:
        omega = (
            detector.pixel_size**2
            / airpath[:, 0] ** 2
            * detector.close_distance
            / airpath[:, 0]
        )
    if beam.nopolar:
        polar = torch.ones_like(omega)
    else:
        polar = polarization_factor(
            beam.kahn,
            incident[0].expand_as(diffracted[:, 0]),
            diffracted[:, 0],
            beam.polar_vector,
        )
    return R_E_SQR * beam.fluence * molecules * intensity / steps * polar * omega


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
