"""Polychromatic Wilson loss with a learned detector map as the mean of the background prior."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integrator.model.loss.polychromatic_wilson_loss import PolychromaticWilsonLoss


class PolychromaticWilsonBgMapLoss(PolychromaticWilsonLoss):
    """`polychromatic_wilson` with the background prior's mean learned over the detector.

    The prior on each shoebox's background rate b (photons per pixel) is

        b ~ Gamma(kappa, kappa / b0(rho, phi, image))       mean b0, sd b0 / sqrt(kappa)
        log b0 = sum_{m <= M} sum_{n} c_mn T_m(x) F_n(phi) + a_image

    with x = 2 (rho - r_min) / (r_max - r_min) - 1 clamped to [-1, 1], rho the
    shoebox centroid's distance from the beam centre in pixels, T_m Chebyshev
    polynomials of degree up to `bg_radial_degree`, F_n the Fourier terms
    1, cos(n phi), sin(n phi) up to `bg_azimuth_order`, and a_image a per-image
    offset with its mean removed (so it cannot trade with c_00). kappa =
    softplus(raw) is learned. c, a and kappa receive gradient only through the
    background KL, so they are fitted to the posteriors q(b) of all shoeboxes
    together: the prior's mean and spread are learned, each shoebox's background
    stays its own posterior. c_00 starts at log `bg_init`, every other
    coefficient at 0, kappa at `bg_kappa_init`.

    The fixed Gamma(`bg_concentration`, `bg_rate`) of the parent is not used.
    Needs metadata `xyzcal.px.0`, `xyzcal.px.1` and, when `bg_per_image`,
    `image_num` (1-based, up to `bg_n_images`).
    """

    def __init__(
        self,
        *,
        bg_beam_center: list[float],
        bg_r_min: float,
        bg_r_max: float,
        bg_radial_degree: int = 8,
        bg_azimuth_order: int = 4,
        bg_per_image: bool = True,
        bg_n_images: int = 1,
        bg_init: float = 1.0,
        bg_kappa_init: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if bg_r_max <= bg_r_min:
            raise ValueError("bg_r_max must exceed bg_r_min")
        if bg_init <= 0 or bg_kappa_init <= 0:
            raise ValueError("bg_init and bg_kappa_init must be positive")
        self.register_buffer("bg_cx", torch.tensor(float(bg_beam_center[0])))
        self.register_buffer("bg_cy", torch.tensor(float(bg_beam_center[1])))
        self.bg_r_min, self.bg_r_max = float(bg_r_min), float(bg_r_max)
        self.bg_radial_degree = int(bg_radial_degree)
        self.bg_azimuth_order = int(bg_azimuth_order)
        n_az = 1 + 2 * self.bg_azimuth_order
        c = torch.zeros(self.bg_radial_degree + 1, n_az)
        c[0, 0] = math.log(float(bg_init))
        self.bg_coef = nn.Parameter(c)
        self.bg_per_image = bool(bg_per_image)
        self.bg_image_offset = nn.Parameter(torch.zeros(int(bg_n_images)), requires_grad=self.bg_per_image)
        self.bg_kappa_raw = nn.Parameter(torch.tensor(math.log(math.expm1(float(bg_kappa_init)))))

    @property
    def bg_kappa(self) -> Tensor:
        return F.softplus(self.bg_kappa_raw) + 1e-4

    def bg_log_mean(self, metadata: dict, device) -> Tensor:
        """log b0 per reflection, shape (B,)."""
        x = metadata["xyzcal.px.0"].reshape(-1).to(device).float() - self.bg_cx
        y = metadata["xyzcal.px.1"].reshape(-1).to(device).float() - self.bg_cy
        rho, phi = torch.hypot(x, y), torch.atan2(y, x)
        u = (2 * (rho - self.bg_r_min) / (self.bg_r_max - self.bg_r_min) - 1).clamp(-1, 1)
        T = [torch.ones_like(u), u]
        for _ in range(2, self.bg_radial_degree + 1):
            T.append(2 * u * T[-1] - T[-2])
        T = torch.stack(T[: self.bg_radial_degree + 1], -1)  # (B, M+1)
        Fz = [torch.ones_like(phi)]
        for n in range(1, self.bg_azimuth_order + 1):
            Fz += [torch.cos(n * phi), torch.sin(n * phi)]
        Fz = torch.stack(Fz, -1)  # (B, 2N+1)
        log_b0 = torch.einsum("bm,mn,bn->b", T, self.bg_coef, Fz)
        if self.bg_per_image:
            img = metadata["image_num"].reshape(-1).to(device).long() - 1
            a = self.bg_image_offset - self.bg_image_offset.mean()
            log_b0 = log_b0 + a[img.clamp(0, a.shape[0] - 1)]
        return log_b0

    def _background_prior(self, group_labels, metadata, device) -> tuple[Tensor, Tensor]:
        kappa = self.bg_kappa
        b0 = self.bg_log_mean(metadata, device).clamp(-20.0, 20.0).exp()
        return kappa.expand_as(b0), kappa / b0
