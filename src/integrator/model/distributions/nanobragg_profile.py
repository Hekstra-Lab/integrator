"""Profile rendered by nanoBragg on each shoebox's own pixels.

The profile of a reflection is the normalized `F_latt^2 * polar * omega` that
nanoBragg sums over subpixels, wavelengths and mosaic domains, evaluated at the
orientation of the image the shoebox came from. Because `F_cell` is constant over
one monochromatic-in-effect Laue spot, it cancels in the normalization, so the
profile depends on the crystal and beam but not on the structure factor, and
`rate = I * profile + bg` is exactly nanoBragg's pixel model.

What is learned is the physics shared by the whole dataset: crystal size along
`a, b, c`, mosaic spread, and the spectrum as a log-density on knots across the
band. Bright reflections pin these down and weak ones inherit them.

A reflection may differ from the physics by a bounded residual: a centroid shift
in pixels with a fixed `N(0, centroid_prior_px^2)` prior and no learned matrix in
front of it, so the prior scale is a physical length and has no gauge. The
encoder emits the shift's Normal posterior; the loss forms its KL against the
prior from `loc`, `scale` and `prior_scale`, as for any latent profile. One
reparameterized draw of the shift is rendered per step and shared by the Monte
Carlo samples of the intensity and background, which keeps the expected
log-likelihood unbiased while rendering once.

Only the reflection's own lattice point is rendered; a neighbour inside the box
is background to this profile.

The dataset must carry a `physics` block in `dataset.yaml` and the per-image
setting matrices it names (written by `scripts/physics/make_sim_shoeboxes.py`).
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integrator.model.distributions.profile_surrogates import ProfileSurrogateOutput
from integrator.physics.nanobragg import (
    Detector,
    mosaic_domains,
    render_shoeboxes,
    render_shoeboxes_continuous,
    rotation_matrices,
)

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class NanoBraggProfileSurrogate(nn.Module):
    """nanoBragg-rendered profile with learned global physics and a bounded centroid residual.

    Args:
        input_dim: Width of the profile encoder's output.
        output_dim: Pixels per shoebox.
        profile_shape: Shoebox grid as `(D, H, W)`; D must be 1.
        data_dir: Shoebox directory holding `dataset.yaml` and the crystal models.
        shape: nanoBragg crystal shape, one of square, round, gauss, tophat.
        init_n_cells: Starting crystal size in unit cells, one value or one per axis.
        init_mosaic_deg: Starting mosaic spread; defaults to the mean in the crystal models.
        n_domains: Mosaic domains drawn once and kept fixed.
        wavelength_integration: "continuous" integrates over wavelength in closed
            form (gauss shape only; exact when the spectrum is smooth over a lattice
            point's crossing), "grid" sums a per-shoebox wavelength grid as nanoBragg
            does. Defaults to continuous for the gauss shape.
        n_lambda: Wavelengths per shoebox on the grid.
        lambda_window_sd: Half-width of each shoebox's wavelength grid, in units of
            the spread the current crystal size and mosaicity give it.
        spectrum_knots: Knots of the learned log-spectrum across the band.
        residual: "centroid" for a per-reflection pixel shift, "none" for pure physics.
        centroid_prior_px: Prior sd of the shift, in pixels.
        learn_globals: Fit crystal size, mosaicity and spectrum through the ELBO.
        domain_chunk: Mosaic domains rendered at a time.
        checkpoint: Recompute domain chunks in the backward pass to save memory.
        dtype: "float32" or "float64" for the render.
        seed: Seed for the mosaic domain draws.
    """

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 441,
        profile_shape: tuple[int, int, int] | list[int] | None = None,
        data_dir: str | None = None,
        shape: str = "gauss",
        init_n_cells: float | list[float] = 10.0,
        init_mosaic_deg: float | None = None,
        n_domains: int = 50,
        wavelength_integration: str | None = None,
        n_lambda: int = 15,
        lambda_window_sd: float = 3.0,
        spectrum_knots: int = 8,
        residual: str = "centroid",
        centroid_prior_px: float = 0.5,
        learn_globals: bool = True,
        domain_chunk: int = 10,
        checkpoint: bool = True,
        dtype: str = "float32",
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        if data_dir is None:
            raise ValueError("NanoBraggProfileSurrogate needs `data_dir` (injected from data_loader.args)")
        if residual not in ("centroid", "none"):
            raise ValueError(f"residual must be 'centroid' or 'none'; got {residual!r}")
        from integrator.io import read_dataset_spec

        spec = read_dataset_spec(data_dir)
        if spec is None or "physics" not in spec:
            raise ValueError(f"{data_dir}/dataset.yaml has no `physics` block; cut it with make_sim_shoeboxes.py")
        physics = spec["physics"]

        if profile_shape is None:
            side = int(round(math.sqrt(output_dim)))
            profile_shape = (1, side, side)
        d, h, w = (int(v) for v in profile_shape)
        if d != 1 or h * w != output_dim:
            raise ValueError(f"profile_shape {profile_shape} must be (1, H, W) with H*W = {output_dim}")
        self.h, self.w = h, w
        self.output_dim = int(output_dim)
        self.shape = shape
        if wavelength_integration is None:
            wavelength_integration = "continuous" if shape == "gauss" else "grid"
        if wavelength_integration not in ("continuous", "grid"):
            raise ValueError(f"wavelength_integration must be 'continuous' or 'grid'; got {wavelength_integration!r}")
        if wavelength_integration == "continuous" and shape != "gauss":
            raise ValueError("continuous wavelength integration needs the gauss shape")
        self.wavelength_integration = wavelength_integration
        self.residual = residual
        self.prior_scale = float(centroid_prior_px)
        self.n_lambda = int(n_lambda)
        self.lambda_window_sd = float(lambda_window_sd)
        self.domain_chunk = int(domain_chunk)
        self.checkpoint = bool(checkpoint)
        self.render_dtype = _DTYPES[dtype]

        det = physics["detector"]
        fdet = torch.tensor(det["fast"], dtype=torch.float64)
        sdet = torch.tensor(det["slow"], dtype=torch.float64)
        pix0 = torch.tensor(det["pix0_mm"], dtype=torch.float64)
        odet = torch.linalg.cross(fdet, sdet)
        odet = odet if torch.dot(pix0, odet) > 0 else -odet
        self.pixel_size = float(det["pixel_size_mm"])
        for name, v in (("pix0", pix0), ("fdet", fdet), ("sdet", sdet), ("odet", odet)):
            self.register_buffer(f"_{name}", v, persistent=False)

        beam = physics["beam"]
        s0 = torch.tensor(beam["s0_hat"], dtype=torch.float64)
        normal = torch.tensor(beam.get("polarization_normal", [0.0, 1.0, 0.0]), dtype=torch.float64)
        self.register_buffer("_s0", s0, persistent=False)
        self.register_buffer("_e_vec", torch.linalg.cross(normal, s0), persistent=False)
        self.kahn = float(beam.get("polarization_fraction", 0.0))
        self.lambda_min = float(beam["lambda_min"])
        self.lambda_max = float(beam["lambda_max"])

        z = np.load(Path(data_dir) / physics["orientation_file"])
        order = np.argsort(z["image_num"])
        self.register_buffer("_image_num", torch.as_tensor(z["image_num"][order]).long(), persistent=False)
        self.register_buffer("_A", torch.as_tensor(z["A"][order], dtype=torch.float64), persistent=False)
        if init_mosaic_deg is None:
            init_mosaic_deg = float(np.mean(z["mos_spread_deg"]))

        n0 = torch.as_tensor(init_n_cells, dtype=torch.float32).expand(3).clone()
        self.log_n_cells = nn.Parameter(n0.log(), requires_grad=learn_globals)
        self.log_mosaic = nn.Parameter(
            torch.tensor(math.log(math.radians(max(init_mosaic_deg, 1e-4)))), requires_grad=learn_globals
        )
        self.log_spectrum = nn.Parameter(torch.zeros(int(spectrum_knots)), requires_grad=learn_globals)

        axes, frac = mosaic_domains(int(n_domains), torch.Generator().manual_seed(seed), dtype=torch.float64)
        self.register_buffer("_axes", axes, persistent=False)
        self.register_buffer("_frac", frac, persistent=False)

        if residual == "centroid":
            self.head = nn.Linear(input_dim, 4)
            nn.init.zeros_(self.head.weight)
            with torch.no_grad():
                self.head.bias.copy_(torch.tensor([0.0, 0.0, 1.0, 1.0]) * _inv_softplus(self.prior_scale))
        else:
            self.head = None

    @property
    def n_cells(self) -> Tensor:
        return self.log_n_cells.exp()

    @property
    def mosaic_rad(self) -> Tensor:
        return self.log_mosaic.exp()

    def _detector(self, dtype) -> Detector:
        return Detector(
            pix0=self._pix0.to(dtype), fdet=self._fdet.to(dtype), sdet=self._sdet.to(dtype),
            odet=self._odet.to(dtype), pixel_size=self.pixel_size,
        )

    def _image_index(self, image_num: Tensor) -> Tensor:
        num = image_num.reshape(-1).round().long().to(self._image_num.device)
        pos = torch.searchsorted(self._image_num, num).clamp(max=len(self._image_num) - 1)
        if not torch.equal(self._image_num[pos], num):
            raise KeyError("batch image numbers missing from the crystal models")
        return pos

    def spectrum(self, lam: Tensor) -> Tensor:
        """Spectral density at `lam`: piecewise-linear log-density on the knots, zero out of band."""
        k = self.log_spectrum.shape[0]
        t = (lam - self.lambda_min) / (self.lambda_max - self.lambda_min) * (k - 1)
        i0 = t.floor().clamp(0, k - 2).long()
        frac = (t - i0).clamp(0.0, 1.0)
        log_s = self.log_spectrum.to(lam.dtype)
        val = torch.exp(log_s[i0] * (1 - frac) + log_s[i0 + 1] * frac)
        inside = (lam >= self.lambda_min) & (lam <= self.lambda_max)
        return val * inside.to(lam.dtype)

    def _wavelengths(self, metadata: dict, dtype) -> tuple[Tensor, Tensor]:
        """Each shoebox's wavelength grid, centred on its own, and the spectral weights."""
        lam0 = metadata["wavelength"].reshape(-1, 1).to(dtype)
        d_sp = metadata["d"].reshape(-1, 1).to(dtype)
        with torch.no_grad():
            sin_t = (lam0 / (2 * d_sp)).clamp(1e-6, 1.0)
            tan_t = sin_t / (1 - sin_t**2).clamp_min(1e-12).sqrt()
            n = self.n_cells.detach().to(dtype)
            if self.shape == "gauss":
                sigma_q = 0.4 / n.prod()
            else:
                cell_len = torch.linalg.det(torch.linalg.inv(self._A[0])).abs().to(dtype) ** (1 / 3)
                sigma_q = 1.0 / (n.mean() * cell_len)
            rel = torch.sqrt((sigma_q * d_sp) ** 2 + (self.mosaic_rad.detach().to(dtype) / tan_t) ** 2)
            half = (self.lambda_window_sd * rel).clamp(1e-4, 0.05)
        t = torch.linspace(-1.0, 1.0, self.n_lambda, dtype=dtype, device=lam0.device)
        lam = lam0 * (1 + half * t)
        return lam, self.spectrum(lam) + 1e-12

    def _render(self, metadata: dict, shift: Tensor | None) -> Tensor:
        dtype = self.render_dtype
        b = metadata["bbox.0"].shape[0]
        idx = self._image_index(metadata["image_num"])
        A = self._A[idx].to(dtype)
        cell = torch.linalg.inv(A)
        rot = rotation_matrices(self._axes.to(dtype), self._frac.to(dtype) * self.mosaic_rad.to(dtype))
        cells = torch.einsum("dij,bkj->bdki", rot, cell)

        x0 = metadata["bbox.0"].reshape(-1, 1).to(dtype)
        y0 = metadata["bbox.2"].reshape(-1, 1).to(dtype)
        ys, xs = torch.meshgrid(
            torch.arange(self.h, dtype=dtype, device=x0.device),
            torch.arange(self.w, dtype=dtype, device=x0.device),
            indexing="ij",
        )
        fpix = x0 + xs.reshape(1, -1)
        spix = y0 + ys.reshape(1, -1)
        if shift is not None:
            # a positive shift moves the spot, so the pixels are sampled from the
            # opposite side of the rendered pattern
            fpix = fpix - shift[:, :1].to(dtype)
            spix = spix - shift[:, 1:].to(dtype)

        hkl = torch.stack([metadata[k].reshape(-1) for k in ("H", "K", "L")], -1).round().to(dtype)
        if self.wavelength_integration == "continuous":
            q_hkl = torch.einsum("bij,bj->bi", A, hkl)
            out = render_shoeboxes_continuous(
                fpix, spix, self._detector(dtype), self._s0.to(dtype), self.spectrum, rot, q_hkl,
                self.n_cells.to(dtype), kahn=self.kahn, polar_vector=self._e_vec.to(dtype),
            )
            assert out.shape == (b, self.output_dim)
            return out

        lam, weight = self._wavelengths(metadata, dtype)
        out = render_shoeboxes(
            fpix, spix, self._detector(dtype), self._s0.to(dtype), lam, weight, cells,
            A.transpose(-1, -2), self.n_cells.to(dtype), shape=self.shape, target_hkl=hkl,
            kahn=self.kahn, polar_vector=self._e_vec.to(dtype),
            domain_chunk=self.domain_chunk, checkpoint=self.checkpoint,
        )
        assert out.shape == (b, self.output_dim)
        return out

    @staticmethod
    def _normalize(raw: Tensor, mask: Tensor) -> Tensor:
        raw = raw * mask
        total = raw.sum(-1, keepdim=True)
        uniform = mask / mask.sum(-1, keepdim=True).clamp_min(1.0)
        # a spot the window misses entirely falls back to uniform rather than 0/0
        return torch.where(total > 0, raw / total.clamp_min(1e-30), uniform)

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        metadata: dict | None = None,
        mask: Tensor | None = None,
        **kwargs,
    ) -> ProfileSurrogateOutput:
        if metadata is None:
            raise ValueError("NanoBraggProfileSurrogate needs batch metadata")
        b = x.shape[0]
        mask = (
            torch.ones(b, self.output_dim, device=x.device) if mask is None else mask.reshape(b, -1).to(x.dtype)
        )

        if self.head is not None:
            out = self.head(x)
            loc = out[:, :2]
            scale = F.softplus(out[:, 2:]) + 1e-4
            shift = loc + scale * torch.randn_like(loc) if self.training else loc
            prior_scale = self.prior_scale
        else:
            # no residual: a posterior equal to its prior, so the loss's KL is zero
            loc = x.new_zeros(b, 1)
            scale = x.new_ones(b, 1)
            shift = None
            prior_scale = 1.0

        profile = self._normalize(self._render(metadata, shift).to(x.dtype), mask)

        return ProfileSurrogateOutput(
            zp=profile.unsqueeze(0).expand(mc_samples, -1, -1),
            # in eval the shift is the posterior mean, so this is the mean profile;
            # in training it is the one draw the step renders
            mean_profile=profile,
            loc=loc,
            scale=scale,
            prior_scale=prior_scale,
        )
