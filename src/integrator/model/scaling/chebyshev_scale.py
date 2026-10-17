import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integrator.model.loss.learned_spectrum import ChebyshevSpectrum


class ChebyshevScale(nn.Module):
    """Smooth learnable scale factor s(frame) via Chebyshev polynomials."""

    def __init__(
        self,
        degree: int = 5,
        frame_min: float = 0.0,
        frame_max: float = 1000.0,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.degree = degree

        frame_mid = (frame_min + frame_max) / 2.0
        frame_half = (frame_max - frame_min) / 2.0

        self.register_buffer("frame_mid", torch.tensor(frame_mid))
        self.register_buffer("frame_half", torch.tensor(frame_half))

        c = torch.zeros(degree + 1)
        c[0] = math.log(math.expm1(init_scale))
        self.c = nn.Parameter(c)

    def forward(self, frame: Tensor) -> Tensor:
        """Evaluate scale at given frame positions.

        Args:
            frame: (B,) frame numbers (e.g. xyzcal.px.2).

        Returns:
            scale: (B,) positive scale factors.
        """
        t = ((frame - self.frame_mid) / self.frame_half).clamp(-1.0, 1.0)
        phi = torch.stack(ChebyshevSpectrum._chebyshev(t, self.degree), dim=-1)
        return F.softplus(phi @ self.c)


class SpatialChebyshevScale(nn.Module):
    """Scale factor s(frame, r) as tensor product of Chebyshev polynomials.

    Extends ChebyshevScale with spatial dependence on detector radius
    to capture absorption and detector efficiency variations.

    s(frame, r) = softplus(sum_jk c_jk T_j(frame_norm) T_k(r_norm))

    where r = distance from beam center.  The tensor product basis
    captures frame-only effects (beam decay), radius-only effects
    (absorption), and cross-terms (e.g. radiation damage that varies
    with scattering angle).
    """

    def __init__(
        self,
        degree_frame: int = 5,
        degree_radius: int = 5,
        frame_min: float = 0.0,
        frame_max: float = 1000.0,
        beam_center: list[float] | None = None,
        r_min: float = 0.0,
        r_max: float = 1500.0,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.degree_frame = degree_frame
        self.degree_radius = degree_radius

        frame_mid = (frame_min + frame_max) / 2.0
        frame_half = (frame_max - frame_min) / 2.0
        self.register_buffer("frame_mid", torch.tensor(frame_mid))
        self.register_buffer("frame_half", torch.tensor(frame_half))

        cx, cy = beam_center or [0.0, 0.0]
        r_mid = (r_min + r_max) / 2.0
        r_half = (r_max - r_min) / 2.0
        self.register_buffer("beam_cx", torch.tensor(cx))
        self.register_buffer("beam_cy", torch.tensor(cy))
        self.register_buffer("r_mid", torch.tensor(r_mid))
        self.register_buffer("r_half", torch.tensor(r_half))

        c = torch.zeros(degree_frame + 1, degree_radius + 1)
        c[0, 0] = math.log(math.expm1(init_scale))
        self.c = nn.Parameter(c)

    def forward(self, frame: Tensor, x: Tensor, y: Tensor) -> Tensor:
        """Evaluate scale at given frame and detector positions.

        Args:
            frame: (B,) frame numbers.
            x: (B,) detector x positions (xyzcal.px.0).
            y: (B,) detector y positions (xyzcal.px.1).

        Returns:
            scale: (B,) positive scale factors.
        """
        t = ((frame - self.frame_mid) / self.frame_half).clamp(-1.0, 1.0)
        r = torch.sqrt((x - self.beam_cx).pow(2) + (y - self.beam_cy).pow(2))
        rn = ((r - self.r_mid) / self.r_half).clamp(-1.0, 1.0)

        phi_t = torch.stack(
            ChebyshevSpectrum._chebyshev(t, self.degree_frame), dim=-1
        )
        phi_r = torch.stack(
            ChebyshevSpectrum._chebyshev(rn, self.degree_radius), dim=-1
        )

        out = (phi_t @ self.c * phi_r).sum(-1)
        return F.softplus(out)


class MLPScale(nn.Module):
    """
    Takes per-observation features (frame, detector x/y, LP, d-spacing)
    and outputs a positive scale factor.

    rate = scale_mlp(features) * F^2 * profile + bg

    Features are normalized to [-1, 1] or [0, 1]
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        frame_min: float = 0.0,
        frame_max: float = 1000.0,
        beam_center: list[float] | None = None,
        r_max: float = 1500.0,
        d_min: float = 1.0,
        d_max: float = 60.0,
        head_init_std: float = 0.0,
        n_abs_sh: int = 0,
        absorption_even_only: bool = True,
        exclude_hkl_constant: bool = False,
        n_extra: int = 0,
        extra_loc: list[float] | None = None,
        extra_scale: list[float] | None = None,
    ):
        super().__init__()

        # d-spacing is a property of the Miller index, so every symmetry mate
        # of a reflection shares it, and on a flat normal detector the radius
        # is a function of d. A scale built from them is therefore constant
        # within each merging group, and `s -> c(d) s`, `I_h -> I_h / c(d)`
        # leaves the likelihood AND the merging consistency untouched: it is
        # the one direction merging cannot see. The Wilson prior is then the
        # only thing fixing it, so the merged intensities come out on the
        # prior's line and the data's own falloff is absorbed into the scale.
        # Dropping both inputs leaves the resolution dependence nowhere to go
        # but into I_h, where it belongs.
        self.exclude_hkl_constant = bool(exclude_hkl_constant)

        frame_mid = (frame_min + frame_max) / 2.0
        frame_half = max((frame_max - frame_min) / 2.0, 1.0)
        self.register_buffer("frame_mid", torch.tensor(frame_mid))
        self.register_buffer("frame_half", torch.tensor(frame_half))

        cx, cy = beam_center or [0.0, 0.0]
        self.register_buffer("beam_cx", torch.tensor(cx))
        self.register_buffer("beam_cy", torch.tensor(cy))
        self.register_buffer("r_max", torch.tensor(max(r_max, 1.0)))

        self.register_buffer("d_min", torch.tensor(d_min))
        self.register_buffer("d_max", torch.tensor(max(d_max, d_min + 1.0)))

        # Optional crystal-frame SH absorption features as extra inputs
        self.n_abs_sh = int(n_abs_sh)
        n_abs_in = 0
        if self.n_abs_sh > 0:
            lmax = int(round(math.sqrt(self.n_abs_sh + 1))) - 1
            if (lmax + 1) ** 2 - 1 != self.n_abs_sh:
                raise ValueError(
                    f"n_abs_sh={self.n_abs_sh} is not (lmax+1)^2-1 for integer "
                    "lmax; pass (scale_sh_lmax+1)^2-1."
                )
            l_of_col = torch.tensor(
                [l for l in range(1, lmax + 1) for _ in range(2 * l + 1)]
            )
            keep = (
                (l_of_col % 2 == 0)
                if absorption_even_only
                else torch.ones_like(l_of_col, dtype=torch.bool)
            )
            self.register_buffer(
                "abs_cols",
                keep.nonzero(as_tuple=False).squeeze(-1),
                persistent=False,
            )
            n_abs_in = int(keep.sum())

        # Arbitrary per-observation metadata columns as further inputs,
        # standardized as feat = (raw - loc) / scale. The scattering vector
        # `s1` goes here: frame is per-image, lp nearly so, and radius and d
        # are HKL-constant, so without a directional input the scale has no
        # freedom that separates h from -h within one image -- which is where
        # the anomalous difference lives.
        self.n_extra = int(n_extra)
        if self.n_extra > 0:
            loc = extra_loc if extra_loc is not None else [0.0] * self.n_extra
            scl = extra_scale if extra_scale is not None else [1.0] * self.n_extra
            if len(loc) != self.n_extra or len(scl) != self.n_extra:
                raise ValueError(
                    "extra_loc/extra_scale must have length "
                    f"n_extra={self.n_extra}"
                )
            self.register_buffer("extra_loc", torch.tensor(loc).float())
            self.register_buffer(
                "extra_scale", torch.tensor(scl).float().clamp(min=1e-8)
            )

        # Input: [frame_norm, radius_norm, d_norm, lp] + selected SH features,
        # or [frame_norm, lp] + SH when the HKL-constant pair is excluded,
        # plus any extra metadata columns.
        n_input = (2 if self.exclude_hkl_constant else 4) + n_abs_in + self.n_extra
        layers = []
        in_dim = n_input
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        # Bias 0 so softplus(0) ~ 0.69 (flat constant scale) at init
        nn.init.zeros_(self.net[-1].bias)
        if head_init_std > 0.0:
            nn.init.normal_(self.net[-1].weight, std=head_init_std)
        else:
            nn.init.zeros_(self.net[-1].weight)

    def forward(
        self,
        frame: Tensor,
        x: Tensor,
        y: Tensor,
        lp: Tensor,
        d: Tensor,
        absorption_sh: Tensor | None = None,
        extra: Tensor | None = None,
    ) -> Tensor:
        frame_norm = (frame - self.frame_mid) / self.frame_half

        if self.exclude_hkl_constant:
            features = torch.stack([frame_norm, lp], dim=-1)
        else:
            r = torch.sqrt(
                (x - self.beam_cx).pow(2) + (y - self.beam_cy).pow(2)
            )
            r_norm = r / self.r_max
            d_norm = (d - self.d_min) / (self.d_max - self.d_min)
            features = torch.stack([frame_norm, r_norm, d_norm, lp], dim=-1)
        if self.n_abs_sh > 0:
            if absorption_sh is None:
                raise ValueError(
                    "MLPScale was built with crystal-frame SH inputs but "
                    "absorption_sh was not provided; point the data loader's "
                    "metadata reference at metadata_sh.pt."
                )
            if absorption_sh.shape[-1] != self.n_abs_sh:
                raise ValueError(
                    f"absorption_sh has {absorption_sh.shape[-1]} harmonics, "
                    f"expected {self.n_abs_sh} (scale_sh_lmax mismatch)."
                )
            features = torch.cat(
                [features, absorption_sh[:, self.abs_cols]], dim=-1
            )
        if self.n_extra > 0:
            if extra is None:
                raise ValueError(
                    "MLPScale was built with extra metadata features but "
                    "`extra` was not provided; check scale_extra_features."
                )
            if extra.shape[-1] != self.n_extra:
                raise ValueError(
                    f"extra has {extra.shape[-1]} features, expected "
                    f"{self.n_extra} (scale_extra_features mismatch)."
                )
            features = torch.cat(
                [features, (extra - self.extra_loc) / self.extra_scale], dim=-1
            )
        return F.softplus(self.net(features).squeeze(-1))


class _ResBlock(nn.Module):
    """Pre-activation residual MLP block: `x + W2 act(W1 act(x))`."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.lin2(self.act(self.lin1(x))))


class ResNetScale(nn.Module):
    """Deep residual-MLP scale over an explicit metadata feature set.

    Differs from `MLPScale` in two ways that matter for anomalous data.

    Detector `x` and `y` go in directly rather than as the radius
    `sqrt((x-cx)^2 + (y-cy)^2)`. A Friedel mate sits at the same radius but a
    different azimuth, so the radius is exactly the reduction that erases the
    one coordinate separating `h` from `-h` within an image; the other MLP
    inputs (frame, and the HKL-constant `d`) cannot separate them either. Raw
    x, y keep that structure, which is why this needs no `scale_extra_features`
    to see the Bijvoet direction.

    `lp` is an input FEATURE, not a multiplicative `/lp` applied afterwards:
    the network learns the whole scale, any lp dependence included.

    Ported from `amortized-integrator-la` (`7f448da`, Jun 2026). The SH
    absorption columns are selected here as `MLPScale` does on this branch, so
    `scale_sh_lmax` and `scale_mlp_absorption_even_only` mean the same thing in
    both.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_blocks: int = 4,
        frame_min: float = 0.0,
        frame_max: float = 1000.0,
        beam_center: list[float] | None = None,
        r_max: float = 1500.0,
        d_min: float = 1.0,
        d_max: float = 60.0,
        n_abs_sh: int = 0,
        absorption_even_only: bool = True,
        use_xy: bool = True,
        # Laue datasets carry no `lp` column -- the Lorentz-polarization term
        # is not separable there and is absorbed elsewhere -- so the feature
        # has to be droppable rather than assumed.
        use_lp: bool = True,
        lp_loc: float = 1.5,
        lp_scale: float = 1.5,
        head_init_std: float = 0.0,
    ):
        super().__init__()
        self.use_xy = bool(use_xy)
        self.use_lp = bool(use_lp)

        frame_mid = (frame_min + frame_max) / 2.0
        frame_half = max((frame_max - frame_min) / 2.0, 1.0)
        self.register_buffer("frame_mid", torch.tensor(frame_mid))
        self.register_buffer("frame_half", torch.tensor(frame_half))
        cx, cy = beam_center or [0.0, 0.0]
        self.register_buffer("beam_cx", torch.tensor(cx))
        self.register_buffer("beam_cy", torch.tensor(cy))
        self.register_buffer("r_max", torch.tensor(max(r_max, 1.0)))
        self.register_buffer("d_min", torch.tensor(d_min))
        self.register_buffer("d_max", torch.tensor(max(d_max, d_min + 1.0)))
        self.register_buffer("lp_loc", torch.tensor(lp_loc))
        self.register_buffer("lp_scale", torch.tensor(max(lp_scale, 1e-6)))

        self.n_abs_sh = int(n_abs_sh)
        n_abs_in = 0
        if self.n_abs_sh > 0:
            lmax = int(round(math.sqrt(self.n_abs_sh + 1))) - 1
            if (lmax + 1) ** 2 - 1 != self.n_abs_sh:
                raise ValueError(
                    f"n_abs_sh={self.n_abs_sh} is not (lmax+1)^2-1 for integer "
                    "lmax; pass (scale_sh_lmax+1)^2-1."
                )
            l_of_col = torch.tensor(
                [l for l in range(1, lmax + 1) for _ in range(2 * l + 1)]
            )
            keep = (
                (l_of_col % 2 == 0)
                if absorption_even_only
                else torch.ones_like(l_of_col, dtype=torch.bool)
            )
            self.register_buffer(
                "abs_cols",
                keep.nonzero(as_tuple=False).squeeze(-1),
                persistent=False,
            )
            n_abs_in = int(keep.sum())

        # frame, d [, lp] [, x, y] [, SH]
        n_input = 2 + (1 if self.use_lp else 0) + (2 if self.use_xy else 0) + n_abs_in
        layers: list[nn.Module] = [nn.Linear(n_input, hidden_dim), nn.SiLU()]
        for _ in range(n_blocks):
            layers.append(_ResBlock(hidden_dim))
        head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(head.bias)
        if head_init_std > 0.0:
            nn.init.normal_(head.weight, std=head_init_std)
        else:
            # a zero head starts every scale at softplus(0) = 0.693, so the
            # field is flat until the data moves it
            nn.init.zeros_(head.weight)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        frame: Tensor,
        x: Tensor,
        y: Tensor,
        lp: Tensor | None,
        d: Tensor,
        absorption_sh: Tensor | None = None,
    ) -> Tensor:
        feats = [
            ((frame - self.frame_mid) / self.frame_half).unsqueeze(-1),
            ((d - self.d_min) / (self.d_max - self.d_min)).unsqueeze(-1),
        ]
        if self.use_lp:
            if lp is None:
                raise ValueError(
                    "ResNetScale was built with use_lp but no `lp` was given; "
                    "set scale_resnet_use_lp: false for data without it."
                )
            feats.append(((lp - self.lp_loc) / self.lp_scale).unsqueeze(-1))
        if self.use_xy:
            feats.append(((x - self.beam_cx) / self.r_max).unsqueeze(-1))
            feats.append(((y - self.beam_cy) / self.r_max).unsqueeze(-1))
        if self.n_abs_sh > 0:
            if absorption_sh is None:
                raise ValueError(
                    "ResNetScale was built with absorption features but "
                    "`absorption_sh` was not provided; check "
                    "scale_mlp_absorption."
                )
            if absorption_sh.shape[-1] != self.n_abs_sh:
                raise ValueError(
                    f"absorption_sh has {absorption_sh.shape[-1]} harmonics, "
                    f"expected {self.n_abs_sh} (scale_sh_lmax mismatch)."
                )
            feats.append(absorption_sh[:, self.abs_cols])
        return F.softplus(self.net(torch.cat(feats, dim=-1)).squeeze(-1))


class LaueMLPScale(nn.Module):
    """Wavelength-aware MLP scale for polychromatic (Laue) stills.

    The monochromatic `MLPScale` assumes a rotation series -- a scale smooth in
    the goniometer angle (`frame`) -- and folds the LP correction in. Laue data
    here is stills: a stationary crystal in a polychromatic beam, one shot per
    image, each with its own refined orientation. Two things change. The dominant
    scale variation is the incident spectrum `G(lambda)`, and there is no rotation
    axis to be smooth along, so a per-image scale (the standard Laue per-shot
    factor) replaces `scale(frame)`. This scale models

        log s_i = MLP([lambda, x, y, d]_i) + image_log_scale[image_i]

    learning the full per-observation scale -- incident spectrum, detector
    geometry, and resolution falloff -- from continuous features, plus an optional
    per-image log-scale embedding. Output is `exp(.)` (not softplus) so the large
    dynamic range of `G(lambda)` is easy to represent, and the scale is flat
    (`s = 1`) at init: the MLP head and the embedding are both zero-initialized.

    Because the MLP owns the whole correction (there is no separate `lp` column
    for Laue stills), the caller does NOT divide by `lp` -- unlike the rotation
    scales. `n_images=None` drops the per-image term (a pure continuous MLP).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        lambda_min: float = 0.95,
        lambda_max: float = 1.25,
        beam_center: list[float] | None = None,
        r_max: float = 1500.0,
        d_min: float = 1.0,
        d_max: float = 60.0,
        n_images: int | None = None,
        head_init_std: float = 0.0,
        n_abs_sh: int = 0,
        absorption_even_only: bool = True,
        normalize_scale: bool = False,
        norm_momentum: float = 0.02,
        use_wavelength: bool = True,
    ):
        super().__init__()

        # Wavelength is the one feature that is also a Wilson-prior parameter
        # under `polychromatic_wilson`, whose G(lambda) is the same spectral
        # correction. With both free they share a gauge, and on hewl_1118 they
        # settled on a prior anti-correlated with the beam (CC -0.72 against
        # the observed intensity-versus-wavelength curve) and a scale that
        # over-corrected, for a product 4.6x too steep. Dropping lambda here
        # leaves the prior as the only spectral component; the dual -- a scalar
        # prior with lambda in the scale -- is what careless does, since its
        # priors carry no wavelength at all.
        self.use_wavelength = bool(use_wavelength)

        # Optional gauge pin: center the scale to geometric mean 1 so I_h carries
        # the absolute photon scale. OFF by default -- it fights the Wilson prior
        # (whose global G is in normalized units, E[I_h]~0.3), forcing I_h to
        # photon units and collapsing the rate -> the NLL roughly doubles. The
        # natural gauge (I_h ~ prior, magnitude in the scale) fits far better; the
        # large alpha_h/beta_h that result are an over-concentration cosmetic, not
        # an NLL problem. Kept here behind the flag for experiments with a
        # photon-scale Wilson G init.
        self.normalize_scale = normalize_scale
        self.norm_momentum = norm_momentum
        self.register_buffer("running_log_mean", torch.zeros(()))

        lam_mid = (lambda_min + lambda_max) / 2.0
        lam_half = max((lambda_max - lambda_min) / 2.0, 1e-6)
        self.register_buffer("lam_mid", torch.tensor(lam_mid))
        self.register_buffer("lam_half", torch.tensor(lam_half))

        cx, cy = beam_center or [0.0, 0.0]
        self.register_buffer("beam_cx", torch.tensor(cx))
        self.register_buffer("beam_cy", torch.tensor(cy))
        self.register_buffer("r_max", torch.tensor(max(r_max, 1.0)))
        self.register_buffer("d_min", torch.tensor(d_min))
        self.register_buffer("d_max", torch.tensor(max(d_max, d_min + 1.0)))

        # Optional crystal-frame SH absorption features as extra MLP inputs (same
        # mechanism as the mono MLPScale). Even-l only by default: Friedel-
        # symmetric (Y_l(-r)=Y_l(r)), so they cannot form the odd-l anomalous-
        # eating band. Needs `absorption_sh` (extract_crystal_frame_sh.py).
        self.n_abs_sh = int(n_abs_sh)
        n_abs_in = 0
        if self.n_abs_sh > 0:
            lmax = int(round(math.sqrt(self.n_abs_sh + 1))) - 1
            if (lmax + 1) ** 2 - 1 != self.n_abs_sh:
                raise ValueError(
                    f"n_abs_sh={self.n_abs_sh} is not (lmax+1)^2-1 for an integer "
                    "lmax; pass (scale_sh_lmax+1)^2-1."
                )
            l_of_col = torch.tensor(
                [l for l in range(1, lmax + 1) for _ in range(2 * l + 1)]
            )
            keep = (
                (l_of_col % 2 == 0)
                if absorption_even_only
                else torch.ones_like(l_of_col, dtype=torch.bool)
            )
            self.register_buffer(
                "abs_cols",
                keep.nonzero(as_tuple=False).squeeze(-1),
                persistent=False,
            )
            n_abs_in = int(keep.sum())

        # Input: [lambda_norm, x_norm, y_norm, d_norm] + selected SH features,
        # less lambda_norm when `use_wavelength` is off.
        layers = []
        in_dim = (4 if self.use_wavelength else 3) + n_abs_in
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        # Flat at init: bias 0 and (by default) zero output weight -> log s = 0.
        # A small head_init_std seeds the structure so it develops from step 0.
        nn.init.zeros_(self.net[-1].bias)
        if head_init_std > 0.0:
            nn.init.normal_(self.net[-1].weight, std=head_init_std)
        else:
            nn.init.zeros_(self.net[-1].weight)

        # Optional per-image (per-shot) log-scale -- the standard Laue per-image
        # scale factor. Zero-initialized so it starts as a no-op.
        self.n_images = int(n_images) if n_images is not None else 0
        if self.n_images > 0:
            self.image_log_scale = nn.Embedding(self.n_images, 1)
            nn.init.zeros_(self.image_log_scale.weight)

    def forward(
        self,
        wavelength: Tensor,
        x: Tensor,
        y: Tensor,
        d: Tensor,
        image_num: Tensor | None = None,
        absorption_sh: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the Laue scale.

        Args:
            wavelength: (B,) per-observation wavelength (Angstrom).
            x: (B,) detector x positions (xyzcal.px.0).
            y: (B,) detector y positions (xyzcal.px.1).
            d: (B,) resolution in Angstrom.
            image_num: (B,) integer image/shot index; required iff the per-image
                embedding is enabled (n_images > 0). Indices are clamped into
                range defensively.
            absorption_sh: (B, n_sh) crystal-frame real-SH features; required iff
                the SH absorption inputs are enabled (n_abs_sh > 0).

        Returns:
            scale: (B,) strictly positive scale factors.
        """
        lam_norm = (wavelength - self.lam_mid) / self.lam_half
        x_norm = (x - self.beam_cx) / self.r_max
        y_norm = (y - self.beam_cy) / self.r_max
        d_norm = (d - self.d_min) / (self.d_max - self.d_min)
        cols = [lam_norm, x_norm, y_norm, d_norm]
        if not self.use_wavelength:
            cols = cols[1:]
        features = torch.stack(cols, dim=-1)
        if self.n_abs_sh > 0:
            if absorption_sh is None:
                raise ValueError(
                    "LaueMLPScale was built with crystal-frame SH inputs "
                    "(n_abs_sh > 0) but absorption_sh was not provided."
                )
            if absorption_sh.shape[-1] != self.n_abs_sh:
                raise ValueError(
                    f"absorption_sh has {absorption_sh.shape[-1]} harmonics, "
                    f"expected {self.n_abs_sh} (scale_sh_lmax mismatch)."
                )
            features = torch.cat(
                [features, absorption_sh[:, self.abs_cols]], dim=-1
            )
        log_s = self.net(features).squeeze(-1)

        if self.n_images > 0:
            if image_num is None:
                raise ValueError(
                    "LaueMLPScale was built with a per-image embedding "
                    "(n_images > 0) but image_num was not provided."
                )
            idx = image_num.long().clamp(0, self.n_images - 1)
            log_s = log_s + self.image_log_scale(idx).squeeze(-1)

        # Center to geometric mean 1 (gauge fix): the offset is detached, so it
        # only fixes the absolute level -- the spectrum/geometry shape and its
        # gradients are untouched.
        if self.normalize_scale:
            if self.training:
                m = log_s.mean().detach()
                self.running_log_mean.mul_(1 - self.norm_momentum).add_(
                    self.norm_momentum * m
                )
            else:
                m = self.running_log_mean
            log_s = log_s - m

        return torch.exp(log_s.clamp(-15.0, 15.0))


class LaueSpectralScale(nn.Module):
    """Structured spectral scale for polychromatic (Laue) merging.

    The Laue per-observation scale is dominated by the incident spectrum
    `G(lambda)`. A black-box `LaueMLPScale` has to discover that systematic law
    from continuous features while separating it from `F^2` through the merge's
    gauge-degenerate consistency, and in practice it stalls near a constant
    (scale CV ~ 0.08, no spectrum). This mirrors the WORKING `polychromatic_wilson`
    integrator -- which represents `G(lambda)` as a low-order `ChebyshevSpectrum`
    in its Wilson prior and fits well -- but places that same structured spectrum
    in the per-observation SCALE, which is where the amortized merge needs it (the
    shared per-HKL `I_h` is the wavelength-independent `F^2`):

        log s = log G(lambda)             # ChebyshevSpectrum, warm-startable
              + [log f_L(2theta)]         # Lorentz (Ren & Moffat eq. 10), optional
              + [log f_P(2theta, phi)]    # polarization (eq. 11), optional
              + [residual(x, y, d)]       # learned geometry/absorption, optional

    The spectrum can be warm-started from a trained polychromatic_wilson model
    (`spectrum_init_from` -> its saved `spectrum.c`) and optionally frozen, so the
    merge starts from an already-correct `G(lambda)` and only settles the gauge.
    Output is `exp(.)` (the large `G(lambda)` dynamic range is easy in log-space);
    `s = 1` at init when the spectrum is zero-init and the residual head is zero.
    The `image_num`/`absorption_sh` args are accepted (so it is a drop-in for
    `LaueMLPScale` in the scale dispatch) but unused.
    """

    def __init__(
        self,
        degree: int = 40,
        lambda_min: float = 0.95,
        lambda_max: float = 1.25,
        spectrum_init_from: str | None = None,
        freeze_spectrum: bool = False,
        lorentz: bool = False,
        polarization: bool = False,
        polarization_fraction: float = 0.99,
        beam_center: list[float] | None = None,
        residual: bool = False,
        residual_hidden: int = 64,
        residual_layers: int = 2,
        residual_init_std: float = 0.0,
        r_max: float = 1500.0,
        d_min: float = 1.0,
        d_max: float = 60.0,
    ):
        super().__init__()
        self.spectrum = ChebyshevSpectrum(
            degree=degree,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            init_from=spectrum_init_from,
        )
        if freeze_spectrum:
            self.spectrum.c.requires_grad_(False)

        self._apply_lorentz = bool(lorentz)
        self._apply_polarization = bool(polarization)
        cx, cy = beam_center or [0.0, 0.0]
        self.register_buffer("beam_cx", torch.tensor(float(cx)))
        self.register_buffer("beam_cy", torch.tensor(float(cy)))
        if self._apply_polarization:
            self.register_buffer(
                "tau_pol", torch.tensor(2.0 * polarization_fraction - 1.0)
            )

        # Optional learned geometry/absorption residual on [x, y, d] (NOT lambda;
        # the spectrum owns lambda). Zero-initialized -> a no-op at the start.
        self.residual = None
        if residual:
            self.register_buffer("r_max", torch.tensor(max(r_max, 1.0)))
            self.register_buffer("d_min", torch.tensor(d_min))
            self.register_buffer(
                "d_max", torch.tensor(max(d_max, d_min + 1.0))
            )
            layers: list[nn.Module] = []
            in_dim = 3
            for _ in range(residual_layers):
                layers += [nn.Linear(in_dim, residual_hidden), nn.SiLU()]
                in_dim = residual_hidden
            head = nn.Linear(in_dim, 1)
            nn.init.zeros_(head.bias)
            if residual_init_std > 0.0:
                nn.init.normal_(head.weight, std=residual_init_std)
            else:
                nn.init.zeros_(head.weight)
            layers.append(head)
            self.residual = nn.Sequential(*layers)

    def _two_theta(self, wavelength: Tensor, d: Tensor) -> Tensor:
        sin_theta = (wavelength / (2.0 * d.clamp(min=1e-6))).clamp(max=1.0)
        return 2.0 * torch.arcsin(sin_theta)

    def forward(
        self,
        wavelength: Tensor,
        x: Tensor,
        y: Tensor,
        d: Tensor,
        image_num: Tensor | None = None,
        absorption_sh: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the structured Laue scale. Returns (B,) positive factors."""
        log_s = self.spectrum.get_log_G(wavelength)

        if self._apply_lorentz or self._apply_polarization:
            two_theta = self._two_theta(wavelength, d)
            if self._apply_lorentz:
                f_L = torch.sin(two_theta).pow(2).clamp(min=1e-8)
                log_s = log_s + torch.log(f_L)
            if self._apply_polarization:
                phi = torch.atan2(y - self.beam_cy, x - self.beam_cx)
                cos2t, sin2t = torch.cos(two_theta), torch.sin(two_theta)
                f_P = 2.0 / (
                    1.0
                    + cos2t.pow(2)
                    - self.tau_pol * torch.cos(2.0 * phi) * sin2t.pow(2)
                ).clamp(min=1e-8)
                log_s = log_s + torch.log(f_P.clamp(min=1e-8))

        if self.residual is not None:
            x_norm = (x - self.beam_cx) / self.r_max
            y_norm = (y - self.beam_cy) / self.r_max
            d_norm = (d - self.d_min) / (self.d_max - self.d_min)
            feats = torch.stack([x_norm, y_norm, d_norm], dim=-1)
            log_s = log_s + self.residual(feats).squeeze(-1)

        return torch.exp(log_s.clamp(-15.0, 15.0))


class PhysicalScale(nn.Module):
    """DIALS-style physical scale: smooth scale(phi) x decay(phi,d) x absorption.

    The unconstrained `MLPScale` captures the bulk frame scale but learns noise in
    the fine, within-image band because it never sees the variable absorption
    actually depends on -- the diffracted-beam direction in the crystal frame --
    and is too flexible to be pinned there by the consistency signal alone. That
    fine band is exactly what gates the anomalous (Bijvoet) signal, so the model's
    merged anomalous differences come out uncorrelated with the truth.

    This mirrors DIALS's physical scaling model instead, as three multiplicative
    components that are additive in log-space:

        log s_i = scale_phi(phi_i) + decay(phi_i, d_i) + (a_i . c_lm)

    - `scale_phi(phi)`: smooth multiplicative scale over the rotation (Chebyshev
      in normalized frame); also carries the gauge constant (l=0 absorption).
    - `decay(phi, d) = -2 B(phi) s^2`, `s^2 = 1/(4 d^2)`: the resolution-dependent
      B-factor falloff with `B(phi)` smooth in frame (the only term with a d
      dependence -- the absorption basis is direction-only).
    - absorption: linear in the precomputed crystal-frame spherical-harmonic basis
      `a_i` (see `scripts/extract_crystal_frame_sh.py`); `c_lm` are the only
      per-harmonic parameters and the fine, anomalous-gating part. Because the
      basis is smooth and low-dimensional (~(lmax+1)^2-1 coefficients) it cannot
      interpolate noise the way the MLP did, and because it is a function of the
      crystal-frame direction it differs between Friedel mates (which sit at
      different phi / detector positions), letting the merge separate the real
      Bijvoet signal from geometry.

    At init `c_lm = 0`, `B = 0`, and `scale_phi = log(init_scale)`, so `s = 1`
    everywhere (the WLS merge starts as an inverse-variance mean). Output is
    `exp(.)`, strictly positive, no softplus needed. The known LP correction is
    applied by the caller (`/lp`), as for the Chebyshev scales.
    """

    def __init__(
        self,
        n_sh: int,
        degree_scale: int = 4,
        degree_decay: int = 2,
        frame_min: float = 0.0,
        frame_max: float = 1000.0,
        init_scale: float = 1.0,
        absorption_init_std: float = 0.0,
    ):
        super().__init__()
        if n_sh < 1:
            raise ValueError(f"n_sh must be >= 1, got {n_sh}")
        self.n_sh = n_sh
        self.degree_scale = degree_scale
        self.degree_decay = degree_decay

        frame_mid = (frame_min + frame_max) / 2.0
        frame_half = max((frame_max - frame_min) / 2.0, 1.0)
        self.register_buffer("frame_mid", torch.tensor(frame_mid))
        self.register_buffer("frame_half", torch.tensor(frame_half))

        sc = torch.zeros(degree_scale + 1)
        sc[0] = math.log(
            init_scale
        )  # log-space constant -> scale starts at init
        self.scale_c = nn.Parameter(sc)
        # B(phi) for decay; 0 -> no resolution falloff at init.
        self.decay_c = nn.Parameter(torch.zeros(degree_decay + 1))
        # Absorption SH coefficients; 0 -> flat absorption at init. A small
        # absorption_init_std seeds the surface so it develops from step 0.
        if absorption_init_std > 0.0:
            self.absorption_c = nn.Parameter(
                torch.randn(n_sh) * absorption_init_std
            )
        else:
            self.absorption_c = nn.Parameter(torch.zeros(n_sh))

        # l index of each SH column (basis ordered l=1..lmax, m=-l..l). Used to
        # weight the restraint by l(l+1) -- the Laplace-Beltrami (smoothness)
        # energy on the sphere. Non-persistent: a constant derived from n_sh.
        lmax = int(round(math.sqrt(n_sh + 1))) - 1
        if (lmax + 1) ** 2 - 1 != n_sh:
            raise ValueError(
                f"n_sh={n_sh} is not (lmax+1)^2-1 for an integer lmax "
                f"(got lmax~{lmax}); pass n_sh from the extractor's --lmax."
            )
        self.lmax = lmax
        l_of_col = [l for l in range(1, lmax + 1) for _ in range(2 * l + 1)]
        self.register_buffer(
            "sh_l",
            torch.tensor(l_of_col, dtype=torch.float32),
            persistent=False,
        )

    def restraint_penalty(self) -> Tensor:
        """Smoothness restraint on the absorption (+ decay) coefficients.

        Returns `sum_lm l(l+1) c_lm^2 + sum_k decay_k^2`; the caller weights it
        and adds it to the loss. The scaling-consistency objective alone has
        nothing opposing an ever-larger absorption surface -- the unconstrained
        Beer-Lambert absorption ran away to `f_A ~ 0` at the detector edges -- and
        the odd-`l` harmonics can absorb the real Bijvoet signal because Friedel
        mates sit at inverted crystal-frame directions (`Y_lm(-r) = (-1)^l
        Y_lm(r)`). Weighting by `l(l+1)` damps the rough, high-`l`, anomalous-
        sensitive band hardest while leaving the low-`l` physical absorption
        nearly free. Exactly 0 at init (`c = 0`, `decay = 0`).
        """
        w = self.sh_l * (self.sh_l + 1.0)
        return (w * self.absorption_c.pow(2)).sum() + self.decay_c.pow(2).sum()

    def _basis(self, frame: Tensor, degree: int) -> Tensor:
        t = ((frame - self.frame_mid) / self.frame_half).clamp(-1.0, 1.0)
        return torch.stack(ChebyshevSpectrum._chebyshev(t, degree), dim=-1)

    def forward(
        self, frame: Tensor, d: Tensor, absorption_sh: Tensor
    ) -> Tensor:
        """Evaluate the physical scale.

        Args:
            frame: (B,) frame numbers (xyzcal.px.2), proxy for rotation angle.
            d: (B,) resolution in Angstrom (for the B-factor decay term).
            absorption_sh: (B, n_sh) precomputed crystal-frame real-SH features.

        Returns:
            scale: (B,) strictly positive scale factors.
        """
        if absorption_sh.shape[-1] != self.n_sh:
            raise ValueError(
                f"absorption_sh has {absorption_sh.shape[-1]} harmonics, expected "
                f"{self.n_sh} (lmax mismatch between the config and the extractor)."
            )
        log_scale = self._basis(frame, self.degree_scale) @ self.scale_c
        b_factor = self._basis(frame, self.degree_decay) @ self.decay_c
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-3).pow(2))
        log_decay = -2.0 * b_factor * s_sq
        log_abs = absorption_sh @ self.absorption_c
        total = (log_scale + log_decay + log_abs).clamp(-15.0, 15.0)
        return torch.exp(total)


# The real tesseral harmonics, unnormalised, l = 1..4 and l = 0 omitted -- the
# constant is the overall scale, which the merge cannot see. Written out rather
# than recursed so this matches `scripts/fit_physical_scale.real_sph_harm`
# column for column: the offline fit and the trained scale have to share one
# basis definition or their results are not comparable.
def _real_sph_harm(xyz: Tensor, lmax: int) -> tuple[Tensor, list[int]]:
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    z2 = z * z
    blocks: list[list[Tensor]] = [
        [z, x, y],
        [3 * z2 - 1, x * z, y * z, x * x - y * y, x * y],
        [z * (5 * z2 - 3), x * (5 * z2 - 1), y * (5 * z2 - 1),
         z * (x * x - y * y), x * y * z,
         x * (x * x - 3 * y * y), y * (3 * x * x - y * y)],
        [35 * z2 * z2 - 30 * z2 + 3, x * z * (7 * z2 - 3), y * z * (7 * z2 - 3),
         (x * x - y * y) * (7 * z2 - 1), x * y * (7 * z2 - 1),
         x * z * (x * x - 3 * y * y), y * z * (3 * x * x - y * y),
         x * x * (x * x - 3 * y * y) - y * y * (3 * x * x - y * y),
         x * y * (x * x - y * y)],
    ]
    if not 1 <= lmax <= len(blocks):
        raise ValueError(f"lmax must be 1..{len(blocks)}, got {lmax}")
    cols, l_of_col = [], []
    for ell in range(1, lmax + 1):
        cols.extend(blocks[ell - 1])
        l_of_col.extend([ell] * len(blocks[ell - 1]))
    return torch.stack(cols, dim=-1), l_of_col


def _rotate_about_axis(v: Tensor, axis: Tensor, angles: Tensor) -> Tensor:
    """Rodrigues rotation of rows of `v` by per-row `angles` about `axis`."""
    k = axis / axis.norm().clamp(min=1e-12)
    c, s = angles.cos().unsqueeze(-1), angles.sin().unsqueeze(-1)
    kv = torch.cross(k.expand_as(v), v, dim=-1)
    kkv = (v * k).sum(-1, keepdim=True) * k
    return v * c + kv * s + kkv * (1.0 - c)


class LauePhysicalScale(nn.Module):
    r"""The Laue scale as physics, fitted in an order that keeps terms apart.

        log s = G(lambda) + log P(2theta, phi; tau, phi0) + K(image)
                - 2 B(image) s^2 + sum_lm c_lm [Y_lm(s0_c) + Y_lm(s1_c)]

    Every arm before this one scaled Laue data either with a generic MLP on
    lab-frame coordinates or with `LaueSpectralScale`, whose residual MLP on
    `[x, y, d]` destroys the spectrum's identifiability -- on a still
    `lambda = 2 d sin(theta(r))`, so a free function of position and resolution
    absorbs 73% of a realistic `G(lambda)`. Nothing here is free in position or
    resolution.

    **Polarisation is supplied, not fitted.** Its azimuthal signature
    `cos(2(phi - phi0))` lies inside the real spherical harmonics at
    `l = 2, m = +-2`, which ARE `cos 2phi` and `sin 2phi`, so an absorption
    surface spans it and only the `sin^2(2theta)` modulation separates them.
    Measured on simulated Laue data where the truth is known: fitting the plane
    lands 22.5 deg away and stays there under refinement, giving the term back
    with CC -0.33 -- the right physics with the sign inverted -- while supplying
    it gives CC +1.00. It is known physics at a synchrotron, so `tau` and
    `phi0` are buffers, not parameters.

    `P = [1 + cos^2(2t) - tau cos(2(phi - phi0)) sin^2(2t)] / 2` (Kahn 1982;
    Ren & Moffat 1995 eq. 11) multiplies the rate. The quantity usually
    tabulated is the correction `1/P`, and applying that as the scale is a sign
    error, which is what `LaueSpectralScale` does at `:762`.

    Absorption is evaluated in the CRYSTAL frame, which no lab-frame scale can
    reach: the stills come from a rotation series, so undoing `phi(image)`
    recovers the coordinate DIALS uses, and a complete real-SH basis is closed
    under rotation so the unknown constant setting does not matter. Both legs
    are summed as DIALS does, and the `lambda^3` dependence of the absorption
    coefficient is applied as known physics rather than learned.

    The `l(l+1)` restraint is `restraint_penalty()`, the Laplace-Beltrami
    energy on the sphere: the consistency objective has nothing opposing an
    ever-larger surface, and odd `l` can eat the real Bijvoet signal because
    Friedel mates sit at inverted crystal-frame directions
    (`Y_lm(-r) = (-1)^l Y_lm(r)`).

    At init every coefficient is zero, so `s = 1` and the merge starts as an
    inverse-variance mean. The basis matches `fit_physical_scale.build_design`
    column for column, so an offline fit and this module are comparable.
    """

    def __init__(
        self,
        lmax: int = 4,
        degree_spectrum: int = 6,
        degree_image: int = 8,
        lambda_min: float = 0.97,
        lambda_max: float = 1.25,
        n_images: int = 1000,
        tau: float = 0.95,
        phi0_deg: float = 0.0,
        beam_center: tuple[float, float] = (0.0, 0.0),
        rotation_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
        deg_per_image: float = 1.0,
        beam_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        lambda_ref: float | None = None,
    ):
        super().__init__()
        self.lmax = int(lmax)
        self.degree_spectrum = int(degree_spectrum)
        self.degree_image = int(degree_image)
        self.degree_decay = max(self.degree_image // 2, 1)
        self.deg_per_image = float(deg_per_image)

        # Chebyshev blocks all skip their constant: an overall level is a gauge
        # the merge absorbs into I_h, and fitting it invites the two to drift.
        self.spectrum_c = nn.Parameter(torch.zeros(self.degree_spectrum))
        self.image_c = nn.Parameter(torch.zeros(self.degree_image))
        self.decay_c = nn.Parameter(torch.zeros(self.degree_decay))

        _, l_of_col = _real_sph_harm(torch.zeros(1, 3), self.lmax)
        self.absorption_c = nn.Parameter(torch.zeros(len(l_of_col)))
        self.register_buffer(
            "sh_l", torch.tensor(l_of_col, dtype=torch.float32), persistent=False
        )
        # roughness weights, matching the offline fit's `pen`
        self.register_buffer(
            "spectrum_pen",
            torch.arange(1, self.degree_spectrum + 1, dtype=torch.float32) ** 4,
            persistent=False,
        )

        for name, value in (
            ("tau", tau), ("phi0", math.radians(phi0_deg)),
            ("lam_mid", (lambda_min + lambda_max) / 2.0),
            ("lam_half", max((lambda_max - lambda_min) / 2.0, 1e-6)),
            ("img_mid", (n_images - 1) / 2.0),
            ("img_half", max((n_images - 1) / 2.0, 1.0)),
            ("beam_cx", beam_center[0]), ("beam_cy", beam_center[1]),
            ("lam_ref", lambda_ref if lambda_ref else (lambda_min + lambda_max) / 2.0),
        ):
            self.register_buffer(name, torch.tensor(float(value)))
        self.register_buffer("axis", torch.tensor(rotation_axis, dtype=torch.float32))
        self.register_buffer("s0_hat", torch.tensor(beam_direction, dtype=torch.float32))

    def restraint_penalty(self) -> Tensor:
        """`sum l(l+1) c_lm^2 + sum k^4 G_k^2 + sum B_k^2`, zero at init.

        The same weights the offline fit uses, so a restraint weight tuned
        there means the same thing here.
        """
        w = self.sh_l * (self.sh_l + 1.0)
        return (
            (w * self.absorption_c.pow(2)).sum()
            + (self.spectrum_pen * self.spectrum_c.pow(2)).sum()
            + self.decay_c.pow(2).sum()
        )

    @staticmethod
    def _cheb(t: Tensor, degree: int) -> Tensor:
        """Chebyshev columns 1..degree on `t` in [-1, 1]; the constant is dropped."""
        cols = ChebyshevSpectrum._chebyshev(t.clamp(-1.0, 1.0), degree)
        return torch.stack(cols[1:], dim=-1)

    def log_terms(
        self,
        wavelength: Tensor,
        d: Tensor,
        x: Tensor,
        y: Tensor,
        s1: Tensor,
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Each term separately, which is how this gets scored against truth."""
        lam = wavelength.clamp(min=1e-6)
        d_safe = d.clamp(min=1e-3)
        sin_t = (lam / (2.0 * d_safe)).clamp(-1.0, 1.0)
        two_t = 2.0 * torch.asin(sin_t)
        s_sq = 1.0 / (4.0 * d_safe.pow(2))

        spectrum = self._cheb(
            (lam - self.lam_mid) / self.lam_half, self.degree_spectrum
        ) @ self.spectrum_c

        phi_det = torch.atan2(y - self.beam_cy, x - self.beam_cx)
        p = 0.5 * (
            1.0
            + two_t.cos().pow(2)
            - self.tau * torch.cos(2.0 * (phi_det - self.phi0)) * two_t.sin().pow(2)
        )
        polarisation = p.clamp(min=1e-8).log()

        img_norm = (image.float() - self.img_mid) / self.img_half
        image_gray = self._cheb(img_norm, self.degree_image) @ self.image_c
        b_factor = self._cheb(img_norm, self.degree_decay) @ self.decay_c
        decay = -2.0 * b_factor * s_sq

        # undo the goniometer rotation to reach the crystal frame
        phi_img = torch.deg2rad(image.float() * self.deg_per_image)
        s1_hat = s1 / s1.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        s1_c = _rotate_about_axis(s1_hat, self.axis, -phi_img)
        s0_c = _rotate_about_axis(
            self.s0_hat.expand_as(s1_hat).contiguous(), self.axis, -phi_img
        )
        y1, _ = _real_sph_harm(s1_c, self.lmax)
        y0, _ = _real_sph_harm(s0_c, self.lmax)
        # mu ~ lambda^3, so the wavelength dependence is known, not learned
        basis = (y0 + y1) * (lam / self.lam_ref).pow(3).unsqueeze(-1)
        absorption = basis @ self.absorption_c

        return {
            "spectrum": spectrum,
            "polarisation": polarisation,
            "image_gray": image_gray,
            "decay": decay,
            "absorption": absorption,
        }

    def forward(
        self,
        wavelength: Tensor,
        d: Tensor,
        x: Tensor,
        y: Tensor,
        s1: Tensor,
        image: Tensor,
    ) -> Tensor:
        terms = self.log_terms(wavelength, d, x, y, s1, image)
        total = sum(terms.values()).clamp(-15.0, 15.0)
        return torch.exp(total)
