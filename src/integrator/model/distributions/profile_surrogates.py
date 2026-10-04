import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from integrator.model.distributions.profile_geometry import (
    N_GEOM_FEATURES,
    geometry_features,
)


def _softplus_inverse(x: float) -> float:
    """Inverse of softplus: log(exp(x) - 1)."""
    return math.log(math.exp(x) - 1.0)


def _sample_and_decode(
    loc: Tensor,
    scale: Tensor,
    W: Tensor,
    b: Tensor,
    mc_samples: int,
) -> tuple[Tensor, Tensor]:
    """Sample h from q(h|x) and decode to profile vectors.

    Args:
        loc: Posterior mean, shape (B, d).
        scale: Posterior std, shape (B, d).
        W: Decoder weight matrix, shape (K, d).
        b: Decoder bias, shape (K,) or (B, K) for per-bin bias.
        mc_samples: Number of Monte Carlo samples.

    Returns:
        zp: Profile samples on the simplex, shape (S, B, K).
        mean_profile: Profile at posterior mean h, shape (B, K).
    """
    q_h = Normal(loc, scale)
    h = q_h.rsample([mc_samples])  # (S, B, d)
    logits = h @ W.T + b  # (S, B, K)
    zp = F.softmax(logits, dim=-1)

    mean_logits = loc @ W.T + b  # (B, K)
    mean_profile = F.softmax(mean_logits, dim=-1)

    return zp, mean_profile


@dataclass
class ProfileSurrogateOutput:
    """Output of a latent-decoder profile surrogate.

    Fields:
        zp: Profile samples on the simplex, shape (S, B, K).
        mean_profile: Profile at posterior mean h, shape (B, K).
        loc: Posterior mean of h, shape (B, d).
        scale: Posterior std of h, shape (B, d).
        prior_scale: Std of the N(0, prior_scale) prior on the latent h
    """

    zp: Tensor
    mean_profile: Tensor
    loc: Tensor
    scale: Tensor
    prior_scale: float = 3.0


class ProfileSurrogate(nn.Module):
    """Profile surrogate with a learned linear decoder.

        prf = softmax(W @ h + b)
        q(h | x) = N(loc(x), diag(scale(x)^2))

    Args:
        input_dim: Dimension of the encoder output.
        latent_dim: Dimension of the latent h. Default 8.
        output_dim: Number of profile pixels (D*H*W). Default 441.
        init_std: Initial posterior std for h.
        prior_scale: Std of the N(0, prior_scale) prior on the latent h.
        smoothness_weight: Penalty on the spatial roughness of each mode.
        orthogonality_weight: Penalty on the angles between modes.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int | None = None,
        output_dim: int = 441,
        init_std: float = 0.5,
        prior_scale: float = 3.0,
        smoothness_weight: float = 0.0,
        orthogonality_weight: float = 0.0,
    ) -> None:
        super().__init__()

        if latent_dim is None:
            latent_dim = 8
        self.d: int = latent_dim
        self.prior_scale = prior_scale
        self.smoothness_weight = smoothness_weight
        self.orthogonality_weight = orthogonality_weight

        self.loc_head = nn.Linear(input_dim, self.d)
        self.scale_head = nn.Linear(input_dim, self.d)
        self.decoder = nn.Linear(self.d, output_dim)

        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, _softplus_inverse(init_std))

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        **kwargs,
    ) -> ProfileSurrogateOutput:
        loc = self.loc_head(x)  # (B, d)
        scale = F.softplus(self.scale_head(x))  # (B, d)

        zp, mean_profile = _sample_and_decode(
            loc, scale, self.decoder.weight, self.decoder.bias, mc_samples
        )
        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            loc=loc,
            scale=scale,
            prior_scale=self.prior_scale,
        )


class GeometricProfileSurrogate(ProfileSurrogate):
    """Learned-basis profile whose weak-reflection fallback is LOCAL, not global.

    The base surrogate decodes `softmax(W h + b)` from a latent `h` inferred from a
    reflection's own pixels, with `h ~ N(0, prior_scale)`. When the counts are too
    weak to constrain `h`, the KL drives `h -> 0` and every such reflection gets
    `softmax(b)` -- a single profile shared by the whole dataset. On Laue stills
    that is an azimuthal average of radial streaks, so it is round where the data
    is elongated, and profile fitting beats it precisely there.

    Two config-gated additions put the detector geometry back, both reusing the
    existing basis rather than replacing it:

    `local_shift`
        `logits = (h + m(s)) @ W.T + b` with `m(s)` a small MLP of the geometry
        features. The KL still pulls `h -> 0`, but the fallback is now
        `softmax(m(s) @ W.T + b)`: the profile of the reflection's neighbourhood.
        `h` becomes the residual from the local profile, so the loss is unchanged.

    `steer`
        The decoded profile is warped by `A(s) = R(phi) diag(s_par, s_perp)`, with
        `phi` the analytic azimuth about the beam centre and the log extents
        predicted from geometry. One canonical shape is learned and pointed along
        the local streak direction, instead of a different shape being relearned
        per detector position. This generalizes steering a Gaussian by the
        eigendecomposition of its covariance to an arbitrary learned profile.

    Both geometry heads are zero-initialized in their last layer, so `local_shift`
    starts as exactly `ProfileSurrogate` and grows away from it. `steer` does not:
    the rotation by `phi` is active from the first step by construction, and only the
    extents start at unity. Either way a `ProfileSurrogate` checkpoint loads with
    `strict=False`, since the geometry heads are the only new parameters.

    Args:
        beam_center: Detector beam centre in pixels; required by `steer`.
        profile_shape: Profile grid as `(D, H, W)`; inferred as a square when unset.
        lambda_min: Low end of the spectrum, for normalizing wavelength.
        lambda_max: High end of the spectrum.
        local_shift: Shift the latent by a geometry-predicted `m(s)`.
        steer: Warp the decoded profile by a geometry-predicted affine.
        geom_hidden: Width of the geometry MLPs.
        max_log_extent: Bound on `|log extent|`, so extents stay in
            `[exp(-max), exp(max)]` and the warp cannot collapse the profile.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int | None = None,
        output_dim: int = 441,
        init_std: float = 0.5,
        prior_scale: float = 3.0,
        smoothness_weight: float = 0.0,
        orthogonality_weight: float = 0.0,
        beam_center: tuple[float, float] | list[float] | None = None,
        profile_shape: tuple[int, int, int] | list[int] | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
        local_shift: bool = False,
        steer: bool = False,
        geom_hidden: int = 32,
        max_log_extent: float = 1.0,
        gate_extent: bool = False,
        extent_half_signal: float = 100.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            init_std=init_std,
            prior_scale=prior_scale,
            smoothness_weight=smoothness_weight,
            orthogonality_weight=orthogonality_weight,
        )

        self.local_shift = bool(local_shift)
        self.steer = bool(steer)
        self.beam_center = tuple(beam_center) if beam_center is not None else None
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.max_log_extent = float(max_log_extent)

        if profile_shape is not None:
            d, h, w = (int(v) for v in profile_shape)
        else:
            side = int(round(math.sqrt(output_dim)))
            if side * side != output_dim:
                raise ValueError(
                    f"output_dim {output_dim} is not square; pass profile_shape"
                )
            d, h, w = 1, side, side
        if d * h * w != output_dim:
            raise ValueError(
                f"profile_shape {(d, h, w)} has {d * h * w} pixels, "
                f"but output_dim is {output_dim}"
            )
        self.profile_shape = (d, h, w)

        if (self.local_shift or self.steer) and self.beam_center is None:
            raise ValueError(
                "GeometricProfileSurrogate needs beam_center when local_shift "
                "or steer is enabled"
            )

        # A rotation maps the output box's corners outside itself, so the canonical
        # profile is decoded on a grid large enough to contain the box at any angle.
        # Sampling zeros from beyond a same-sized grid would instead shave the
        # corners by an amount that depends on the azimuth -- a position-correlated
        # error, and position is exactly what Friedel mates differ in.
        self.canonical_shape = self.profile_shape
        if self.steer:
            ch, cw = (self._cover(h), self._cover(w))
            self.canonical_shape = (d, ch, cw)
            self.decoder = nn.Linear(self.d, d * ch * cw)

        self.shift_head = (
            self._geometry_mlp(geom_hidden, self.d) if self.local_shift else None
        )
        self.extent_head = (
            self._geometry_mlp(geom_hidden, 2) if self.steer else None
        )

        # Measured on PDZ2: the geometry-predicted extent rescues weak reflections
        # (effective pixel count 196 -> 49) but fits BRIGHT spots worse than a freely
        # inferred profile does -- cos(profile, data) 0.956 against 0.982 at the top
        # of the I/sigma range. A bright shoebox has the photons to determine its own
        # shape, and constraining it to the neighbourhood's costs first-order
        # normalisation bias, which scales with I. So the extent gets a residual that
        # only opens up where the shoebox can pay for it.
        self.gate_extent = bool(gate_extent) and self.steer
        self.extent_residual_head = None
        if self.gate_extent:
            head = nn.Linear(input_dim, 2)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.extent_residual_head = head
            self.register_buffer(
                "extent_half_signal",
                torch.tensor(float(extent_half_signal)),
                persistent=True,
            )

    @staticmethod
    def _cover(side: int) -> int:
        """Smallest odd grid that contains a `side`-wide box rotated to any angle."""
        n = math.ceil(side * math.sqrt(2.0))
        return n + 1 if n % 2 == 0 else n

    @staticmethod
    def _geometry_mlp(hidden: int, out_dim: int) -> nn.Module:
        """Geometry MLP whose output starts at exactly zero."""
        mlp = nn.Sequential(
            nn.Linear(N_GEOM_FEATURES, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(mlp[-1].weight)
        nn.init.zeros_(mlp[-1].bias)
        return mlp

    def _extent_gate(self, shape_signal: Tensor | None, like: Tensor) -> Tensor:
        """How much this shoebox is allowed to set its own extent, in [0, 1).

        The information a shoebox carries about its own profile shape grows with the
        number of signal photons in it, so the gate is a saturating function of that
        count: `s / (s + k)`, half-open at `s = k`. It is deliberately built from the
        counts rather than from the latent's posterior width, because the latent is
        gauge-equivalent under `h -> h/c`, `W -> cW` -- any gate read off `scale`
        would inherit that freedom and mean nothing.

        Without a signal estimate the gate is closed, which reproduces the ungated
        geometry warp exactly.
        """
        if shape_signal is None:
            return torch.zeros(like.shape[0], 1, device=like.device, dtype=like.dtype)
        s = shape_signal.to(like.dtype).to(like.device).reshape(-1, 1).clamp(min=0.0)
        return s / (s + self.extent_half_signal.clamp(min=1e-3))

    def _warp(self, profile: Tensor, phi: Tensor, log_extent: Tensor) -> Tensor:
        """Affinely steer profiles that are already on the simplex.

        The profile is warped rather than the logits so that the zero padding
        outside the box means zero mass, which renormalizing then handles
        correctly; zero-padded logits would instead read as mass `exp(0)`.

        Args:
            profile: Canonical profiles, shape (S, B, K_canonical), rows summing to 1.
            phi: Azimuth per reflection, shape (B,).
            log_extent: Log extents along and across `phi`, shape (B, 2).

        Returns:
            Steered profiles on the output grid, shape (S, B, K), summing to 1.
        """
        s, b, _ = profile.shape
        d, h, w = self.profile_shape
        cd, ch, cw = self.canonical_shape

        log_extent = log_extent.clamp(-self.max_log_extent, self.max_log_extent)
        inv_par = torch.exp(-log_extent[:, 0])
        inv_perp = torch.exp(-log_extent[:, 1])
        cos, sin = torch.cos(phi), torch.sin(phi)

        # grid_sample reads output(u) = input(theta u), so theta is A^-1 with
        # A = R(phi) diag(s_par, s_perp) and A^-1 = diag(1/s_par, 1/s_perp) R(-phi).
        # The canonical grid's normalized coordinates are wider than the output's by
        # canonical/output, so theta carries that ratio to keep both in pixel units.
        theta = profile.new_zeros(b, 2, 3)
        theta[:, 0, 0] = cos * inv_par * (w / cw)
        theta[:, 0, 1] = sin * inv_par * (h / cw)
        theta[:, 1, 0] = -sin * inv_perp * (w / ch)
        theta[:, 1, 1] = cos * inv_perp * (h / ch)

        theta = theta.unsqueeze(0).expand(s, b, 2, 3).reshape(s * b, 2, 3)
        vol = profile.reshape(s * b, cd, ch, cw)
        grid = F.affine_grid(theta, (s * b, cd, h, w), align_corners=False)
        warped = F.grid_sample(
            vol, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        warped = warped.reshape(s, b, d * h * w).clamp_min(0)
        return warped / warped.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        metadata: dict | None = None,
        **kwargs,
    ) -> ProfileSurrogateOutput:
        loc = self.loc_head(x)
        scale = F.softplus(self.scale_head(x))

        needs_geometry = self.local_shift or self.steer
        if needs_geometry and metadata is None:
            raise ValueError(
                "GeometricProfileSurrogate needs metadata when local_shift "
                "or steer is enabled"
            )

        features = phi = None
        if needs_geometry:
            features, phi = geometry_features(
                metadata, self.beam_center, self.lambda_min, self.lambda_max
            )
            features = features.to(loc.dtype).to(loc.device)
            phi = phi.to(loc.dtype).to(loc.device)

        # the KL still shrinks `loc` to zero, but the decoded fallback is now local
        decode_loc = loc
        if self.shift_head is not None:
            decode_loc = loc + self.shift_head(features)

        zp, mean_profile = _sample_and_decode(
            decode_loc, scale, self.decoder.weight, self.decoder.bias, mc_samples
        )

        if self.extent_head is not None:
            log_extent = self.extent_head(features)
            if self.extent_residual_head is not None:
                log_extent = log_extent + self._extent_gate(
                    kwargs.get("shape_signal"), loc
                ) * self.extent_residual_head(x)
            zp = self._warp(zp, phi, log_extent)
            mean_profile = self._warp(
                mean_profile.unsqueeze(0), phi, log_extent
            ).squeeze(0)

        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            loc=loc,
            scale=scale,
            prior_scale=self.prior_scale,
        )
