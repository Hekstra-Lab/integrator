"""Profile as a fixed canonical density moved by a quasiconformal map of the plane.

A learned-basis profile `softmax(W h + b)` is linear in log-density, so two profiles
interpolate as a geometric mixture, and a streak that rotates or lengthens across the
detector has to be spelled out as a sum of basis images. Here the shape and the motion
are separated:

    p(x) = p0(g(x)) |det Dg(x)| / Z

with `p0` an isotropic density of unit variance and `g` a map from detector pixels to
canonical coordinates. Every orientation-preserving map of the plane is determined, up to
a conformal one, by its Beltrami coefficient `mu = g_zbar / g_z`, `|mu| < 1`. Its
argument is twice the direction of stretch and `K = (1 + |mu|) / (1 - |mu|)` the aspect
ratio, so a streak's direction and elongation are one complex number. `g` is

    z'  = exp(-i phi) (z - z_c) - t                 local radial frame, centred
    u   = z' / R                                    box-normalized
    q   = u + c u^2 + d u ubar + e ubar^2           bending and taper
    v   = (q - mu qbar) / sqrt(1 - |mu|^2)          area-preserving stretch
    g   = R v / s                                   size

with `phi` the analytic azimuth about the beam centre, `z_c` the predicted centroid and
`R` the box half-width. With `c = d = e = 0` and constant `mu` this is the affine `steer`
of `GeometricProfileSurrogate`. Positive real `mu` stretches along the radial direction.

The eleven parameters `theta = (t, log s, w, c, d, e)` are `theta0 + m(s) + h`: a global
offset, a geometry MLP and a per-reflection latent `h ~ q(h | x)`. Each has a prior width
in its own physical unit (pixels, log units, dimensionless), so there is no basis to trade
scale with and the KL is well-posed. The density is evaluated in closed form at pixel
centres: no resampling, no padding, and no hard bound on the elongation.

Constraints, all smooth:
    |mu| = mu_max tanh(|w|) < mu_max, default 0.9, so the aspect ratio K <= 19
    |c| + |d| + |e| <= 0.45, which keeps |q_ubar| < |q_u| on the box, so g has no fold
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, kl_divergence

from integrator.model.distributions.profile_geometry import (
    N_GEOM_FEATURES,
    geometry_features,
)
from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
    renormalize_masked,
)

# theta layout. DRIFT and DEPTH are appended, so the 2D slices and any
# checkpoint written before the depth axis existed keep their meaning.
T, LOG_S, W, CURV = slice(0, 2), 2, slice(3, 5), slice(5, 11)
N_PARAMS = 11
DRIFT = slice(11, 13)
_CURV_MAX = 0.45


def n_params(depth: int) -> int:
    """Parameter count for a shoebox of `depth` frames.

    `depth = 1` gives exactly `N_PARAMS`, so the 2D model is unchanged. Beyond
    that the map stays planar and gains a rocking curve: by Liouville's theorem
    a conformal map in dimension >= 3 is a Mobius transformation, so there is no
    Beltrami coefficient in 3D and the `mu` construction has no 3D analogue.
    The frames are therefore separable -- `D` free simplex weights -- with a
    linear centroid drift so a spot may walk across frames.
    """
    return N_PARAMS if depth <= 1 else N_PARAMS + 2 + depth


class QuasiconformalProfileSurrogate(nn.Module):
    """Canonical density warped by a geometry-steered quasiconformal map.

    Args:
        input_dim: Width of the profile-encoder embedding.
        beam_center: Detector beam centre in pixels.
        profile_shape: Shoebox grid `(D, H, W)`. `D > 1` keeps the planar map
            and adds `D` simplex weights over frames plus a linear centroid
            drift, which is the only extension available: see `n_params`.
        lambda_min: Low end of the spectrum, for the geometry features.
        lambda_max: High end of the spectrum.
        prior_scale: Prior std of `h`, either one value for all eleven components or
            a mapping with keys `t` (pixels), `log_s`, `mu` (applies to `w`) and
            `curv`.
        learned_prior_scale: Predict `log sigma(s)` per component from geometry,
            starting at `prior_scale`.
        max_log_prior_scale: Bound on `|log sigma(s) - log prior_scale|`.
        init_size: Initial `s` in pixels, the per-axis std of the profile before
            stretching.
        init_mu: Initial `|mu|`, along the radial direction.
        mu_max: Ceiling on `|mu|`. An unconstrained fit to a weak shoebox drives `|mu|`
            to 1 and draws an arc through the noise (K = 835 on a simulated DHFR
            reflection with I/sigma 4); 0.9 allows K up to 19.
        init_beta: Initial shape exponent of `p0(r) ~ exp(-(r / alpha)^beta)`;
            2 is Gaussian, smaller is heavier-tailed.
        init_std: Initial posterior std of each component of `h`, as a fraction of
            that component's prior std, so the KL starts small for every unit.
        geom_hidden: Width of the geometry MLPs.
        supersample: Evaluate `ss x ss` points per pixel and average.
        use_centroid: Centre the map on the predicted centroid from `xyzcal.px`
            and `bbox` rather than on the box centre.
    """

    def __init__(
        self,
        input_dim: int,
        beam_center: tuple[float, float] | list[float] | None = None,
        profile_shape: tuple[int, int, int] | list[int] | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
        prior_scale: float | dict | None = None,
        learned_prior_scale: bool = False,
        max_log_prior_scale: float = 3.0,
        init_size: float = 1.5,
        init_mu: float = 0.3,
        mu_max: float = 0.9,
        init_beta: float = 2.0,
        init_std: float = 0.5,
        geom_hidden: int = 32,
        supersample: int = 1,
        use_centroid: bool = True,
        init_depth_peak: float = 1.5,
        output_dim: int | None = None,
        # accepted so a config copied from a basis profile does not fail; unused
        latent_dim: int | None = None,
    ) -> None:
        super().__init__()
        if beam_center is None:
            raise ValueError("QuasiconformalProfileSurrogate needs beam_center")
        if profile_shape is None:
            if output_dim is None:
                raise ValueError("pass profile_shape or output_dim")
            side = int(round(math.sqrt(output_dim)))
            profile_shape = (1, side, side)
        d, h, w = (int(v) for v in profile_shape)

        if output_dim is not None and output_dim != d * h * w:
            raise ValueError(f"output_dim {output_dim} != {d}x{h}x{w}")
        self.profile_shape = (d, h, w)
        self.beam_center = tuple(float(v) for v in beam_center)
        self.lambda_min, self.lambda_max = lambda_min, lambda_max
        self.n_par = n_params(d)
        self.d = self.n_par
        self.supersample = int(supersample)
        self.use_centroid = bool(use_centroid)
        self.learned_prior_scale = bool(learned_prior_scale)
        self.max_log_prior_scale = float(max_log_prior_scale)

        self.register_buffer("prior_sd", self._prior_vector(prior_scale, d), persistent=True)
        self.prior_scale = float(self.prior_sd.mean())

        self.loc_head = nn.Linear(input_dim, self.n_par)
        self.scale_head = nn.Linear(input_dim, self.n_par)
        nn.init.normal_(self.loc_head.weight, std=1e-3)
        nn.init.zeros_(self.loc_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        with torch.no_grad():
            self.scale_head.bias.copy_(torch.log(torch.expm1(init_std * self.prior_sd)))

        theta0 = torch.zeros(self.n_par)
        if d > 1 and init_depth_peak:
            # Start the rocking curve peaked on the central frame. Uniform
            # depth logits are a flat fixed point -- the same one that
            # collapsed the steerable arms -- and on 9b7c the rocking curve
            # spans about one frame, so a flat start is also far from the data.
            centre = (d - 1) / 2.0
            idx = torch.arange(d, dtype=torch.float32)
            theta0[N_PARAMS + 2:] = -init_depth_peak * (idx - centre).abs()
        theta0[LOG_S] = math.log(init_size)
        if not 0 < init_mu < mu_max < 1:
            raise ValueError("need 0 < init_mu < mu_max < 1")
        self.mu_max = float(mu_max)
        theta0[3] = math.atanh(init_mu / mu_max)
        self.theta0 = nn.Parameter(theta0)
        self.shift_head = self._geometry_mlp(geom_hidden, self.n_par)
        self.prior_scale_head = (
            self._geometry_mlp(geom_hidden, self.n_par) if self.learned_prior_scale else None
        )
        lo, hi = 0.5, 4.0
        self._beta_range = (lo, hi)
        self.beta_raw = nn.Parameter(
            torch.tensor(math.log((init_beta - lo) / (hi - init_beta)))
        )

        # pixel-centre offsets from the box centre, x along columns (fast), y along rows
        ss = self.supersample
        sub = (torch.arange(ss) + 0.5) / ss - 0.5
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        xs = (xx.reshape(-1, 1, 1) + sub.reshape(1, 1, ss)).expand(-1, ss, ss)
        ys = (yy.reshape(-1, 1, 1) + sub.reshape(1, ss, 1)).expand(-1, ss, ss)
        self.register_buffer("gx", (xs - (w - 1) / 2).reshape(h * w, ss * ss), persistent=False)
        self.register_buffer("gy", (ys - (h - 1) / 2).reshape(h * w, ss * ss), persistent=False)
        self.R = (min(h, w) - 1) / 2

    # ---- construction helpers ------------------------------------------------

    @staticmethod
    def _prior_vector(prior_scale, depth: int = 1) -> Tensor:
        default = {"t": 1.0, "log_s": 0.5, "mu": 0.5, "curv": 0.2,
                   "drift": 0.3, "depth": 1.0}
        if prior_scale is None:
            spec = default
        elif isinstance(prior_scale, (int, float)):
            spec = {k: float(prior_scale) for k in default}
        else:
            spec = {**default, **{k: float(v) for k, v in dict(prior_scale).items()}}
            unknown = set(spec) - set(default)
            if unknown:
                raise ValueError(f"unknown prior_scale keys {sorted(unknown)}")
        v = torch.empty(n_params(depth))
        v[T] = spec["t"]
        v[LOG_S] = spec["log_s"]
        v[W] = spec["mu"]
        v[CURV] = spec["curv"]
        if depth > 1:
            # drift in pixels per frame; depth logits are dimensionless, and a
            # width of 1.0 lets the rocking curve reach a single frame
            v[DRIFT] = spec["drift"]
            v[N_PARAMS + 2:] = spec["depth"]
        return v

    @staticmethod
    def _geometry_mlp(hidden: int, out_dim: int) -> nn.Module:
        mlp = nn.Sequential(
            nn.Linear(N_GEOM_FEATURES, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
        )
        nn.init.zeros_(mlp[-1].weight)
        nn.init.zeros_(mlp[-1].bias)
        return mlp

    @property
    def beta(self) -> Tensor:
        lo, hi = self._beta_range
        return lo + (hi - lo) * torch.sigmoid(self.beta_raw)

    # ---- the map and the density --------------------------------------------

    def physical(self, theta: Tensor) -> dict[str, Tensor]:
        """Constrained parameters from `theta`, shape (..., 11)."""
        t = torch.complex(theta[..., 0], theta[..., 1])
        s = theta[..., LOG_S].exp()
        w = torch.complex(theta[..., 3], theta[..., 4])
        aw = w.abs()
        # tanh(|w|)/|w| -> 1 as |w| -> 0
        ratio = self.mu_max * torch.where(
            aw > 1e-6, torch.tanh(aw) / aw.clamp_min(1e-6), 1 - aw**2 / 3
        )
        mu = w * ratio
        cc = theta[..., CURV]
        c = torch.complex(cc[..., 0], cc[..., 1])
        d = torch.complex(cc[..., 2], cc[..., 3])
        e = torch.complex(cc[..., 4], cc[..., 5])
        tot = c.abs() + d.abs() + e.abs()
        k = torch.where(
            tot > 1e-6, _CURV_MAX * torch.tanh(tot / _CURV_MAX) / tot.clamp_min(1e-6),
            torch.ones_like(tot),
        )
        out = dict(t=t, s=s, mu=mu, c=c * k, d=d * k, e=e * k)
        if self.profile_shape[0] > 1:
            out["drift"] = torch.complex(theta[..., 11], theta[..., 12])
            out["depth"] = theta[..., N_PARAMS + 2:]
        return out

    def _plane(self, p: dict, zp: Tensor) -> Tensor:
        """Unnormalized planar density at the shifted coordinates `zp`.

        `zp` carries the per-reflection dims followed by any number of grid
        dims, so the parameters are expanded to however many `zp` has: two for
        the 2D path `(S, B, K, ss2)` and three for the per-frame path
        `(S, B, D, K, ss2)`.
        """
        n_extra = zp.dim() - p["mu"].dim()
        ex = lambda t: t.reshape(*t.shape, *([1] * n_extra))  # noqa: E731
        u = zp / self.R
        ub = u.conj()
        c, d, e = (ex(p[k]) for k in ("c", "d", "e"))
        q = u + c * u * u + d * u * ub + e * ub * ub
        q_u = 1 + 2 * c * u + d * ub
        q_ub = d * u + 2 * e * ub
        mu = ex(p["mu"])
        v = (q - mu * q.conj()) / torch.sqrt(1 - mu.abs() ** 2)
        g = v * (self.R / ex(p["s"]))
        beta = self.beta
        alpha = torch.sqrt(
            2 * torch.exp(torch.lgamma(2 / beta) - torch.lgamma(4 / beta))
        )
        det = (q_u.abs() ** 2 - q_ub.abs() ** 2).clamp_min(1e-6)
        return -(g.abs() / alpha).clamp_min(1e-12) ** beta + det.log()

    def density(self, theta: Tensor, phi: Tensor, zc: Tensor, mask: Tensor | None) -> Tensor:
        """Profiles for `theta` (S, B, n_par); returns (S, B, D*H*W)."""
        p = self.physical(theta)
        depth = self.profile_shape[0]
        z = torch.complex(self.gx, self.gy)                          # (K, ss2)
        rot = torch.polar(torch.ones_like(phi), -phi)                # (B,)
        zp0 = rot[None, :, None, None] * (z[None, None] - zc[None, :, None, None])
        zp0 = zp0 - p["t"][..., None, None]

        if depth == 1:
            logw = self._plane(p, zp0)
            logw = logw - logw.amax(dim=(-2, -1), keepdim=True)
            prof = logw.exp().mean(-1)
            prof = prof / prof.sum(-1, keepdim=True).clamp_min(1e-30)
            return renormalize_masked(prof, mask)

        # One planar map per frame, centred on a track linear in the frame
        # index, weighted by a free rocking curve over frames. The weights are
        # a simplex, so the result is normalized over the whole shoebox and the
        # profile can put all of its mass on a single frame -- which is the
        # 9b7c regime, where the rocking curve spans about one frame.
        off = torch.arange(depth, device=zp0.device, dtype=self.gx.dtype)
        off = off - (depth - 1) / 2.0
        drift = p["drift"][..., None, None, None] * off[:, None, None]
        logw = self._plane(p, (zp0[..., None, :, :] - drift))        # (S,B,D,K,ss2)
        logw = logw - logw.amax(dim=(-3, -2, -1), keepdim=True)
        frames = logw.exp().mean(-1)                                 # (S,B,D,K)
        wts = torch.softmax(p["depth"], dim=-1)[..., None]           # (S,B,D,1)
        prof = frames / frames.sum(-1, keepdim=True).clamp_min(1e-30) * wts
        prof = prof.reshape(*prof.shape[:-2], depth * prof.shape[-1])
        return renormalize_masked(prof, mask)

    def _centroid(self, metadata: dict, b: int, like: Tensor) -> Tensor:
        """Predicted centroid relative to the box centre, as x + iy in pixels."""
        if not self.use_centroid or "bbox.0" not in metadata:
            return torch.zeros(b, dtype=torch.complex64, device=like.device)
        _, h, w = self.profile_shape
        x = metadata["xyzcal.px.0"].reshape(-1).float() - metadata["bbox.0"].reshape(-1).float()
        y = metadata["xyzcal.px.1"].reshape(-1).float() - metadata["bbox.2"].reshape(-1).float()
        # xyzcal is in pixel-edge coordinates, so pixel j's centre sits at j + 0.5
        return torch.complex(x - 0.5 - (w - 1) / 2, y - 0.5 - (h - 1) / 2).to(like.device)

    # ---- forward -------------------------------------------------------------

    def forward(
        self, x: Tensor, mc_samples: int = 1, metadata: dict | None = None, **kwargs
    ) -> ProfileSurrogateOutput:
        if metadata is None:
            raise ValueError("QuasiconformalProfileSurrogate needs metadata")
        mask = kwargs.get("mask")
        b = x.shape[0]
        loc = self.loc_head(x)
        scale = F.softplus(self.scale_head(x)) + 1e-6

        feats, phi = geometry_features(
            metadata, self.beam_center, self.lambda_min, self.lambda_max
        )
        feats = feats.to(x.dtype).to(x.device)
        phi = phi.to(x.dtype).to(x.device)
        base = self.theta0 + self.shift_head(feats)                  # (B, 11)
        zc = self._centroid(metadata, b, x)

        h = Normal(loc, scale).rsample([mc_samples])                 # (S, B, 11)
        zp = self.density(base + h, phi, zc, mask)
        mean_profile = self.density((base + loc)[None], phi, zc, mask)[0]

        sd = self.prior_sd.to(x.dtype)
        if self.prior_scale_head is not None:
            sd = sd * self.prior_scale_head(feats).clamp(
                -self.max_log_prior_scale, self.max_log_prior_scale
            ).exp()
        kl = kl_divergence(Normal(loc, scale), Normal(torch.zeros_like(loc), sd)).sum(-1)

        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            loc=loc,
            scale=scale,
            prior_scale=self.prior_scale,
            kl=kl,
        )
