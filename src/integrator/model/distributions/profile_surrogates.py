import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from integrator.model.distributions.profile_geometry import (
    N_GEOM_FEATURES,
    detector_polar,
    geometry_features,
)

logger = logging.getLogger(__name__)


def _peek_basis_latent_dim(path: str) -> int | None:
    """Width of a saved warm-start basis, or None if it cannot be read.

    Read before the decoder is sized, so the basis can set the latent
    dimension rather than being rejected for disagreeing with it. Returns None
    on any failure so construction proceeds and `_apply_warmstart` raises the
    specific error instead.
    """
    try:
        basis = torch.load(path, weights_only=False, map_location="cpu")
        return int(basis["W"].shape[1])
    except Exception:
        return None


def _softplus_inverse(x: float) -> float:
    """Inverse of softplus: log(exp(x) - 1)."""
    return math.log(math.exp(x) - 1.0)


# Filling masked logits with this rather than -inf keeps a fully masked
# shoebox finite: softmax then returns the uniform profile instead of NaN.
_MASK_FILL = -1e9


def _broadcast_mask(mask: Tensor, like: Tensor) -> Tensor:
    """Reshape a shoebox mask to `(B, K)`, adding a sample axis if needed."""
    m = mask.reshape(mask.shape[0], -1)
    return m.unsqueeze(0) if like.dim() == 3 else m


def masked_softmax(logits: Tensor, mask: Tensor | None) -> Tensor:
    """Softmax over the VALID pixels only.

    The likelihood zeroes masked pixels, so profile mass placed there is never
    paid for; the intensity then has to grow by `1 / (mass on valid pixels)` to
    explain the counts that are seen. Normalizing over the valid pixels removes
    that, and the bias it removes is per-reflection and tied to where the
    shoebox sits on the detector, so no downstream per-image scale can absorb
    it.
    """
    if mask is None:
        return F.softmax(logits, dim=-1)
    m = _broadcast_mask(mask, logits).bool()
    return F.softmax(logits.masked_fill(~m, _MASK_FILL), dim=-1)


class _Entmax15(torch.autograd.Function):
    """1.5-entmax: the exact closed form from Peters, Niculae & Martins (2019).

    arXiv:1905.05702. The alpha-entmax family runs from softmax (alpha = 1) to
    sparsemax (alpha = 2); at alpha = 1.5 the solution has a closed form found
    by sorting, and the result is a point ON the simplex with EXACT zeros.

    That is the property a Bragg profile needs and softmax cannot provide.
    `softmax` is strictly positive everywhere, so "no mass here" is
    unreachable, and the gradient d(pi)/d(z) = pi (1 - pi) vanishes as a pixel
    gets small -- the loss stops pushing long before the mass is gone. Spread
    over a 625-pixel box, 2e-4 a pixel that nothing can remove is 12.5% of the
    profile, which is what the softmax families actually show (13.5% beyond
    5 px, against 0.2% for the shapelet family).

    The backward is the known closed form rather than autograd through the
    sort: on the support, the Jacobian of entmax-1.5 is
    diag(s) - s s^T / sum(s) with s = sqrt(y).
    """

    @staticmethod
    def forward(ctx, X: Tensor, dim: int = -1) -> Tensor:
        X = X / 2  # the alpha = 1.5 scaling
        X = X - X.max(dim=dim, keepdim=True).values  # shift for stability
        Xsrt, _ = torch.sort(X, dim=dim, descending=True)

        rho = torch.arange(
            1, X.shape[dim] + 1, device=X.device, dtype=X.dtype
        )
        shape = [1] * X.dim()
        shape[dim] = -1
        rho = rho.reshape(shape)

        mean = Xsrt.cumsum(dim) / rho
        mean_sq = (Xsrt**2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean**2)
        delta = (1 - ss) / rho
        tau = mean - torch.sqrt(torch.clamp(delta, min=0))

        support_size = (tau <= Xsrt).sum(dim=dim, keepdim=True)
        tau_star = tau.gather(dim, support_size - 1)

        Y = torch.clamp(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y)
        ctx.dim = dim
        return Y

    @staticmethod
    def backward(ctx, dY: Tensor):
        (Y,) = ctx.saved_tensors
        dim = ctx.dim
        gppr = Y.sqrt()  # zero off the support, so the gradient stays there
        dX = dY * gppr
        q = dX.sum(dim=dim, keepdim=True) / gppr.sum(dim=dim, keepdim=True).clamp_min(
            1e-12
        )
        dX = dX - q * gppr
        return dX, None


def entmax15(logits: Tensor, dim: int = -1) -> Tensor:
    """`softmax`'s sparse counterpart: sums to 1, but may be exactly 0."""
    return _Entmax15.apply(logits, dim)


def masked_entmax15(logits: Tensor, mask: Tensor | None) -> Tensor:
    """1.5-entmax over the VALID pixels only, as `masked_softmax` but sparse."""
    if mask is None:
        return entmax15(logits, dim=-1)
    m = _broadcast_mask(mask, logits).bool()
    return entmax15(logits.masked_fill(~m, _MASK_FILL), dim=-1)


_NORMALIZERS = {"softmax": masked_softmax, "entmax15": masked_entmax15}


def get_normalizer(name: str):
    """Map a config name to the simplex map the profile is built with."""
    try:
        return _NORMALIZERS[name]
    except KeyError:
        raise ValueError(
            f"unknown profile normalization {name!r}; "
            f"expected one of {sorted(_NORMALIZERS)}"
        ) from None


def renormalize_masked(p: Tensor, mask: Tensor | None) -> Tensor:
    """Drop the masked pixels from an already-normalized profile and rescale.

    For the families that build the profile directly rather than through a
    softmax over logits, and after any warp that moves mass across the box.
    """
    if mask is None:
        return p
    m = _broadcast_mask(mask, p).to(p.dtype)
    p = p * m
    return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _sample_and_decode(
    loc: Tensor,
    scale: Tensor,
    W: Tensor,
    b: Tensor,
    mc_samples: int,
    mask: Tensor | None = None,
    normalize=masked_softmax,
) -> tuple[Tensor, Tensor]:
    """Sample h from q(h|x) and decode to profile vectors.

    Args:
        loc: Posterior mean, shape (B, d).
        scale: Posterior std, shape (B, d).
        W: Decoder weight matrix, shape (K, d).
        b: Decoder bias, shape (K,) or (B, K) for per-bin bias.
        mc_samples: Number of Monte Carlo samples.
        mask: Shoebox mask, shape `(B, K)` or `(B, K, 1)`. When given, the
            profile is normalized over the valid pixels only.

    Returns:
        zp: Profile samples on the simplex, shape (S, B, K).
        mean_profile: Profile at posterior mean h, shape (B, K).
    """
    q_h = Normal(loc, scale)
    h = q_h.rsample([mc_samples])  # (S, B, d)
    logits = h @ W.T + b  # (S, B, K)
    zp = normalize(logits, mask)

    mean_logits = loc @ W.T + b  # (B, K)
    mean_profile = normalize(mean_logits, mask)

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
        kl: Per-reflection KL, shape (B,), for surrogates whose prior is not a
            single scalar scale; `None` leaves the caller to form it from
            `prior_scale`.
    """

    zp: Tensor
    mean_profile: Tensor
    loc: Tensor
    scale: Tensor
    prior_scale: float = 3.0
    kl: Tensor | None = None


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
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int | None = None,
        output_dim: int = 441,
        init_std: float = 0.5,
        prior_scale: float = 3.0,
        # Warm-start the decoder from a precomputed basis. Without it the
        # decoder starts near uniform, and on Laue shoeboxes (~96.6%
        # background) a uniform profile makes intensity x profile a flat field
        # the model cannot tell from the background -- the documented
        # identifiability collapse. `prepare_profile_basis` writes the file.
        warmstart_basis_path: str | None = None,
        freeze_bias: bool = False,
        # How the decoded logits are mapped onto the simplex. `softmax` is
        # strictly positive, so it cannot place exactly zero mass on a pixel
        # and its gradient pi (1 - pi) stops pushing long before a small tail
        # is gone; `entmax15` can, and does so with exact zeros.
        normalization: str = "softmax",
    ) -> None:
        super().__init__()
        self.normalization = normalization
        self._normalize = get_normalizer(normalization)

        if latent_dim is None:
            latent_dim = 8
        if warmstart_basis_path is not None:
            # A warm-start basis fixes its own width, so it wins over the
            # requested latent_dim. There is no configuration where both can
            # hold: a (output_dim, d) basis cannot load into a decoder of a
            # different width. The old behaviour raised and advised dropping
            # latent_dim from the config, which is impossible when the value
            # comes from BaseIntegrator.DEFAULT_SURROGATES rather than the
            # config -- as it does for every learned_basis_profile run.
            basis_d = _peek_basis_latent_dim(warmstart_basis_path)
            if basis_d is not None and basis_d != latent_dim:
                logger.info(
                    "warmstart basis %s has d=%d; using it instead of "
                    "latent_dim=%d",
                    Path(warmstart_basis_path).name,
                    basis_d,
                    latent_dim,
                )
                latent_dim = basis_d
        self.d: int = latent_dim
        self.prior_scale = prior_scale

        self.loc_head = nn.Linear(input_dim, self.d)
        self.scale_head = nn.Linear(input_dim, self.d)
        self.decoder = nn.Linear(self.d, output_dim)
        if warmstart_basis_path is not None:
            self._apply_warmstart(warmstart_basis_path, output_dim)
        if freeze_bias:
            self.decoder.bias.requires_grad_(False)

        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, _softplus_inverse(init_std))


    def _apply_warmstart(self, path: str, output_dim: int) -> None:
        """Copy a precomputed (W, b) basis into the decoder.

        Shapes are checked rather than broadcast: a basis built for a different
        shoebox would otherwise warm-start the model into a quietly wrong
        profile.
        """
        basis = torch.load(path, weights_only=False)
        W, b = basis["W"].float(), basis["b"].float()
        if W.shape[0] != output_dim or b.shape[0] != output_dim:
            raise ValueError(
                f"warmstart basis from {path!r} has K={W.shape[0]} (bias "
                f"{b.shape[0]}) but output_dim={output_dim}"
            )
        if W.shape[1] != self.d:
            raise ValueError(
                f"warmstart basis has d={W.shape[1]} but latent_dim={self.d}; "
                f"drop latent_dim from the config to infer it from the basis"
            )
        with torch.no_grad():
            self.decoder.weight.data.copy_(W)
            self.decoder.bias.data.copy_(b)

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        **kwargs,
    ) -> ProfileSurrogateOutput:
        loc = self.loc_head(x)  # (B, d)
        # +1e-6 because a plain softplus UNDERFLOWS to exactly 0.0 in
        # float32 once the pre-activation goes far negative, and
        # Normal rejects scale == 0. Adding the floor rather than
        # clamping keeps the gradient alive below it. Same fix as
        # ShapeletProfile below and folded_normal.py.
        scale = F.softplus(self.scale_head(x)) + 1e-6  # (B, d)

        zp, mean_profile = _sample_and_decode(
            loc,
            scale,
            self.decoder.weight,
            self.decoder.bias,
            mc_samples,
            mask=kwargs.get("mask"),
            normalize=self._normalize,
        )
        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            loc=loc,
            scale=scale,
            prior_scale=self.prior_scale,
        )


def _monomial_basis(
    shape: tuple[int, int, int], max_order: int = 2
) -> tuple[Tensor, list[tuple[int, int, int]]]:
    """Monomials in normalized pixel coordinates, constant term excluded.

    The constant is left out because softmax is invariant to it and the decoder bias
    already carries whatever shape every reflection shares. Columns are scaled to
    unit norm so the latent lives in fixed logit units.

    Returns the `(P, d)` basis and the exponent tuple of each column.
    """
    d, h, w = shape
    axes = [
        torch.linspace(-1.0, 1.0, n) if n > 1 else torch.zeros(1)
        for n in (d, h, w)
    ]
    zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
    coords = {"z": zz.reshape(-1), "y": yy.reshape(-1), "x": xx.reshape(-1)}

    exponents = [
        (i, j, k)
        for i in range(max_order + 1)
        for j in range(max_order + 1)
        for k in range(max_order + 1)
        if 1 <= i + j + k <= max_order
        and (d > 1 or i == 0)
        and (h > 1 or j == 0)
        and (w > 1 or k == 0)
    ]
    cols = [
        coords["z"].pow(i) * coords["y"].pow(j) * coords["x"].pow(k)
        for i, j, k in exponents
    ]
    basis = torch.stack(cols, dim=-1)
    return basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-12), exponents


class FixedBasisProfileSurrogate(ProfileSurrogate):
    """Learned-basis profile with the basis FROZEN to an analytic polynomial family.

    `softmax(W h + b)` with a learned `W` carries a gauge: `h -> h/c`, `W -> cW`
    leaves the likelihood unchanged. Nothing then pins the scale of `h`, so the KL
    weight, the prior scale, and any penalty written in terms of `W` are not
    well-posed quantities. Freezing `W` removes the gauge and puts `h` in logit
    units.

    It also bounds which profiles are reachable. A free softmax over `P` pixels has
    two attractors, one-hot and uniform, and falls into both; with quadratic columns
    the family is the anisotropic Gaussians, so a near-delta needs a coefficient of
    order `P^(2/3)` logits and uniform needs every coefficient at zero against a
    bias that the likelihood is pushing away from flat.

    The linear columns are a sub-pixel translation and the quadratic ones the inverse
    covariance, so `max_order=2` is a Gaussian whose centre and anisotropy the
    encoder predicts. Higher orders add skew at the cost of reopening sharper
    features.

    Args:
        max_order: Highest total monomial degree. 2 gives the Gaussians.
        profile_shape: Profile grid as `(D, H, W)`; inferred as a square when unset.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 441,
        init_std: float = 0.5,
        prior_scale: float = 1.0,
        max_order: int = 2,
        profile_shape: tuple[int, int, int] | list[int] | None = None,
        **kwargs,
    ) -> None:
        if profile_shape is not None:
            shape = tuple(int(v) for v in profile_shape)
        else:
            side = int(round(math.sqrt(output_dim)))
            if side * side != output_dim:
                raise ValueError(
                    f"output_dim {output_dim} is not square; pass profile_shape"
                )
            shape = (1, side, side)
        if shape[0] * shape[1] * shape[2] != output_dim:
            raise ValueError(
                f"profile_shape {shape} has {shape[0] * shape[1] * shape[2]} "
                f"pixels, but output_dim is {output_dim}"
            )

        basis, exponents = _monomial_basis(shape, max_order)
        super().__init__(
            input_dim=input_dim,
            latent_dim=basis.shape[1],
            output_dim=output_dim,
            init_std=init_std,
            prior_scale=prior_scale,
            **kwargs,
        )
        self.profile_shape = shape
        self.max_order = int(max_order)
        self.exponents = exponents

        with torch.no_grad():
            self.decoder.weight.copy_(basis)
            nn.init.zeros_(self.decoder.bias)
        self.decoder.weight.requires_grad_(False)


def _hermite_1d(u: Tensor, n_max: int) -> Tensor:
    """Orthonormal Hermite functions `phi_0..phi_n_max` at `u`, stacked on dim -2.

    Uses the stable three-term recursion rather than explicit Hermite polynomials,
    which overflow by order 6. The `1/sqrt(beta)` normalization of the continuous
    basis is omitted: it is common to every column, so it cancels when the profile
    is normalized over the grid.
    """
    phi = [math.pi**-0.25 * torch.exp(-0.5 * u * u)]
    if n_max >= 1:
        phi.append(math.sqrt(2.0) * u * phi[0])
    for n in range(1, n_max):
        phi.append(
            math.sqrt(2.0 / (n + 1)) * u * phi[n]
            - math.sqrt(n / (n + 1)) * phi[n - 1]
        )
    return torch.stack(phi, dim=-2)


class ShapeletProfileSurrogate(nn.Module):
    """Squared Gauss-Hermite profile in a geometry-aligned anisotropic frame.

        pi ~ (sum_k c_k Phi_k)^2,   c = (1, eps)/sqrt(1 + |eps|^2),   eps ~ q

    Three properties the learned-softmax profile does not have.

    **It cannot be uniform or a near-delta.** The Gaussian envelope bounds the
    support from above and the Christoffel inequality `pi <= sum_n phi_n^2` bounds
    the peak from below, so both attractors of a free softmax are outside the
    family. Truncation order alone does not do this -- the envelope and the
    order-decaying prior do.

    **It has no gauge.** The basis is analytic and the coefficient vector is
    normalized, with `c_0` fixed at 1, so there is no `h -> h/c, W -> cW` freedom
    and the KL weight and prior scale are well-posed for the first time.

    **Its coefficients mean something.** Order 1 is a centroid shift, order 2 an
    ellipticity, order 3 the skew -- which is what a wavelength-dependent Laue
    streak needs and a symmetric Gaussian cannot express. Each order therefore gets
    its own prior width, `tau_n = tau0 (1+n)^-decay`. The decay is load-bearing, not
    cosmetic: with a flat prior across orders the degeneracy returns as the order
    rises.

    The frame -- azimuth, centre and the two extents -- is predicted from detector
    geometry rather than inferred per reflection, because a free per-reflection
    scale re-admits the `(beta, c)` non-identification that the analytic basis
    otherwise removes.

    Args:
        n_par: Hermite order along the streak (azimuthal) direction.
        n_perp: Hermite order across it.
        n_z: Hermite order along the frame axis; 0 for 2D stills.
        beta_par: Envelope scale along the streak, in pixels, before the
            geometry-predicted correction.
        beta_perp: Envelope scale across it.
        beta_z: Envelope scale along frames.
        max_log_beta: Bound on the geometry correction to each log extent.
        tau0: Prior width of the order-1 coefficients.
        tau_decay: Exponent in `tau_n = tau0 (1+n)^-decay`.
        quad: Gauss-Legendre points per axis for pixel integration. 1 samples pixel
            centres, which costs ~0.3 in basis error once an extent falls below a
            pixel; 3 brings that to ~4e-4.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 441,
        profile_shape: tuple[int, int, int] | list[int] | None = None,
        beam_center: tuple[float, float] | list[float] | None = None,
        n_par: int = 4,
        n_perp: int = 2,
        n_z: int = 0,
        beta_par: float = 2.0,
        beta_perp: float = 1.2,
        beta_z: float = 1.0,
        max_log_beta: float = 1.0,
        tau0: float = 0.5,
        tau_decay: float = 1.5,
        init_std: float = 0.1,
        quad: int = 3,
        geom_hidden: int = 32,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if profile_shape is not None:
            shape = tuple(int(v) for v in profile_shape)
        else:
            side = int(round(math.sqrt(output_dim)))
            if side * side != output_dim:
                raise ValueError(
                    f"output_dim {output_dim} is not square; pass profile_shape"
                )
            shape = (1, side, side)
        if shape[0] * shape[1] * shape[2] != output_dim:
            raise ValueError(
                f"profile_shape {shape} has {shape[0] * shape[1] * shape[2]} "
                f"pixels, but output_dim is {output_dim}"
            )
        self.profile_shape = shape
        self.output_dim = int(output_dim)

        if shape[0] == 1:
            n_z = 0
        self.n_par, self.n_perp, self.n_z = int(n_par), int(n_perp), int(n_z)
        self.n_coeff = (self.n_par + 1) * (self.n_perp + 1) * (self.n_z + 1)
        self.d = self.n_coeff - 1

        self.beta_par, self.beta_perp, self.beta_z = (
            float(beta_par), float(beta_perp), float(beta_z),
        )
        self.max_log_beta = float(max_log_beta)
        self.quad = max(1, int(quad))
        self.beam_center = tuple(beam_center) if beam_center is not None else None
        self.lambda_min, self.lambda_max = lambda_min, lambda_max
        self.prior_scale = float(tau0)

        self.loc_head = nn.Linear(input_dim, self.d)
        self.scale_head = nn.Linear(input_dim, self.d)
        nn.init.zeros_(self.loc_head.weight)
        nn.init.zeros_(self.loc_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, _softplus_inverse(init_std))

        # the frame is a function of geometry, never a per-reflection latent
        n_frame = 4 + (1 if self.n_z > 0 else 0)
        self.frame_head = None
        if beam_center is not None:
            mlp = nn.Sequential(
                nn.Linear(N_GEOM_FEATURES, geom_hidden),
                nn.SiLU(),
                nn.Linear(geom_hidden, n_frame),
            )
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
            self.frame_head = mlp

        orders = torch.tensor(
            [
                i + j + k
                for i in range(self.n_par + 1)
                for j in range(self.n_perp + 1)
                for k in range(self.n_z + 1)
            ],
            dtype=torch.float32,
        )
        tau = float(tau0) * (1.0 + orders).pow(-float(tau_decay))
        self.register_buffer("tau", tau[1:].clone(), persistent=True)

        self.register_buffer("_grid", self._pixel_grid(shape), persistent=False)
        pts, wts = np.polynomial.legendre.leggauss(self.quad)
        self.register_buffer(
            "_qp", torch.tensor(pts / 2.0, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_qw", torch.tensor(wts / 2.0, dtype=torch.float32), persistent=False
        )

    @staticmethod
    def _pixel_grid(shape: tuple[int, int, int]) -> Tensor:
        """Pixel-centre offsets from the grid centre, shape (3, P) as (z, y, x)."""
        d, h, w = shape
        zz, yy, xx = torch.meshgrid(
            torch.arange(d, dtype=torch.float32) - (d - 1) / 2,
            torch.arange(h, dtype=torch.float32) - (h - 1) / 2,
            torch.arange(w, dtype=torch.float32) - (w - 1) / 2,
            indexing="ij",
        )
        return torch.stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], 0)

    def _frame(self, features: Tensor | None, batch: int, device, dtype):
        """Centre offset and log extents, from geometry when it is available."""
        zeros = torch.zeros(batch, 1, device=device, dtype=dtype)
        if self.frame_head is None or features is None:
            dx = dy = zeros
            lp = lq = zeros
            lz = zeros
        else:
            raw = self.frame_head(features)
            dx, dy = raw[:, 0:1], raw[:, 1:2]
            lp = self.max_log_beta * torch.tanh(raw[:, 2:3])
            lq = self.max_log_beta * torch.tanh(raw[:, 3:4])
            lz = (
                self.max_log_beta * torch.tanh(raw[:, 4:5])
                if raw.shape[1] > 4
                else zeros
            )
        return dx, dy, lp, lq, lz

    def _amplitude(self, c: Tensor, phi: Tensor, frame, offset) -> Tensor:
        """`sum_k c_k Phi_k` on the grid for one quadrature offset, shape (S, B, P)."""
        dx, dy, lp, lq, lz = frame
        ox, oy = offset
        grid = self._grid
        gz, gy, gx = grid[0], grid[1], grid[2]

        ddx = gx.unsqueeze(0) - dx - ox
        ddy = gy.unsqueeze(0) - dy - oy
        cos, sin = torch.cos(phi).unsqueeze(-1), torch.sin(phi).unsqueeze(-1)
        u = (cos * ddx + sin * ddy) / (self.beta_par * torch.exp(lp))
        v = (-sin * ddx + cos * ddy) / (self.beta_perp * torch.exp(lq))

        pu = _hermite_1d(u, self.n_par)
        pv = _hermite_1d(v, self.n_perp)
        if self.n_z > 0:
            w = gz.unsqueeze(0) / (self.beta_z * torch.exp(lz))
            pw = _hermite_1d(w, self.n_z)
            return torch.einsum("bip,bjp,bkp,sbijk->sbp", pu, pv, pw, c)
        return torch.einsum("bip,bjp,sbij->sbp", pu, pv, c[..., 0])

    def _profile(self, eps: Tensor, phi: Tensor, frame) -> Tensor:
        """Pixel-integrated squared amplitude, normalized, shape (S, B, P)."""
        s, b = eps.shape[0], eps.shape[1]
        ones = eps.new_ones(s, b, 1)
        c = torch.cat([ones, eps], dim=-1)
        c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        c = c.reshape(s, b, self.n_par + 1, self.n_perp + 1, self.n_z + 1)

        p = None
        for ix, wx in zip(self._qp, self._qw):
            for iy, wy in zip(self._qp, self._qw):
                g = self._amplitude(c, phi, frame, (ix, iy))
                term = (wx * wy) * g * g
                p = term if p is None else p + term
        return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        metadata: dict | None = None,
        **kwargs,
    ) -> ProfileSurrogateOutput:
        loc = self.loc_head(x)
        scale = F.softplus(self.scale_head(x)) + 1e-6

        phi = x.new_zeros(x.shape[0])
        features = None
        if self.frame_head is not None:
            if metadata is None:
                raise ValueError(
                    "ShapeletProfileSurrogate needs metadata when beam_center is set"
                )
            features, phi = geometry_features(
                metadata, self.beam_center, self.lambda_min, self.lambda_max
            )
            features = features.to(loc.dtype).to(loc.device)
            phi = phi.to(loc.dtype).to(loc.device)

        frame = self._frame(features, x.shape[0], loc.device, loc.dtype)

        eps = loc.unsqueeze(0) + scale.unsqueeze(0) * torch.randn(
            mc_samples, *loc.shape, device=loc.device, dtype=loc.dtype
        )
        mask = kwargs.get("mask")
        zp = renormalize_masked(self._profile(eps, phi, frame), mask)
        mean_profile = renormalize_masked(
            self._profile(loc.unsqueeze(0), phi, frame).squeeze(0), mask
        )

        tau = self.tau.to(loc.dtype)
        kl = (
            torch.log(tau / scale)
            + (scale.pow(2) + loc.pow(2)) / (2 * tau.pow(2))
            - 0.5
        ).sum(dim=-1)

        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            loc=loc,
            scale=scale,
            prior_scale=self.prior_scale,
            kl=kl,
        )


class PhysicalProfileSurrogate(nn.Module):
    """Profile computed from diffraction geometry, with the physics fitted globally.

    The data do not identify the intensity: `pi' = (f/f')pi + (1-f/f')u` gives an
    identical likelihood at every signal level, so `I` is pinned only by the profile
    family. A profile with per-reflection freedom in the uniform and delta directions
    therefore leaves `I` unidentified, and a free `softmax(Wh + b)` falls into both
    of those directions in practice.

    This surrogate removes that freedom. Each reflection's profile is *computed* from
    its own geometry; the only learned quantities are a handful of numbers shared by
    the whole dataset, which bright reflections pin down. There is no per-reflection
    latent and no basis matrix, so there is no `h -> h/c, W -> cW` gauge either.

    The form is the one that convolution of near-Gaussian kernels gives, with
    mosaicity entering through the scattering angle and divergence, crystal size and
    detector PSF as a floor:

        sigma_tangential^2 = a sin^2(theta) + b
        sigma_radial^2     = a sin^2(theta) + b + c

    `a` carries the mosaic spread, `b` the angle-independent terms, and `c` the
    wavelength spread, which acts along the radial direction only and is therefore
    zero for monochromatic data. Measured on real bright reflections this holds with
    R^2 0.96-0.97 on HEWL Laue and 0.56-0.59 on monochromatic HEWL; the fitted radial
    excess is 2.59 px^2 on the Laue data against 0.06 px^2 monochromatic, which is
    what `c` represents.

    Args:
        beam_center: Detector beam centre in pixels; sets the radial direction.
        profile_shape: Profile grid as `(D, H, W)`.
        init_a, init_b, init_c: Starting values in px^2. Defaults are the measured
            Laue values; a method-of-moments fit on bright reflections is a better
            start, and the globals are cheap to refit.
        init_z: Starting variance along the frame axis, for 3D data.
        quad: Gauss-Legendre points per axis for pixel integration. Spots can be
            narrower than a pixel -- monochromatic HEWL sits at sigma 0.76-1.27 px --
            so sampling pixel centres is not adequate there.
        learn_globals: Fit the globals through the ELBO. False freezes them, which is
            the warm-start-and-hold used for the first epoch.
        lambda_min, lambda_max: Spectrum band from the dataset manifest. Used only
            to supply a wavelength when the batch carries none.
    """

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 441,
        profile_shape: tuple[int, int, int] | list[int] | None = None,
        beam_center: tuple[float, float] | list[float] | None = None,
        init_a: float = 250.0,
        init_b: float = 3.0,
        init_c: float = 2.6,
        init_z: float = 1.0,
        wavelength: float | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
        quad: int = 2,
        learn_globals: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        if profile_shape is not None:
            shape = tuple(int(v) for v in profile_shape)
        else:
            side = int(round(math.sqrt(output_dim)))
            if side * side != output_dim:
                raise ValueError(
                    f"output_dim {output_dim} is not square; pass profile_shape"
                )
            shape = (1, side, side)
        if shape[0] * shape[1] * shape[2] != output_dim:
            raise ValueError(
                f"profile_shape {shape} has {shape[0] * shape[1] * shape[2]} "
                f"pixels, but output_dim is {output_dim}"
            )
        self.profile_shape = shape
        self.output_dim = int(output_dim)
        self.is_3d = shape[0] > 1
        self.beam_center = tuple(beam_center) if beam_center is not None else None
        # a monochromatic batch carries no per-reflection wavelength, so fall back
        # to the band centre when the dataset manifest supplies one
        if wavelength is None and None not in (lambda_min, lambda_max):
            wavelength = 0.5 * (float(lambda_min) + float(lambda_max))
        self.wavelength = wavelength
        self.quad = max(1, int(quad))
        self.prior_scale = 1.0

        def _p(v):
            return nn.Parameter(
                torch.tensor(math.log(max(float(v), 1e-6))), requires_grad=learn_globals
            )

        self.log_a = _p(init_a)
        self.log_b = _p(init_b)
        self.log_c = _p(init_c)
        self.log_z = _p(init_z)

        self.register_buffer("_grid", self._pixel_grid(shape), persistent=False)
        pts, wts = np.polynomial.legendre.leggauss(self.quad)
        self.register_buffer("_qp", torch.tensor(pts / 2.0, dtype=torch.float32), persistent=False)
        self.register_buffer("_qw", torch.tensor(wts / 2.0, dtype=torch.float32), persistent=False)

    @staticmethod
    def _pixel_grid(shape: tuple[int, int, int]) -> Tensor:
        """Pixel-centre offsets from the grid centre, shape (3, P) as (z, y, x)."""
        d, h, w = shape
        zz, yy, xx = torch.meshgrid(
            torch.arange(d, dtype=torch.float32) - (d - 1) / 2,
            torch.arange(h, dtype=torch.float32) - (h - 1) / 2,
            torch.arange(w, dtype=torch.float32) - (w - 1) / 2,
            indexing="ij",
        )
        return torch.stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], 0)

    def widths(self, metadata: dict, device, dtype):
        """Radial, tangential and frame-axis variances, in px^2, per reflection."""
        d_sp = metadata["d"].reshape(-1).to(device=device, dtype=dtype).clamp_min(1e-6)
        if "wavelength" in metadata:
            lam = metadata["wavelength"].reshape(-1).to(device=device, dtype=dtype)
        elif self.wavelength is not None:
            lam = torch.full_like(d_sp, float(self.wavelength))
        else:
            raise ValueError(
                "PhysicalProfileSurrogate needs metadata['wavelength'] or a "
                "`wavelength` argument to form sin(theta)"
            )
        sin2 = (lam / (2.0 * d_sp)).clamp(0.0, 1.0).pow(2)

        a, b = self.log_a.exp(), self.log_b.exp()
        # the wavelength term acts along the radial direction only, and is absent
        # when the beam is monochromatic
        c = self.log_c.exp() if "wavelength" in metadata else torch.zeros_like(a)
        v_tan = a * sin2 + b
        v_rad = v_tan + c
        return v_rad, v_tan, self.log_z.exp().expand_as(v_tan)

    def _profile(self, metadata: dict, mask: Tensor) -> Tensor:
        """Pixel-integrated oriented Gaussian on the shoebox grid, shape (B, P)."""
        device, dtype = mask.device, torch.float32
        v_rad, v_tan, v_z = self.widths(metadata, device, dtype)
        b = v_rad.shape[0]

        if self.beam_center is not None:
            _, _, phi = detector_polar(metadata, self.beam_center)
            phi = phi.to(device=device, dtype=dtype)
        else:
            phi = torch.zeros(b, device=device, dtype=dtype)

        gz, gy, gx = self._grid[0], self._grid[1], self._grid[2]
        cos, sin = torch.cos(phi).unsqueeze(-1), torch.sin(phi).unsqueeze(-1)
        vr, vt, vz = (v.unsqueeze(-1).clamp_min(1e-6) for v in (v_rad, v_tan, v_z))

        acc = None
        for ix, wx in zip(self._qp, self._qw):
            for iy, wy in zip(self._qp, self._qw):
                dx = gx.unsqueeze(0) - ix
                dy = gy.unsqueeze(0) - iy
                u = cos * dx + sin * dy
                v = -sin * dx + cos * dy
                e = u * u / vr + v * v / vt
                if self.is_3d:
                    e = e + gz.unsqueeze(0).pow(2) / vz
                term = (wx * wy) * torch.exp(-0.5 * e)
                acc = term if acc is None else acc + term

        acc = acc * mask.reshape(b, -1).to(dtype)
        return acc / acc.sum(-1, keepdim=True).clamp_min(1e-12)

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        metadata: dict | None = None,
        mask: Tensor | None = None,
        **kwargs,
    ) -> ProfileSurrogateOutput:
        if metadata is None:
            raise ValueError("PhysicalProfileSurrogate needs metadata")
        if mask is None:
            mask = torch.ones(x.shape[0], self.output_dim, device=x.device)

        profile = self._profile(metadata, mask)
        b = profile.shape[0]
        zeros = profile.new_zeros(b, 1)

        return ProfileSurrogateOutput(
            # the profile is deterministic given the geometry, so every draw is
            # the same; the ELBO then carries no profile entropy, which is correct
            zp=profile.unsqueeze(0).expand(mc_samples, -1, -1),
            mean_profile=profile,
            loc=zeros,
            scale=profile.new_ones(b, 1),
            prior_scale=self.prior_scale,
            kl=profile.new_zeros(b),
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
        normalization: str = "softmax",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            init_std=init_std,
            prior_scale=prior_scale,
            normalization=normalization,
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
        # +1e-6 because a plain softplus UNDERFLOWS to exactly 0.0 in
        # float32 once the pre-activation goes far negative, and
        # Normal rejects scale == 0. Adding the floor rather than
        # clamping keeps the gradient alive below it. Same fix as
        # ShapeletProfile below and folded_normal.py.
        scale = F.softplus(self.scale_head(x)) + 1e-6

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

        mask = kwargs.get("mask")
        zp, mean_profile = _sample_and_decode(
            decode_loc,
            scale,
            self.decoder.weight,
            self.decoder.bias,
            mc_samples,
            # NOT the mask: when steering is on the decoder writes to the
            # CANONICAL grid, which is larger than the output box (37x37
            # against 25x25 on hewl_1118), and the mask indexes the output.
            # The mask is applied below instead, once `_warp` has brought the
            # profile back onto the output grid.
            normalize=self._normalize,
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

        # On the output grid either way, so this is the one place the mask
        # can be applied for both the warped and unwarped paths.
        zp = renormalize_masked(zp, mask)
        mean_profile = renormalize_masked(mean_profile, mask)

        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            loc=loc,
            scale=scale,
            prior_scale=self.prior_scale,
        )
