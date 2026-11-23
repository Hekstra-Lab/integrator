import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, kl_divergence

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


def _border_median(counts: Tensor, shape: tuple[int, int, int]) -> Tensor:
    """Median of the shoebox border, per reflection, shape `(..., 1)`.

    Profile-free by construction, so using it to background-subtract before a
    projection cannot leak the shape being estimated.
    """
    d, h, w = shape
    idx = torch.arange(h * w, device=counts.device).reshape(h, w)
    edge = torch.zeros(h, w, dtype=torch.bool, device=counts.device)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    sel = idx[edge].reshape(-1)
    per_frame = counts.reshape(counts.shape[0], d, h * w)
    return per_frame[:, :, sel].median(-1).values.mean(-1, keepdim=True)


def _lin(n: int, device) -> Tensor:
    """`n` points spanning [-1, 1], the coordinate convention `grid_sample` uses."""
    return torch.linspace(-1.0, 1.0, n, device=device)


def _softplus_inverse_tensor(x: Tensor) -> Tensor:
    """Inverse of softplus, elementwise.

    `log(exp(x) - 1)` overflows for large `x` and loses every digit for small
    `x`; `x + log1p(-exp(-x))` is exact in both tails, which matters because a
    sqrt-density initializer spans many orders of magnitude.
    """
    return x + torch.log1p(-torch.exp(-x))


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
) -> tuple[Tensor, Tensor, Tensor]:
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
        mean_logits: Pre-normalization logits at posterior mean h, shape
            (B, K). `SVAEMergingIntegrator` forms its per-pixel responsibility
            from these; a per-row shift is immaterial there because its gate
            carries one.
    """
    q_h = Normal(loc, scale)
    h = q_h.rsample([mc_samples])  # (S, B, d)
    logits = h @ W.T + b  # (S, B, K)
    zp = normalize(logits, mask)

    mean_logits = loc @ W.T + b  # (B, K)
    mean_profile = normalize(mean_logits, mask)

    return zp, mean_profile, mean_logits


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
        mean_logits: Pre-normalization logits at the posterior mean, shape
            (B, K). `None` for surrogates that do not decode through logits.
    """

    zp: Tensor
    mean_profile: Tensor
    loc: Tensor
    scale: Tensor
    prior_scale: float = 3.0
    kl: Tensor | None = None
    mean_logits: Tensor | None = None


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
        # Mean-squared finite difference penalty on the decoder basis W,
        # applied by the integrator (the surrogate only carries the weight).
        # June's merging runs used 0.02; this branch dropped the term, and it
        # is the one config key from those runs with nowhere to go.
        smoothness_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.normalization = normalization
        self.smoothness_weight = float(smoothness_weight)
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

        zp, mean_profile, mean_logits = _sample_and_decode(
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
            mean_logits=mean_logits,
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

    def profile_and_grad(
        self,
        metadata: dict,
        mask: Tensor,
        offset: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Profile at a shifted centre, and its derivative in that shift.

        `_profile` puts the Gaussian at the grid centre, so a reflection whose
        predicted centroid is wrong gets its profile in the wrong place and its
        intensity biased low. This evaluates the same profile at a per-reflection
        offset and returns the two derivatives needed to solve for that offset.

        The offset is in the `(radial, tangential)` frame rather than in
        detector pixels, which is what makes the derivative cheap: rotating the
        grid gives `u = u0 - mu_r` and `v = v0 - mu_t`, so

            d(term)/d(mu_r) = term * u / v_rad,
            d(term)/d(mu_t) = term * v / v_tan,

        and the two directions are the ones the widths are expressed in, so a
        prior on the shift can be set per direction.

        The returned derivative is of the NORMALIZED profile, which makes each
        column sum to zero over the valid pixels:

            sum_j p_j = 1  =>  sum_j dp_j/dmu = d/dmu (1) = 0.

        That is what keeps the intensity identified when the columns are added
        to the design. A column with a non-zero sum trades against `I` -- the
        uniform direction of `p' = (f/f')p + (1-f/f')u` -- while a zero-sum
        column cannot move total mass and so carries only the shift.

        The quadrature loop is written out again rather than shared with
        `_profile`, so that the no-offset path the trained models use stays
        bit-identical.

        Args:
            metadata: Needs `d`, `wavelength` and, for the radial direction,
                whatever `detector_polar` reads.
            mask: Valid-pixel mask, shape `(B, P)`.
            offset: Shift in `(radial, tangential)` pixels for a still, or
                `(radial, tangential, frame)` for a rotation series, shape
                `(B, n_shift_directions)`; zero when omitted.

        Returns:
            `(profile, grad)` of shapes `(B, P)` and `(B, S, P)`, the latter
            ordered `(radial, tangential[, frame])` with `S` given by
            `n_shift_directions`.
        """
        device, dtype = mask.device, torch.float32
        v_rad, v_tan, v_z = self.widths(metadata, device, dtype)
        b = v_rad.shape[0]

        if self.beam_center is not None:
            _, _, phi = detector_polar(metadata, self.beam_center)
            phi = phi.to(device=device, dtype=dtype)
        else:
            phi = torch.zeros(b, device=device, dtype=dtype)

        n_dir = self.n_shift_directions
        if offset is None:
            off = torch.zeros(b, n_dir, device=device, dtype=dtype)
        else:
            off = offset.to(device=device, dtype=dtype).reshape(b, n_dir)
        mu_r, mu_t = off[:, 0:1], off[:, 1:2]
        mu_z = off[:, 2:3] if n_dir > 2 else None

        gz, gy, gx = self._grid[0], self._grid[1], self._grid[2]
        cos, sin = torch.cos(phi).unsqueeze(-1), torch.sin(phi).unsqueeze(-1)
        vr, vt, vz = (v.unsqueeze(-1).clamp_min(1e-6) for v in (v_rad, v_tan, v_z))

        q = None
        acc_dir: list[Tensor | None] = [None] * n_dir
        for ix, wx in zip(self._qp, self._qw):
            for iy, wy in zip(self._qp, self._qw):
                dx = gx.unsqueeze(0) - ix
                dy = gy.unsqueeze(0) - iy
                u = cos * dx + sin * dy - mu_r
                v = -sin * dx + cos * dy - mu_t
                e = u * u / vr + v * v / vt
                dz = None
                if self.is_3d:
                    # the frame axis is not rotated by phi: it is the rocking
                    # direction, orthogonal to the detector face
                    w_z = gz.unsqueeze(0)
                    if mu_z is not None:
                        w_z = w_z - mu_z
                    e = e + w_z * w_z / vz
                term = (wx * wy) * torch.exp(-0.5 * e)
                contrib = [term * u / vr, term * v / vt]
                if n_dir > 2:
                    dz = term * w_z / vz
                    contrib.append(dz)
                q = term if q is None else q + term
                for j, cj in enumerate(contrib):
                    acc_dir[j] = cj if acc_dir[j] is None else acc_dir[j] + cj

        m = mask.reshape(b, -1).to(dtype)
        q = q * m
        s = q.sum(-1, keepdim=True).clamp_min(1e-12)
        p = q / s
        # d/dmu [q/S] = [dq/dmu - p * sum_j dq_j/dmu] / S, which sums to zero
        grads = []
        for g in acc_dir:
            gm = g * m
            grads.append((gm - p * gm.sum(-1, keepdim=True)) / s)
        return p, torch.stack(grads, dim=1)

    @property
    def n_shift_directions(self) -> int:
        """Shift directions this profile exposes: `(radial, tangential[, frame])`.

        Two for a single-frame shoebox, three for a rotation series, where the
        reflection can also be mis-centred along the rocking direction because
        the predicted `phi` is wrong. A still has no third direction to be wrong
        in, so asking for one would add an unidentified column.
        """
        return 3 if self.is_3d else 2

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
        latent_gain: Scales the per-reflection latent `h`, which is the only
            path by which this reflection's own pixels reach its profile. The
            default 1.0 leaves the decode bit-identical. Setting 0.0 makes the
            profile a pure function of detector geometry, `softmax(m(s) W^T +
            b)`, which is the controlled version of the accidental
            geometry-only arm that [[svae-only-exists-on-sim]]'s dead ReLU
            encoder produced -- and the arm the pixel-access comparison needs,
            since that one confounded encoder liveness with the integrator.
        steer: Warp the decoded profile by a geometry-predicted affine.
        geom_hidden: Width of the geometry MLPs.
        max_log_extent: Bound on `|log extent|`, so extents stay in
            `[exp(-max), exp(max)]` and the warp cannot collapse the profile.
            It bounds each extent on its own, so it does NOT bound the
            anisotropy: the ratio `s_par/s_perp` is free up to
            `exp(2 * max_log_extent)`.
        max_log_anisotropy: Bound on half the log of `s_par/s_perp`, applied
            as a `tanh` squash on the anisotropic part of `log_extent` while
            the mean, the overall size, stays free. The ratio is then confined
            to `[exp(-2m), exp(2m)]`, so `m = 0.32` caps it at 1.9. `None`,
            the default, applies no bound and leaves the decode
            bit-identical.
        basis_norm: `none` (default) decodes with the raw `W`. `rms` divides each
            column by its root-mean-square over pixels, so a unit change in `h_k`
            moves the logits by 1 in rms. This removes the gauge `h -> h/c`,
            `W -> cW`, under which `prior_scale`, the KL and the size of `m(s)` are
            otherwise not well-posed quantities.
        learned_prior_scale: Replace the fixed `N(0, prior_scale^2)` prior on `h`
            with `N(0, sigma(s)^2)`, `sigma` per latent component predicted from
            geometry and started at `prior_scale`. With `local_shift` this is the
            prior `c ~ N(m(s), sigma(s)^2)` on the decoded coefficients, so the data
            set how much reflections depart from their neighbourhood. Requires
            `basis_norm: rms`, without which `sigma` and `W` trade scale freely.
        max_log_prior_scale: Bound on `|log sigma(s) - log prior_scale|`.
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
        latent_gain: float = 1.0,
        geom_hidden: int = 32,
        max_log_extent: float = 1.0,
        max_log_anisotropy: float | None = None,
        gate_extent: bool = False,
        extent_half_signal: float = 100.0,
        normalization: str = "softmax",
        basis_norm: str = "none",
        learned_prior_scale: bool = False,
        max_log_prior_scale: float = 3.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            init_std=init_std,
            prior_scale=prior_scale,
            normalization=normalization,
        )

        self.latent_gain = float(latent_gain)
        self.local_shift = bool(local_shift)
        self.steer = bool(steer)
        if basis_norm not in ("none", "rms"):
            raise ValueError(f"basis_norm must be none or rms, got {basis_norm!r}")
        self.basis_norm = basis_norm
        self.learned_prior_scale = bool(learned_prior_scale)
        self.max_log_prior_scale = float(max_log_prior_scale)
        if self.learned_prior_scale and basis_norm == "none":
            raise ValueError(
                "learned_prior_scale needs basis_norm: rms; with a raw W the prior "
                "scale and the basis trade scale freely and sigma(s) means nothing"
            )
        self.beam_center = tuple(beam_center) if beam_center is not None else None
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.max_log_extent = float(max_log_extent)
        self.max_log_anisotropy = (
            None if max_log_anisotropy is None else float(max_log_anisotropy)
        )
        if self.max_log_anisotropy is not None and self.max_log_anisotropy <= 0:
            raise ValueError(
                "max_log_anisotropy must be positive or None; got "
                f"{max_log_anisotropy!r}"
            )

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

        if (
            self.local_shift or self.steer or self.learned_prior_scale
        ) and self.beam_center is None:
            raise ValueError(
                "GeometricProfileSurrogate needs beam_center when local_shift, "
                "steer or learned_prior_scale is enabled"
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
        # zero last layer, so log sigma(s) starts at log prior_scale everywhere
        self.prior_scale_head = (
            self._geometry_mlp(geom_hidden, self.d)
            if self.learned_prior_scale
            else None
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

    def _basis(self) -> Tensor:
        """Decoder basis `W`, shape (K, d), normalized as `basis_norm` asks."""
        w = self.decoder.weight
        if self.basis_norm == "rms":
            w = w / w.pow(2).mean(dim=0, keepdim=True).sqrt().clamp_min(1e-8)
        return w

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

        # `log_extent` carries two separate things: its MEAN is the overall
        # size of the profile and its DIFFERENCE is the anisotropy, and only
        # the second one needs holding down. `max_log_extent` alone bounds
        # each extent independently, which leaves the ratio
        # s_par/s_perp free up to exp(2 * max_log_extent) -- 7.4 at the
        # default, against a measured 1.85 in the bright reflections this is
        # meant to imitate. Measured on 9b7c, `steer` points the profile the
        # right way (6.5 degrees from the local bright profiles against a 45
        # degree null) and then overshoots the width, and the width is what
        # costs I/sigma, because the RB fit pays for every unit of profile
        # mass sitting on a signal-free pixel.
        #
        # tanh rather than a second clamp: a clamped head has zero gradient
        # once it saturates, and this one does saturate.
        if self.max_log_anisotropy is not None:
            size = log_extent.mean(dim=-1, keepdim=True)
            aniso = log_extent - size
            m = self.max_log_anisotropy
            log_extent = size + m * torch.tanh(aniso / m)
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
        loc = self.latent_gain * self.loc_head(x)
        # +1e-6 because a plain softplus UNDERFLOWS to exactly 0.0 in
        # float32 once the pre-activation goes far negative, and
        # Normal rejects scale == 0. Adding the floor rather than
        # clamping keeps the gradient alive below it. Same fix as
        # ShapeletProfile below and folded_normal.py.
        scale = self.latent_gain * F.softplus(self.scale_head(x)) + 1e-6

        needs_geometry = self.local_shift or self.steer or self.learned_prior_scale
        if needs_geometry and metadata is None:
            raise ValueError(
                "GeometricProfileSurrogate needs metadata when local_shift, "
                "steer or learned_prior_scale is enabled"
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
        zp, mean_profile, mean_logits = _sample_and_decode(
            decode_loc,
            scale,
            self._basis(),
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

        # None keeps the caller's KL against the fixed N(0, prior_scale^2)
        kl = None
        if self.prior_scale_head is not None:
            log_sigma = math.log(self.prior_scale) + self.prior_scale_head(
                features
            ).clamp(-self.max_log_prior_scale, self.max_log_prior_scale)
            kl = kl_divergence(
                Normal(loc, scale), Normal(torch.zeros_like(loc), log_sigma.exp())
            ).sum(dim=-1)

        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            loc=loc,
            scale=scale,
            prior_scale=self.prior_scale,
            kl=kl,
        )


class StreakBaseProfileSurrogate(ProfileSurrogate):
    """Learned detector-frame basis whose fallback is a geometry-computed streak.

        logits_j = log(g_j(s) + floor / K) + gamma * (W h)_j,    p = softmax over valid pixels

    `g` is a separable generalised Gaussian in the radial frame at the analytic
    azimuth `phi` about the beam centre, integrated over each pixel by a
    `quad x quad` sub-grid:

        u =  cos(phi) dx + sin(phi) dy - c_u,    v = -sin(phi) dx + cos(phi) dy - c_v
        g ~ exp(-|u / (alpha_u s_u)|^beta_u - |v / (alpha_v s_v)|^beta_v)

    with `alpha(beta) = sqrt(Gamma(1/beta) / Gamma(3/beta))`, so `s_u` and `s_v`
    are the base's second-moment standard deviations in pixels and nothing else
    carries its size. This replaces `geometric_basis_profile`'s canonical grid
    and affine warp, in which the canonical shape's width and the warp extents
    trade freely (on mos010 the canonical softmax went one-hot and the stretch
    sat on its clamp); here there is no canonical grid and no resampling.

    - Extents: `log s = log s0 + m tanh(f / m)`, `f` a small MLP of the radius
      features (`extent_features: radius`) or all nine geometry features
      (`all`); smooth and bounded, so no gradient is lost at a bound.
    - Centre: the predicted sub-pixel centroid (`xyzcal.px - bbox`) rotated into
      the radial frame, plus a geometry head bounded at `max_shift` pixels.
    - Shape exponents `beta` (one along, one across), global, in (0.5, 4).
    - Residual: the parent's raw detector-frame `W h`, `h ~ N(loc(x), scale(x))`
      against the fixed prior `N(0, prior_scale^2)`. The decoder bias is frozen at
      zero, since the streak takes its place. It enters after the base in the
      detector frame, so no resampling smooths pixel-level detail.
    - Gate: `gamma = z^k / (z^k + z0^k)`, `z = S / sqrt(b / sum g^2 + S)` from a
      profile fit of the base to the shoebox's own counts with the mean of the
      `border`-pixel frame as background, computed without gradient. Faint
      shoeboxes get the streak; the residual opens as the signal can pay for it.

    `residual: false` removes the residual (the base alone; the profile encoder
    then gets no gradient). `residual_gate: false` sets `gamma = 1`.

    Args:
        input_dim: Encoder output width.
        latent_dim: Width of the residual latent `h`. Default 8 (parent).
        output_dim: Profile pixels `D*H*W`.
        init_std: Initial posterior std of `h`.
        prior_scale: Std of the fixed prior on `h`.
        beam_center: Beam centre in pixels.
        profile_shape: `(D, H, W)`; `D` must be 1.
        lambda_min, lambda_max: Spectrum band, for the wavelength features.
        init_extent: Starting `(s_u, s_v)` in pixels.
        max_log_extent: Half-range `m` of the tanh bound on `log s - log s0`.
        extent_features: `radius` or `all`.
        init_beta: Starting `(beta_u, beta_v)`.
        max_shift: Bound on the learned centre offset, in pixels.
        use_centroid: Start the centre at the predicted sub-pixel centroid.
        residual: Include the pixel residual `W h`.
        residual_gate: Gate the residual by the signal-to-noise; else `gamma = 1`.
        residual_half_snr: `z0`, the signal-to-noise at which `gamma = 1/2`.
        gate_power: `k`.
        base_floor: Mass spread evenly under the base before the log, over `K`.
        quad: Sub-pixel samples per axis for the pixel integral.
        geom_hidden: Width of the geometry heads.
        border: Width of the frame whose mean is the gate's background.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int | None = None,
        output_dim: int = 625,
        init_std: float = 0.5,
        prior_scale: float = 3.0,
        beam_center: tuple[float, float] | list[float] | None = None,
        profile_shape: tuple[int, int, int] | list[int] | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
        init_extent: tuple[float, float] | list[float] = (4.0, 1.0),
        max_log_extent: float = 1.2,
        extent_features: str = "radius",
        init_beta: tuple[float, float] | list[float] = (2.0, 2.0),
        max_shift: float = 2.0,
        use_centroid: bool = True,
        residual: bool = True,
        residual_gate: bool = True,
        residual_half_snr: float = 12.0,
        gate_power: float = 2.0,
        base_floor: float = 1e-3,
        quad: int = 3,
        geom_hidden: int = 16,
        border: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            init_std=init_std,
            prior_scale=prior_scale,
        )
        if beam_center is None:
            raise ValueError("streak_base_profile needs beam_center")
        if profile_shape is None:
            side = int(round(math.sqrt(output_dim)))
            profile_shape = (1, side, side)
        d, h, w = (int(v) for v in profile_shape)
        if d != 1 or h * w != output_dim:
            raise ValueError(
                f"streak_base_profile is 2-D: profile_shape {profile_shape} "
                f"against output_dim {output_dim}"
            )
        if extent_features not in ("radius", "all"):
            raise ValueError(f"extent_features must be radius or all, got {extent_features!r}")
        self.profile_shape = (d, h, w)
        self.beam_center = tuple(float(v) for v in beam_center)
        self.lambda_min, self.lambda_max = lambda_min, lambda_max
        self.max_log_extent = float(max_log_extent)
        self.max_shift = float(max_shift)
        self.use_centroid = bool(use_centroid)
        self.residual = bool(residual)
        self.residual_gate = bool(residual_gate)
        self.residual_half_snr = float(residual_half_snr)
        self.gate_power = float(gate_power)
        self.base_floor = float(base_floor)
        self.extent_features = extent_features

        n_in = 2 if extent_features == "radius" else N_GEOM_FEATURES
        self.extent_head = nn.Sequential(
            nn.Linear(n_in, geom_hidden), nn.SiLU(), nn.Linear(geom_hidden, 2)
        )
        self.shift_head = nn.Sequential(
            nn.Linear(N_GEOM_FEATURES, geom_hidden), nn.SiLU(), nn.Linear(geom_hidden, 2)
        )
        for head in (self.extent_head, self.shift_head):
            # small, not zero: with an exactly zero last layer the first layer
            # has no gradient at step 0
            nn.init.normal_(head[-1].weight, std=1e-3)
            nn.init.zeros_(head[-1].bias)
        self.log_extent0 = nn.Parameter(
            torch.tensor([math.log(float(init_extent[0])), math.log(float(init_extent[1]))])
        )
        self._beta_lo, self._beta_hi = 0.5, 4.0
        self.beta_raw = nn.Parameter(
            torch.tensor(
                [math.log((float(b) - self._beta_lo) / (self._beta_hi - float(b))) for b in init_beta]
            )
        )
        # the streak replaces the decoder bias
        nn.init.zeros_(self.decoder.bias)
        self.decoder.bias.requires_grad_(False)
        if not self.residual:
            for p in (*self.loc_head.parameters(), *self.scale_head.parameters(), self.decoder.weight):
                p.requires_grad_(False)

        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32) - (h - 1) / 2,
            torch.arange(w, dtype=torch.float32) - (w - 1) / 2,
            indexing="ij",
        )
        sub = (torch.arange(quad, dtype=torch.float32) + 0.5) / quad - 0.5
        oy, ox = torch.meshgrid(sub, sub, indexing="ij")
        self.register_buffer("_gx", (xx.reshape(-1, 1) + ox.reshape(1, -1)).contiguous(), persistent=False)
        self.register_buffer("_gy", (yy.reshape(-1, 1) + oy.reshape(1, -1)).contiguous(), persistent=False)
        frame = torch.ones(h, w, dtype=torch.bool)
        frame[border:-border, border:-border] = False
        self.register_buffer("_frame", frame.reshape(-1).float(), persistent=False)

    @property
    def beta(self) -> Tensor:
        return self._beta_lo + (self._beta_hi - self._beta_lo) * torch.sigmoid(self.beta_raw)

    def geometry(self, metadata: dict, dtype, device) -> tuple[Tensor, Tensor, Tensor]:
        """Azimuth `phi` (B,), log extents (B, 2) and centre `(c_u, c_v)` (B, 2)."""
        feats, phi = geometry_features(metadata, self.beam_center, self.lambda_min, self.lambda_max)
        feats = feats.to(dtype).to(device)
        phi = phi.to(dtype).to(device)
        ext_in = feats[:, :2] if self.extent_features == "radius" else feats
        m = self.max_log_extent
        log_s = self.log_extent0 + m * torch.tanh(self.extent_head(ext_in) / m)
        shift = self.max_shift * torch.tanh(self.shift_head(feats) / self.max_shift)
        if self.use_centroid and "bbox.0" in metadata and "bbox.2" in metadata:
            _, h, w = self.profile_shape
            fx = (metadata["xyzcal.px.0"].reshape(-1) - metadata["bbox.0"].reshape(-1)).to(dtype).to(device)
            fy = (metadata["xyzcal.px.1"].reshape(-1) - metadata["bbox.2"].reshape(-1)).to(dtype).to(device)
            fx = fx - 0.5 - (w - 1) / 2
            fy = fy - 0.5 - (h - 1) / 2
            c, s = torch.cos(phi), torch.sin(phi)
            shift = shift + torch.stack([c * fx + s * fy, -s * fx + c * fy], dim=-1)
        return phi, log_s, shift

    def base(self, phi: Tensor, log_s: Tensor, shift: Tensor, mask: Tensor | None) -> Tensor:
        """The pixel-integrated streak on the valid pixels, (B, K), rows summing to 1."""
        c, s = torch.cos(phi)[:, None, None], torch.sin(phi)[:, None, None]
        u = c * self._gx[None] + s * self._gy[None] - shift[:, 0, None, None]
        v = -s * self._gx[None] + c * self._gy[None] - shift[:, 1, None, None]
        beta = self.beta
        alpha = torch.exp(0.5 * (torch.lgamma(1 / beta) - torch.lgamma(3 / beta)))
        su = log_s[:, 0].exp()[:, None, None]
        sv = log_s[:, 1].exp()[:, None, None]
        e = (u.abs() / (alpha[0] * su) + 1e-6) ** beta[0] + (v.abs() / (alpha[1] * sv) + 1e-6) ** beta[1]
        g = torch.exp(-e).mean(-1)
        if mask is not None:
            g = g * mask.reshape(g.shape).to(g.dtype)
        return g / g.sum(-1, keepdim=True).clamp_min(1e-12)

    def gate(self, g: Tensor, counts: Tensor | None, mask: Tensor | None) -> tuple[Tensor, Tensor]:
        """Residual gate `gamma` and the base profile-fit signal-to-noise `z`; no gradient."""
        with torch.no_grad():
            if not self.residual_gate or counts is None:
                one = torch.ones(g.shape[0], device=g.device, dtype=g.dtype)
                return one, torch.full_like(one, float("nan"))
            c = counts.reshape(g.shape).to(g.dtype)
            m = torch.ones_like(g) if mask is None else mask.reshape(g.shape).to(g.dtype)
            fm = m * self._frame
            bg = (c * fm).sum(-1) / fm.sum(-1).clamp_min(1.0)
            s2 = (g.pow(2) * m).sum(-1).clamp_min(1e-12)
            s_fit = ((g * (c - bg[:, None]) * m).sum(-1) / s2).clamp_min(0.0)
            z = s_fit / (bg.clamp_min(1e-3) / s2 + s_fit).sqrt().clamp_min(1e-6)
            k = self.gate_power
            return z.pow(k) / (z.pow(k) + self.residual_half_snr**k), z

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        metadata: dict | None = None,
        **kwargs,
    ) -> ProfileSurrogateOutput:
        if metadata is None:
            raise ValueError("streak_base_profile needs metadata")
        mask = kwargs.get("mask")
        phi, log_s, shift = self.geometry(metadata, x.dtype, x.device)
        g = self.base(phi, log_s, shift, mask)
        base_logits = torch.log(g + self.base_floor / g.shape[-1])
        if not self.residual:
            prof = masked_softmax(base_logits, mask)
            zeros = prof.new_zeros(prof.shape[0], self.d)
            return ProfileSurrogateOutput(
                zp=prof.unsqueeze(0).expand(mc_samples, -1, -1),
                mean_profile=prof,
                loc=zeros,
                scale=torch.ones_like(zeros),
                prior_scale=self.prior_scale,
                kl=prof.new_zeros(prof.shape[0]),
            )
        gamma, _ = self.gate(g, kwargs.get("counts"), mask)
        loc = self.loc_head(x)
        scale = F.softplus(self.scale_head(x)) + 1e-6
        h = Normal(loc, scale).rsample([mc_samples])
        W = self.decoder.weight
        zp = masked_softmax(base_logits[None] + gamma[None, :, None] * (h @ W.T), mask)
        mean_profile = masked_softmax(base_logits + gamma[:, None] * (loc @ W.T), mask)
        return ProfileSurrogateOutput(
            zp=zp, mean_profile=mean_profile, loc=loc, scale=scale, prior_scale=self.prior_scale
        )


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class StreakMixtureProfileSurrogate(ProfileSurrogate):
    """Core-and-halo streak base from metadata, with an optional learned shape posterior and residual.

        logits_j = log(g_j(z) + floor / K) + r_j(h),    p = softmax over valid pixels

    The base is a core and a halo, both Gaussian, centred at one point and
    aligned with the analytic detector azimuth phi:

        g_j = w N(u_j; a1^2 + 1/12) N(v_j; c1^2 + 1/12)
              + (1 - w) N(u_j; a2^2 + 1/12) N(v_j; c2^2 + 1/12)

    at pixel centres, each component normalised on the valid pixels so that w
    is the core's share exactly; adding 1/12 to each variance is the second
    moment of the pixel integral (exact for a component wider than a pixel). The shape vector z has seven components: log a1, the
    halo-to-core ratio along (a2 = a1 (1 + softplus)), log c1, the ratio across,
    the logit of the core weight w in [0.02, 0.98], and a centre offset (u, v)
    in pixels from `xyzcal`.

        prior      p(z | m)    = N(mu(m), diag(sigma^2)),  mu(m) = z0 + b tanh(f(m) / b)
        posterior  q(z | x, m) = N(mu(m) + sigma delta(x), diag((sigma s(x))^2))

    with f an MLP on the nine geometry features and z0, b, sigma fixed buffers
    (a trainable bound centre drifted onto its own bound in
    `streak_base_profile`). delta = 0, s = 1 is the metadata base at zero KL.

    `shape_posterior`: `none` (z = mu(m)), `amortized` (above, KL added to the
    profile KL), `deterministic` (z = mu + sigma delta without KL; the control).
    `residual`: `none`, `detector` (W h in the detector frame, not gated),
    `streak` (a fixed dictionary of 45 Gaussian bumps in streak coordinates
    mixed by a learned matrix from h).

    Designed from fits to the true mos010 profiles: a one-scale generalised
    Gaussian recovers 0.85-0.88 of I_true below 300 photons under the expected
    Poisson NLL, two aligned Gaussians 0.96-1.00. `init_shape` and
    `shape_prior_sd` are the median and the post-metadata spread of those fits.
    """

    NZ = 7

    def __init__(
        self,
        input_dim: int,
        latent_dim: int | None = None,
        output_dim: int = 625,
        init_std: float = 0.5,
        prior_scale: float = 3.0,
        beam_center=None,
        profile_shape=None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
        # core sd along, halo sd along, core sd across, halo sd across, core weight, centre u, centre v
        init_shape=(1.9, 5.7, 0.20, 1.05, 0.38, -0.15, 0.0),
        # half-range of the tanh bound on mu(m) - z0, per component of z
        mean_bound=(1.2, 2.5, 1.2, 2.5, 2.5, 2.0, 0.5),
        # prior sd of z about mu(m): the remainder of the per-reflection fits after metadata
        shape_prior_sd=(0.35, 1.27, 0.36, 1.48, 0.46, 0.65, 0.05),
        shape_posterior: str = "amortized",
        max_delta: float = 3.0,
        shape_samples: int = 8,
        residual: str = "detector",
        base_floor: float = 1e-3,
        geom_hidden: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim,
                         init_std=init_std, prior_scale=prior_scale)
        if beam_center is None:
            raise ValueError("streak_mixture_profile needs beam_center")
        if profile_shape is None:
            side = int(round(math.sqrt(output_dim)))
            profile_shape = (1, side, side)
        d, h, w = (int(v) for v in profile_shape)
        if d != 1 or h * w != output_dim:
            raise ValueError("streak_mixture_profile is 2-D")
        if shape_posterior not in ("none", "amortized", "deterministic"):
            raise ValueError(shape_posterior)
        if residual not in ("none", "detector", "streak"):
            raise ValueError(residual)
        self.profile_shape = (d, h, w)
        self.beam_center = tuple(float(v) for v in beam_center)
        self.lambda_min, self.lambda_max = lambda_min, lambda_max
        self.shape_posterior, self.residual = shape_posterior, residual
        self.max_delta, self.shape_samples, self.base_floor = float(max_delta), int(shape_samples), float(base_floor)

        a1, a2, c1, c2, w0, du, dv = (float(v) for v in init_shape)
        z0 = [math.log(a1), _inv_softplus(a2 / a1 - 1), math.log(c1), _inv_softplus(c2 / c1 - 1),
              math.log((w0 - 0.02) / (0.98 - w0)), du, dv]
        self.register_buffer("z0", torch.tensor(z0))
        self.register_buffer("mean_bound", torch.tensor([float(v) for v in mean_bound]))
        self.register_buffer("shape_prior_sd", torch.tensor([float(v) for v in shape_prior_sd]))

        self.shape_head = nn.Sequential(
            nn.Linear(N_GEOM_FEATURES, geom_hidden), nn.SiLU(),
            nn.Linear(geom_hidden, geom_hidden), nn.SiLU(),
            nn.Linear(geom_hidden, self.NZ),
        )
        nn.init.normal_(self.shape_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.shape_head[-1].bias)
        if shape_posterior != "none":
            self.delta_head = nn.Linear(input_dim, self.NZ)
            nn.init.normal_(self.delta_head.weight, std=1e-3)
            nn.init.zeros_(self.delta_head.bias)
        if shape_posterior == "amortized":
            self.zscale_head = nn.Linear(input_dim, self.NZ)
            nn.init.normal_(self.zscale_head.weight, std=1e-3)
            nn.init.constant_(self.zscale_head.bias, _inv_softplus(1.0))

        nn.init.zeros_(self.decoder.bias)
        self.decoder.bias.requires_grad_(False)
        if residual != "detector":
            self.decoder.weight.requires_grad_(False)
        if residual == "none":
            for p in (*self.loc_head.parameters(), *self.scale_head.parameters()):
                p.requires_grad_(False)
        if residual == "streak":
            cu = torch.tensor([0.0, 1.5, -1.5, 3.5, -3.5, 6.5, -6.5, 10.5, -10.5])
            wu = torch.tensor([0.9, 0.9, 0.9, 1.3, 1.3, 2.0, 2.0, 2.6, 2.6])
            cv = torch.tensor([0.0, 1.0, -1.0, 2.5, -2.5])
            wv = torch.tensor([0.6, 0.6, 0.6, 1.1, 1.1])
            for n, t in (("_cu", cu), ("_wu", wu), ("_cv", cv), ("_wv", wv)):
                self.register_buffer(n, t, persistent=False)
            self.streak_mix = nn.Parameter(torch.randn(len(cu) * len(cv), self.d) * 0.05)

        yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32) - (h - 1) / 2,
                                torch.arange(w, dtype=torch.float32) - (w - 1) / 2, indexing="ij")
        self.register_buffer("_px", xx.reshape(-1).contiguous(), persistent=False)
        self.register_buffer("_py", yy.reshape(-1).contiguous(), persistent=False)

    # ---- pieces -----------------------------------------------------------------------
    def prior_mean(self, metadata: dict, dtype, device):
        """phi (B,), mu(m) (B, 7), predicted centroid in the streak frame (B, 2)."""
        feats, phi = geometry_features(metadata, self.beam_center, self.lambda_min, self.lambda_max)
        feats, phi = feats.to(dtype).to(device), phi.to(dtype).to(device)
        mu = self.z0 + self.mean_bound * torch.tanh(self.shape_head(feats) / self.mean_bound)
        _, h, w = self.profile_shape
        fx = (metadata["xyzcal.px.0"].reshape(-1) - metadata["bbox.0"].reshape(-1)).to(dtype).to(device) - 0.5 - (w - 1) / 2
        fy = (metadata["xyzcal.px.1"].reshape(-1) - metadata["bbox.2"].reshape(-1)).to(dtype).to(device) - 0.5 - (h - 1) / 2
        c, s = torch.cos(phi), torch.sin(phi)
        return phi, mu, torch.stack([c * fx + s * fy, -s * fx + c * fy], dim=-1)

    @staticmethod
    def unpack(z: Tensor):
        a1 = z[..., 0].exp()
        a2 = a1 * (1 + F.softplus(z[..., 1]))
        c1 = z[..., 2].exp()
        c2 = c1 * (1 + F.softplus(z[..., 3]))
        w = 0.02 + 0.96 * torch.sigmoid(z[..., 4])
        return a1, a2, c1, c2, w

    def streak_coords(self, phi: Tensor, centre: Tensor):
        """Pixel-centre coordinates (u along, v across) about `centre`; centre (..., B, 2) -> (..., B, K)."""
        c, s = torch.cos(phi)[:, None], torch.sin(phi)[:, None]
        u0 = c * self._px[None] + s * self._py[None]
        v0 = -s * self._px[None] + c * self._py[None]
        return u0 - centre[..., 0, None], v0 - centre[..., 1, None]

    def base(self, phi: Tensor, z: Tensor, centroid: Tensor, mask: Tensor | None) -> Tensor:
        """Core + halo on the valid pixels; z (..., B, 7) -> (..., B, K), rows summing to 1."""
        a1, a2, c1, c2, w = (t[..., None] for t in self.unpack(z))
        u, v = self.streak_coords(phi, centroid + z[..., 5:7])
        m = None if mask is None else mask.reshape(u.shape[-2:]).to(u.dtype)

        # each component is normalised on the valid pixels before mixing, so w is
        # the core's share of the photons exactly: a 0.35 px core sampled at pixel
        # centres sums to more than 1 under its continuous constant, by an amount
        # that moves with its sub-pixel position
        def comp(a, c):
            va, vc = a * a + 1.0 / 12, c * c + 1.0 / 12
            e = torch.exp(-0.5 * (u * u / va + v * v / vc))
            if m is not None:
                e = e * m
            return e / e.sum(-1, keepdim=True).clamp_min(1e-30)

        return w * comp(a1, c1) + (1 - w) * comp(a2, c2)

    def residual_logits(self, h: Tensor, phi: Tensor, centre: Tensor) -> Tensor:
        """h (..., B, d) -> (..., B, K)."""
        if self.residual == "detector":
            return h @ self.decoder.weight.T
        u, v = self.streak_coords(phi, centre)                      # (B, K)
        bu = torch.exp(-0.5 * ((u[..., None] - self._cu) / self._wu) ** 2)   # (B, K, 9)
        bv = torch.exp(-0.5 * ((v[..., None] - self._cv) / self._wv) ** 2)   # (B, K, 5)
        phi_b = (bu[..., :, None] * bv[..., None, :]).flatten(-2)             # (B, K, 45)
        return torch.einsum("bkm,...bm->...bk", phi_b, h @ self.streak_mix.T)

    # ---- forward ----------------------------------------------------------------------
    def forward(self, x: Tensor, mc_samples: int = 1, metadata: dict | None = None, **kwargs) -> ProfileSurrogateOutput:
        if metadata is None:
            raise ValueError("streak_mixture_profile needs metadata")
        mask = kwargs.get("mask")
        phi, mu, centroid = self.prior_mean(metadata, x.dtype, x.device)
        B = x.shape[0]
        kl = x.new_zeros(B)
        sig = self.shape_prior_sd
        if self.shape_posterior == "none":
            z_mean, z_s = mu, mu[None]
        else:
            delta = self.max_delta * torch.tanh(self.delta_head(x) / self.max_delta)
            z_mean = mu + sig * delta
            if self.shape_posterior == "amortized":
                s = F.softplus(self.zscale_head(x)) + 1e-4
                n_z = max(1, min(self.shape_samples, mc_samples))
                z_s = z_mean[None] + sig * s[None] * torch.randn(n_z, B, self.NZ, device=x.device, dtype=x.dtype)
                kl = kl + 0.5 * (delta.pow(2) + s.pow(2) - 1 - 2 * s.log()).sum(-1)
            else:
                z_s = z_mean[None]
        floor = self.base_floor / self.decoder.weight.shape[0]
        base_mean = torch.log(self.base(phi, z_mean, centroid, mask) + floor)             # (B, K)
        base_s = torch.log(self.base(phi, z_s, centroid, mask) + floor)                   # (n_z, B, K)
        reps = -(-mc_samples // base_s.shape[0])
        base_s = base_s.repeat(reps, 1, 1)[:mc_samples]                                   # (S, B, K)
        if self.residual == "none":
            zeros = x.new_zeros(B, self.d)
            return ProfileSurrogateOutput(zp=masked_softmax(base_s, mask), mean_profile=masked_softmax(base_mean, mask),
                                          loc=zeros, scale=torch.ones_like(zeros), prior_scale=self.prior_scale, kl=kl)
        loc = self.loc_head(x)
        scale = F.softplus(self.scale_head(x)) + 1e-6
        h = Normal(loc, scale).rsample([mc_samples])
        centre = centroid + z_mean[:, 5:7]
        kl = kl + (torch.log(self.prior_scale / scale) + (scale.pow(2) + loc.pow(2)) / (2 * self.prior_scale**2) - 0.5).sum(-1)
        zp = masked_softmax(base_s + self.residual_logits(h, phi, centre), mask)
        mean_profile = masked_softmax(base_mean + self.residual_logits(loc, phi, centre), mask)
        return ProfileSurrogateOutput(zp=zp, mean_profile=mean_profile, loc=loc, scale=scale, prior_scale=self.prior_scale, kl=kl)


class HellingerBasisProfileSurrogate(nn.Module):
    """Learned-basis profile in the square-root chart, orthonormal in Fisher-Rao.

    `softmax(W h + b)` is linear in log-density and carries a gauge: `h -> h/c`,
    `W -> cW` leaves the likelihood unchanged, so `prior_scale` and the KL weight
    are not well-posed quantities and nothing pins the profile's breadth. Four
    runs of one config gave `n_eff` 56 / 106 / 211 / 323.

    Conditioning a Poisson pixel vector on its total gives a multinomial, so the
    profile's Fisher metric is the multinomial one, and in `psi = sqrt(p)` it is
    four times the Euclidean inner product:

        g_ij = sum_k (1/p_k) dp_k/dh_i dp_k/dh_j = 4 <dpsi/dh_i, dpsi/dh_j>

    This surrogate therefore works in `psi`, along a geodesic of the unit sphere
    from a base point:

        v   = sum_m h_m e_m,   r = |v|
        psi = cos(r) psi0 + sin(r) v / r
        p   = psi^2

    with `{e_m}` orthonormalized against `psi0`. Three consequences. The Fisher
    information for `h` is `4 I`, so the coordinates are statistically
    uncorrelated and `prior_scale` is in radians of Bhattacharyya angle, where
    `cos r = sum_k sqrt(p_k p0_k)`. The gauge is closed, because `|e_m| = 1` is
    imposed rather than learned. And `p >= 0` with `sum p = 1` hold by
    construction, with no softmax, clamp or Dirichlet.

    `project_breadth` also removes the two directions along which the profile's
    breadth moves, so the learned coefficients cannot change it to first order
    and the width is left to whatever sets `psi0`. A functional `phi(p)` is
    stationary in `h` at the base point exactly when the span omits the tangent
    part of its gradient in `psi`, and the two breadth measures give different
    directions:

        Renyi-2, phi = sum_k p_k^2 (so n_eff = 1/phi):  grad = 4 psi0^3
        Shannon, phi = -sum_k p_k log p_k:              grad = -2 psi0 (log p0 + 1)

    Both are removed. Projecting only the Shannon direction leaves `n_eff`
    moving at half rate, since the two gradients are not parallel; `n_eff` is
    the quantity four runs of one config spread 5.8-fold, so it is the one that
    has to be pinned. Both vanish for a uniform `psi0`, which the Gaussian
    initialization avoids.

    `psi` may leave the positive orthant; `p = psi^2` is still a density, and a
    sign change is a node rather than an invalid profile.

    Orthogonality holds on the full grid. A per-reflection mask makes it
    approximate, since restoring it exactly would cost one QR per reflection.

    Args:
        input_dim: Encoder output width.
        latent_dim: Number of basis directions. Default 8.
        output_dim: Profile pixels `D*H*W`. Default 441.
        profile_shape: Grid as `(D, H, W)`; inferred as a square when unset.
        init_std: Initial posterior std of `h`.
        prior_radius: One-sigma geodesic radius of the prior, in radians of
            Bhattacharyya angle. The per-coordinate `prior_scale` the caller
            needs for the KL is `prior_radius / sqrt(latent_dim)`, because `|h|`
            over `d` independent coordinates grows as `sqrt(d)`. At 0.4 the
            Bhattacharyya coefficient `cos r = sum_k sqrt(p_k p0_k)` against
            `psi0` stays above 0.92 at one sigma; the softmax family's 3.0 is in
            unit-free logits and has no such reading.
        init_base_sigma: Gaussian sigma in pixels for the initial `psi0`. 1.0
            sits inside the 0.43-1.47 px range measured on hewl_9b7c.
        project_breadth: Remove the two breadth directions from the span. The
            breadth is then a property of `psi0` alone -- one shared quantity
            per run that `n_eff(psi0)` reports directly -- rather than a
            per-reflection latent direction.
        freeze_base: Hold a learned `psi0` at its Gaussian initialization, which
            pins the breadth outright at `init_base_sigma`. No effect under
            `base_mode='physical'`, where `base_raw` is unused.
        base_mode: Where `psi0` comes from.

            `learned` -- one shared base point for the whole dataset. Simplest,
            but the profile then cannot vary with detector position at all.

            `geometric` -- the base point is carried away from a global
            `psi_bar` along a geodesic whose direction and length are a learned
            function of the detector geometry:

                psi0(s) = cos(r) psi_bar + sin(r) ghat(s),
                r = geom_radius * tanh(|c(s)|),  c(s) = MLP(geometry features)

            with `ghat(s)` in a learned `geom_dim`-dimensional tangent basis.
            Nothing about the shape is prescribed: the curve `s -> psi0(s)` is
            fitted, where a width law would have dictated it. What makes the
            width identifiable is not a formula but the split -- `c(s)` is
            shared by every reflection at the same detector position, so the
            width is pooled, while `h` is per-reflection and the width
            directions are absent from its span. A reflection too weak to
            constrain `h` falls back to the local profile.

            `physical` -- places it with a composed `PhysicalProfileSurrogate`,
            i.e. an `a sin^2(theta) + b` width law. Kept as a control arm for
            `geometric`, which contains it as one particular curve.
        geom_dim: Tangent dimension of the geometry map under `geometric`.
        geom_radius: Bound in radians on how far geometry may carry the base
            point from `psi_bar`, through a `tanh`.
        geom_hidden: Width of the geometry MLP's hidden layer.
        beam_center: Detector beam centre in pixels, required by `geometric`.
        lambda_min, lambda_max: Spectrum band, used only to scale a wavelength
            feature when the batch carries one.
        base_args: Constructor arguments for the composed surrogate under
            `physical`. `input_dim`, `output_dim` and `profile_shape` are
            supplied.
        project_shift: Also remove the translation directions, for runs where a
            recentring solve downstream owns the centre.
        init_basis_std: Scale of the raw basis before orthonormalization.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int | None = None,
        output_dim: int = 441,
        profile_shape: tuple[int, int, int] | list[int] | None = None,
        init_std: float = 0.5,
        prior_radius: float = 0.4,
        init_base_sigma: float = 1.0,
        project_breadth: bool = True,
        project_shift: bool = False,
        init_basis_std: float = 0.1,
        base_mode: str = "learned",
        freeze_base: bool = False,
        base_args: dict | None = None,
        geom_dim: int = 6,
        geom_radius: float = 0.8,
        geom_hidden: int = 32,
        beam_center: tuple[float, float] | list[float] | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
    ) -> None:
        super().__init__()
        if profile_shape is not None:
            shape = tuple(int(v) for v in profile_shape)
            if len(shape) != 3:
                raise ValueError(f"profile_shape {shape} is not (D, H, W)")
            if shape[0] * shape[1] * shape[2] != output_dim:
                raise ValueError(
                    f"profile_shape {shape} has "
                    f"{shape[0] * shape[1] * shape[2]} pixels but output_dim="
                    f"{output_dim}"
                )
        else:
            side = int(round(math.sqrt(output_dim)))
            if side * side != output_dim:
                raise ValueError(
                    f"output_dim {output_dim} is not square; pass profile_shape"
                )
            shape = (1, side, side)
        self.profile_shape = shape
        self.d = 8 if latent_dim is None else int(latent_dim)
        self.prior_radius = float(prior_radius)
        self.prior_scale = self.prior_radius / math.sqrt(self.d)
        self.project_breadth = bool(project_breadth)
        self.project_shift = bool(project_shift)

        self.loc_head = nn.Linear(input_dim, self.d)
        self.scale_head = nn.Linear(input_dim, self.d)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, _softplus_inverse(init_std))

        # psi0 starts as a centred Gaussian rather than uniform: a uniform base
        # point makes the breadth direction identically zero, and on Laue
        # shoeboxes a uniform profile is the documented identifiability
        # collapse.
        axes = [
            torch.arange(n, dtype=torch.float32) - 0.5 * (n - 1) for n in shape
        ]
        zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
        s2 = max(float(init_base_sigma), 1e-3) ** 2
        logp = -0.5 * (zz**2 + yy**2 + xx**2) / s2
        p0 = torch.softmax(logp.reshape(-1), dim=0)
        self.base_raw = nn.Parameter(
            _softplus_inverse_tensor(p0.sqrt().clamp_min(1e-6))
        )
        self.basis_raw = nn.Parameter(
            init_basis_std * torch.randn(output_dim, self.d)
        )
        self.register_buffer(
            "_grid", torch.stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)])
        )
        if freeze_base:
            self.base_raw.requires_grad_(False)

        if base_mode not in ("learned", "geometric", "physical"):
            raise ValueError(
                f"base_mode {base_mode!r} is not 'learned', 'geometric' or "
                "'physical'"
            )
        self.base_mode = base_mode
        self.base = None
        self.geom_head = None
        self.beam_center = tuple(beam_center) if beam_center is not None else None
        self.lambda_min, self.lambda_max = lambda_min, lambda_max
        self.geom_radius = float(geom_radius)
        if base_mode == "geometric":
            if beam_center is None:
                raise ValueError(
                    "base_mode='geometric' needs beam_center to form the "
                    "detector geometry features"
                )
            self.geom_dim = int(geom_dim)
            self.geom_raw = nn.Parameter(
                init_basis_std * torch.randn(output_dim, self.geom_dim)
            )
            self.geom_head = nn.Sequential(
                nn.Linear(N_GEOM_FEATURES, geom_hidden),
                nn.SiLU(),
                nn.Linear(geom_hidden, self.geom_dim),
            )
            # Start at psi_bar: a geometry map that begins somewhere arbitrary
            # would move every profile before the shape coefficients have
            # learned anything to correct.
            nn.init.zeros_(self.geom_head[-1].weight)
            nn.init.zeros_(self.geom_head[-1].bias)
        if base_mode == "physical":
            # Composed, not reimplemented: the base point of the sphere becomes
            # the same pixel-integrated oriented Gaussian, rotated frame and
            # `a sin^2(theta) + b` width law the direct fit uses, evaluated at
            # the reflection's own detector position and resolution. So the
            # breadth is a pooled function of geometry rather than a
            # per-reflection latent, and a reflection too weak to constrain `h`
            # falls back to the local profile instead of a global average.
            # `input_dim` is unused on the `_profile` path.
            self.base = PhysicalProfileSurrogate(
                input_dim=1,
                output_dim=output_dim,
                profile_shape=list(shape),
                **(base_args or {}),
            )

    @staticmethod
    def _unit(col: Tensor, psi0: Tensor) -> Tensor:
        """Make `col` tangent at `psi0` and scale it to unit norm."""
        col = col - (col * psi0).sum(-1, keepdim=True) * psi0
        return col / col.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _shift_columns(self, psi0: Tensor) -> list[Tensor]:
        """Translation generators at `psi0`: the spatial gradient along each axis.

        Translating `p` translates `psi`, so the generator is the derivative of
        `psi0` on the grid. Axes of extent 1 carry no translation and are
        dropped.
        """
        lead = psi0.shape[:-1]
        vol = psi0.reshape(*lead, *self.profile_shape)
        cols = []
        for axis, n in enumerate(self.profile_shape):
            if n < 2:
                continue
            dim = len(lead) + axis
            lo = torch.roll(vol, 1, dims=dim)
            hi = torch.roll(vol, -1, dims=dim)
            cols.append(
                self._unit((0.5 * (hi - lo)).reshape(*psi0.shape), psi0)
            )
        return cols

    def _frame(self, psi0: Tensor) -> Tensor:
        """Orthonormal basis of pure-shape directions at `psi0`.

        Batched over any leading dimensions of `psi0`, so a base point that
        varies per reflection gets its own span.
        """
        cols = [psi0]
        if self.project_breadth:
            p0 = (psi0 * psi0).clamp_min(1e-12)
            logp0 = p0.log()
            n = psi0.shape[-1]
            # Renyi-2 then Shannon: the two breadth functionals' gradients.
            cols.append(self._unit(psi0**3, psi0))
            cols.append(
                self._unit(
                    psi0 * (logp0 - (p0 * logp0).sum(-1, keepdim=True)), psi0
                )
            )
            # The exact likelihood gauge: with a free flat background,
            # p -> alpha p + (1 - alpha) u, I -> I / alpha and
            # b -> b - I (1 - alpha) / (alpha P) leaves every pixel rate
            # unchanged, so the counts carry no information about where on this
            # orbit the fit sits. Its tangent at alpha = 1 is
            # (p0 - u) / (2 sqrt(p0)), floored because a peaked base point
            # drives sqrt(p0) toward zero over most of the box -- which is also
            # why n_eff, a tail-weighted statistic, is what drifts.
            floor = (p0 + 1.0 / n**2).sqrt()
            cols.append(self._unit((p0 - 1.0 / n) / (2.0 * floor), psi0))
        if self.project_shift:
            cols.extend(self._shift_columns(psi0))
        removed, _ = torch.linalg.qr(torch.stack(cols, dim=-1))

        raw = self.basis_raw
        if psi0.dim() > 1:
            raw = raw.expand(*psi0.shape[:-1], -1, -1)
        m = raw - removed @ (removed.transpose(-1, -2) @ raw)
        q, r = torch.linalg.qr(m)
        # Householder QR leaves the column signs to the algorithm, so pin
        # diag(R) > 0: an unpinned sign flip between steps would move the
        # profile discontinuously under a fixed h.
        sgn = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        return q * sgn.unsqueeze(-2)

    def _base_point(self, metadata: dict | None, mask: Tensor | None) -> Tensor:
        """`psi0`, shape `(P,)` when shared and `(B, P)` when from geometry."""
        psi_bar = F.softplus(self.base_raw)
        psi_bar = psi_bar / psi_bar.norm().clamp_min(1e-12)
        if self.base_mode == "learned":
            return psi_bar

        if self.base_mode == "geometric":
            if metadata is None:
                raise ValueError(
                    "base_mode='geometric' needs the batch metadata to place "
                    "the base point; the integrator passes it as `metadata`"
                )
            features, _ = geometry_features(
                metadata, self.beam_center, self.lambda_min, self.lambda_max
            )
            features = features.to(psi_bar.dtype)
            coef = self.geom_head(features)
            # The tangent basis is shared, so one QR for the batch.
            f = self.geom_raw - psi_bar.unsqueeze(-1) * (
                psi_bar @ self.geom_raw
            ).unsqueeze(0)
            f, _ = torch.linalg.qr(f)
            norm = coef.norm(dim=-1, keepdim=True)
            direction = (coef / norm.clamp_min(1e-12)) @ f.transpose(-1, -2)
            # tanh bounds the geodesic distance from psi_bar, so geometry
            # cannot carry the base point to the far side of the sphere.
            r = self.geom_radius * torch.tanh(norm)
            return torch.cos(r) * psi_bar + torch.sin(r) * direction

        if metadata is None:
            raise ValueError(
                "base_mode='physical' needs the batch metadata to place the "
                "base point; the integrator passes it as `metadata`"
            )
        p0 = self.base._profile(metadata, mask)
        p0 = p0 / p0.sum(-1, keepdim=True).clamp_min(1e-12)
        # sum p0 == 1 already makes |psi0| == 1.
        return p0.clamp_min(0.0).sqrt()

    def _decode(
        self, h: Tensor, psi0: Tensor, basis: Tensor, mask: Tensor | None
    ) -> Tensor:
        if basis.dim() > 2:
            v = torch.einsum("sbd,bpd->sbp", h, basis)
        else:
            v = h @ basis.transpose(-1, -2)
        r = v.norm(dim=-1, keepdim=True)
        # sinc(r/pi) == sin(r)/r, which torch evaluates stably at r = 0.
        psi = torch.cos(r) * psi0 + torch.sinc(r / math.pi) * v
        p = psi * psi
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return renormalize_masked(p, mask)

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        **kwargs,
    ) -> ProfileSurrogateOutput:
        loc = self.loc_head(x)
        # +1e-6 because a plain softplus underflows to exactly 0.0 in float32
        # once the pre-activation goes far negative, and Normal rejects
        # scale == 0.
        scale = F.softplus(self.scale_head(x)) + 1e-6

        mask = kwargs.get("mask")
        if mask is None and self.base is not None:
            mask = x.new_ones(x.shape[0], self.basis_raw.shape[0], dtype=torch.bool)
        psi0 = self._base_point(kwargs.get("metadata"), mask)
        basis = self._frame(psi0)
        eps = torch.randn(
            mc_samples, *loc.shape, device=loc.device, dtype=loc.dtype
        )
        h = loc.unsqueeze(0) + scale.unsqueeze(0) * eps
        zp = self._decode(h, psi0, basis, mask)
        mean_profile = self._decode(loc.unsqueeze(0), psi0, basis, mask)[0]

        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            loc=loc,
            scale=scale,
            prior_scale=self.prior_scale,
        )


def _tucker_basis(
    core: Tensor, u_d: Tensor, u_h: Tensor, u_w: Tensor
) -> Tensor:
    """Basis `(d, D*H*W)` from per-axis factors and one core per basis vector.

    `W_m = core_m x1 U_d x2 U_h x3 U_w`. The factor matrices are shared across
    basis vectors, so what is learned per vector is only a small core.
    """
    out = torch.einsum("mabc,ia,jb,kc->mijk", core, u_d, u_h, u_w)
    return out.reshape(core.shape[0], -1)


class TuckerBasisProfileSurrogate(nn.Module):
    """Linear-basis profile with a Tucker-structured basis and a basis-tied encoder.

    Two measured facts set this design. On hewl_9b7c, swapping the profile
    family moved the iodine peaks 19.944 -> 27.203, +36%, while swapping the
    integrator across four architectures moved them 27.20 -> 27.86, inside the
    0.86 sigma single-run spread of [[training-unseeded-run-variance]]. And in
    the configuration that produced those numbers, `encoders.profile` holds
    758,368 parameters, 63.5% of the model, to emit a 12-dimensional latent,
    against 47,147 for the basis it decodes through. The budget is allocated
    25x away from the component that carries the signal.

    `softmax(W h + b)` is an exponential family, so the maximum-likelihood `h`
    depends on the counts only through `W^T c`. An encoder reading those `d`
    projections is therefore exactly as powerful as one reading all `P` pixels,
    for this family, and it needs no parameters of its own because `W` is
    already the decoder. That is what `tie_encoder` does. Dense pixel access
    buys something only for structure the projections cannot summarize -- an
    overlapping neighbour, a bimodal spot -- which is a local question, so
    `conv_channels` adds a two-layer stub rather than a deep encoder.

    The Tucker structure is a shape prior, not only a parameter count. A free
    `d x P` basis may be speckle; a low-rank separable one may not, while
    elongation and skew remain reachable because they are separable. On the
    `(3, 21, 21)` grid at ranks `(3, 6, 6)` the basis costs 261 shared plus 864
    in cores against 10,584 free, 9.4x fewer. A rotated elongated spot is not
    rank-1 separable in detector `x, y`, so the ranks are the knob that trades
    parameters against rotation freedom; rotating the grid to the radial frame
    first would let rank 1 suffice.

    Args:
        input_dim: Width of the encoder representation, used only when
            `tie_encoder` is false.
        latent_dim: Number of basis vectors `d`.
        output_dim: Profile pixels `D*H*W`.
        profile_shape: Grid as `(D, H, W)`; inferred as a square when unset.
        ranks: Tucker ranks per axis, clipped to each axis extent.
        init_std: Initial posterior std of `h`.
        prior_scale: Std of the `N(0, prior_scale)` prior on `h`.
        tie_encoder: Build the features from `W^T c` -- the sufficient
            statistic -- instead of from the upstream encoder output.
        n_quantiles: Count quantiles appended to the features. Quantiles rather
            than moments because the second-moment width is tail-dominated: the
            measured ratio swung 1.394 -> 0.832 as the brightness floor went
            10 -> 100.
        conv_channels: Channels in the two-layer conv stub; 0 disables it.
        conv_layers: Layers in that stub. Two 3x3 layers reach 5x5, which
            covers a neighbouring spot's core on a 21x21 box.
        hidden: Width of the MLP that maps features to `loc` and `scale`.
        beam_center: Detector beam centre, enabling the geometry features.
        lambda_min, lambda_max: Spectrum band for the wavelength feature.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int | None = None,
        output_dim: int = 441,
        profile_shape: tuple[int, int, int] | list[int] | None = None,
        ranks: tuple[int, int, int] | list[int] = (3, 6, 6),
        init_std: float = 0.5,
        prior_scale: float = 3.0,
        tie_encoder: bool = True,
        n_quantiles: int = 5,
        conv_channels: int = 0,
        conv_layers: int = 2,
        hidden: int = 32,
        beam_center: tuple[float, float] | list[float] | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
    ) -> None:
        super().__init__()
        if profile_shape is not None:
            shape = tuple(int(v) for v in profile_shape)
            if len(shape) != 3 or shape[0] * shape[1] * shape[2] != output_dim:
                raise ValueError(
                    f"profile_shape {shape} does not hold output_dim={output_dim}"
                )
        else:
            side = int(round(math.sqrt(output_dim)))
            if side * side != output_dim:
                raise ValueError(
                    f"output_dim {output_dim} is not square; pass profile_shape"
                )
            shape = (1, side, side)
        self.profile_shape = shape
        self.d = 8 if latent_dim is None else int(latent_dim)
        self.prior_scale = float(prior_scale)
        self.tie_encoder = bool(tie_encoder)
        self.n_quantiles = int(n_quantiles)
        self.beam_center = tuple(beam_center) if beam_center is not None else None
        self.lambda_min, self.lambda_max = lambda_min, lambda_max

        r = [min(int(a), int(b)) for a, b in zip(ranks, shape)]
        self.ranks = tuple(r)
        self.u_d = nn.Parameter(torch.randn(shape[0], r[0]) / math.sqrt(r[0]))
        self.u_h = nn.Parameter(torch.randn(shape[1], r[1]) / math.sqrt(r[1]))
        self.u_w = nn.Parameter(torch.randn(shape[2], r[2]) / math.sqrt(r[2]))
        self.core = nn.Parameter(
            torch.randn(self.d, r[0], r[1], r[2]) * 0.1
        )
        self.bias = nn.Parameter(torch.zeros(output_dim))

        self.conv = None
        if conv_channels > 0:
            kz = 3 if shape[0] >= 3 else 1
            pad = (kz // 2, 1, 1)
            layers: list[nn.Module] = []
            c_in = 1
            for _ in range(max(1, conv_layers)):
                layers += [
                    nn.Conv3d(c_in, conv_channels, (kz, 3, 3), padding=pad),
                    nn.SiLU(),
                ]
                c_in = conv_channels
            self.conv = nn.Sequential(*layers)
        self.conv_channels = int(conv_channels)

        n_feat = (self.d if tie_encoder else input_dim) + 4 + self.n_quantiles
        if self.beam_center is not None:
            n_feat += N_GEOM_FEATURES
        n_feat += self.conv_channels
        self.n_feat = n_feat
        self.head = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.SiLU(), nn.Linear(hidden, 2 * self.d)
        )
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.zero_()
            self.head[-1].bias[self.d :] = _softplus_inverse(init_std)

    def basis(self) -> Tensor:
        return _tucker_basis(self.core, self.u_d, self.u_h, self.u_w)

    def _features(
        self,
        x: Tensor,
        basis: Tensor,
        counts: Tensor | None,
        mask: Tensor | None,
        metadata: dict | None,
    ) -> Tensor:
        if counts is None:
            raise ValueError(
                "TuckerBasisProfileSurrogate needs `counts`; the integrator "
                "passes it as the `counts` keyword"
            )
        c = counts.reshape(counts.shape[0], -1).to(basis.dtype)
        m = (
            mask.reshape(c.shape).to(c.dtype)
            if mask is not None
            else torch.ones_like(c)
        )
        c = c * m
        total = c.sum(-1, keepdim=True)
        # The multinomial conditions on the total, so the shape statistic is
        # the normalized shoebox and no background estimate is needed here.
        p_obs = c / total.clamp_min(1.0)

        feats = [
            p_obs @ basis.transpose(0, 1) if self.tie_encoder else x,
            torch.log1p(total),
            self._centroid(p_obs),
        ]
        if self.n_quantiles > 0:
            q = torch.linspace(
                0.5, 1.0, self.n_quantiles + 1, device=c.device, dtype=c.dtype
            )[:-1]
            feats.append(torch.quantile(p_obs, q, dim=-1).transpose(0, 1))
        if self.beam_center is not None:
            g, _ = geometry_features(
                metadata, self.beam_center, self.lambda_min, self.lambda_max
            )
            feats.append(g.to(c.dtype).to(c.device))
        if self.conv is not None:
            vol = p_obs.reshape(-1, 1, *self.profile_shape)
            feats.append(self.conv(vol).mean(dim=(-3, -2, -1)))
        return torch.cat(feats, dim=-1)

    def _centroid(self, p_obs: Tensor) -> Tensor:
        d, h, w = self.profile_shape
        vol = p_obs.reshape(-1, d, h, w)
        out = []
        for axis, n in zip((1, 2, 3), (d, h, w)):
            if n < 2:
                out.append(torch.zeros_like(p_obs[:, :1]))
                continue
            coord = torch.linspace(
                -1.0, 1.0, n, device=p_obs.device, dtype=p_obs.dtype
            )
            dims = [i for i in (1, 2, 3) if i != axis]
            out.append((vol.sum(dim=dims) * coord).sum(-1, keepdim=True))
        return torch.cat(out, dim=-1)

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        metadata: dict | None = None,
        **kwargs,
    ) -> ProfileSurrogateOutput:
        basis = self.basis()
        mask = kwargs.get("mask")
        feats = self._features(x, basis, kwargs.get("counts"), mask, metadata)
        raw = self.head(feats)
        loc, raw_scale = raw[:, : self.d], raw[:, self.d :]
        scale = F.softplus(raw_scale) + 1e-6

        zp, mean_profile, mean_logits = _sample_and_decode(
            loc,
            scale,
            basis.transpose(0, 1),
            self.bias,
            mc_samples,
            mask=mask,
            normalize=get_normalizer("softmax"),
        )
        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            loc=loc,
            scale=scale,
            prior_scale=self.prior_scale,
            mean_logits=mean_logits,
        )


class ControlMeshProfileSurrogate(nn.Module):
    """Profile as a template pushed through a warp defined on a coarse control mesh.

    A different prior from `TuckerBasisProfileSurrogate`, at comparable cost.
    Tucker says the basis is separable and low rank; this says the profile is a
    smooth deformation of one template. Those admit different shape families, so
    which is right is a measurement rather than an argument.

    The mesh is the right place for a mesh in this problem. A diffraction spot
    is not a surface and has no boundary to mesh, and meshing an iso-intensity
    contour discards the falloff -- which `[[oracle-is-not-a-precision-ceiling]]`
    records as a real error, since the signal falls off steeply across its
    support and the falloff is what sets the estimator's weights. What a mesh
    does parameterize well is the DEFORMATION: a coarse control lattice whose
    displacements define a smooth map, regularized by a discrete energy on the
    lattice. That is also how graphics parameterizes complex shapes -- a coarse
    cage deforming a detailed object, not a mesh of the object.

        p(x) = template(x + v(x)) * |det J(x)| / Z

    `v` is a trilinear interpolation of displacements at `(mz, my, mx)` control
    points, so a `4 x 4 x 2` lattice with three components is 96 numbers.

    The gauge enters cleanly here. `det J` is the local area change, so the mean
    of `log det J` IS the concentration direction that the counts cannot
    identify -- a 10x intensity error costs zero log-likelihood with a free flat
    background. The trace-free part of the deformation is shear and skew, which
    is asymmetry, and the constant part of `v` is translation. Three orthogonal
    components of one object, mapping onto the three requirements: pin the first
    to a pooled function of geometry, leave the other two free.

    Args:
        input_dim: Width of the upstream encoder representation.
        latent_dim: Unused; present so the factory's shared defaults apply.
        output_dim: Profile pixels `D*H*W`.
        profile_shape: Grid as `(D, H, W)`.
        lattice: Control points per axis, clipped to each axis extent.
        init_std: Initial posterior std of the displacement latent.
        prior_scale: Std of the prior on the displacement latent, in pixels.
        bending_weight: Weight on the discrete Dirichlet energy of the lattice,
            which is what makes a rough warp expensive and a smooth one cheap.
        pin_log_det: Subtract the mean of `log det J` so the warp cannot change
            the profile's concentration, closing the gauge direction while
            leaving shear and translation free.
        template_sigma: Gaussian sigma in pixels for the initial template, which
            is a learned parameter from there.
        hidden: Width of the MLP mapping features to displacements.
        beam_center: Detector beam centre, enabling geometry features.
        lambda_min, lambda_max: Spectrum band for the wavelength feature.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int | None = None,
        output_dim: int = 441,
        profile_shape: tuple[int, int, int] | list[int] | None = None,
        lattice: tuple[int, int, int] | list[int] = (2, 4, 4),
        init_std: float = 0.3,
        prior_scale: float = 1.0,
        bending_weight: float = 0.1,
        pin_log_det: bool = True,
        template_sigma: float = 2.0,
        hidden: int = 32,
        beam_center: tuple[float, float] | list[float] | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
    ) -> None:
        super().__init__()
        if profile_shape is None:
            side = int(round(math.sqrt(output_dim)))
            if side * side != output_dim:
                raise ValueError(
                    f"output_dim {output_dim} is not square; pass profile_shape"
                )
            profile_shape = (1, side, side)
        shape = tuple(int(v) for v in profile_shape)
        if shape[0] * shape[1] * shape[2] != output_dim:
            raise ValueError(
                f"profile_shape {shape} does not hold output_dim={output_dim}"
            )
        self.profile_shape = shape
        self.lattice = tuple(
            min(int(a), int(b)) for a, b in zip(lattice, shape)
        )
        self.n_ctrl = int(np.prod(self.lattice))
        self.d = 3 * self.n_ctrl
        self.prior_scale = float(prior_scale)
        self.bending_weight = float(bending_weight)
        self.pin_log_det = bool(pin_log_det)
        self.beam_center = tuple(beam_center) if beam_center is not None else None
        self.lambda_min, self.lambda_max = lambda_min, lambda_max

        d_, h_, w_ = shape
        axes = [
            torch.arange(n, dtype=torch.float32) - 0.5 * (n - 1) for n in shape
        ]
        zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
        s2 = max(float(template_sigma), 1e-3) ** 2
        self.template_logits = nn.Parameter(
            -0.5 * (zz**2 + yy**2 + xx**2).reshape(-1) / s2
        )
        # normalized pixel coordinates in [-1, 1]; `grid_sample` orders the
        # trailing coordinate axis (x, y, z), opposite to the volume's (z, y, x)
        nz, ny, nx = torch.meshgrid(
            *[
                _lin(n, zz.device) if n > 1 else torch.zeros(1)
                for n in (d_, h_, w_)
            ],
            indexing="ij",
        )
        self.register_buffer(
            "_norm_grid",
            torch.stack([nx, ny, nz], dim=-1).reshape(1, d_, h_, w_, 3),
        )

        n_feat = input_dim + 4
        if self.beam_center is not None:
            n_feat += N_GEOM_FEATURES
        self.head = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.SiLU(), nn.Linear(hidden, 2 * self.d)
        )
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.zero_()
            self.head[-1].bias[self.d :] = _softplus_inverse(init_std)

    def bending_energy(self, disp: Tensor) -> Tensor:
        """Discrete Dirichlet energy of the displacement field on the lattice.

        The sum of squared differences between neighbouring control points,
        which is the graph Laplacian quadratic form -- a rough warp is expensive
        and a smooth one is cheap. On a regular lattice this is all that the
        discrete exterior calculus reduces to, which is why the interesting
        differential geometry lives in the warp and not in the grid.
        """
        v = disp.reshape(-1, 3, *self.lattice)
        total = v.new_zeros(v.shape[0])
        for axis, n in zip((2, 3, 4), self.lattice):
            if n < 2:
                continue
            total = total + (
                v.diff(dim=axis).pow(2).sum(dim=(1, 2, 3, 4))
            )
        return total

    def _warp(self, disp: Tensor) -> tuple[Tensor, Tensor]:
        """Profile and its per-reflection log-determinant penalty.

        `p(x) = template(x + v(x)) |det J(x)|` with `J = I + grad v`, which is
        the density transform rather than a plain resample -- without the
        determinant the warp would not conserve mass.
        """
        b = disp.shape[0]
        d_, h_, w_ = self.profile_shape
        ctrl = disp.reshape(b, 3, *self.lattice)
        mode = "trilinear" if d_ > 1 else "trilinear"
        v = F.interpolate(
            ctrl, size=(d_, h_, w_), mode=mode, align_corners=True
        )  # (B, 3, D, H, W), displacement in PIXELS, order (z, y, x)

        # Jacobian of the displacement by central differences, in pixel units
        grads = []
        for axis, n in zip((2, 3, 4), (d_, h_, w_)):
            if n < 2:
                grads.append(torch.zeros_like(v))
                continue
            lo = torch.roll(v, 1, dims=axis)
            hi = torch.roll(v, -1, dims=axis)
            grads.append(0.5 * (hi - lo))
        # jac[..., i, j] = d v_i / d x_j
        jac = torch.stack(grads, dim=-1).permute(0, 2, 3, 4, 1, 5)
        eye = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(jac)
        det = torch.linalg.det(eye + jac).abs().clamp_min(1e-6)
        log_det = det.log().reshape(b, -1)
        if self.pin_log_det:
            # The mean of log det J is the concentration of the profile, and
            # that is the one direction the counts cannot identify: a 10x
            # intensity error costs zero log-likelihood against a free flat
            # background. Removing its mean leaves shear and translation free.
            log_det = log_det - log_det.mean(dim=-1, keepdim=True)

        # grid_sample wants normalized (x, y, z) and a trailing coordinate axis
        scale = torch.tensor(
            [
                2.0 / max(w_ - 1, 1),
                2.0 / max(h_ - 1, 1),
                2.0 / max(d_ - 1, 1),
            ],
            device=v.device,
            dtype=v.dtype,
        )
        v_norm = torch.stack(
            [v[:, 2], v[:, 1], v[:, 0]], dim=-1
        ) * scale  # (B, D, H, W, 3)
        grid = self._norm_grid.to(v.dtype) + v_norm

        template = torch.softmax(self.template_logits, dim=0)
        vol = template.reshape(1, 1, d_, h_, w_).expand(b, -1, -1, -1, -1)
        warped = F.grid_sample(
            vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).reshape(b, -1)
        p = warped.clamp_min(0.0) * log_det.exp()
        return p / p.sum(-1, keepdim=True).clamp_min(1e-12), log_det

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        metadata: dict | None = None,
        **kwargs,
    ) -> ProfileSurrogateOutput:
        feats = [x]
        counts = kwargs.get("counts")
        if counts is not None:
            c = counts.reshape(counts.shape[0], -1).to(x.dtype)
            total = c.sum(-1, keepdim=True)
            feats += [torch.log1p(total), c.new_zeros(c.shape[0], 3)]
        else:
            feats.append(x.new_zeros(x.shape[0], 4))
        if self.beam_center is not None:
            g, _ = geometry_features(
                metadata, self.beam_center, self.lambda_min, self.lambda_max
            )
            feats.append(g.to(x.dtype).to(x.device))
        raw = self.head(torch.cat(feats, dim=-1))
        loc, raw_scale = raw[:, : self.d], raw[:, self.d :]
        scale = F.softplus(raw_scale) + 1e-6

        mask = kwargs.get("mask")
        samples, energies = [], []
        for _ in range(mc_samples):
            disp = loc + scale * torch.randn_like(scale)
            prof, _ = self._warp(disp)
            samples.append(renormalize_masked(prof, mask))
            energies.append(self.bending_energy(disp))
        zp = torch.stack(samples, dim=0)
        mean_profile = renormalize_masked(self._warp(loc)[0], mask)

        kl = kl_divergence(
            Normal(loc, scale),
            Normal(torch.zeros_like(loc), torch.full_like(scale, self.prior_scale)),
        ).sum(dim=-1)
        if self.bending_weight > 0.0:
            # Carried in `kl` because that is the channel the integrator already
            # adds to the loss, so a rough warp costs something without touching
            # the loss classes.
            kl = kl + self.bending_weight * torch.stack(energies, 0).mean(0)
        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            loc=loc,
            scale=scale,
            prior_scale=self.prior_scale,
            kl=kl,
        )


class MixtureGaussianProfileSurrogate(nn.Module):
    """Profile as a mixture of oriented Gaussians, pixel-integrated.

    Aimed at the one deficit the truth figure measures. Across six resolution
    bins the true footprints have axis ratios 2.6 to 6.6 and the best arm we
    have reaches 2.6, and raising the Tucker rank from 4 to 12 buys elongation
    (2.1 -> 2.6) only by spending rank on the rotation, which costs intensity
    (I/sigma 0.71 -> 0.34). A mixture does not have that problem: one component
    is already an oriented ellipse, so elongation is free, and several strung
    along the streak give a bent or asymmetric shape that no single ellipse and
    no low-rank separable basis reaches cheaply.

        p = sum_k w_k N(x; mu_k, Sigma_k),    w on the simplex

    `K` components cost `6K` numbers in 2D -- weight, two centre offsets, two
    log extents and an angle -- against 1681 pixels, so `K = 4` is 24 numbers
    per reflection. Non-negative and normalized by construction, with no
    softmax over pixels and no positivity clamp.

    GAUGE. A mixture can still go flat by inflating its covariances, which is
    the direction the counts cannot identify, so `max_log_scale` bounds each
    extent around `base_sigma` through a tanh rather than leaving it free.

    ROTATED FRAME. With `beam_center` set, each component's angle is measured
    from the analytic azimuth about the beam centre rather than from the
    detector axes, so a radial streak is the zero-angle case and the model
    starts aligned with it instead of discovering it.

    Args:
        input_dim: Encoder width, used only when `tie_encoder` is false.
        n_components: Mixture size `K`.
        output_dim: Profile pixels `D*H*W`.
        profile_shape: Grid as `(D, H, W)`; inferred as a square when unset.
        base_sigma: Centre of the allowed extent range, in pixels.
        max_log_scale: Bound on `|log(sigma / base_sigma)|`.
        max_shift: Bound on a component's centre offset, in pixels.
        quad: Sub-pixel quadrature points per axis. A spot narrower than a
            pixel is badly misrepresented by its value at the pixel centre.
        rotated_frame: Measure component angles from the beam-centre azimuth.
        tie_encoder: Build features from the shoebox summaries rather than the
            upstream encoder output.
        n_quantiles: Count quantiles in the feature vector.
        hidden: Width of the MLP mapping features to mixture parameters.
    """

    def __init__(
        self,
        input_dim: int,
        n_components: int = 4,
        output_dim: int = 441,
        profile_shape: tuple[int, int, int] | list[int] | None = None,
        init_std: float = 0.5,
        prior_scale: float = 3.0,
        base_sigma: float = 2.0,
        max_log_scale: float = 1.5,
        max_shift: float = 6.0,
        quad: int = 2,
        rotated_frame: bool = True,
        tie_encoder: bool = True,
        n_quantiles: int = 5,
        hidden: int = 32,
        beam_center: tuple[float, float] | list[float] | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
    ) -> None:
        super().__init__()
        if profile_shape is None:
            side = int(round(math.sqrt(output_dim)))
            if side * side != output_dim:
                raise ValueError(
                    f"output_dim {output_dim} is not square; pass profile_shape"
                )
            profile_shape = (1, side, side)
        shape = tuple(int(v) for v in profile_shape)
        if shape[0] * shape[1] * shape[2] != output_dim:
            raise ValueError(
                f"profile_shape {shape} does not hold output_dim={output_dim}"
            )
        self.profile_shape = shape
        self.is_3d = shape[0] > 1
        self.k = int(n_components)
        self.base_sigma = float(base_sigma)
        self.max_log_scale = float(max_log_scale)
        self.max_shift = float(max_shift)
        self.quad = max(1, int(quad))
        self.rotated_frame = bool(rotated_frame) and beam_center is not None
        self.tie_encoder = bool(tie_encoder)
        self.n_quantiles = int(n_quantiles)
        self.prior_scale = float(prior_scale)
        self.beam_center = tuple(beam_center) if beam_center is not None else None
        self.lambda_min, self.lambda_max = lambda_min, lambda_max

        # weight, centre (2 or 3), log extents (2 or 3), one in-plane angle
        self.n_par = 1 + (3 if self.is_3d else 2) * 2 + 1
        self.d = self.k * self.n_par

        d_, h_, w_ = shape
        axes = [
            torch.arange(n, dtype=torch.float32) - 0.5 * (n - 1) for n in shape
        ]
        zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
        # `.contiguous()` is load-bearing: `torch.meshgrid` returns EXPANDED
        # views, so on a depth-1 grid `zz.reshape(-1)` is 1681 elements over a
        # single storage element with stride 0. Training is unaffected, and
        # `load_state_dict` then fails with "more than one element of the
        # written-to tensor refers to a single memory location" -- so the
        # failure only appears at predict, after the run has already cost its
        # full training time.
        self.register_buffer("_gz", zz.reshape(-1).contiguous())
        self.register_buffer("_gy", yy.reshape(-1).contiguous())
        self.register_buffer("_gx", xx.reshape(-1).contiguous())
        off = (torch.arange(self.quad, dtype=torch.float32) + 0.5) / self.quad - 0.5
        self.register_buffer("_qoff", off)

        n_feat = (6 if tie_encoder else input_dim) + self.n_quantiles
        if beam_center is not None:
            n_feat += N_GEOM_FEATURES
        self.n_feat = n_feat
        self.head = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.SiLU(), nn.Linear(hidden, self.d)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        # spread the components along one line at init, so they do not start
        # stacked on each other with nothing to break the symmetry
        with torch.no_grad():
            spread = torch.linspace(-1.0, 1.0, self.k) * 0.4
            self.head[-1].bias[1 :: self.n_par] = spread
        self.loc_scale = nn.Parameter(torch.full((self.d,), _softplus_inverse(init_std)))

    def _features(self, x, counts, mask, metadata):
        if self.tie_encoder and counts is None:
            raise ValueError(
                "MixtureGaussianProfileSurrogate with tie_encoder needs `counts`"
            )
        feats = []
        if self.tie_encoder:
            c = counts.reshape(counts.shape[0], -1).to(self._gx.dtype)
            m = (
                mask.reshape(c.shape).to(c.dtype)
                if mask is not None
                else torch.ones_like(c)
            )
            c = c * m
            total = c.sum(-1, keepdim=True)
            p_obs = c / total.clamp_min(1.0)
            cy = (p_obs * self._gy).sum(-1, keepdim=True)
            cx = (p_obs * self._gx).sum(-1, keepdim=True)
            vy = (p_obs * (self._gy - cy) ** 2).sum(-1, keepdim=True)
            vx = (p_obs * (self._gx - cx) ** 2).sum(-1, keepdim=True)
            feats += [torch.log1p(total), cy, cx, vy.sqrt(), vx.sqrt(),
                      (p_obs * (self._gy - cy) * (self._gx - cx)).sum(-1, keepdim=True)]
            if self.n_quantiles > 0:
                q = torch.linspace(
                    0.5, 1.0, self.n_quantiles + 1, device=c.device, dtype=c.dtype
                )[:-1]
                feats.append(torch.quantile(p_obs, q, dim=-1).transpose(0, 1))
        else:
            feats.append(x)
            if self.n_quantiles > 0:
                feats.append(x.new_zeros(x.shape[0], self.n_quantiles))
        phi = None
        if self.beam_center is not None:
            g, phi = geometry_features(
                metadata, self.beam_center, self.lambda_min, self.lambda_max
            )
            feats.append(g.to(feats[0].dtype).to(feats[0].device))
            phi = phi.to(feats[0].dtype).to(feats[0].device)
        return torch.cat(feats, dim=-1), phi

    def _components(self, loc: Tensor, phi: Tensor | None) -> Tensor:
        """The `K` component densities, shape `(B, K, P)`.

        Built from the posterior MEAN of the pose, not from each Monte Carlo
        sample. With `mc_samples = 50` a sampled pose rebuilds every component
        per sample, which makes one `(50, 512, 4, 1681)` intermediate of 688 MB
        and several of them -- the 19 GiB that would not fit on a 20 GiB MIG
        slice. The steerable family never had this problem because its basis
        depends on the reflection and not on the sample, and the same split
        applies here: the pose is a deterministic function of the features and
        only the mixing weights carry the uncertainty.
        """
        t = loc.reshape(loc.shape[0], self.k, self.n_par)
        return self._gaussians(t, phi)

    def _gaussians(self, t: Tensor, phi: Tensor | None) -> Tensor:
        """Component densities from `(B, K, n_par)` poses, shape `(B, K, P)`."""
        b = t.shape[0]
        n_c = 3 if self.is_3d else 2
        shift = self.max_shift * torch.tanh(t[..., 1 : 1 + n_c])
        sig = self.base_sigma * (
            self.max_log_scale * torch.tanh(t[..., 1 + n_c : 1 + 2 * n_c])
        ).exp()
        ang = t[..., -1]
        if phi is not None:
            ang = ang + phi.reshape(b, 1)
        cos, sin = torch.cos(ang).unsqueeze(-1), torch.sin(ang).unsqueeze(-1)

        if self.is_3d:
            dz0 = self._gz.reshape(1, 1, -1) - shift[..., 0:1]
            dy0 = self._gy.reshape(1, 1, -1) - shift[..., 1:2]
            dx0 = self._gx.reshape(1, 1, -1) - shift[..., 2:3]
            sz, sy, sx = sig[..., 0:1], sig[..., 1:2], sig[..., 2:3]
        else:
            dz0 = None
            dy0 = self._gy.reshape(1, 1, -1) - shift[..., 0:1]
            dx0 = self._gx.reshape(1, 1, -1) - shift[..., 1:2]
            sy, sx = sig[..., 0:1], sig[..., 1:2]

        wq = 1.0 / (self.quad ** (3 if self.is_3d else 2))
        z_offs = self._qoff if self.is_3d else torch.zeros(1, device=t.device)
        acc = None
        for oy in self._qoff:
            for ox in self._qoff:
                for oz in z_offs:
                    dy, dx = dy0 + oy, dx0 + ox
                    u = (cos * dx + sin * dy) / sx.clamp_min(1e-3)
                    v = (-sin * dx + cos * dy) / sy.clamp_min(1e-3)
                    q2 = u * u + v * v
                    if self.is_3d:
                        dz = (dz0 + oz) / sz.clamp_min(1e-3)
                        q2 = q2 + dz * dz
                    term = torch.exp(-0.5 * q2) * wq
                    acc = term if acc is None else acc + term
        return acc / acc.sum(-1, keepdim=True).clamp_min(1e-12)

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        metadata: dict | None = None,
        **kwargs,
    ) -> ProfileSurrogateOutput:
        mask = kwargs.get("mask")
        feats, phi = self._features(x, kwargs.get("counts"), mask, metadata)
        loc = self.head(feats)
        scale = F.softplus(self.loc_scale).expand_as(loc) + 1e-6

        comps = self._components(loc, phi)                     # (B, K, P)
        w_loc = loc.reshape(loc.shape[0], self.k, self.n_par)[..., 0]
        w_scale = scale.reshape(scale.shape[0], self.k, self.n_par)[..., 0]
        eps = torch.randn(
            mc_samples, *w_loc.shape, device=loc.device, dtype=loc.dtype
        )
        w = torch.softmax(w_loc.unsqueeze(0) + w_scale.unsqueeze(0) * eps, dim=-1)
        p = torch.einsum("sbk,bkp->sbp", w, comps)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
        zp = renormalize_masked(p, mask)
        p0 = torch.einsum("bk,bkp->bp", torch.softmax(w_loc, dim=-1), comps)
        mean_profile = renormalize_masked(
            (p0 / p0.sum(-1, keepdim=True).clamp_min(1e-12)).unsqueeze(0), mask
        )[0]
        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            loc=loc,
            scale=scale,
            prior_scale=self.prior_scale,
        )


def _zernike_radial(r: Tensor, n: int, m: int) -> Tensor:
    """Zernike radial polynomial `R_n^m(r)`, zero unless `n - m` is even."""
    out = torch.zeros_like(r)
    for k in range((n - m) // 2 + 1):
        c = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)
            )
        )
        out = out + c * r.pow(n - 2 * k)
    return out


def zernike_basis(
    coords_u: Tensor, coords_v: Tensor, max_order: int
) -> tuple[Tensor, list[tuple[int, int, str]]]:
    """Zernike functions on the unit disc, evaluated at given coordinates.

    Returns `(..., K)` and the `(n, m, kind)` label of each column. Orthogonal
    on the disc, and STEERABLE: a rotation by `alpha` mixes the `cos(m phi)`
    and `sin(m phi)` columns of order `m` by a plane rotation through `m
    alpha`, exactly, with no interpolation and no extra terms. That is why the
    streak's orientation costs this family nothing, where a separable basis in
    detector x, y has to build a diagonal out of many terms -- which is what
    made the Tucker arms spend rank 4 -> 12 buying axis ratio 2.1 -> 2.6.
    """
    # `hypot` and `atan2` are both singular at the origin: the forward value is
    # fine but the gradient is 0/0, and a centred odd grid has a pixel at
    # exactly (0, 0) with the pose's shift starting at zero. That is a silent
    # NaN the moment anything backpropagates. So the radius is floored and the
    # angular functions come from a Chebyshev recurrence on (cos phi, sin phi)
    # rather than from an angle, which never forms the angle at all.
    r2 = coords_u * coords_u + coords_v * coords_v
    r = torch.sqrt(r2 + 1e-12)
    cos1, sin1 = coords_u / r, coords_v / r
    inside = (r <= 1.0).to(r.dtype)

    harmonics = {0: (torch.ones_like(r), torch.zeros_like(r))}
    for m in range(1, max_order + 1):
        cp, sp = harmonics[m - 1]
        harmonics[m] = (cp * cos1 - sp * sin1, sp * cos1 + cp * sin1)

    cols, names = [], []
    for n in range(max_order + 1):
        for m in range(n + 1):
            if (n - m) % 2:
                continue
            rad = _zernike_radial(r.clamp(max=1.0), n, m) * inside
            cm, sm = harmonics[m]
            if m == 0:
                cols.append(rad)
                names.append((n, 0, "c"))
            else:
                cols.append(rad * cm)
                names.append((n, m, "c"))
                cols.append(rad * sm)
                names.append((n, m, "s"))
    return torch.stack(cols, dim=-1), names


class SteerableProfileSurrogate(nn.Module):
    """Profile as a canonical Zernike shape carried by a pose read from geometry.

        p(x) = ( sum_k c_k psi_k( A^-1 (x - mu) ) )^2,  normalized

    SHAPE AND POSE ARE SEPARATED, which is the point. Every basis family tried
    so far spends capacity on pose: the Tucker arms bought axis ratio 2.1 ->
    2.6 only by raising rank 4 -> 12, because a separable basis in detector
    x, y has to construct a diagonal streak out of many terms. Here the stretch
    and the rotation are `A`, two and one numbers, exact; the centre is `mu`,
    two numbers, exact; and the basis only has to carry what is left, which is
    the residual shape. Off-centre spots and elongation therefore cost the
    basis nothing.

    THE BASIS IS ANALYTIC AND ORTHOGONAL on the disc, so the coefficients come
    from a PROJECTION rather than a network: `c = Psi^T sqrt(p_obs)` evaluated
    in the posed frame. Squaring puts the profile on the Hellinger sphere,
    where the natural inner product is the one the multinomial likelihood
    induces, and makes it non-negative without a softmax or a clamp. A small
    geometry head adds a learned residual to `c`, so the model is still
    amortized and still sees where the reflection sits, but the capability
    lives in the basis and the pose rather than in an encoder.

    SPEED. No convolution and no per-pixel decoder: one analytic basis
    evaluation, one matrix multiply for the projection, one square.

    Args:
        input_dim: Encoder width; used only by the optional residual head.
        max_order: Highest Zernike radial order. 4 gives 9 functions, 6 gives
            16, which is the whole shape budget.
        output_dim: Profile pixels `D*H*W`.
        profile_shape: Grid as `(D, H, W)`.
        base_radius: Disc radius in pixels at unit scale.
        max_log_scale: Bound on `|log| of each pose extent.
        max_shift: Bound on the centre offset, in pixels.
        learned_residual: Add a geometry-and-summary head to the projected
            coefficients.
        project: Take the coefficients from the data by projection. With it
            off the head supplies them alone, which is the ablation that says
            how much the projection is worth.
        beam_center: Detector beam centre; the pose's rotation is measured
            from the analytic azimuth about it, so a radial streak is the
            zero-angle case.
        lambda_min, lambda_max: Spectrum band for the wavelength feature.
    """

    def __init__(
        self,
        input_dim: int,
        max_order: int = 6,
        output_dim: int = 441,
        profile_shape: tuple[int, int, int] | list[int] | None = None,
        base_radius: float = 8.0,
        max_log_scale: float = 1.2,
        max_shift: float = 6.0,
        init_std: float = 0.3,
        prior_scale: float = 1.0,
        learned_residual: bool = True,
        project: bool = True,
        depth_mode: str = "simplex",
        hidden: int = 32,
        beam_center: tuple[float, float] | list[float] | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
    ) -> None:
        super().__init__()
        if profile_shape is None:
            side = int(round(math.sqrt(output_dim)))
            if side * side != output_dim:
                raise ValueError(
                    f"output_dim {output_dim} is not square; pass profile_shape"
                )
            profile_shape = (1, side, side)
        shape = tuple(int(v) for v in profile_shape)
        if shape[0] * shape[1] * shape[2] != output_dim:
            raise ValueError(
                f"profile_shape {shape} does not hold output_dim={output_dim}"
            )
        if depth_mode not in ("simplex", "shared"):
            raise ValueError(f"depth_mode {depth_mode!r} is not 'simplex' or 'shared'")
        self.profile_shape = shape
        self.max_order = int(max_order)
        self.base_radius = float(base_radius)
        self.max_log_scale = float(max_log_scale)
        self.max_shift = float(max_shift)
        self.project = bool(project)
        self.learned_residual = bool(learned_residual)
        self.prior_scale = float(prior_scale)
        self.beam_center = tuple(beam_center) if beam_center is not None else None
        self.lambda_min, self.lambda_max = lambda_min, lambda_max

        d_, h_, w_ = shape
        axes = [
            torch.arange(n, dtype=torch.float32) - 0.5 * (n - 1) for n in shape
        ]
        zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
        # `.contiguous()` is load-bearing: `torch.meshgrid` returns EXPANDED
        # views, so on a depth-1 grid `zz.reshape(-1)` is 1681 elements over a
        # single storage element with stride 0. Training is unaffected, and
        # `load_state_dict` then fails with "more than one element of the
        # written-to tensor refers to a single memory location" -- so the
        # failure only appears at predict, after the run has already cost its
        # full training time.
        self.register_buffer("_gz", zz.reshape(-1).contiguous())
        self.register_buffer("_gy", yy.reshape(-1).contiguous())
        self.register_buffer("_gx", xx.reshape(-1).contiguous())

        probe, names = zernike_basis(
            torch.zeros(1), torch.zeros(1), self.max_order
        )
        self.n_basis = probe.shape[-1]
        self.basis_names = names
        # pose: two log extents, one angle offset, two centre offsets
        self.n_pose = 5
        self.d = self.n_basis

        n_feat = N_GEOM_FEATURES if beam_center is not None else input_dim
        n_feat += 4
        self.pose_head = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.SiLU(), nn.Linear(hidden, self.n_pose)
        )
        nn.init.zeros_(self.pose_head[-1].weight)
        nn.init.zeros_(self.pose_head[-1].bias)

        # A thin axis gets a FREE categorical rather than a parametric curve.
        # On hewl_9b7c the fitted rocking width was sqrt(0.158) = 0.40 px over
        # three frames -- sub-frame, essentially all the intensity in one. A
        # Gaussian there spends two numbers (mu, sigma) on three data points
        # AND assumes a shape the rocking curve need not have, since a curve
        # sampled at three points can be asymmetric. A simplex over D frames is
        # D - 1 numbers, no more, strictly more expressive, and cannot be wrong
        # about the functional form. Use the nonparametric model wherever the
        # dimension is small enough that it is the cheaper one.
        self.depth_head = None
        if self.profile_shape[0] > 1 and depth_mode == "simplex":
            self.depth_head = nn.Linear(n_feat, self.profile_shape[0])
            nn.init.zeros_(self.depth_head.weight)
            nn.init.zeros_(self.depth_head.bias)
        self.depth_mode = depth_mode
        self.res_head = None
        if learned_residual:
            self.res_head = nn.Sequential(
                nn.Linear(n_feat, hidden), nn.SiLU(), nn.Linear(hidden, self.n_basis)
            )
            nn.init.zeros_(self.res_head[-1].weight)
            nn.init.zeros_(self.res_head[-1].bias)
        # the canonical shape the projection is a correction to
        # The canonical shape must start PEAKED. `Z_0^0` is 1 over the whole
        # disc, so `c0 = [1, 0, ...]` is exactly the uniform-disc profile --
        # the flat attractor, and the first run collapsed onto it (n_eff 928
        # of 1681 against a uniform profile's 1681). Adding a negative
        # `Z_2^0 = 2r^2 - 1` makes the amplitude `1 - a(2r^2 - 1)`, which peaks
        # at the centre and vanishes near the rim.
        self.c0 = nn.Parameter(torch.zeros(self.n_basis))
        with torch.no_grad():
            self.c0[0] = 1.0
            for j, (n, m, kind) in enumerate(names):
                if (n, m, kind) == (2, 0, "c"):
                    self.c0[j] = -0.85
                elif (n, m, kind) == (4, 0, "c"):
                    self.c0[j] = 0.25
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.loc_scale = nn.Parameter(
            torch.full((self.n_basis,), _softplus_inverse(init_std))
        )

    def _posed_basis(self, pose: Tensor, phi: Tensor | None) -> Tensor:
        """Zernike columns evaluated at `A^-1 (x - mu)`, shape `(B, P, K)`."""
        mu = self.max_shift * torch.tanh(pose[:, 0:2])
        logs = self.max_log_scale * torch.tanh(pose[:, 2:4])
        ang = pose[:, 4:5]
        if phi is not None:
            ang = ang + phi.reshape(-1, 1)
        s = self.base_radius * self.log_scale.exp() * logs.exp()

        dy = self._gy.reshape(1, -1) - mu[:, 0:1]
        dx = self._gx.reshape(1, -1) - mu[:, 1:2]
        cos, sin = torch.cos(ang), torch.sin(ang)
        u = (cos * dx + sin * dy) / s[:, 0:1].clamp_min(1e-3)
        v = (-sin * dx + cos * dy) / s[:, 1:2].clamp_min(1e-3)
        psi, _ = zernike_basis(u, v, self.max_order)
        return psi

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        metadata: dict | None = None,
        **kwargs,
    ) -> ProfileSurrogateOutput:
        mask = kwargs.get("mask")
        counts = kwargs.get("counts")
        phi = None
        if self.beam_center is not None:
            g, phi = geometry_features(
                metadata, self.beam_center, self.lambda_min, self.lambda_max
            )
            feats = g.to(x.dtype).to(x.device)
            phi = phi.to(x.dtype).to(x.device)
        else:
            feats = x
        if counts is not None:
            c = counts.reshape(counts.shape[0], -1).to(x.dtype)
            m = (
                mask.reshape(c.shape).to(c.dtype)
                if mask is not None
                else torch.ones_like(c)
            )
            c = c * m
            # Subtract the background BEFORE projecting. Laue shoeboxes are
            # 96.6% background, so the raw normalized box is nearly uniform and
            # projecting it hands the basis background rather than shape --
            # which is how the first run ended up flat. The border median is
            # profile-free, so it cannot smuggle the answer in.
            bg = _border_median(c, self.profile_shape)
            sig = (c - bg).clamp_min(0.0) * m
            tot = sig.sum(-1, keepdim=True)
            p_obs = sig / tot.clamp_min(1.0)
            cy = (p_obs * self._gy).sum(-1, keepdim=True)
            cx = (p_obs * self._gx).sum(-1, keepdim=True)
            feats = torch.cat([feats, torch.log1p(tot), cy, cx,
                               p_obs.max(-1, keepdim=True).values], dim=-1)
        else:
            p_obs = None
            feats = torch.cat([feats, x.new_zeros(x.shape[0], 4)], dim=-1)

        psi = self._posed_basis(self.pose_head(feats), phi)

        coef = self.c0.reshape(1, -1).expand(x.shape[0], -1)
        if self.project and p_obs is not None:
            # the projection: the basis is orthogonal on the disc, so the
            # amplitude's coefficients are an inner product, not a fit
            coef = coef + torch.einsum("bp,bpk->bk", p_obs.clamp_min(0).sqrt(), psi)
        if self.res_head is not None:
            coef = coef + self.res_head(feats)
        loc = coef
        scale = F.softplus(self.loc_scale).expand_as(loc) + 1e-6

        depth = None
        if self.depth_head is not None:
            d_, h_, w_ = self.profile_shape
            depth = torch.softmax(self.depth_head(feats), dim=-1)
            depth = depth.reshape(-1, d_, 1).expand(-1, d_, h_ * w_).reshape(
                -1, d_ * h_ * w_
            )

        def decode(cf: Tensor) -> Tensor:
            amp = torch.einsum("sbk,bpk->sbp", cf, psi)
            p = amp * amp
            if depth is not None:
                # separable: the rocking direction and the detector plane are
                # different physics, so they get different parameterizations
                p = p * depth.unsqueeze(0)
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
            return renormalize_masked(p, mask)

        eps = torch.randn(
            mc_samples, *loc.shape, device=loc.device, dtype=loc.dtype
        )
        zp = decode(loc.unsqueeze(0) + scale.unsqueeze(0) * eps)
        mean_profile = decode(loc.unsqueeze(0))[0]
        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            loc=loc,
            scale=scale,
            prior_scale=self.prior_scale,
        )


def solve_intensity_background(
    counts: Tensor,
    mask: Tensor,
    profile: Tensor,
    n_iter: int = 5,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Profile out `I` and the flat background by Poisson IRLS.

    For a fixed profile the Poisson log-likelihood is concave in `(I, b)` --
    the rate is linear in them and the log-likelihood is concave in the rate --
    so these two solve exactly in a few passes and need no variational family.
    The variance is the Schur complement, which is the correct frequentist one
    rather than a choice of family:

        Var(I) = f_bb / (f_ii f_bb - f_ib^2) = 1 / (sum_j w_j Var_w(p))

    verified against `fit.direct.solve_profile_fit` to 1.1e-6. The determinant
    vanishes exactly when the profile is flat, which is the statement that a
    uniform profile carries no information about the intensity.

    Returns `(I, b, Var(I))`, each shape `(..., )` over the leading dims.
    """
    m = mask.to(counts.dtype)
    c = counts * m
    p = profile * m
    i_hat = (c - (c * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1.0)).sum(-1).clamp_min(1.0)
    b_hat = (c.sum(-1) / m.sum(-1).clamp_min(1.0)).clamp_min(eps)
    for _ in range(max(1, n_iter)):
        rate = (i_hat.unsqueeze(-1) * p + b_hat.unsqueeze(-1)).clamp_min(eps)
        w = m / rate
        f_ii = (w * p * p).sum(-1)
        f_ib = (w * p).sum(-1)
        f_bb = (w).sum(-1).clamp_min(eps)
        r_i = (w * p * c).sum(-1)
        r_b = (w * c).sum(-1)
        det = (f_ii * f_bb - f_ib * f_ib)
        det = torch.where(det.abs() < 1e-30, torch.full_like(det, 1e-30), det)
        i_hat = (r_i * f_bb - r_b * f_ib) / det
        b_hat = ((r_b * f_ii - r_i * f_ib) / det).clamp_min(0.0)
        i_hat = torch.nan_to_num(i_hat, nan=0.0)
    rate = (i_hat.clamp_min(0.0).unsqueeze(-1) * p + b_hat.unsqueeze(-1)).clamp_min(eps)
    w = m / rate
    f_ii = (w * p * p).sum(-1)
    f_ib = (w * p).sum(-1)
    f_bb = (w).sum(-1).clamp_min(eps)
    var = f_bb / (f_ii * f_bb - f_ib * f_ib).clamp_min(1e-30)
    return i_hat, b_hat, var


class ShiftedBasisProfileSurrogate(ProfileSurrogate):
    """The free learned basis, plus the one operation a linear basis cannot do.

    The measurements say the plain learned basis is the most capable family we
    have: against the simulation's truth it reaches cos 0.958-0.962 on bright
    reflections where the structured families reach 0.927-0.951, recovers twice
    as much off-centring (slope 0.431 against 0.205-0.260) and is the only one
    that is appreciably asymmetric (symmetry 0.928 against 0.975-0.990, with
    the truth at 0.574). Every constraint added on top of it -- low rank,
    separability, a pose, co-located Gaussians -- bought identifiability by
    removing exactly the freedom that off-centre and asymmetric spots need.

    So this adds capability instead of structure, and only where a linear basis
    is provably weak. `softmax(W h + b)` is linear in log-density, and a
    TRANSLATION is not: shifting a blob by one pixel requires a coordinated
    change in every basis vector, which is why the basis recovers less than
    half the true offset. Here the shift is applied directly, by a phase ramp
    in the Fourier domain, which is exact for a band-limited profile.

    WHERE THE SHIFT COMES FROM. The observed centroid is the sufficient
    statistic for a translation, but it is unusable raw: over all reflections
    it tracks the true offset with slope 0.071, because at low signal it is
    measuring background. On the bright quartile it reaches 0.536. So it is
    shrunk toward the predicted centre by the counts,

        shift = n / (n + kappa) * centroid_obs + learned correction,

    with one fitted `kappa`. Bright reflections use their own centroid, weak
    ones fall back, and the weight is set by the data rather than by a
    hyperparameter.

    Args:
        shift_kappa: Initial signal-count scale at which the observed centroid
            is half-trusted. Fitted in log space from there.
        max_shift: Bound on the applied shift, in pixels.
        learn_shift_residual: Add a small geometry-and-summary correction on
            top of the shrunk centroid.
        profile_shape: Grid as `(D, H, W)`; inferred as a square when unset.
        beam_center: Enables the geometry features for the residual head.
        Remaining arguments are `ProfileSurrogate`'s and behave identically.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int | None = None,
        output_dim: int = 441,
        init_std: float = 0.5,
        prior_scale: float = 3.0,
        warmstart_basis_path: str | None = None,
        freeze_bias: bool = False,
        normalization: str = "softmax",
        smoothness_weight: float = 0.0,
        shift_kappa: float = 200.0,
        max_shift: float = 8.0,
        learn_shift_residual: bool = True,
        scale_groups: list[int] | None = None,
        tau_decay: float = 0.45,
        init_group_sigma: float = 6.0,
        fixed_embedding: int = 0,
        fixed_embedding_kind: str = "rff",
        rff_bandwidth: float = 4.0,
        profile_shape: tuple[int, int, int] | list[int] | None = None,
        beam_center: tuple[float, float] | list[float] | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
        hidden: int = 32,
    ) -> None:
        # The parent's arguments are declared explicitly rather than taken
        # through `*args, **kwargs`: the config validator reads this
        # signature, and with them hidden it rejected `latent_dim`,
        # `output_dim`, `init_std` and `prior_scale` as unknown.
        super().__init__(
            input_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            init_std=init_std,
            prior_scale=prior_scale,
            warmstart_basis_path=warmstart_basis_path,
            freeze_bias=freeze_bias,
            normalization=normalization,
            smoothness_weight=smoothness_weight,
        )
        out_dim = self.decoder.weight.shape[0]
        if profile_shape is not None:
            shape = tuple(int(v) for v in profile_shape)
        else:
            side = int(round(math.sqrt(out_dim)))
            if side * side != out_dim:
                raise ValueError(
                    f"output_dim {out_dim} is not square; pass profile_shape"
                )
            shape = (1, side, side)
        if shape[0] * shape[1] * shape[2] != out_dim:
            raise ValueError(f"profile_shape {shape} does not hold {out_dim}")
        self.profile_shape = shape
        self.max_shift = float(max_shift)
        self.beam_center = tuple(beam_center) if beam_center is not None else None
        self.lambda_min, self.lambda_max = lambda_min, lambda_max
        self.log_kappa = nn.Parameter(torch.tensor(math.log(max(shift_kappa, 1.0))))

        d_, h_, w_ = shape
        axes = [
            torch.arange(n, dtype=torch.float32) - 0.5 * (n - 1) for n in shape
        ]
        zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
        self.register_buffer("_gy", yy.reshape(-1).contiguous())
        self.register_buffer("_gx", xx.reshape(-1).contiguous())

        # Multiscale prior on the SAME free basis. Nothing is constrained --
        # `softmax(W h + b)` is untouched and every profile remains reachable.
        # What changes is the price: a coefficient in a fine-scale group has a
        # tighter prior, so it costs more KL, and it therefore activates only
        # where the likelihood gain exceeds that cost. Since the KL cost is
        # fixed while the Poisson information at a pixel is its expected count,
        # the model keeps detail where there are counts and reverts to the
        # coarse groups where there are not. The regions never have to be
        # named: the selection falls out of a constant cost meeting a varying
        # evidence. This is Donoho-Johnstone shrinkage written as a prior.
        # A FIXED embedding of the shoebox, in place of the learned encoder.
        # Two kinds, both training-free and both with a guarantee:
        #
        # `rff` -- the kernel mean embedding. A shoebox IS a distribution over
        #   pixel positions, and for a characteristic kernel the mean embedding
        #   `mu_P = E[phi(X)]` is INJECTIVE, so distinct profiles map to
        #   distinct codes and nothing is lost. Random Fourier features give
        #   the finite-dimensional approximation with a uniform error bound.
        #
        # `rp` -- a random projection. Johnson-Lindenstrauss: `m ~ log n / eps^2`
        #   dimensions preserve every pairwise distance to `(1 +- eps)`,
        #   distribution-free.
        #
        # The empirical case for trying this at all: the profile surrogates
        # that IGNORE the encoder output entirely matched or beat the ones that
        # use it, and for `softmax(W h + b)` the sufficient statistic is
        # `W^T c`, so no encoder can extract more than a projection for that
        # family.
        self.fixed_embedding = int(fixed_embedding)
        self.register_buffer("_embed", None)
        if self.fixed_embedding > 0:
            if fixed_embedding_kind not in ("rff", "rp"):
                raise ValueError(
                    f"fixed_embedding_kind {fixed_embedding_kind!r} is not "
                    "'rff' or 'rp'"
                )
            n_px = shape[0] * shape[1] * shape[2]
            g = torch.Generator().manual_seed(0)
            m_dim = self.fixed_embedding
            if fixed_embedding_kind == "rp":
                emb = torch.randn(n_px, m_dim, generator=g) / math.sqrt(m_dim)
            else:
                ax = [
                    torch.arange(n, dtype=torch.float32) - 0.5 * (n - 1)
                    for n in shape
                ]
                zz, yy, xx = torch.meshgrid(*ax, indexing="ij")
                coord = torch.stack(
                    [zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], dim=-1
                )
                w = torch.randn(3, m_dim, generator=g) / max(rff_bandwidth, 1e-3)
                b = 2 * math.pi * torch.rand(m_dim, generator=g)
                emb = math.sqrt(2.0 / m_dim) * torch.cos(coord @ w + b)
            self._embed = emb
            # the coefficient heads now read the embedding, not the encoder
            self.loc_head = nn.Linear(m_dim, self.d)
            self.scale_head = nn.Linear(m_dim, self.d)
            nn.init.zeros_(self.scale_head.weight)
            nn.init.constant_(self.scale_head.bias, _softplus_inverse(init_std))
        self.fixed_embedding_kind = fixed_embedding_kind

        # registered as a buffer up front: a plain `self.tau = None` would
        # block `register_buffer` later, and it has to be a buffer so it moves
        # with the module and survives a state-dict round trip
        self.register_buffer("tau", None)
        if scale_groups:
            if sum(scale_groups) != self.d:
                raise ValueError(
                    f"scale_groups {scale_groups} sum to {sum(scale_groups)}, "
                    f"not latent_dim {self.d}"
                )
            tau = torch.cat([
                torch.full((n,), self.prior_scale * tau_decay**g)
                for g, n in enumerate(scale_groups)
            ])
            self.tau = tau
            # and give the columns matching spatial frequencies, so the
            # grouping refers to something: group g is a random field smoothed
            # to sigma / 2^g, coarse first
            with torch.no_grad():
                col = 0
                for g, n in enumerate(scale_groups):
                    sig = max(init_group_sigma / (2.0**g), 0.6)
                    for _ in range(n):
                        f = torch.randn(1, 1, shape[1], shape[2])
                        k = int(max(3, 2 * round(2 * sig) + 1))
                        ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
                        w1 = torch.exp(-0.5 * (ax / sig) ** 2)
                        w1 = (w1 / w1.sum()).reshape(1, 1, 1, -1)
                        f = F.conv2d(f, w1, padding=(0, k // 2))
                        f = F.conv2d(f, w1.transpose(-1, -2), padding=(k // 2, 0))
                        f = (f - f.mean()) / f.std().clamp_min(1e-6)
                        self.decoder.weight[:, col] = (
                            0.3 * f.reshape(-1).repeat(shape[0])
                        )
                        col += 1

        self.shift_head = None
        if learn_shift_residual:
            n_feat = 4 + (N_GEOM_FEATURES if beam_center is not None else 0)
            self.shift_head = nn.Sequential(
                nn.Linear(n_feat, hidden), nn.SiLU(), nn.Linear(hidden, 2)
            )
            nn.init.zeros_(self.shift_head[-1].weight)
            nn.init.zeros_(self.shift_head[-1].bias)

    def _shift_of(self, counts, mask, metadata):
        """Shrunk observed centroid, shape `(B, 2)` in pixels, order (y, x)."""
        c = counts.reshape(counts.shape[0], -1).to(self._gy.dtype)
        m = (
            mask.reshape(c.shape).to(c.dtype)
            if mask is not None
            else torch.ones_like(c)
        )
        c = c * m
        bg = _border_median(c, self.profile_shape)
        sig = (c - bg).clamp_min(0.0) * m
        n = sig.sum(-1, keepdim=True)
        p_obs = sig / n.clamp_min(1.0)
        cy = (p_obs * self._gy).sum(-1, keepdim=True)
        cx = (p_obs * self._gx).sum(-1, keepdim=True)
        # n / (n + kappa): the centroid is trusted in proportion to the signal
        # that measured it. Raw it tracks the true offset at slope 0.071
        # because at low signal it measures background.
        w = n / (n + self.log_kappa.exp().clamp_min(1.0))
        shift = w * torch.cat([cy, cx], dim=-1)
        if self.shift_head is not None:
            feats = [torch.log1p(n), cy, cx, p_obs.max(-1, keepdim=True).values]
            if self.beam_center is not None:
                g, _ = geometry_features(
                    metadata, self.beam_center, self.lambda_min, self.lambda_max
                )
                feats.append(g.to(c.dtype).to(c.device))
            shift = shift + self.shift_head(torch.cat(feats, dim=-1))
        return self.max_shift * torch.tanh(shift / self.max_shift)

    def forward(self, x, mc_samples: int = 1, **kwargs):
        if self._embed is not None:
            counts = kwargs.get("counts")
            if counts is None:
                raise ValueError(
                    "fixed_embedding needs `counts`; the integrator passes it "
                    "as the `counts` keyword"
                )
            c = counts.reshape(counts.shape[0], -1).to(self._embed.dtype)
            mask = kwargs.get("mask")
            m = (
                mask.reshape(c.shape).to(c.dtype)
                if mask is not None
                else torch.ones_like(c)
            )
            c = c * m
            sig = (c - _border_median(c, self.profile_shape)).clamp_min(0.0) * m
            p_obs = sig / sig.sum(-1, keepdim=True).clamp_min(1.0)
            x = p_obs @ self._embed
        out = super().forward(x, mc_samples=mc_samples, **kwargs)
        if self.tau is not None:
            # per-dimension prior, so the caller cannot form the KL from one
            # scalar scale; supply it here
            out.kl = kl_divergence(
                Normal(out.loc, out.scale),
                Normal(torch.zeros_like(out.loc), self.tau.expand_as(out.loc)),
            ).sum(dim=-1)
        counts = kwargs.get("counts")
        if counts is None:
            return out
        shift = self._shift_of(counts, kwargs.get("mask"), kwargs.get("metadata"))
        d_, h_, w_ = self.profile_shape

        def move(p: Tensor) -> Tensor:
            lead = p.shape[:-1]
            vol = p.reshape(-1, d_, h_, w_)
            s = shift
            if vol.shape[0] != s.shape[0]:
                s = s.unsqueeze(0).expand(
                    vol.shape[0] // s.shape[0], -1, -1
                ).reshape(-1, 2)
            fy = torch.fft.fftfreq(h_, device=p.device).reshape(1, h_, 1)
            fx = torch.fft.fftfreq(w_, device=p.device).reshape(1, 1, w_)
            phase = -2.0 * math.pi * (
                s[:, 0].reshape(-1, 1, 1) * fy + s[:, 1].reshape(-1, 1, 1) * fx
            )
            ker = torch.polar(torch.ones_like(phase), phase).unsqueeze(1)
            moved = torch.fft.ifft2(torch.fft.fft2(vol) * ker).real
            moved = moved.clamp_min(0.0).reshape(*lead, -1)
            return moved / moved.sum(-1, keepdim=True).clamp_min(1e-12)

        mask = kwargs.get("mask")
        out.zp = renormalize_masked(move(out.zp), mask)
        out.mean_profile = renormalize_masked(move(out.mean_profile), mask)
        return out
