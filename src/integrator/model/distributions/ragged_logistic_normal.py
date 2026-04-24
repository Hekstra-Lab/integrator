"""Ragged-compatible LogisticNormal profile surrogate.

Mirror of `LogisticNormalSurrogate` in `distributions/logistic_normal.py` but
with the fixed (441, d_basis) Hermite W replaced by a learned coord→basis MLP.

Key ideas:
  - Each voxel in a shoebox has a normalized coord (z, y, x) in [-1, 1]^3.
  - A shared small MLP maps (fourier-encoded coord) -> d_basis vector.
  - Profile logits = W_i @ h_i   where W_i is evaluated on this reflection's grid.
  - Masked softmax over voxels gives a profile summing to 1 on real voxels.
  - q(h | x) = Normal(mu(x), diag(sigma^2(x))); closed-form Gaussian KL to
    a Normal(0, sigma_prior^2) prior.

The output `RaggedProfilePosterior` exposes the same attributes the existing
loss code reads (`concentration = None`, `mean`, `rsample`, `kl_to_prior`).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _fourier_encode(coords: Tensor, n_freqs: int) -> Tensor:
    """Map (..., 3) coords in [-1, 1] to (..., 6*n_freqs) Fourier features.

    Uses log-spaced frequencies so the network sees both coarse (position)
    and fine (profile curvature) structure.
    """
    # Frequencies: pi, 2*pi, 4*pi, ... up to 2^(n_freqs-1) * pi
    freqs = (2.0 ** torch.arange(n_freqs, device=coords.device, dtype=coords.dtype)) * math.pi
    # (..., 3, n_freqs)
    phase = coords.unsqueeze(-1) * freqs
    pe = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)  # (..., 3, 2*n_freqs)
    return pe.flatten(-2, -1)  # (..., 6 * n_freqs)


def _build_normalized_coords(shapes: Tensor, pad_shape: tuple[int, int, int]) -> Tensor:
    """Build per-shoebox normalized coord grids, padded to batch max.

    shapes:    (B, 3) int — per-reflection (D, H, W)
    pad_shape: (Dmax, Hmax, Wmax)
    returns:   (B, Dmax, Hmax, Wmax, 3) float, coords in [-1, 1] inside the
               real region and 0 in the padded region (ignored via mask).
    """
    B = shapes.shape[0]
    Dmax, Hmax, Wmax = pad_shape
    device = shapes.device

    coords = shapes.new_zeros((B, Dmax, Hmax, Wmax, 3), dtype=torch.float32)

    # Vectorizing across B with variable ranges is clunky; loop is fine:
    # B is typically 64-256, each assignment is O(D*H*W*3).
    for i in range(B):
        D, H, W = int(shapes[i, 0]), int(shapes[i, 1]), int(shapes[i, 2])
        if D < 1 or H < 1 or W < 1:
            continue
        # linspace gives exact endpoints, degenerating to single 0 if dim==1
        zs = torch.linspace(-1.0, 1.0, D, device=device) if D > 1 else torch.zeros(1, device=device)
        ys = torch.linspace(-1.0, 1.0, H, device=device) if H > 1 else torch.zeros(1, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device) if W > 1 else torch.zeros(1, device=device)
        gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
        coords[i, :D, :H, :W, 0] = gz
        coords[i, :D, :H, :W, 1] = gy
        coords[i, :D, :H, :W, 2] = gx

    return coords


class RaggedProfilePosterior:
    """Output of `RaggedLogisticNormalSurrogate`. Holds per-reflection profile
    samples plus the moments needed for closed-form Gaussian KL.

    Attributes:
        profile:       (mc, B, Dmax, Hmax, Wmax) profile samples (each summing to 1 over valid voxels)
        profile_mean:  (B, Dmax, Hmax, Wmax) profile at posterior mean mu
        mu:            (B, d_basis) posterior mean of h
        logvar:        (B, d_basis) posterior log-variance of h
        mask:          (B, Dmax, Hmax, Wmax) bool — real & valid voxels
        concentration: None  (Dirichlet-compat shim for existing loss code)
    """

    __slots__ = ("profile", "profile_mean", "mu", "logvar", "mask", "concentration")

    def __init__(self, profile, profile_mean, mu, logvar, mask):
        self.profile = profile
        self.profile_mean = profile_mean
        self.mu = mu
        self.logvar = logvar
        self.mask = mask
        self.concentration = None  # keeps the Dirichlet isinstance check paths happy

    @property
    def mean(self) -> Tensor:
        """Profile at posterior mean. Shape (B, Dmax, Hmax, Wmax)."""
        return self.profile_mean

    def rsample(self, sample_shape):
        """Compat shim — profiles are already sampled in the forward pass.

        We ignore `sample_shape` and return the pre-computed samples. The
        caller should pass `mc_samples` to the surrogate's forward instead.
        """
        return self.profile

    def kl_to_prior(self, sigma_prior: Tensor) -> Tensor:
        """KL( N(mu, diag(exp(logvar))) || N(0, diag(sigma_prior^2)) ) per reflection.

        sigma_prior: (d_basis,) or scalar.
        Returns: (B,) tensor of per-reflection KL.
        """
        if sigma_prior.ndim == 0:
            sigma_prior = sigma_prior.expand_as(self.mu[0])
        var = self.logvar.exp()
        var_p = sigma_prior.pow(2).clamp(min=1e-8)
        # Gaussian KL, diagonal
        kl = 0.5 * (
            (var + self.mu.pow(2)) / var_p
            - 1.0
            + var_p.log()
            - self.logvar
        )
        return kl.sum(dim=-1)


class RaggedLogisticNormalSurrogate(nn.Module):
    """q(profile | shoebox) with learned coord→basis MLP.

    Args:
        input_dim:     dimension of the encoder feature (from RaggedShoeboxEncoder).
        d_basis:       latent size; number of basis functions. Default 29 to
                       match the Hermite truncation used by the fixed version.
        basis_hidden:  hidden width of the coord→basis MLP.
        fourier_freqs: number of log-spaced Fourier frequencies. Each freq
                       contributes 6 channels (sin+cos for each of z, y, x).
        logvar_clamp:  (min, max) for the posterior logvar, matches existing
                       LogisticNormal behavior to prevent variance collapse.
    """

    def __init__(
        self,
        input_dim: int,
        d_basis: int = 29,
        basis_hidden: int = 64,
        fourier_freqs: int = 6,
        logvar_clamp: tuple[float, float] = (-10.0, 4.0),
    ):
        super().__init__()
        self.d_basis = d_basis
        self.fourier_freqs = fourier_freqs
        self.logvar_clamp = logvar_clamp

        pos_dim = 6 * fourier_freqs  # sin + cos over 3 axes

        # Encoder feature -> posterior mu, logvar for h
        self.head_mu = nn.Linear(input_dim, d_basis)
        self.head_logvar = nn.Linear(input_dim, d_basis)

        # Shared coord -> basis MLP
        self.basis_mlp = nn.Sequential(
            nn.Linear(pos_dim, basis_hidden),
            nn.GELU(),
            nn.Linear(basis_hidden, basis_hidden),
            nn.GELU(),
            nn.Linear(basis_hidden, d_basis),
        )
        # Scalar bias on logits
        self.log_bias = nn.Parameter(torch.zeros(1))

    def _eval_basis(self, coords: Tensor) -> Tensor:
        """coords: (B, Dmax, Hmax, Wmax, 3) in [-1, 1]
        Returns: (B, Dmax, Hmax, Wmax, d_basis)
        """
        pe = _fourier_encode(coords, self.fourier_freqs)
        return self.basis_mlp(pe)

    def _profile_from_h(self, W: Tensor, h: Tensor, mask: Tensor) -> Tensor:
        """
        W:    (..., B, Dmax, Hmax, Wmax, d_basis)
        h:    (..., B, d_basis)
        mask: (B, Dmax, Hmax, Wmax)
        returns: (..., B, Dmax, Hmax, Wmax), softmaxed over the spatial axes
                 with padded voxels set to zero.
        """
        # Logits = einsum over d_basis
        # h has shape (..., B, d) — broadcast spatially
        h_b = h[..., None, None, None, :]  # (..., B, 1, 1, 1, d)
        logits = (W * h_b).sum(dim=-1) + self.log_bias  # (..., B, Dmax, Hmax, Wmax)

        # Masked softmax: set padded voxels to -inf, softmax across spatial
        m = mask
        while m.ndim < logits.ndim:
            m = m.unsqueeze(0)
        logits = logits.masked_fill(~m, float("-inf"))
        flat = logits.flatten(start_dim=-3)  # (..., B, V)
        prof_flat = torch.softmax(flat, dim=-1)
        return prof_flat.view_as(logits)

    def forward(
        self,
        features: Tensor,
        shapes: Tensor,
        mask: Tensor,
        mc_samples: int = 1,
        group_labels: Tensor | None = None,
    ) -> RaggedProfilePosterior:
        """
        features: (B, input_dim) — encoder output
        shapes:   (B, 3) int — per-reflection (D, H, W)
        mask:     (B, Dmax, Hmax, Wmax) bool
        mc_samples: number of Monte Carlo samples of h to draw

        Returns a RaggedProfilePosterior with profile shape (mc, B, Dmax, Hmax, Wmax).
        """
        del group_labels  # unused here; kept for interface parity

        B = features.shape[0]
        _, Dmax, Hmax, Wmax = mask.shape

        # Posterior parameters
        mu = self.head_mu(features)                                  # (B, d_basis)
        logvar = self.head_logvar(features).clamp(*self.logvar_clamp)

        # Normalized coord grid + per-voxel basis
        coords = _build_normalized_coords(shapes, (Dmax, Hmax, Wmax))
        coords = coords.to(features.device, dtype=features.dtype)
        W = self._eval_basis(coords)  # (B, Dmax, Hmax, Wmax, d_basis)

        # Sample h ~ N(mu, diag(exp(logvar))) via reparameterization
        std = (0.5 * logvar).exp()
        eps = torch.randn(mc_samples, *mu.shape, device=mu.device, dtype=mu.dtype)
        h_samples = mu.unsqueeze(0) + std.unsqueeze(0) * eps           # (mc, B, d_basis)

        # Broadcast W over mc samples and compute profiles
        W_exp = W.unsqueeze(0).expand(mc_samples, *W.shape)
        profile = self._profile_from_h(W_exp, h_samples, mask)          # (mc, B, Dmax, Hmax, Wmax)

        # Profile at posterior mean (no sampling)
        profile_mean = self._profile_from_h(W, mu, mask)               # (B, Dmax, Hmax, Wmax)

        return RaggedProfilePosterior(
            profile=profile,
            profile_mean=profile_mean,
            mu=mu,
            logvar=logvar,
            mask=mask,
        )
