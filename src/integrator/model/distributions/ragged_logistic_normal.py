"""Ragged-compatible LogisticNormal profile surrogate.

Mirror of `LogisticNormalSurrogate` in `distributions/logistic_normal.py` but
with the fixed (V, d_basis) Hermite W replaced by a learned coord→basis MLP.

Returns the same `ProfileSurrogateOutput` dataclass used by the fixed-size
surrogates, so it plugs straight into `WilsonLoss.compute_profile_kl`. The
only ragged-specific wrinkle: output tensors are flat along the voxel axis
(K = Dmax * Hmax * Wmax in the batch's padded shape); the integrator's
existing `rate = zI * zp + zbg` broadcast works unchanged.

Key ideas:
  - Each voxel in a shoebox has a normalized coord (z, y, x) in [-1, 1]^3.
  - A shared small MLP maps (fourier-encoded coord) -> d_basis vector.
  - Profile logits = W_i @ h_i   where W_i is evaluated on this reflection's grid.
  - Masked softmax over voxels: padded voxels get logits = -inf, softmaxed to 0,
    so `zp` sums to 1 over valid voxels and contributes nothing at padded ones.
  - q(h | x) = Normal(mu_h, diag(std_h^2)); std_h is parameterized via softplus
    to match `FixedBasisProfileSurrogate`'s convention for `compute_profile_kl`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)


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


def _softplus_inverse(y: float) -> float:
    """log(exp(y) - 1); used to initialize a softplus head to produce y."""
    return float(math.log(math.exp(y) - 1.0))


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


class RaggedLogisticNormalSurrogate(nn.Module):
    """q(profile | shoebox) with learned coord→basis MLP.

    Args:
        input_dim:     dimension of the encoder feature (from RaggedShoeboxEncoder).
        d_basis:       latent size; number of basis functions. Default 29 to
                       match the Hermite truncation used by the fixed version.
        basis_hidden:  hidden width of the coord→basis MLP.
        fourier_freqs: number of log-spaced Fourier frequencies. Each freq
                       contributes 6 channels (sin+cos for each of z, y, x).
        init_std:      initial posterior std for h; the softplus head's bias
                       is set so that F.softplus(bias) == init_std.
    """

    def __init__(
        self,
        input_dim: int,
        d_basis: int = 29,
        basis_hidden: int = 64,
        fourier_freqs: int = 6,
        init_std: float = 0.5,
    ):
        super().__init__()
        self.d_basis = d_basis
        self.fourier_freqs = fourier_freqs

        pos_dim = 6 * fourier_freqs  # sin + cos over 3 axes

        # Encoder feature -> posterior mu_h and std_h for latent h
        # std_h parameterized via softplus (matches FixedBasisProfileSurrogate)
        self.mu_head = nn.Linear(input_dim, d_basis)
        self.std_head = nn.Linear(input_dim, d_basis)
        nn.init.zeros_(self.std_head.weight)
        nn.init.constant_(self.std_head.bias, _softplus_inverse(init_std))

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

    def _eval_basis_flat(self, coords_flat: Tensor) -> Tensor:
        """coords_flat: (B, K, 3) in [-1, 1]
        Returns: W flat (B, K, d_basis) — per-voxel basis row.
        """
        pe = _fourier_encode(coords_flat, self.fourier_freqs)
        return self.basis_mlp(pe)

    def _profile_from_h(
        self, W_flat: Tensor, h: Tensor, mask_flat: Tensor
    ) -> Tensor:
        """
        W_flat:     (B, K, d_basis)                 — per-voxel basis
        h:          (..., B, d_basis)               — latent sample(s); the
                    leading dims (e.g. mc samples) are arbitrary.
        mask_flat:  (B, K)  bool                    — valid voxels
        returns:    (..., B, K) softmaxed over K, zero at padded voxels.

        Implementation note: we explicitly avoid broadcasting W to
        (..., B, K, d_basis) — that intermediate is huge for large mc and K
        (mc=100 × B=256 × K=4000 × d=29 ~ 12 GB in float32, plus its grad).
        Instead we do a batched matmul through einsum, which goes straight
        to the (..., B, K) output without materializing the 4-D tensor.
        """
        if h.ndim == 2:
            # Single-sample case: (B, d_basis)
            # W_flat: (B, K, d_basis) -> logits: (B, K)
            logits = torch.einsum("bkd,bd->bk", W_flat, h)
        else:
            # Sampled case: leading dims (S1, S2, ...) before (B, d_basis).
            # Flatten leading dims to a single sample axis for einsum.
            sample_shape = h.shape[:-2]
            B, d = h.shape[-2:]
            h_flat = h.reshape(-1, B, d)                                  # (S, B, d)
            logits_flat = torch.einsum("bkd,sbd->sbk", W_flat, h_flat)    # (S, B, K)
            logits = logits_flat.reshape(*sample_shape, B, W_flat.shape[1])

        logits = logits + self.log_bias

        m = mask_flat
        while m.ndim < logits.ndim:
            m = m.unsqueeze(0)
        logits = logits.masked_fill(~m, float("-inf"))
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        features: Tensor,
        shapes: Tensor,
        mask: Tensor,
        mc_samples: int = 1,
        group_labels: Tensor | None = None,
    ) -> ProfileSurrogateOutput:
        """
        features: (B, input_dim) — encoder output
        shapes:   (B, 3) int — per-reflection (D, H, W)
        mask:     (B, Dmax, Hmax, Wmax) bool
        mc_samples: number of Monte Carlo samples of h to draw

        Returns `ProfileSurrogateOutput` with flat voxel shapes:
          zp:           (mc, B, K)       K = Dmax*Hmax*Wmax
          mean_profile: (B, K)
          mu_h:         (B, d_basis)
          std_h:        (B, d_basis)     softplus-parameterized
        """
        del group_labels  # unused here; kept for interface parity

        _, Dmax, Hmax, Wmax = mask.shape
        K = Dmax * Hmax * Wmax

        # Posterior parameters — softplus std matches existing surrogates
        mu_h = self.mu_head(features)                           # (B, d_basis)
        std_h = F.softplus(self.std_head(features))             # (B, d_basis)

        # Normalized coord grid → flat (B, K, 3) → per-voxel basis (B, K, d_basis)
        coords_3d = _build_normalized_coords(shapes, (Dmax, Hmax, Wmax))
        coords_3d = coords_3d.to(features.device, dtype=features.dtype)
        coords_flat = coords_3d.reshape(coords_3d.shape[0], K, 3)
        W_flat = self._eval_basis_flat(coords_flat)             # (B, K, d_basis)

        mask_flat = mask.reshape(mask.shape[0], K)              # (B, K)

        # Sample h ~ N(mu_h, std_h^2) via reparameterization
        eps = torch.randn(
            mc_samples, *mu_h.shape, device=mu_h.device, dtype=mu_h.dtype
        )
        h_samples = mu_h.unsqueeze(0) + std_h.unsqueeze(0) * eps  # (mc, B, d_basis)

        zp = self._profile_from_h(W_flat, h_samples, mask_flat)   # (mc, B, K)
        mean_profile = self._profile_from_h(W_flat, mu_h, mask_flat)  # (B, K)

        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            mu_h=mu_h,
            std_h=std_h,
        )
