import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal


def _softplus_inverse(x: float) -> float:
    """Inverse of softplus: log(exp(x) - 1)."""
    return math.log(math.exp(x) - 1.0)


def _sample_and_decode(
    mu_h: Tensor,
    std_h: Tensor,
    W: Tensor,
    b: Tensor,
    mc_samples: int,
) -> tuple[Tensor, Tensor]:
    """Sample h from q(h|x) and decode to profile vectors.

    Args:
        mu_h: Posterior mean, shape (B, d).
        std_h: Posterior std, shape (B, d).
        W: Decoder weight matrix, shape (K, d).
        b: Decoder bias, shape (K,) or (B, K) for per-bin bias.
        mc_samples: Number of Monte Carlo samples.

    Returns:
        zp: Profile samples on the simplex, shape (S, B, K).
        mean_profile: Profile at posterior mean h, shape (B, K).
    """
    q_h = Normal(mu_h, std_h)
    h = q_h.rsample([mc_samples])  # (S, B, d)
    logits = h @ W.T + b  # (S, B, K)
    zp = F.softmax(logits, dim=-1)

    mean_logits = mu_h @ W.T + b  # (B, K)
    mean_profile = F.softmax(mean_logits, dim=-1)

    return zp, mean_profile


@dataclass
class ProfileSurrogateOutput:
    """Output of a latent-decoder profile surrogate.

    Fields:
        zp: Profile samples on the simplex, shape (S, B, K).
        mean_profile: Profile at posterior mean h, shape (B, K).
        mu_h: Posterior mean of h, shape (B, d).
        std_h: Posterior std of h, shape (B, d).
        shift: Optional per-reflection shift in normalized grid coords,
            shape (B, ndim) where ndim = 2 or 3. Present when the
            surrogate has a translation head; None otherwise. Consumers
            (e.g. the integrator) can use this to apply a Gaussian prior
            penalty on the shift magnitude.
    """

    zp: Tensor
    mean_profile: Tensor
    mu_h: Tensor
    std_h: Tensor
    # Shift outputs. `shift` is the actual tensor applied to logits
    # (post-sampling + tanh if variational, or deterministic if MAP).
    # `shift_mu` and `shift_sigma` are only set when shift_variational=True;
    # they parameterize the amortized posterior q(shift_raw|x) =
    # N(shift_mu, shift_sigma²) whose KL vs N(0, σ_prior²) the loss
    # incorporates into the ELBO.
    shift: Tensor | None = None
    shift_mu: Tensor | None = None
    shift_sigma: Tensor | None = None


# %%
class FixedBasisProfileSurrogate(nn.Module):
    """Profile surrogate with a fixed basis (Hermite or PCA).

    Args:
        input_dim: Dimension of the encoder output.
        basis_path: Path to profile_basis.pt.
        init_std: Initial posterior std for h.
    """

    def __init__(
        self,
        input_dim: int,
        basis_path: str,
        init_std: float = 0.5,
    ) -> None:
        super().__init__()

        basis = torch.load(basis_path, weights_only=False)
        self.W: Tensor
        self.b: Tensor
        self.register_buffer("W", basis["W"])  # (K, d)
        self.register_buffer("b", basis["b"])  # (K,)
        self.d: int = int(basis["d"])

        self.mu_head = nn.Linear(input_dim, self.d)
        self.std_head = nn.Linear(input_dim, self.d)

        nn.init.zeros_(self.std_head.weight)
        nn.init.constant_(self.std_head.bias, _softplus_inverse(init_std))

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        group_labels: Tensor | None = None,
    ) -> ProfileSurrogateOutput:
        mu_h = self.mu_head(x)  # (B, d)
        std_h = F.softplus(self.std_head(x))  # (B, d)

        zp, mean_profile = _sample_and_decode(
            mu_h, std_h, self.W, self.b, mc_samples
        )

        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            mu_h=mu_h,
            std_h=std_h,
        )


# %%
class LearnedBasisProfileSurrogate(nn.Module):
    """Profile surrogate with a learned linear decoder.

        prf = softmax(W @ h + b)
        q(h | x) = N(mu_h(x), diag(sigma_h(x)^2))

    Args:
        input_dim: Dimension of the encoder output.
        latent_dim: Dimension of the latent h. Default 8.
        output_dim: Number of profile pixels (H * W). Default 441.
        warmstart_basis_path: Optional .pt file with keys 'W' (K, d) and
            'b' (K,) to warm-start the decoder. When `latent_dim` differs
            from the basis's `d`, the first `min(latent_dim, basis_d)`
            columns of W are copied; any extra columns stay randomly
            initialized.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        output_dim: int = 441,
        init_std: float = 0.5,
        warmstart_basis_path: str | None = None,
        freeze_bias: bool = False,
        shift_head: bool = False,
        sbox_shape: tuple[int, ...] | list[int] | None = None,
        max_shift_norm: float = 1.0,
        shift_variational: bool = False,
        shift_init_std: float = 0.1,
    ) -> None:
        super().__init__()

        self.d: int = latent_dim

        self.mu_head = nn.Linear(input_dim, self.d)
        self.std_head = nn.Linear(input_dim, self.d)
        self.decoder = nn.Linear(self.d, output_dim)

        nn.init.zeros_(self.std_head.weight)
        nn.init.constant_(self.std_head.bias, _softplus_inverse(init_std))

        if warmstart_basis_path is not None:
            self._warmstart_from_basis(warmstart_basis_path, output_dim)

        if freeze_bias:
            self.decoder.bias.requires_grad_(False)

        # Optional per-reflection translation head. The shift is applied
        # to the basis logits before softmax — linear-in-logits, so this
        # is equivalent to translating b AND every column of W by the
        # same offset.
        #
        # Two modes:
        #   shift_variational=False (default): deterministic MAP head.
        #     shift = max_shift_norm * tanh(Linear(x)). Regularize via
        #     `shift_prior_weight` in the integrator config (L2 penalty).
        #   shift_variational=True: amortized posterior
        #     q(shift_raw|x) = N(shift_mu(x), shift_sigma(x)^2).
        #     Reparameterized sampling, tanh-bounded output, and the
        #     KL(q || N(0, sigma_prior^2)) gets added to the ELBO by the
        #     loss module (see wilson_loss.py's compute_shift_kl path).
        self.has_shift = shift_head
        self.shift_variational = shift_variational
        if shift_head:
            if sbox_shape is None:
                raise ValueError(
                    "shift_head=True requires sbox_shape (e.g. [3, 21, 21])"
                )
            sbox = tuple(int(s) for s in sbox_shape)
            if len(sbox) not in (2, 3):
                raise ValueError(f"sbox_shape must be 2D or 3D, got {sbox}")
            prod = 1
            for s in sbox:
                prod *= s
            if prod != output_dim:
                raise ValueError(
                    f"sbox_shape {sbox} product {prod} != output_dim {output_dim}"
                )
            self.sbox_shape = sbox
            ndim = len(sbox)
            self.shift_dim = ndim
            self.max_shift_norm = float(max_shift_norm)

            if shift_variational:
                # Mean and log-sigma heads for q(shift_raw|x).
                self.shift_mu_layer = nn.Linear(input_dim, ndim)
                self.shift_logsigma_layer = nn.Linear(input_dim, ndim)
                # Init: mean at zero (identity at step 0), sigma starts
                # narrow (shift_init_std) so the initial posterior commits
                # and the KL vs a looser prior (σ_prior ≥ shift_init_std)
                # is small. The logsigma head starts with zero weights
                # and a fixed bias so σ is input-independent at init.
                nn.init.zeros_(self.shift_mu_layer.bias)
                nn.init.normal_(self.shift_mu_layer.weight, std=0.01)
                nn.init.zeros_(self.shift_logsigma_layer.weight)
                nn.init.constant_(
                    self.shift_logsigma_layer.bias,
                    math.log(float(shift_init_std)),
                )
            else:
                self.shift_layer = nn.Linear(input_dim, ndim)
                nn.init.zeros_(self.shift_layer.bias)
                nn.init.normal_(self.shift_layer.weight, std=0.01)

    def _warmstart_from_basis(
        self, basis_path: str, output_dim: int
    ) -> None:
        basis = torch.load(basis_path, weights_only=False)
        W = basis["W"].float()   # (K, d_basis)
        b = basis["b"].float()   # (K,)
        if W.shape[0] != output_dim:
            raise ValueError(
                f"warmstart basis W has K={W.shape[0]} but output_dim="
                f"{output_dim}. Basis shape doesn't match the surrogate's "
                "profile size."
            )
        if b.shape[0] != output_dim:
            raise ValueError(
                f"warmstart basis b has K={b.shape[0]} but output_dim="
                f"{output_dim}."
            )
        d_basis = W.shape[1]
        d_copy = min(self.d, d_basis)
        # nn.Linear.weight has shape (out_features, in_features) = (K, d).
        # Basis W is also (K, d). Copy the first d_copy columns into the
        # corresponding decoder columns; leave extra columns at random init.
        with torch.no_grad():
            self.decoder.weight.data[:, :d_copy].copy_(W[:, :d_copy])
            self.decoder.bias.data.copy_(b)

    def _predict_shift_map(self, x: Tensor) -> Tensor:
        """Deterministic shift in [-max_shift_norm, +max_shift_norm]."""
        raw = self.shift_layer(x)
        return self.max_shift_norm * torch.tanh(raw)

    def _predict_shift_variational(
        self, x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Variational shift: q(shift_raw|x) = N(mu(x), sigma(x)^2).

        Returns (shift, shift_mu, shift_sigma):
          shift:       sampled + tanh-bounded, used for grid_sample
          shift_mu:    pre-tanh posterior mean (for KL)
          shift_sigma: pre-tanh posterior std  (for KL)

        KL is computed in the pre-tanh (unbounded) space — the prior
        N(0, sigma_prior^2) is defined on the raw latent; tanh is just
        a monotonic output squashing to prevent the sampling grid from
        escaping the shoebox.
        """
        shift_mu = self.shift_mu_layer(x)
        shift_sigma = F.softplus(self.shift_logsigma_layer(x)) + 1e-6
        eps = torch.randn_like(shift_mu)
        shift_raw = shift_mu + shift_sigma * eps
        shift = self.max_shift_norm * torch.tanh(shift_raw)
        return shift, shift_mu, shift_sigma

    def _shift_logits(self, logits: Tensor, shift_norm: Tensor) -> Tensor:
        """Translate logits spatially via affine_grid + grid_sample.

        Zero-padding outside the shoebox is the right choice: mass shifted
        past the edge becomes 0. Softmax *after* the shift renormalizes
        the remaining mass to sum to 1.

        Accepts logits with or without an MC-sample leading dim.
        """
        sbox = self.sbox_shape
        ndim = self.shift_dim
        K = logits.shape[-1]
        squeeze_S = False

        if logits.dim() == 2:
            S, B = 1, logits.shape[0]
            logits = logits.unsqueeze(0)  # (1, B, K)
            squeeze_S = True
        else:
            S, B = logits.shape[0], logits.shape[1]
        assert K == logits.shape[-1]

        # Expand per-reflection shift across MC samples → (S*B, ndim)
        shift = shift_norm.unsqueeze(0).expand(S, -1, -1).reshape(S * B, ndim)

        # Build affine matrix (identity + translation). affine_grid maps
        # output coords to input coords, so translation is the *negation*
        # of the desired output shift.
        if ndim == 3:
            D, H, W = sbox
            theta = torch.zeros(S * B, 3, 4, device=logits.device, dtype=logits.dtype)
            theta[:, 0, 0] = 1.0   # x
            theta[:, 1, 1] = 1.0   # y
            theta[:, 2, 2] = 1.0   # z
            # Predicted shift order: (dz, dy, dx); theta translation order: (tx, ty, tz)
            theta[:, 0, 3] = -shift[:, 2]
            theta[:, 1, 3] = -shift[:, 1]
            theta[:, 2, 3] = -shift[:, 0]
            grid = F.affine_grid(
                theta, size=[S * B, 1, D, H, W], align_corners=True
            )
            vol = logits.reshape(S * B, 1, D, H, W)
            shifted = F.grid_sample(
                vol, grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
        else:  # 2D
            H, W = sbox
            theta = torch.zeros(S * B, 2, 3, device=logits.device, dtype=logits.dtype)
            theta[:, 0, 0] = 1.0
            theta[:, 1, 1] = 1.0
            theta[:, 0, 2] = -shift[:, 1]  # dx
            theta[:, 1, 2] = -shift[:, 0]  # dy
            grid = F.affine_grid(
                theta, size=[S * B, 1, H, W], align_corners=True
            )
            vol = logits.reshape(S * B, 1, H, W)
            shifted = F.grid_sample(
                vol, grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )

        shifted = shifted.reshape(S, B, K)
        if squeeze_S:
            shifted = shifted.squeeze(0)
        return shifted

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        group_labels: Tensor | None = None,
    ) -> ProfileSurrogateOutput:
        mu_h = self.mu_head(x)  # (B, d)
        std_h = F.softplus(self.std_head(x))  # (B, d)

        if not self.has_shift:
            zp, mean_profile = _sample_and_decode(
                mu_h, std_h, self.decoder.weight, self.decoder.bias, mc_samples
            )
            return ProfileSurrogateOutput(
                zp=zp,
                mean_profile=mean_profile,
                mu_h=mu_h,
                std_h=std_h,
            )

        # Shifted path: compute logits first, translate, THEN softmax so
        # the profile still sums to 1 after the shift (any mass pushed
        # past the shoebox edge is dropped; the remaining mass normalizes).
        q_h = Normal(mu_h, std_h)
        h = q_h.rsample([mc_samples])  # (S, B, d)
        logits_sample = h @ self.decoder.weight.T + self.decoder.bias   # (S, B, K)
        mean_logits = mu_h @ self.decoder.weight.T + self.decoder.bias  # (B, K)

        if self.shift_variational:
            shift, shift_mu, shift_sigma = self._predict_shift_variational(x)
        else:
            shift = self._predict_shift_map(x)
            shift_mu, shift_sigma = None, None

        logits_sample = self._shift_logits(logits_sample, shift)
        mean_logits = self._shift_logits(mean_logits, shift)

        zp = F.softmax(logits_sample, dim=-1)
        mean_profile = F.softmax(mean_logits, dim=-1)

        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            mu_h=mu_h,
            std_h=std_h,
            shift=shift,
            shift_mu=shift_mu,
            shift_sigma=shift_sigma,
        )
