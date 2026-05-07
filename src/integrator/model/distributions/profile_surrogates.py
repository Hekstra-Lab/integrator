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
    """

    zp: Tensor
    mean_profile: Tensor
    loc: Tensor
    scale: Tensor


# %%
class FixedBasisProfileSurrogate(nn.Module):
    """Profile surrogate with a fixed basis (Hermite basis).

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

        self.loc_head = nn.Linear(input_dim, self.d)
        self.scale_head = nn.Linear(input_dim, self.d)

        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, _softplus_inverse(init_std))

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        group_labels: Tensor | None = None,
        **kwargs,
    ) -> ProfileSurrogateOutput:
        loc = self.loc_head(x)  # (B, d)
        scale = F.softplus(self.scale_head(x))  # (B, d)

        zp, mean_profile = _sample_and_decode(
            loc, scale, self.W, self.b, mc_samples
        )

        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            loc=loc,
            scale=scale,
        )


# %%
class LearnedBasisProfileSurrogate(nn.Module):
    """Profile surrogate with a learned linear decoder.

        prf = softmax(W @ h + b)
        q(h | x) = N(loc(x), diag(scale(x)^2))

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
        latent_dim: int | None = None,
        output_dim: int = 441,
        init_std: float = 0.5,
        warmstart_basis_path: str | None = None,
        freeze_bias: bool = False,
    ) -> None:
        super().__init__()

        self._basis_W: Tensor | None = None
        self._basis_b: Tensor | None = None

        if warmstart_basis_path is not None:
            basis = torch.load(warmstart_basis_path, weights_only=False)
            self._basis_W = basis["W"].float()  # (K, d_basis)
            self._basis_b = basis["b"].float()  # (K,)
            d_basis = self._basis_W.shape[1]
            if latent_dim is None:
                latent_dim = d_basis

        # TODO: remove latent_dim fallback once legacy configs (ablation
        # runs Apr 2026) are done — latent_dim should always be inferred
        # from the basis file when warmstarting.
        if latent_dim is None:
            latent_dim = 8

        self.d: int = latent_dim

        self.loc_head = nn.Linear(input_dim, self.d)
        self.scale_head = nn.Linear(input_dim, self.d)
        self.decoder = nn.Linear(self.d, output_dim)

        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, _softplus_inverse(init_std))

        if self._basis_W is not None:
            self._apply_warmstart(output_dim)

        if freeze_bias:
            self.decoder.bias.requires_grad_(False)

    def _apply_warmstart(self, output_dim: int) -> None:
        W, b = self._basis_W, self._basis_b
        if W.shape[0] != output_dim:
            raise ValueError(
                f"warmstart basis W has K={W.shape[0]} but output_dim="
                f"{output_dim}."
            )
        if b.shape[0] != output_dim:
            raise ValueError(
                f"warmstart basis b has K={b.shape[0]} but output_dim="
                f"{output_dim}."
            )
        d_copy = min(self.d, W.shape[1])
        with torch.no_grad():
            self.decoder.weight.data[:, :d_copy].copy_(W[:, :d_copy])
            self.decoder.bias.data.copy_(b)
        del self._basis_W, self._basis_b

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        group_labels: Tensor | None = None,
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
        )
