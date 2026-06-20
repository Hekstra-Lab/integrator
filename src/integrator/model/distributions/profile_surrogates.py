import math
from dataclasses import dataclass

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
) -> tuple[Tensor, Tensor, Tensor]:
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
        mean_logits: Pre-softmax logits at posterior mean h, shape (B, K).
    """
    q_h = Normal(loc, scale)
    h = q_h.rsample([mc_samples])  # (S, B, d)
    logits = h @ W.T + b  # (S, B, K)
    zp = F.softmax(logits, dim=-1)

    mean_logits = loc @ W.T + b  # (B, K)
    mean_profile = F.softmax(mean_logits, dim=-1)

    return zp, mean_profile, mean_logits


@dataclass
class ProfileSurrogateOutput:
    """Output of a latent-decoder profile surrogate.

    Fields:
        zp: Profile samples on the simplex, shape (S, B, K).
        mean_profile: Profile at posterior mean h, shape (B, K).
        mean_logits: Pre-softmax profile logits at posterior mean, shape (B, K).
            The shared shape field: softmax -> profile, sigmoid(. + gate) ->
            per-pixel signal responsibility.
        loc: Posterior mean of h, shape (B, d).
        scale: Posterior std of h, shape (B, d).
        prior_scale: Std of the N(0, prior_scale) prior on the latent h
    """

    zp: Tensor
    mean_profile: Tensor
    mean_logits: Tensor
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
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int | None = None,
        output_dim: int = 441,
        init_std: float = 0.5,
        prior_scale: float = 3.0,
        smoothness_weight: float = 0.0,
    ) -> None:
        super().__init__()

        if latent_dim is None:
            latent_dim = 8
        self.d: int = latent_dim
        self.prior_scale = prior_scale
        self.smoothness_weight = smoothness_weight

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

        zp, mean_profile, mean_logits = _sample_and_decode(
            loc, scale, self.decoder.weight, self.decoder.bias, mc_samples
        )
        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            mean_logits=mean_logits,
            loc=loc,
            scale=scale,
            prior_scale=self.prior_scale,
        )
