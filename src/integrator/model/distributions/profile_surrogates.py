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


class LearnedBasisProfileSurrogate(nn.Module):
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
    ) -> None:
        super().__init__()

        if latent_dim is None:
            latent_dim = 8
        self.d: int = latent_dim

        self.loc_head = nn.Linear(input_dim, self.d)
        self.scale_head = nn.Linear(input_dim, self.d)
        self.decoder = nn.Linear(self.d, output_dim)

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
            loc, scale, self.decoder.weight, self.decoder.bias, mc_samples
        )
        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            loc=loc,
            scale=scale,
        )
