"""Convolutional profile surrogate.

Drop-in alternative to LearnedBasisProfileSurrogate. Same variational machinery
on a low-dim latent h (closed-form Normal KL against a global N(0, sigma_prior^2 I)
prior), but the decoder is a small 3D conv stack instead of a single linear
layer. Weight sharing across spatial positions gives an implicit
spatial-smoothness inductive bias that the basis decoder lacks.

    q(h | x) = N(mu_h(x), sigma_h(x)^2)
    prf      = softmax(ConvDecoder(h))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
    _softplus_inverse,
)


class ConvProfileSurrogate(nn.Module):
    """Profile surrogate with a convolutional decoder.

    Emits a :class:`ProfileSurrogateOutput`, so the loss-side KL path is
    identical to :class:`LearnedBasisProfileSurrogate`.

    Architecture:
        1. Encoder features x -> (mu_h, sigma_h) via two Linear heads.
        2. Sample h ~ N(mu_h, sigma_h^2).
        3. project(h) -> low-resolution 3D feature volume (C, D_seed, H_seed, W_seed).
        4. Trilinear upsample to shoebox shape (D, H, W).
        5. 3x3x3 Conv3d stack refines to a single-channel logit volume.
        6. softmax over all D*H*W pixels gives the profile simplex.

    Args:
        input_dim: Dim of the encoder output (the x in q(h|x)).
        latent_dim: Dim of the latent h.
        sbox_shape: Output shoebox shape (D, H, W). For 2D, pass (1, H, W).
        channels: Channel width of the conv stack.
        init_std: Initial posterior std for h.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        sbox_shape: tuple[int, int, int] | list[int] = (3, 21, 21),
        channels: int = 16,
        init_std: float = 0.5,
    ) -> None:
        super().__init__()

        sbox = tuple(int(s) for s in sbox_shape)
        if len(sbox) != 3:
            raise ValueError(
                f"sbox_shape must be a 3-tuple (D, H, W); got {sbox}"
            )
        D, H, W = sbox
        self.sbox_shape = sbox
        self.d = latent_dim
        self.output_dim = D * H * W

        # Variational heads on the latent h (same pattern as
        # LearnedBasisProfileSurrogate).
        self.mu_head = nn.Linear(input_dim, latent_dim)
        self.std_head = nn.Linear(input_dim, latent_dim)
        nn.init.zeros_(self.std_head.weight)
        nn.init.constant_(self.std_head.bias, _softplus_inverse(init_std))

        # Seed feature volume. Spatial dims start at ~1/3 of the shoebox so
        # the upsample does the bulk of the spatial fill; depth stays full
        # resolution (typically 3 frames).
        seed_d = D
        seed_h = max(1, H // 3)
        seed_w = max(1, W // 3)
        self.seed_shape = (seed_d, seed_h, seed_w)
        self.channels = channels

        seed_numel = channels * seed_d * seed_h * seed_w
        self.project = nn.Linear(latent_dim, seed_numel)

        # Fixed trilinear upsample to full shoebox shape.
        self.upsample = nn.Upsample(
            size=sbox, mode="trilinear", align_corners=False
        )

        # Refinement conv stack. GroupNorm tolerates tiny batches (MC samples).
        n_groups = min(channels, 4)
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, 1, kernel_size=3, padding=1),
        )

    def _decode(self, h: Tensor) -> Tensor:
        """Map latent samples to per-pixel logits.

        Args:
            h: Latent tensor with trailing dim = latent_dim. Leading dims are
                preserved; e.g. ``(B, d)`` or ``(S, B, d)`` are both accepted.

        Returns:
            Logits tensor with the same leading dims and trailing dim = D*H*W.
        """
        orig = h.shape[:-1]
        flat = h.reshape(-1, self.d)
        N = flat.shape[0]
        x = self.project(flat).reshape(N, self.channels, *self.seed_shape)
        x = self.upsample(x)
        x = self.refine(x)
        logits = x.reshape(N, self.output_dim)
        return logits.reshape(*orig, self.output_dim)

    def forward(
        self,
        x: Tensor,
        mc_samples: int = 1,
        group_labels: Tensor | None = None,
    ) -> ProfileSurrogateOutput:
        del group_labels  # not used; accepted for signature parity with other qp surrogates.

        mu_h = self.mu_head(x)  # (B, d)
        std_h = F.softplus(self.std_head(x))  # (B, d)

        q_h = Normal(mu_h, std_h)
        h = q_h.rsample(torch.Size([mc_samples]))  # (S, B, d)

        logits_sample = self._decode(h)  # (S, B, K)
        mean_logits = self._decode(mu_h)  # (B, K)

        zp = F.softmax(logits_sample, dim=-1)
        mean_profile = F.softmax(mean_logits, dim=-1)

        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            mu_h=mu_h,
            std_h=std_h,
        )
