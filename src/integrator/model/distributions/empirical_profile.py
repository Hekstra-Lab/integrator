import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
from torch import Tensor

from .profile_surrogates import (
    ProfileSurrogateOutput,
    _sample_and_decode,
    _softplus_inverse,
)


class EmpiricalProfileSurrogate(nn.Module):
    """Profile surrogate with per-bin empirical bias.

    Loads a basis file (`empirical_profile_basis_per_bin.pt`) containing:

    - `W`             (K, d):       basis matrix (Hermite)
    - `b_per_group`   (n_bins, K):  per-bin log(mean_profile)
    - `d`             int:          latent dimensionality

    Args:
        input_dim: Dimension of the encoder output.
        basis_path: Path to the empirical_profile_basis_per_bin.pt file.
        learn_W: If True, W is a learnable parameter; otherwise a frozen buffer.
        orthogonal_W: If True, W is a learnable parameter constrained to have
            orthonormal columns (Stiefel manifold) via Householder
            parametrization. Implies `learn_W=True`.
    """

    def __init__(
        self,
        input_dim: int,
        basis_path: str,
        learn_W: bool = False,
        orthogonal_W: bool = False,
        init_std: float = 0.5,
    ) -> None:
        super().__init__()

        basis = torch.load(basis_path, weights_only=False)

        if orthogonal_W or learn_W:
            self.W = nn.Parameter(basis["W"].clone())  # (K, d)
            if orthogonal_W:
                P.orthogonal(self, name="W")
        else:
            self.register_buffer("W", basis["W"])  # (K, d)
        self.register_buffer(
            "b_per_group", basis["b_per_group"]
        )  # (n_bins, K)

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
        """Map encoder output to profile samples.

        Args:
            x: Encoder output, shape (B, input_dim).
            mc_samples: Number of Monte Carlo profile samples.
            group_labels: Bin index per reflection, shape (B,) long tensor.
                Selects the per-bin empirical bias for each reflection.
                When None, uses the mean bias across all bins as a fallback.
        """
        mu_h = self.mu_head(x)  # (B, d)
        std_h = F.softplus(self.std_head(x))  # (B, d)

        if group_labels is not None:
            b = self.b_per_group[group_labels.long()]  # (B, K)
        else:
            b = self.b_per_group.mean(dim=0)  # (K,) fallback

        zp, mean_profile = _sample_and_decode(
            mu_h, std_h, self.W, b, mc_samples
        )

        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            mu_h=mu_h,
            std_h=std_h,
        )
