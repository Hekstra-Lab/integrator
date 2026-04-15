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
    """

    zp: Tensor
    mean_profile: Tensor
    mu_h: Tensor
    std_h: Tensor


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
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        output_dim: int = 441,
        sigma_prior: float = 3.0,
        init_std: float = 0.5,
    ) -> None:
        super().__init__()

        self.d: int = latent_dim

        self.mu_head = nn.Linear(input_dim, self.d)
        self.std_head = nn.Linear(input_dim, self.d)
        self.decoder = nn.Linear(self.d, output_dim)

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
            mu_h, std_h, self.decoder.weight, self.decoder.bias, mc_samples
        )

        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            mu_h=mu_h,
            std_h=std_h,
        )


class CPProfileSurrogate(nn.Module):
    """Profile surrogate with per-bin learned decoder via CP decomposition.

    Each 2D bin gets its own decoder W_n and bias b_n, where W_n is
    factored as a rank-R CP decomposition to share structure across bins:

        W_n[k, j] = sum_r A[n, r] * B[k, r] * C[j, r]

    The forward pass never materializes the full W_n. Instead:

        z = h @ C              (project latent to rank-R)
        z_n = A[n] * z         (per-bin scaling)
        logits = z_n @ B.T + b_n   (decode to pixels)

    Args:
        input_dim: Dimension of the encoder output.
        n_bins: Number of resolution x angle bins.
        latent_dim: Dimension of the latent h.
        output_dim: Number of profile pixels (H * W * D).
        rank: CP rank.
        init_std: Initial posterior std for h.
    """

    def __init__(
        self,
        input_dim: int,
        n_bins: int = 300,
        latent_dim: int = 8,
        output_dim: int = 1323,
        rank: int = 4,
        init_std: float = 0.5,
    ) -> None:
        super().__init__()

        self.d: int = latent_dim

        self.A = nn.Parameter(torch.randn(n_bins, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(output_dim, rank) * 0.01)
        self.C = nn.Parameter(torch.randn(latent_dim, rank) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_bins, output_dim))

        self.mu_head = nn.Linear(input_dim, latent_dim)
        self.std_head = nn.Linear(input_dim, latent_dim)

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

        q_h = Normal(mu_h, std_h)
        h = q_h.rsample([mc_samples])  # (S, B, d)

        A_n = self.A[group_labels]  # (B, R)
        b_n = self.b[group_labels]  # (B, K)

        z = h @ self.C  # (S, B, R)
        z_n = A_n * z  # (S, B, R) — A_n broadcasts over S
        logits = z_n @ self.B.T + b_n  # (S, B, K)
        zp = F.softmax(logits, dim=-1)

        z_mean = mu_h @ self.C  # (B, R)
        z_mean_n = A_n * z_mean  # (B, R)
        mean_logits = z_mean_n @ self.B.T + b_n  # (B, K)
        mean_profile = F.softmax(mean_logits, dim=-1)

        return ProfileSurrogateOutput(
            zp=zp,
            mean_profile=mean_profile,
            mu_h=mu_h,
            std_h=std_h,
        )


class PerBinProfileSurrogate(nn.Module):
    """Profile surrogate with fixed basis (Hermite or PCA).

    Loads a `profile_basis_per_bin.pt` file containing:
        - W (K, d): basis matrix (Hermite functions or PCA components)
        - b (K,): bias (log of reference profile or mean of log-profiles)
        - d (int): latent dimensionality

    Args:
        input_dim: Dimension of the encoder output.
        basis_path: Path to profile_basis_per_bin.pt.
    """

    def __init__(
        self,
        input_dim: int,
        basis_path: str,
        init_std: float = 0.5,
    ) -> None:
        super().__init__()

        basis = torch.load(basis_path, weights_only=False)

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
