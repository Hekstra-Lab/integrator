"""Empirical profile surrogate with per-bin bias from data.

Instead of a single symmetric Gaussian bias shared across all bins
(as in `PerBinLogisticNormalSurrogate`), this surrogate uses per-bin
empirical biases computed from the mean bg-subtracted profile in each
resolution/azimuthal bin.  At z=0, the profile reproduces the empirical
average for that bin; the model only learns per-reflection corrections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
from torch import Tensor

from .logistic_normal import (
    PerBinProfilePosterior,
    ProfilePosterior,
    _softplus_inverse,
)


class EmpiricalProfileSurrogate(nn.Module):
    """Profile surrogate with per-bin empirical bias.

    Loads a basis file (`empirical_profile_basis_per_bin.pt`) containing:

    - `W`             (K, d):       basis matrix (Hermite)
    - `b_per_group`   (n_bins, K):  per-bin log(mean_profile)
    - `mu_per_group`  (n_bins, d):  per-bin prior mean in latent space
    - `std_per_group` (n_bins, d):  per-bin prior std in latent space
    - `sigma_prior`   float:        global prior std
    - `d`             int:          latent dimensionality

    Args:
        input_dim: Dimension of the encoder output.
        basis_path: Path to the empirical_profile_basis_per_bin.pt file.
        learn_W: If True, W is a learnable parameter; otherwise a frozen buffer.
        orthogonal_W: If True, W is a learnable parameter constrained to have
            orthonormal columns (Stiefel manifold) via Householder
            parametrization. Implies `learn_W=True`.
        global_prior: If True, use a single global N(0, sigma_prior^2 I) prior
            on h instead of per-bin N(mu_k, diag(std_k^2)) priors.
    """

    def __init__(
        self,
        input_dim: int,
        basis_path: str,
        learn_W: bool = False,
        orthogonal_W: bool = False,
        global_prior: bool = False,
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
        self.register_buffer(
            "mu_per_group", basis["mu_per_group"]
        )  # (n_bins, d)
        self.register_buffer(
            "std_per_group", basis["std_per_group"]
        )  # (n_bins, d)

        self.d: int = int(basis["d"])
        self.sigma_prior: float = float(basis.get("sigma_prior", 3.0))
        self.global_prior: bool = global_prior

        self.mu_head = nn.Linear(input_dim, self.d)
        self.std_head = nn.Linear(input_dim, self.d)

        nn.init.zeros_(self.std_head.weight)
        nn.init.constant_(self.std_head.bias, _softplus_inverse(init_std))

    def forward(
        self,
        x: Tensor,
        group_labels: Tensor | None = None,
    ) -> ProfilePosterior:
        """Map encoder output to a profile posterior.

        Args:
            x: Encoder output, shape (B, input_dim).
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

        if self.global_prior:
            return ProfilePosterior(
                mu_h=mu_h,
                std_h=std_h,
                W=self.W,
                b=b,
                sigma_prior=self.sigma_prior,
            )

        return PerBinProfilePosterior(
            mu_h=mu_h,
            std_h=std_h,
            W=self.W,
            b=b,
            sigma_prior=self.sigma_prior,
            mu_prior=self.mu_per_group,
            std_prior=self.std_per_group,
        )
