"""Empirical profile surrogate with per-bin bias from data.

Instead of a single symmetric Gaussian bias shared across all bins
(as in ``PerBinLogisticNormalSurrogate``), this surrogate uses per-bin
empirical biases computed from the mean bg-subtracted profile in each
resolution/azimuthal bin.  At z=0, the profile reproduces the empirical
average for that bin — the model only learns per-reflection corrections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .logistic_normal import PerBinProfilePosterior


class EmpiricalProfileSurrogate(nn.Module):
    """Profile surrogate with per-bin empirical bias and shared Hermite basis.

    Loads a basis file (``empirical_profile_basis_per_bin.pt``) containing:

    - ``W``             (K, d):       shared Hermite basis matrix
    - ``b_per_group``   (n_bins, K):  per-bin log(mean_profile)
    - ``mu_per_group``  (n_bins, d):  per-bin prior mean in latent space
    - ``std_per_group`` (n_bins, d):  per-bin prior std in latent space
    - ``sigma_prior``   float:        global fallback prior std
    - ``d``             int:          latent dimensionality

    Parameters
    ----------
    input_dim : int
        Dimension of the encoder output.
    basis_path : str
        Path to the empirical_profile_basis_per_bin.pt file.
    """

    def __init__(self, input_dim: int, basis_path: str, learn_W: bool = False) -> None:
        super().__init__()

        basis = torch.load(basis_path, weights_only=False)

        if learn_W:
            self.W = nn.Parameter(basis["W"].clone())                # (K, d)
        else:
            self.register_buffer("W", basis["W"])                    # (K, d)
        self.register_buffer("b_per_group", basis["b_per_group"])    # (n_bins, K)
        self.register_buffer("mu_per_group", basis["mu_per_group"])  # (n_bins, d)
        self.register_buffer("std_per_group", basis["std_per_group"])  # (n_bins, d)

        self.d: int = int(basis["d"])
        self.sigma_prior: float = float(basis.get("sigma_prior", 3.0))

        self.mu_head = nn.Linear(input_dim, self.d)
        self.std_head = nn.Linear(input_dim, self.d)

        # Initialise std_head so initial std ≈ softplus(-0.81) ≈ 0.37
        nn.init.zeros_(self.std_head.weight)
        nn.init.constant_(self.std_head.bias, -0.81)

    def forward(
        self, x: Tensor, group_labels: Tensor | None = None,
    ) -> PerBinProfilePosterior:
        """Map encoder output to a profile posterior.

        Parameters
        ----------
        x : (B, input_dim)
            Encoder output.
        group_labels : (B,) long tensor or None
            Bin index per reflection.  When provided, selects the per-bin
            empirical bias for each reflection.  When ``None``, uses the
            mean bias across all bins as a fallback.
        """
        mu_h = self.mu_head(x)                # (B, d)
        std_h = F.softplus(self.std_head(x))  # (B, d)

        if group_labels is not None:
            b = self.b_per_group[group_labels.long()]  # (B, K)
        else:
            b = self.b_per_group.mean(dim=0)  # (K,) fallback

        return PerBinProfilePosterior(
            mu_h=mu_h,
            std_h=std_h,
            W=self.W,
            b=b,
            sigma_prior=self.sigma_prior,
            mu_prior=self.mu_per_group,
            std_prior=self.std_per_group,
        )
