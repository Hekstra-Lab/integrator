import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Gamma

from .gamma import _softplus_inverse_shifted
from .utils import get_positive_constraint


@dataclass
class ZeroInflatedGammaOutput:
    """Output of the ZeroInflatedGamma surrogate.

    Uses the straight-through Gumbel-Sigmoid estimator for the
    detection decision: hard {0, 1} in the forward pass, but
    gradients flow through the soft sigmoid in the backward pass.

    Attributes:
        gamma: Underlying Gamma distribution for intensity given detection.
        pi: Detection probability ∈ (0, 1), shape (B,).
    """

    gamma: Gamma
    pi: Tensor

    @property
    def mean(self) -> Tensor:
        return self.pi * self.gamma.mean

    @property
    def variance(self) -> Tensor:
        gamma_mean = self.gamma.mean
        gamma_var = self.gamma.variance
        return self.pi * (gamma_var + gamma_mean**2) - (self.pi * gamma_mean) ** 2

    def rsample(self, sample_shape=torch.Size()) -> Tensor:
        z = self.gamma.rsample(sample_shape)
        if sample_shape:
            pi = self.pi.unsqueeze(0).expand_as(z)
        else:
            pi = self.pi
        return pi * z


class ZeroInflatedGammaA(nn.Module):
    """GammaA with a learned detection probability π.

    The detection head receives BOTH encoder features (concatenated)
    plus is independent from the k and r heads.
    """

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        k_min: float = 0.01,
        positive_constraint: str = "softplus",
        k_init: float = 1.0,
        r_init: float | None = None,
        pi_init: float = 0.7,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_min = k_min
        self._constrain = get_positive_constraint(positive_constraint)
        self._constraint_name = positive_constraint

        self.linear_k = nn.Linear(in_features, 1)
        self.linear_r = nn.Linear(in_features, 1)

        # Detection head: takes BOTH x and x_ concatenated
        self.detection_head = nn.Sequential(
            nn.Linear(in_features * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self._init_head_bias(self.linear_k, k_init, k_min)
        if r_init is not None:
            self._init_head_bias(self.linear_r, r_init, eps)

        # Initialize detection output bias to logit(pi_init)
        with torch.no_grad():
            logit = math.log(pi_init / (1 - pi_init))
            self.detection_head[-1].bias.fill_(logit)

    def _init_head_bias(self, linear: nn.Linear, target: float, floor: float) -> None:
        if linear.bias is None:
            return
        delta = max(target - floor, 1e-12)
        if self._constraint_name == "log":
            val = math.log(delta)
        elif delta > 30.0:
            val = float(delta)
        else:
            val = math.log(math.expm1(delta))
        with torch.no_grad():
            linear.bias.fill_(val)

    def forward(self, x: Tensor, x_: Tensor) -> ZeroInflatedGammaOutput:
        k = self._constrain(self.linear_k(x)) + self.k_min
        r = self._constrain(self.linear_r(x_)) + self.eps
        pi = torch.sigmoid(self.detection_head(torch.cat([x, x_], dim=-1)))

        gamma = Gamma(concentration=k.flatten(), rate=r.flatten())
        return ZeroInflatedGammaOutput(gamma=gamma, pi=pi.flatten())


class ZeroInflatedGammaB(nn.Module):
    """GammaB (mu/fano) with a learned detection probability π.

    The detection head receives BOTH encoder features (concatenated)
    plus is independent from the mu and fano heads.
    """

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        k_min: float = 0.1,
        mean_init: float | None = None,
        fano_init: float = 1.0,
        positive_constraint: str = "softplus",
        mu_positive_constraint: str | None = None,
        pi_init: float = 0.7,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_min = k_min
        constraint = mu_positive_constraint or positive_constraint
        self._mu_constrain = get_positive_constraint(constraint)
        self._mu_constraint_name = constraint

        self.linear_mu = nn.Linear(in_features, 1)
        self.linear_fano = nn.Linear(in_features, 1)

        # Detection head: takes BOTH x and x_ concatenated
        self.detection_head = nn.Sequential(
            nn.Linear(in_features * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        if mean_init is not None:
            if constraint == "log":
                bias_val = math.log(max(mean_init, 1e-12))
            else:
                bias_val = _softplus_inverse_shifted(mean_init, eps)
            with torch.no_grad():
                self.linear_mu.bias.fill_(bias_val)
                self.linear_mu.weight.zero_()
                self.linear_fano.weight.zero_()

        # Initialize detection output bias to logit(pi_init)
        with torch.no_grad():
            logit = math.log(pi_init / (1 - pi_init))
            self.detection_head[-1].bias.fill_(logit)

    def forward(self, x: Tensor, x_: Tensor) -> ZeroInflatedGammaOutput:
        mu = self._mu_constrain(self.linear_mu(x))
        if self._mu_constraint_name == "softplus":
            mu = mu + self.eps
        fano = F.softplus(self.linear_fano(x_)) + self.eps

        r = 1.0 / fano
        k = (mu * r).clamp(min=self.k_min)
        pi = torch.sigmoid(self.detection_head(torch.cat([x, x_], dim=-1)))

        gamma = Gamma(concentration=k.flatten(), rate=r.flatten())
        return ZeroInflatedGammaOutput(gamma=gamma, pi=pi.flatten())
