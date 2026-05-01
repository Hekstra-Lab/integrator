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

    Behaves like a Distribution for the integrator:
    - .mean returns π * gamma_mean
    - .variance returns the mixture variance
    - .rsample() returns π * gamma_sample
    - .gamma gives the underlying Gamma distribution (for KL)
    - .pi gives the detection probability
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

    Forward returns ZeroInflatedGammaOutput with:
    - gamma: Gamma(k, r) from the k/r heads
    - pi: sigmoid(detection_head(x)) ∈ (0, 1)

    The effective intensity is I = π · I_gamma.
    """

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        k_min: float = 0.01,
        positive_constraint: str = "softplus",
        k_init: float = 1.0,
        r_init: float | None = None,
        pi_init: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_min = k_min
        self._constrain = get_positive_constraint(positive_constraint)
        self._constraint_name = positive_constraint

        self.linear_k = nn.Linear(in_features, 1)
        self.linear_r = nn.Linear(in_features, 1)
        self.detection_head = nn.Linear(in_features, 1)

        self._init_head_bias(self.linear_k, k_init, k_min)
        if r_init is not None:
            self._init_head_bias(self.linear_r, r_init, eps)

        with torch.no_grad():
            logit = math.log(pi_init / (1 - pi_init))
            self.detection_head.bias.fill_(logit)
            self.detection_head.weight.zero_()

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
        pi = torch.sigmoid(self.detection_head(x))

        gamma = Gamma(concentration=k.flatten(), rate=r.flatten())
        return ZeroInflatedGammaOutput(gamma=gamma, pi=pi.flatten())


class ZeroInflatedGammaB(nn.Module):
    """GammaB (mu/fano parameterization) with a learned detection probability π.

    Forward returns ZeroInflatedGammaOutput.
    """

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        k_min: float = 0.1,
        mean_init: float | None = None,
        fano_init: float = 1.0,
        mu_positive_constraint: str = "softplus",
        pi_init: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.k_min = k_min
        self._mu_constrain = get_positive_constraint(mu_positive_constraint)
        self._mu_constraint_name = mu_positive_constraint

        self.linear_mu = nn.Linear(in_features, 1)
        self.linear_fano = nn.Linear(in_features, 1)
        self.detection_head = nn.Linear(in_features, 1)

        if mean_init is not None:
            if mu_positive_constraint == "log":
                bias_val = math.log(max(mean_init, 1e-12))
            else:
                bias_val = _softplus_inverse_shifted(mean_init, eps)
            with torch.no_grad():
                self.linear_mu.bias.fill_(bias_val)
                self.linear_mu.weight.zero_()
                self.linear_fano.weight.zero_()

        with torch.no_grad():
            logit = math.log(pi_init / (1 - pi_init))
            self.detection_head.bias.fill_(logit)
            self.detection_head.weight.zero_()

    def forward(self, x: Tensor, x_: Tensor) -> ZeroInflatedGammaOutput:
        mu = self._mu_constrain(self.linear_mu(x))
        if self._mu_constraint_name == "softplus":
            mu = mu + self.eps
        fano = F.softplus(self.linear_fano(x_)) + self.eps

        r = 1.0 / fano
        k = (mu * r).clamp(min=self.k_min)
        pi = torch.sigmoid(self.detection_head(x))

        gamma = Gamma(concentration=k.flatten(), rate=r.flatten())
        return ZeroInflatedGammaOutput(gamma=gamma, pi=pi.flatten())
