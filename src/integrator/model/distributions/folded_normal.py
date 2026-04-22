from math import pi, sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import (
    Normal,
    TransformedDistribution,
    constraints,
)
from torch.distributions.transforms import AbsTransform


class FoldedNormal(TransformedDistribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

    def __init__(self, loc, scale, validate_args=None):
        self._normal = Normal(loc, scale, validate_args=validate_args)
        super().__init__(
            self._normal, AbsTransform(), validate_args=validate_args
        )

    @property
    def has_rsample(self) -> bool:
        return True

    @property
    def support(self) -> constraints.Constraint:
        return constraints.nonnegative

    @property
    def loc(self) -> Tensor:
        return self._normal.loc

    @property
    def scale(self) -> Tensor:
        return self._normal.scale

    @property
    def mean(self) -> Tensor:
        loc, scale = self._normal.loc, self._normal.scale
        a = loc / scale
        return scale * sqrt(2 / pi) * torch.exp(-0.5 * a**2) + loc * (
            1 - 2 * torch.distributions.Normal(0.0, 1.0).cdf(-a)
        )

    @property
    def variance(self) -> Tensor:
        loc, scale = self._normal.loc, self._normal.scale
        return loc**2 + scale**2 - self.mean**2

    def cdf(self, value) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        rt2 = torch.sqrt(torch.tensor(2.0))
        a = (value + self.loc) / (self.scale * rt2)
        b = (value - self.loc) / (self.scale * rt2)
        return 0.5 * (torch.erf(a) + torch.erf(b))

    def log_prob(self, value) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        n = self._normal
        return torch.logaddexp(n.log_prob(value), n.log_prob(-value))


class FoldedNormalDistribution(nn.Module):
    """
    FoldedNormal distribution surrogate.

    When `separate_inputs=False` (default), uses a single Linear(in, 2) and
    splits.  When `True`, uses two separate Linear(in, 1).
    """

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 0.1,
        separate_inputs: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.separate_inputs = separate_inputs

        if separate_inputs:
            self.linear_loc = nn.Linear(in_features, 1)
            self.linear_scale = nn.Linear(in_features, 1)
        else:
            self.fc = nn.Linear(in_features, 2)

    def forward(
        self,
        x: Tensor,
        x_: Tensor | None = None,
    ) -> FoldedNormal:
        if self.separate_inputs:
            raw_loc = self.linear_loc(x)
            raw_scale = self.linear_scale(x_ if x_ is not None else x)
        else:
            raw_loc, raw_scale = self.fc(x).chunk(2, dim=-1)

        loc = (F.softplus(raw_loc) + self.eps).squeeze()
        scale = (F.softplus(raw_scale) + self.eps).squeeze()

        return FoldedNormal(loc, scale)
