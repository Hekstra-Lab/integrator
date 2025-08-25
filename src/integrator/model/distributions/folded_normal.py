from math import pi, sqrt

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal, constraints
from torch.distributions.constraints import Constraint
from torch.distributions.transformed_distribution import (
    TransformedDistribution,
)
from torch.distributions.transforms import AbsTransform

from integrator.model.distributions import BaseDistribution, MetaData


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
    def support(self) -> Constraint:
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


class FoldedNormalDistribution(BaseDistribution[FoldedNormal]):
    """
    FoldedNormal distribution with parameters predicted by a linear layer.
    """

    def __init__(
        self,
        in_features: int,
        eps: float = 1e-12,
        beta: float = 1.0,
    ):
        """
        Args:
        """
        super().__init__(in_features=in_features, eps=eps, beta=1.0)
        self.fc = nn.Linear(in_features, 2)

    def _transform_loc_scale(
        self, raw_loc, raw_scale
    ) -> tuple[Tensor, Tensor]:
        loc = torch.exp(raw_loc)
        scale = self.constraint(raw_scale)
        return loc, scale

    def forward(
        self, x: Tensor, *, meta_data: MetaData | None = None
    ) -> FoldedNormal:
        assert meta_data is None  # remove if you want to use metadata or masks

        raw_loc, raw_scale = self.fc(x).unbind(-1)
        loc, scale = self._transform_loc_scale(raw_loc, raw_scale)
        return FoldedNormal(loc, scale)


if __name__ == "main":
    foldednormal = FoldedNormalDistribution(in_features=64)
    representation = torch.randn(10, 64)
    q = foldednormal(representation)
