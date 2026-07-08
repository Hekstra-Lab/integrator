import math

import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal
from torch.distributions import constraints

from .utils import get_positive_constraint


class FoldedNormal(Distribution):
    """
    FoldedNormal distribution.

    If:
        X ~ Normal(loc, scale)
        Y = |X|

    then Y follows a folded normal distribution.

    We implement this manually instead of using AbsTransform because
    PyTorch's AbsTransform does not provide everything needed for log_prob.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    support = constraints.nonnegative
    has_rsample = True

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.base_dist = Normal(loc=loc, scale=scale)
        super().__init__(batch_shape=loc.shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterized sample.

        Draw X from Normal(loc, scale), then return |X|.
        """
        return self.base_dist.rsample(sample_shape).abs()

    def sample(self, sample_shape=torch.Size()):
        """
        Non-gradient sample.

        Draw X from Normal(loc, scale), then return |X|.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor):
        """
        Log probability of FoldedNormal.

        For y >= 0:

            p_Y(y) = Normal(y | loc, scale) + Normal(-y | loc, scale)

        because both +y and -y fold into the same absolute value y.
        """
        value = value.clamp_min(0.0)

        log_p_pos = self.base_dist.log_prob(value)
        log_p_neg = self.base_dist.log_prob(-value)

        return torch.logaddexp(log_p_pos, log_p_neg)

    @property
    def mean(self):
        """
        E[|X|] where X ~ Normal(loc, scale).
        """
        z = self.loc / (math.sqrt(2.0) * self.scale)

        return (
            self.scale * math.sqrt(2.0 / math.pi) * torch.exp(-z.pow(2))
            + self.loc * torch.erf(z)
        )

    @property
    def variance(self):
        """
        Var(|X|) = E[X^2] - E[|X|]^2

        For X ~ Normal(loc, scale):
            E[X^2] = loc^2 + scale^2
        """
        return self.loc.pow(2) + self.scale.pow(2) - self.mean.pow(2)


class FoldedNormalDistribution(nn.Module):
    """
    FoldedNormal posterior distribution.

    The network predicts:
      loc   = mean of the underlying Normal distribution
      scale = standard deviation of the underlying Normal distribution

    Then the distribution samples positive intensity values by folding:
      y = abs(z)
    """

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-6,
        scale_min: float = 0.01,
        positive_constraint: str = "softplus",
        **kwargs,
    ):
        super().__init__()

        self.eps = eps
        self.scale_min = scale_min

        self.scale_constrain = get_positive_constraint(positive_constraint)

        self.linear_loc = nn.Linear(in_features, 1)
        self.linear_scale = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor, x_: torch.Tensor):
        loc = self.linear_loc(x)
        scale = self.scale_constrain(self.linear_scale(x_))
        scale = scale + self.scale_min + self.eps

        loc = loc.flatten()
        scale = scale.flatten()

        return FoldedNormal(loc=loc, scale=scale)


def build_folded_normal(**kwargs) -> nn.Module:
    """
    Factory function used by the surrogate registry.
    """
    _reject_unknown_folded_normal_args(kwargs)
    return FoldedNormalDistribution(**kwargs)


def _folded_normal_valid_args() -> set[str]:
    return {
        "in_features",
        "eps",
        "scale_min",
        "positive_constraint",
    }


def _reject_unknown_folded_normal_args(kwargs: dict) -> None:
    valid = _folded_normal_valid_args()
    unknown = set(kwargs) - valid
    if unknown:
        raise ValueError(
            f"Unknown folded_normal surrogate arg(s): {sorted(unknown)}. "
            f"Valid args: {sorted(valid)}."
        )


build_folded_normal.arg_names = _folded_normal_valid_args()  # type: ignore[attr-defined]