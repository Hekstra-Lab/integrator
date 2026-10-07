import torch
import torch.nn as nn
from torch.distributions import LogNormal

from .utils import get_positive_constraint


"""
Pattern:
class SomeDistribution(nn.Module)
  -> define linear heads
  -> constrain scale/positive params
  -> forward(...) returns a torch.distributions object
  -> build_* factory function
  -> arg_names for YAML/factory
"""


"""
YAML/config or registry
        |
        v
build_lognormal(**kwargs)
        |
        v
_reject_unknown_lognormal_args(kwargs)
        |
        v
LogNormalDistribution(**kwargs)
        |
        v
__init__(...)
        |
        v
model is created with:
  - linear_loc
  - linear_scale
  - scale_constrain
        |
        v
forward(x, x_)
        |
        v
loc   = linear_loc(x)
scale = linear_scale(x_)
scale = positive constraint(scale)
scale = scale + scale_min + eps
        |
        v
return LogNormal(loc, scale)
"""


class LogNormalDistribution(nn.Module):
    """
    LogNormal posterior distribution.

    A LogNormal random variable is always positive.

    This can be useful for intensity-like quantities because intensities are
    often positive and skewed.

    The network predicts two values:
      loc   = mean of the underlying Normal distribution
      scale = standard deviation of the underlying Normal distribution

    Then PyTorch defines:

      z ~ Normal(loc, scale)
      y = exp(z) > 0

    so:

      y ~ LogNormal(loc, scale)
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

        # Returns a function that makes scale positive.
        self.scale_constrain = get_positive_constraint(positive_constraint)

        # Predicts loc. Loc can be negative, zero, or positive.
        self.linear_loc = nn.Linear(in_features, 1)

        # Predicts scale. Scale must be positive.
        self.linear_scale = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor, x_: torch.Tensor):
        """
        x   -> linear_loc   -> loc
        x_  -> linear_scale -> positive scale
        """

        # Predict loc from one encoder feature vector.
        loc = self.linear_loc(x)

        # Predict positive scale from another encoder feature vector.
        scale = self.scale_constrain(self.linear_scale(x_))

        # Keep scale away from zero.
        scale = scale + self.scale_min + self.eps

        # Each sample needs one scalar loc and one scalar scale.

        """
        LogNormal:
        z ~ Normal(loc, scale)
        y = exp(z)
        """
        return LogNormal(loc=loc.flatten(), scale=scale.flatten())


def build_lognormal(**kwargs) -> nn.Module:
    """
    Helper function that creates the surrogate object for the registry.
    """
    _reject_unknown_lognormal_args(kwargs)
    return LogNormalDistribution(**kwargs)


def _lognormal_valid_args() -> set[str]:
    return {
        "in_features",
        "eps",
        "scale_min",
        "positive_constraint",
    }


def _reject_unknown_lognormal_args(kwargs: dict) -> None:
    valid = _lognormal_valid_args()

    # set(kwargs) converts the dictionary's keys into a set.
    # unknown = keys passed in kwargs but not allowed by valid.
    unknown = set(kwargs) - valid

    if unknown:
        raise ValueError(
            f"Unknown lognormal surrogate arg(s): {sorted(unknown)}. "
            f"Valid args: {sorted(valid)}."
        )


build_lognormal.arg_names = _lognormal_valid_args()  # type: ignore[attr-defined]
