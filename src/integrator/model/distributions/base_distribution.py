import enum
from dataclasses import dataclass
from typing import TypeVar, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Distribution


@dataclass(slots=True)
class MetaData:
    masks: Tensor | None = None
    metadata: Tensor | None = None


T = TypeVar("T", bound=torch.distributions.Distribution)


class ConstrainFn(str, enum.Enum):
    softplus = "softplus"
    exp = "exp"


class BaseDistribution[T: Distribution](nn.Module):
    """
    Base class for parametric distributions.
    Ensures buffers like eps/beta are device-safe and used consistently.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int | tuple[int, ...],
        constraint: str = "softplus",
        eps: float = 1e-12,
        beta: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Register buffers so they move with the module
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

        # Store constraint kind (not a closure)
        if constraint is None:
            self._constraint_kind = None
        else:
            self._constraint_kind = ConstrainFn[constraint]

    def _constrain(self, x: Tensor) -> Tensor:
        """Device/dtype-safe constraint using registered buffers."""
        if self._constraint_kind is None:
            return x
        if self._constraint_kind is ConstrainFn.softplus:
            return F.softplus(x, beta=float(self.beta)) + self.eps
        elif self._constraint_kind is ConstrainFn.exp:
            return torch.exp(x) + self.eps
        else:
            raise ValueError(
                f"Unknown constraint kind: {self._constraint_kind!r}"
            )

    def forward(self, x: Tensor) -> T:
        """To be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, x: Tensor) -> T:
        return cast(T, super().__call__(x))
