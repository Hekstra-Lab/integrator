from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Distribution


class ConstrainFn(StrEnum):
    softplus = auto()
    exp = auto()


@dataclass(slots=True, kw_only=True)
class ConstraintSpec:
    kind: ConstrainFn
    eps: float = 1e-12
    beta: float = 1.0  # softplus beta

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ConstrainFn):
            try:
                self.kind = ConstrainFn[self.kind]  # by name, case-sensitive
            except (KeyError, TypeError):
                try:
                    self.kind = ConstrainFn(str(self.kind))  # by value
                except ValueError as e:
                    valid = [m.value for m in ConstrainFn]
                    raise ValueError(
                        f"Invalid kind {self.kind!r}. Valid options: {valid}"
                    ) from e

        if self.kind is ConstrainFn.softplus and self.beta <= 0:
            raise ValueError("beta must be > 0 for softplus")
        if self.eps < 0:
            raise ValueError("eps must be >= 0")


def constraint_fn(params: ConstraintSpec) -> Callable[[Tensor], Tensor]:
    eps_t = torch.as_tensor(params.eps)

    match params.kind:
        case ConstrainFn.softplus:
            beta = float(params.beta)
            return lambda x: F.softplus(x, beta=beta) + eps_t
        case ConstrainFn.exp:
            return lambda x: torch.exp(x) + eps_t
        case _:
            # Should be unreachable due to __post_init__
            raise ValueError(f"Unknown constraint kind: {params.kind!r}")


@dataclass(slots=True)
class MetaData:
    masks: Tensor | None = None
    metadata: Tensor | None = None


class BaseDistribution[T: Distribution](nn.Module):
    in_features: int
    """Dimension of the input `shoebox`."""
    out_features: int | tuple[int, ...]
    """Dimension of the paramters for `qp`."""
    eps: Tensor
    """Registed buffer: Small offset to prevent division by zero."""
    beta: Tensor
    """Registed buffer: Beta parameter used in the softplus constraint."""

    def __init__(
        self,
        in_features: int,
        out_features: int | tuple[int, ...],
        constraint: str = "softplus",
        eps: float = 1e-12,
        beta: float = 1.0,
    ):
        """
        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            constraint: String name of positivity constraint function.
            eps: Optional epsilon used for numerical stability. If ``None``, defaults to ``1e-12``.
            beta: Optional beta parameter for the softplus constraint. If ``None``, defaults to ``1.0``.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("eps", torch.tensor(1e-6 if eps is None else eps))
        self.register_buffer(
            "beta", torch.tensor(1.0 if beta is None else beta)
        )

        self._constrain_fn: Callable[[Tensor], Tensor]
        if constraint is None:
            self._constrain_fn = lambda x: x

        else:
            self._constrain_fn = constraint_fn(
                params=ConstraintSpec(
                    kind=ConstrainFn[constraint],
                    eps=self.eps.item(),
                    beta=self.beta.item(),
                )
            )

    def forward(self, x: Tensor) -> T:
        """
        Forward pass to be implemented by subclass

        Args:
            x: Input tensor of shape ``(batch, in_features)`` or compatible.

        Returns:
            A distribution instance of type ``T``.
        """
        raise NotImplementedError

    def __call__(self, x: Tensor) -> T:
        return cast(T, super().__call__(x))
