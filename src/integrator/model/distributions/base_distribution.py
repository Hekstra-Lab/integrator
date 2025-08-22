from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Distribution


@dataclass(slots=True)
class MetaData:
    masks: Tensor | None = None
    metadata: Tensor | None = None


# 2) Generic base for “distribution heads” that return a Distribution of type T.
class BaseDistribution[T: Distribution](nn.Module):
    """Neural head that returns a Distribution of type T."""

    def __init__(
        self,
        eps: float | None = None,
        beta: float | None = None,
    ):
        super().__init__()

        self.eps: Tensor
        self.beta: Tensor

        if eps is not None:
            self.register_buffer("eps", torch.tensor(eps))
        else:
            self.register_buffer("eps", torch.tensor(1e-12))
        if beta is not None:
            self.register_buffer("beta", torch.tensor(beta))
        else:
            self.register_buffer("beta", torch.tensor(1.0))

    def constraint(self, x):
        beta = (self.beta + self.eps).item()
        return F.softplus(x, beta=beta)

    def forward(self, x: Tensor, *, meta_data: MetaData | None = None) -> T:  # subclasses implement
        raise NotImplementedError

    # Refine __call__ typing while preserving nn.Module hooks/autocast semantics.
    def __call__(self, x: Tensor, *, meta_data: MetaData | None = None) -> T:
        return cast(T, super().__call__(x, meta_data=meta_data))
