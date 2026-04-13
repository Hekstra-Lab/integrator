from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Constrain(nn.Module):
    def __init__(
        self,
        constraint_fn: Literal["exp", "softplus"] | None,
        eps: float,
        beta: int,
    ):
        super().__init__()
        self.constraint_fn = constraint_fn
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        if self.constraint_fn is None:
            return x
        if self.constraint_fn == "softplus":
            return F.softplus(x, beta=self.beta) + self.eps
        elif self.constraint_fn == "exp":
            return torch.exp(x) + self.eps
        else:
            raise ValueError(
                f"Unknown constraint kind: {self.constraint_fn!r}"
            )
