import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import LogNormal

from integrator.layers import Linear


class LogNormalDistribution(nn.Module):
    """LogNormal distribution with parameters from a single linear layer.

    For use with single-encoder models (IntegratorModelA).

    ``loc`` (mu in log-space) is unconstrained — the linear output is used
    directly, allowing the mean ``exp(mu)`` to cover any positive range.
    ``scale`` (sigma in log-space) is constrained positive via softplus.
    """

    def __init__(
        self,
        in_features: int,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__()

        self.fc = Linear(
            in_features=in_features,
            out_features=2,
            bias=False,
        )
        self.eps = eps

    def forward(
        self,
        x: Tensor,
    ) -> LogNormal:
        raw_loc, raw_scale = self.fc(x).chunk(2, dim=-1)
        loc = raw_loc.squeeze()  # unconstrained: mu in log-space
        scale = F.softplus(raw_scale).squeeze() + self.eps
        return LogNormal(loc=loc, scale=scale)


class LogNormalA(nn.Module):
    """LogNormal distribution with separate linear layers for loc and scale.

    For use with two-encoder models (IntegratorModelB) where ``x`` feeds
    into ``loc`` and ``x_`` feeds into ``scale``.

    ``loc`` (mu in log-space) is unconstrained.
    ``scale`` (sigma in log-space) is constrained positive via softplus.
    """

    def __init__(
        self,
        in_features: int = 64,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.linear_loc = nn.Linear(in_features, 1)
        self.linear_scale = nn.Linear(in_features, 1)
        self.eps = eps

    def forward(
        self,
        x: Tensor,
        x_: Tensor | None = None,
    ) -> LogNormal:
        loc = self.linear_loc(x).squeeze()  # unconstrained: mu in log-space

        if x_ is not None:
            raw_scale = self.linear_scale(x_)
        else:
            raw_scale = self.linear_scale(x)

        scale = F.softplus(raw_scale).squeeze() + self.eps
        return LogNormal(loc=loc, scale=scale)
