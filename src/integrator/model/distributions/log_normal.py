import torch
from torch import Tensor
from torch.distributions import LogNormal

from integrator.layers import Linear
from integrator.model.distributions import BaseDistribution


class LogNormalDistribution(BaseDistribution[LogNormal]):
    """
    LogNormal distribution with parameters predicted by a linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 2,
        constraint: str = "softplus",
        eps: float = 1e-12,
        beta: float = 1.0,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            constraint=constraint,
            eps=eps,
            beta=beta,
        )

        self.fc = Linear(
            in_features=in_features,
            out_features=out_features,
        )

    def forward(
        self,
        x: Tensor,
    ) -> LogNormal:
        params = self.fc(x)
        lognormal = LogNormal(
            loc=params[..., 0],
            scale=self._constrain_fn(params[..., 1]),
        )

        return lognormal


if __name__ == "__main__":
    # generate a batch of 10 representation vectors
    representation = torch.randn(10, 64)
    metarep = torch.randn(10, 64)

    # initialize a LogNormalDistribution object
    lognormal = LogNormalDistribution(in_features=64, constraint="softplus")
