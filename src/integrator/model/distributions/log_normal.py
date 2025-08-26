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
        out_features: int = 2,
        constraint: str = "softplus",
        eps: float = 1e-12,
        beta: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            out_features=out_features,
            constraint=constraint,
            eps=eps,
            beta=beta,
            **kwargs,
        )

        self.fc = Linear(
            in_features=self.in_features,
            out_features=self.out_features,
        )

    def distribution(self, loc, scale) -> LogNormal:
        """
        Args:
            loc ():
            scale ():

        Returns:

        """
        scale = self._constrain_fn(scale)
        return LogNormal(loc=loc.flatten(), scale=scale.flatten())

    def forward(
        self,
        x: Tensor,
    ) -> LogNormal:
        params = self.fc(x)
        lognormal = self.distribution(params[..., 0], params[..., 1])

        return lognormal


if __name__ == "__main__":
    # generate a batch of 10 representation vectors
    representation = torch.randn(10, 64)
    metarep = torch.randn(10, 64)

    # initialize a LogNormalDistribution object
    lognormal = LogNormalDistribution(in_features=64, constraint="softplus")
