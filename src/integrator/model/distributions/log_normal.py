import torch
from torch import Tensor
from torch.distributions import LogNormal

from integrator.layers import Linear
from integrator.model.distributions import BaseDistribution


class LogNormalDistribution(BaseDistribution[LogNormal]):
    def __init__(
        self,
        in_features: int,
        constraint: str = "softplus",
        eps: float = 1e-12,
        beta: float = 1.0,
    ):
        """
        Args:
            in_features:
            use_metarep:
        """
        super().__init__(
            in_features=in_features,
            eps=eps,
            beta=beta,
            constraint=constraint,
        )

        self.fc = Linear(
            in_features=in_features,
            out_features=2,
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
