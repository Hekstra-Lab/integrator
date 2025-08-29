import torch
from torch import Tensor
from torch.distributions.half_normal import HalfNormal

from integrator.layers import Linear
from integrator.model.distributions import BaseDistribution


class HalfNormalDistribution(BaseDistribution[HalfNormal]):
    def __init__(
        self,
        in_features,
        out_features=1,
        constraint="softplus",
    ):
        super().__init__(
            in_features=in_features,
            out_features=1,
            constraint=constraint,
        )
        self.fc = Linear(
            in_features=self.in_features,
            out_features=out_features,
        )

    def forward(
        self,
        x: Tensor,
    ) -> HalfNormal:
        scale = self.fc(x)
        scale = self._constrain_fn(scale)
        return HalfNormal(scale=scale)


if __name__ == "__main__":
    # Example usage

    in_features = 64
    half_normal_dist = HalfNormalDistribution(in_features, out_features=1)

    representation = torch.randn(10, in_features)
    qbg = half_normal_dist(representation)
