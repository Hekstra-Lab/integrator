import torch
from torch import Tensor
from torch.distributions import Normal

from integrator.layers import Linear
from integrator.model.distributions import BaseDistribution, MetaData


class NormalDistribution(BaseDistribution[Normal]):
    def __init__(
        self,
        dmodel: int,
        out_features: int = 2,
        use_metarep: bool = False,
        q: type[Normal] = Normal,
    ):
        super().__init__()

        self.use_metarep = use_metarep
        self.q = q

        if self.use_metarep:
            # separate layers for params1 and params2
            self.fc1 = Linear(
                in_features=dmodel,
                out_features=1,
            )
            self.fc2 = Linear(
                in_features=dmodel * 2,
                out_features=1,
            )
        else:
            self.fc = Linear(
                in_features=dmodel,
                out_features=out_features,
            )

    def distribution(self, loc, scale):
        scale = self.constraint(scale)
        return self.q(loc=loc.flatten(), scale=scale.flatten())

    def forward(self, x: Tensor, *, meta_data: MetaData | None = None) -> Normal:
        assert meta_data is None
        if self.use_metarep:
            assert metarep is not None and self.fc2 is not None
            params1 = self.fc1(x)
            combined_rep = torch.cat([x, metarep], dim=1)
            params2 = self.fc2(combined_rep)
            normal = self.distribution(params1, params2)

        else:
            params = self.fc(x)
            normal = self.distribution(params[..., 0], params[..., 1])
        return normal


if __name__ == "__main__":
    x = torch.randn(10, 64)
    metarep = torch.randn(10, 64)
    model = NormalDistribution(dmodel=64, use_metarep=False)
    normal = model(x)
