from integrator.model.distributions import BaseDistribution
from integrator.layers import Linear, Constraint
from torch.distributions import Normal
import torch


class NormalDistribution(BaseDistribution):
    def __init__(
        self,
        dmodel,
        constraint=Constraint(),
        out_features=2,
        use_metarep=False,
    ):
        super().__init__(q=Normal)
        self.use_metarep = use_metarep

        self.constraint = constraint

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

    def forward(self, representation, metarep=None):
        if self.use_metarep:
            assert metarep is not None, "metarep required when use_metarep=True"
            params1 = self.fc1(representation)
            combined_rep = torch.cat([representation, metarep], dim=1)
            params2 = self.fc2(combined_rep)
            normal = self.distribution(params1, params2)

        else:
            params = self.fc(representation)
            normal = self.distribution(params[..., 0], params[..., 1])
        return normal


if __name__ == "__main__":
    representation = torch.randn(10, 64)
    metarep = torch.randn(10, 64)
    model = NormalDistribution(dmodel=64, use_metarep=True)
    normal = model(representation, metarep)


rate = torch.randn(10, 100, 1323)
std = rate.std(dim=1, keepdim=True)


torch.distributions.Normal(rate, std).log_prob(torch.randn(10, 100, 1323))
