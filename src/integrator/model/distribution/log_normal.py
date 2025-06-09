from integrator.model.distribution import BaseDistribution
from integrator.layers import Linear, Constraint
from torch.distributions import Distribution
from torch.distributions import LogNormal
import torch


class LogNormalDistribution(BaseDistribution):
    def __init__(
        self,
        dmodel,
        constraint=Constraint(),
        out_features: int = 2,
        use_metarep: bool = False,
    ):
        super().__init__(
            q=LogNormal,
        )
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
            lognormal = self.distribution(params1, params2)

        else:
            params = self.fc(representation)
            lognormal = self.distribution(params[..., 0], params[..., 1])
        return lognormal


if __name__ == "__main__":
    # generate a batch of 10 representation vectors
    representation = torch.randn(10, 64)
    metarep = torch.randn(10, 64)

    # initialize a LogNormalDistribution object
    model = LogNormalDistribution(dmodel=64, use_metarep=True)

    # get the parameterized torch.distributions.LogNormal object
    lognormal = model(representation, metarep)

    # sample from the distribution
    lognormal.rsample([100])
