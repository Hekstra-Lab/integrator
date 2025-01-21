from integrator.model.distribution import BaseDistribution
from integrator.layers import Linear, Constraint
from torch.distributions import Gamma


class GammaDistribution(BaseDistribution):
    def __init__(
        self,
        dmodel,
        constraint=Constraint(),
        out_features=2,
    ):
        super().__init__(q=Gamma)
        self.fc = Linear(
            in_features=dmodel,
            out_features=out_features,
        )
        self.constraint = constraint

    def distribution(self, params):
        concentration = self.constraint(params[..., 0])
        rate = self.constraint(params[..., 1])
        return self.q(concentration, rate)

    def forward(self, representation):
        params = self.fc(representation)
        gamma = self.distribution(params)
        return gamma
