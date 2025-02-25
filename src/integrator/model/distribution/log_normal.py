from integrator.model.distribution import BaseDistribution
from integrator.layers import Linear, Constraint
from torch.distributions import LogNormal


class LogNormalDistribution(BaseDistribution):
    def __init__(
        self,
        dmodel,
        constraint=Constraint(),
        out_features=2,
    ):
        super().__init__(
            q=LogNormal,
        )
        self.fc = Linear(
            in_features=dmodel,
            out_features=out_features,
        )
        self.constraint = constraint

    def distribution(self, params):
        loc = params[..., 0]
        scale = self.constraint(params[..., 1])
        return self.q(loc=loc, scale=scale)

    def forward(self, representation):
        params = self.fc(representation)
        lognormal = self.distribution(params)
        return lognormal


if __name__ == "__main__":
    data = torch.randn(10, 3, 21, 21)
