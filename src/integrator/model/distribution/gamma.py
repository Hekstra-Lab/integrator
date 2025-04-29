from integrator.model.distribution import BaseDistribution
import torch
from integrator.layers import Linear, Constraint
from torch.distributions import Gamma


class GammaDistribution(BaseDistribution):
    def __init__(
        self,
        dmodel,
        constraint=Constraint(),
        out_features=2,
        use_metarep=False,
    ):
        super().__init__(q=Gamma)

        self.use_metarep = use_metarep
        self.constraint = constraint
        self.min_value = 1e-3
        self.max_value = 100.0

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
            # single layer for both params
            self.fc = Linear(
                in_features=dmodel,
                out_features=out_features,
            )

    def distribution(self, concentration, rate):
        concentration = self.constraint(concentration)
        rate = self.constraint(rate)
        return self.q(concentration.flatten(), rate.flatten())

    def forward(self, rep, metarep=None):
        if self.use_metarep:
            assert metarep is not None, "metarep required when use_metarep=True"
            params1 = self.fc1(rep)
            combined_rep = torch.cat([rep, metarep], dim=1)
            params2 = self.fc2(combined_rep)
            gamma = self.distribution(params1, params2)
        else:
            params = self.fc(rep)
            gamma = self.distribution(params[..., 0], params[..., 1])

        return gamma


if __name__ == "__main__":
    # Example usage
    dmodel = 64
    gamma_dist = GammaDistribution(dmodel)
    representation = torch.randn(10, dmodel)  # Example input
    metarep = torch.randn(10, dmodel * 2)  # Example metadata representation
    qbg = gamma_dist(representation, metarep=metarep)
    qbg.rsample([100]).shape  # Sample from the distribution

