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
    ):
        super().__init__(q=Gamma)
        self.fc = Linear(
            in_features=dmodel,
            out_features=out_features,
        )
        self.constraint = constraint
        self.min_value = 1e-3
        self.max_value = 20.0

    def distribution(self, params):
        concentration = torch.clamp(
            self.constraint(params[..., 0]), min=self.min_value, max=self.max_value
        )
        rate = torch.clamp(
            self.constraint(params[..., 1]), min=self.min_value, max=self.max_value
        )
        return self.q(concentration, rate)

    def forward(self, representation):
        params = self.fc(representation)
        gamma = self.distribution(params)
        return gamma
