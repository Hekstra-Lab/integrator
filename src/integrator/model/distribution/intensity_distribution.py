from integrator.model.distribution import BaseDistribution
import torch.nn as nn
import torch.nn.functional as F
from integrator.layers import Linear, Constraint


class IntensityDistribution(BaseDistribution):
    def __init__(
        self,
        dmodel,
        q_I,
        constraint=Constraint(),
    ):
        super().__init__(q_I)
        self.fc = Linear(dmodel, 2)
        self.constraint = constraint

    def distribution(self, params):
        loc = self.constraint(params[..., 0])
        scale = self.constraint(params[..., 1])
        return self.q(loc, scale)

    def forward(self, representation):
        params = self.fc(representation)
        q_I = self.distribution(params)
        return q_I
