import torch
from integrator.layers import Linear, Constraint


class BackgroundDistribution(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        q_bg,
        constraint=Constraint(),
    ):
        super().__init__()
        self.fc = Linear(dmodel, 2)
        self.q_bg = q_bg
        self.constraint = constraint

    def background(self, params):
        loc = self.constraint(params[..., 0])
        scale = self.constraint(params[..., 1])
        return self.q_bg(loc, scale)

    def forward(self, representation):
        params = self.fc(representation)
        q_bg = self.background(params)
        return q_bg


class IntensityDistribution(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        q_I,
        constraint=Constraint(),
    ):
        super().__init__()
        self.fc = Linear(dmodel, 2)
        self.q_I = q_I
        self.constraint = constraint

    def intensity(self, params):
        loc = self.constraint(params[..., 0])
        scale = self.constraint(params[..., 1])
        return self.q_I(loc, scale)

    def forward(self, representation):
        params = self.fc(representation)
        q_I = self.intensity(params)
        return q_I
