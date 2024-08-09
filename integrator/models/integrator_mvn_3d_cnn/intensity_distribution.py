import torch
from integrator.layers import Linear
from integrator.layers import Constraint


class IntensityDistribution(torch.nn.Module):
    def __init__(self, dmodel, intensity_dist):
        super().__init__()
        self.linear_intensity_params = Linear(dmodel, 2)
        self.intensity_dist = intensity_dist
        self.constraint = Constraint()

    def intensity_distribution(self, intensity_params):
        loc = self.constraint(intensity_params[..., 0])
        # loc = intensity_params[..., 0]
        scale = self.constraint(intensity_params[..., 1])
        return self.intensity_dist(loc, scale)

    def forward(self, representation):
        intensity_params = self.linear_intensity_params(representation)
        q_I = self.intensity_distribution(intensity_params)
        return q_I
