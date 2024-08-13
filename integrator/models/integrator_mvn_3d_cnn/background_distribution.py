import torch
from integrator.layers import Linear
from integrator.layers import Constraint


class BackgroundDistribution(torch.nn.Module):
    def __init__(self, dmodel, background_distribution, constraint=Constraint()):
        super().__init__()
        self.linear_bg_params = Linear(dmodel, 2)
        self.background_distribution = background_distribution
        self.constraint = constraint

    def background(self, bgparams):
        mu = self.constraint(bgparams[..., 0])
        sigma = self.constraint(bgparams[..., 1])
        return self.background_distribution(mu, sigma)

    def forward(self, representation):
        bgparams = self.linear_bg_params(representation)
        q_bg = self.background(bgparams)
        return q_bg
