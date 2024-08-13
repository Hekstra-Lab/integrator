import torch
from integrator.layers import Linear
import torch.nn.functional as F


class Constraint(torch.nn.Module):
    def __init__(self, eps=1e-12, beta=1.0):
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

    def forward(self, x):
        return F.softplus(x, beta=self.beta) + self.eps


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


class IntensityDistribution(torch.nn.Module):
    def __init__(self, dmodel, intensity_dist):
        super().__init__()
        self.linear_intensity_params = Linear(dmodel, 2)
        self.intensity_dist = intensity_dist
        self.constraint = Constraint()

    def intensity_distribution(self, intensity_params):
        loc = self.constraint(intensity_params[..., 0])
        scale = self.constraint(intensity_params[..., 1])
        return self.intensity_dist(loc, scale)

    def forward(self, representation):
        intensity_params = self.linear_intensity_params(representation)
        q_I = self.intensity_distribution(intensity_params)
        return q_I


class Builder(torch.nn.Module):
    def __init__(
        self,
        intensity_distribution,
        background_distribution,
        spot_profile_model,
        bg_indicator=None,
        eps=1e-12,
        beta=1.0,
    ):
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        self.intensity_distribution = intensity_distribution
        self.background_distribution = background_distribution
        self.spot_profile_model = spot_profile_model
        self.bg_indicator = bg_indicator if bg_indicator is not None else None

    def forward(
        self,
        representation,
        dxyz,
    ):

        bg_profile = (
            self.bg_indicator(representation) if self.bg_indicator is not None else None
        )

        spot_profile, L = self.spot_profile_model(representation, dxyz)

        q_bg = self.background_distribution(representation)

        q_I = self.intensity_distribution(representation)

        return q_bg, q_I, spot_profile, L, bg_profile
