from pylab import *
import torch
from integrator.layers import Linear
from integrator.models import MLP


class DistributionBuilder(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        intensity_dist,
        background_dist,
        eps=1e-12,
        beta=1.0,
        output_dim=10,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
        self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)
        self.output_dim = output_dim
        self.input_dim = dmodel
        self.linear1 = Linear(self.input_dim, self.output_dim)
        self.intensity_dist = intensity_dist
        self.background_dist = background_dist

    def constraint(self, x):
        return torch.nn.functional.softplus(x, beta=self.beta) + self.eps

    def intensity_distribution(self, params):
        loc = params[..., 0]
        scale = params[..., 1]
        scale = self.constraint(scale)
        q_I = self.background_dist(loc, scale)
        return q_I

    def background(self, params):
        mu = params[..., 2]
        sigma = params[..., 3]
        sigma = self.constraint(sigma)
        q_bg = self.background_dist(mu, sigma)
        return q_bg

    def get_params(self, representation):
        return self.linear(representation)

    def MVNProfile3D(self, params, dxy):
        factory_kwargs = {
            "device": dxy.device,
            "dtype": dxy.dtype,
        }
        chol = torch.distributions.transforms.CorrCholeskyTransform(cache_size=0)
        L = chol(params[..., 4:7])
        mu = params[..., 7:]
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, scale_tril=L
        )
        profile = torch.exp(mvn.log_prob(dxy))
        return profile

    def forward(self, representation, dxy):
        params = self.linear1(representation)
        profile = self.MVNProfile3D(params, dxy)

        # variational background distribution
        q_bg = self.background(params)

        # variational intensity distribution
        q_I = self.intensity_distribution(params)

        return q_bg, q_I, profile
