from pylab import *
import torch
from integrator.layers import Linear
from integrator.models import MLP
from rs_distributions.transforms import FillScaleTriL


class DistributionBuilder(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        intensity_dist,
        background_dist,
        eps=1e-12,
        beta=1.0,
        output_dim=10,
        batch_size=100,
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
        self.L_transfrom = FillScaleTriL()
        self.batch_size = batch_size

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

    # def get_params(self, representation):
    # return self.linear1(representation)

    def MVNProfile3D(self, params, dxy, mask):
        factory_kwargs = {
            "device": dxy.device,
            "dtype": dxy.dtype,
        }
        # chol = torch.distributions.transforms.CorrCholeskyTransform(cache_size=0)
        # mu = params[..., 4:7]
        mu = torch.zeros((self.batch_size, 1, 3), requires_grad=False, **factory_kwargs)
        L = params[..., 4:]
        L = self.L_transfrom(L)
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, scale_tril=L
        )
        log_probs = mvn.log_prob(dxy)
        max_log_prob = torch.max(log_probs, dim=-1, keepdim=True)[0]
        log_probs_ = log_probs - max_log_prob
        masked_log_probs = log_probs * mask.squeeze(-1)
        profile = torch.exp(masked_log_probs)
        norm_factor = torch.sum(profile * mask.squeeze(-1), dim=-1, keepdim=True)
        profile = profile * mask.squeeze(-1) / norm_factor
        return profile

    def forward(self, representation, dxy, mask):
        params = self.linear1(representation)
        profile = self.MVNProfile3D(params, dxy, mask)

        # variational background distribution
        q_bg = self.background(params)

        # variational intensity distribution
        q_I = self.intensity_distribution(params)
        # print(f'loc_min:{loc.min()},loc_max{loc.max()},scale_min:{scale.min()},scale_max:{scale.max()}')

        return q_bg, q_I, profile


class VariationalDistributionBuilder(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        intensity_dist,
        background_dist,
        eps=1e-12,
        beta=1.0,
        output_dim=4,
        batch_size=100,
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
        self.L_transfrom = FillScaleTriL()
        self.batch_size = batch_size

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

    # def get_params(self, representation):
    # return self.linear1(representation)

    def forward(self, representation, dxy):
        params = self.linear1(representation)

        # variational background distribution
        q_bg = self.background(params)

        # variational intensity distribution
        q_I = self.intensity_distribution(params)
        # print(f'loc_min:{loc.min()},loc_max{loc.max()},scale_min:{scale.min()},scale_max:{scale.max()}')

        return q_bg, q_I
