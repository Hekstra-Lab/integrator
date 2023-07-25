from pylab import *
import torch
from integrator.layers import Linear, ResidualLayer
from integrator.models import MLP


class EllipticalProfileBase(torch.nn.Module):
    def __init__(self, dmodel, eps=1e-12, beta=1.0, dtype=None, device=None):
        super().__init__()
        self.linear = Linear(
            1024,  # use dmodel for non-transformer model
            #  bg  cov dxy  I  SigI
            3 + 3 + 2 + 1 + 1,
        )
        self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
        self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)

    def constraint(self, x):
        return torch.nn.functional.softplus(x, beta=self.beta) + self.eps

    def distribution(self, parameters):
        raise NotImplementedError(
            "Derived classes must implement self.distribution(tensor) -> distribution"
        )

    def background(self, params, dxy):
        m = params[..., :2]
        b = params[..., 2]
        bg = (m * dxy).sum(-1) + b
        bg = self.constraint(bg)
        return bg

    def centroid_offset(self, params):
        ddxy = params[..., 6:8]
        return ddxy

    def profile(self, params, dxy):
        factory_kwargs = {
            "device": dxy.device,
            "dtype": dxy.dtype,
        }
        diag = self.constraint(params[..., 3:5])
        L = diag[:, None, :] * torch.eye(2, **factory_kwargs)[None, ...]
        L[..., 1, 0] = params[..., 5]
        precision = torch.matmul(L, L.transpose(-1, -2))
        ddxy = self.centroid_offset(params)
        X = dxy - ddxy
        profile = torch.exp(-X[..., None, :] @ precision @ X[..., :, None])
        profile = torch.squeeze(profile, [-2, -1])
        return profile

    def get_params(self, representation):
        return self.linear(representation)

    def forward(self, representation, dxy, mask=None, mc_samples=10):
        params = self.get_params(representation)
        # profile = self.profile(params, dxy)
        bg = self.background(params, dxy)
        q = self.distribution(params)
        return bg, q


class EllipticalProfile(EllipticalProfileBase):
    def distribution(self, parameters):
        loc = parameters[..., -2]
        scale = parameters[..., -1]
        scale = self.constraint(scale)
        q = torch.distributions.LogNormal(loc, scale)
        return q
