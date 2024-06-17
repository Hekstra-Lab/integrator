from pylab import *
import torch
import math
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
        batch_size=10,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
        self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)
        self.output_dim = output_dim
        self.linear1 = Linear(dmodel, 4)
        self.linear_L = Linear(dmodel,6)
        # self.linear_L_2D = Linear(dmodel, 3)
        # self.linear_L_3D = Linear(dmodel, 6)
        self.intensity_dist = intensity_dist
        self.background_dist = background_dist
        self.L_transform = FillScaleTriL(diag_transform=ExpTransform())
        self.batch_size = batch_size

    def constraint(self, x):
        return torch.nn.functional.softplus(x, beta=self.beta) + self.eps

    def intensity_distribution(self, params):
        loc = params[..., 0]
        loc = torch.exp(loc)
        scale = params[..., 1]
        scale = self.constraint(scale)
        q_I = self.intensity_dist(loc, scale)
        return q_I

    # def init_weights(self):
        # L_weights = self.linear_L.weight.data
        # L_bias = self.linear_L.bias.data

        # scale = 0.01
        # torch.nn.init.uniform_(L_weights, -scale, scale)
        # torch.nn.init.uniform_(L_bias, 1 - scale, 1 + scale)

        # L_weights.zero_()
        # L_bias.zero_()
        # L_bias[0] = 1.0
        # L_bias[2] = 1.0
        # L_bias[5] = 1.0

    def background(self, params):
        mu = params[..., 2]
        # mu = self.constraint(mu)
        sigma = params[..., 3]
        sigma = self.constraint(sigma)
        q_bg = self.background_dist(mu, sigma)
        return q_bg

    def MVNProfile3D(self, L_params, dxyz, mask):
        factory_kwargs = {
            "device": dxyz.device,
            "dtype": dxyz.dtype,
        }
        batch_size = L_params.size(0)
        mu = torch.zeros((batch_size, 1, 3), requires_grad=False, **factory_kwargs)
        mu = mu.to(torch.float32)
        # mu = params[...,4:7]
        # L = params[..., 7:]
        # L = self.Linear_L([..., 4:]
        # L_params = torch.cat([L_params, torch.zeros(1, 1, 3)], dim=-1)
        L = self.L_transform(L_params)
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, scale_tril=L
        )
        log_probs = mvn.log_prob(dxyz)
        profile = torch.exp(log_probs)
        return profile,L

    # def MVNProfile2D(self,L_params,dxy,mask):
        # factory_kwargs={
            # "device":dxy.device,
            # "dtype":dxy.dtype
            # }

        # batch_size = L_params.size(0)
        # mu = torch.zeros((batch_size, 1, 2), requires_grad=False, **factory_kwargs)
        # L_2d = self.L_transform(L_params)
        # mu = torch.zeros((batch_size, 1, 2), requires_grad=False, **factory_kwargs)
        # mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            # mu, scale_tril=L_2d
        # )
        # log_probs = mvn.log_prob(dxy)
        # profile = torch.exp(log_probs)

        # L = torch.zeros((batch_size,1,3,3),**factory_kwargs)
        # L[:,:,:2,:2] = L_2d
        # return profile,L

    def MVNProfile2D(self, L_params, dxy, mask):
        factory_kwargs = {
            "device": dxy.device,
            "dtype": dxy.dtype
        }

        batch_size = L_params.size(0)
        mu = torch.zeros((batch_size, 1, 2), requires_grad=False, **factory_kwargs)
        mu = mu.to(torch.float32)
        L_2d = self.L_transform(L_params[..., :3])  # Use only the first 3 parameters
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, scale_tril=L_2d
        )
        log_probs = mvn.log_prob(dxy)
        profile = torch.exp(log_probs)

        L = torch.zeros((batch_size, 1, 3, 3), **factory_kwargs)
        L[:, :, :2, :2] = L_2d
        return profile, L

    def forward(self, representation, dxyz, mask, isflat):
        params = self.linear1(representation)
        q_bg = self.background(params)
        q_I = self.intensity_distribution(params)


        if (isflat).all() == True:
            L_params = self.linear_L(representation)
            dxy = dxyz[..., :2]
            dxy = dxy.to(torch.float32)
            profile, L = self.MVNProfile2D(L_params, dxy, mask)
            return q_bg, q_I, profile, L

        elif(isflat).all() == False:
            L_params = self.linear_L(representation)
            dxyz = dxyz.to(torch.float32)
            profile,L = self.MVNProfile3D(L_params,dxyz,mask)
            return q_bg,q_I,profile,L
        else:
            rep_2d = representation[isflat]
            rep_3d = representation[~isflat]
            dxyz_3d = dxyz[~isflat]
            dxyz_3d = dxyz_3d.to(torch.float32)
            dxy = dxyz[isflat][..., :2]
            dxy = dxy.to(torch.float32)


            L_params = self.linear_L(representation)
            L_params_2d = L_params[isflat]
            L_params_3d = L_params[~isflat]


            if is_flat.all() == True:
                prof_2d, L_2 = self.MVNProfile2D(L_params_2d, dxy, mask[isflat])
                batch_size = representation.size(0)
                profile = torch.zeros(batch_size, mask.size(1), dtype=prof_2d.dtype, device=prof_2d.device)
                profile[isflat] = prof_2d
                profile[~isflat] = prof_3d

                L = torch.zeros(batch_size, 3, 3, dtype=L_2.dtype, device=L_2.device)
                L[isflat] = L_2.squeeze(1)

                return q_bg, q_I, profile, L

            else:
                prof_2d, L_2 = self.MVNProfile2D(L_params_2d, dxy, mask[isflat])
                prof_3d, L_3 = self.MVNProfile3D(L_params_3d, dxyz_3d, mask[~isflat])

                batch_size = representation.size(0)
                profile = torch.zeros(batch_size, mask.size(1), dtype=prof_3d.dtype, device=prof_3d.device)
                profile[isflat] = prof_2d
                profile[~isflat] = prof_3d

                L = torch.zeros(batch_size, 3, 3, dtype=L_3.dtype, device=L_3.device)
                L[isflat] = L_2.squeeze(1)
                L[~isflat] = L_3.squeeze(1)

                return q_bg, q_I, profile, L
