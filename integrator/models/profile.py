from pylab import *
import torch
import math
from integrator.layers import Linear
from integrator.models import MLP
from rs_distributions.transforms import FillScaleTriL
from torch.distributions.transforms import ExpTransform
import rs_distributions.distributions as rsd
import torch.distributions.constraints as constraints


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
    ):
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        self.output_dim = output_dim
        self.linear1 = Linear(dmodel, 4)
        self.linear_L_2D = Linear(dmodel, 3)
        self.linear_L_3D = Linear(dmodel, 6)
        # self.linear_L = Linear(dmodel, 6)
        self.intensity_dist = intensity_dist
        self.background_dist = background_dist
        self.L_transform = FillScaleTriL(diag_transform=ExpTransform())
        self.batch_size = batch_size

    def constraint(self, x, constraint_type):
        if isinstance(constraint_type, constraints.Positive) or isinstance(
            constraint_type, constraints.GreaterThan
        ):
            return torch.nn.functional.softplus(x, beta=self.beta) + self.eps
        elif isinstance(constraint_type, constraints.Real):
            return x
        else:
            raise NotImplementedError(f"Constraint {constraint_type} not implemented")

    # def intensity_distribution(self, params):
    # #loc = self.constraint(params[..., 0])
    # loc = params[..., 0]
    # loc = torch.exp(loc)
    # scale = self.constraint(params[..., 1])
    # return self.intensity_dist(loc, scale)

    # def background(self, params):
    # #mu = self.constraint(params[..., 2])
    # mu = params[..., 2]
    # sigma = self.constraint(params[..., 3])
    # return self.background_dist(mu, sigma)

    def intensity_distribution(self, params):
        arg_constraints = self.intensity_dist.arg_constraints
        constrained_params = [
            self.constraint(params[..., i], constraint)
            for i, constraint in enumerate(arg_constraints.values())
        ]
        return self.intensity_dist(*constrained_params)

    def background(self, params):
        arg_constraints = self.background_dist.arg_constraints
        constrained_params = [
            self.constraint(params[..., i + 2], constraint)
            for i, constraint in enumerate(arg_constraints.values())
        ]
        return self.background_dist(*constrained_params)

    def MVNProfile3D(self, L_params, dxyz, mask, device):
        batch_size = L_params.size(0)
        mu = torch.zeros((batch_size, 1, 3), requires_grad=False, device=device).to(
            torch.float32
        )
        L = self.L_transform(L_params).to(torch.float32)
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, scale_tril=L
        )
        log_probs = mvn.log_prob(dxyz)
        profile = torch.exp(log_probs)
        return profile, L

    def MVNProfile2D(self, L_params, dxy, mask, device):
        batch_size = L_params.size(0)
        mu = torch.zeros((batch_size, 1, 2), requires_grad=False, device=device).to(
            torch.float32
        )
        #        L_2d = self.L_transform(L_params[..., :3]).to(torch.float32)
        L_2d = self.L_transform(L_params).to(torch.float32)
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, scale_tril=L_2d
        )
        log_probs = mvn.log_prob(dxy)
        profile = torch.exp(log_probs)
        L = torch.zeros((batch_size, 1, 3, 3), device=device)
        L[:, :, :2, :2] = L_2d
        return profile, L

    def forward(self, representation, dxyz, mask, isflat):
        device = representation.device
        params = self.linear1(representation)
        q_bg = self.background(params)
        q_I = self.intensity_distribution(params)

        if isflat.all():
            L_params = self.linear_L_2D(representation)
            dxy = dxyz[..., :2]
            profile, L = self.MVNProfile2D(L_params, dxy, mask, device)
            return q_bg, q_I, profile, L

        elif not isflat.any():
            L_params = self.linear_L_3D(representation)
            profile, L = self.MVNProfile3D(L_params, dxyz, mask, device)
            return q_bg, q_I, profile, L

        else:
            rep_2d = representation[isflat]
            rep_3d = representation[~isflat]
            dxyz_3d = dxyz[~isflat]
            dxy = dxyz[isflat][..., :2]

            L_params_2d = self.linear_L_2D(rep_2d)
            L_params_3d = self.linear_L_3D(rep_3d)

            prof_2d, L_2 = self.MVNProfile2D(L_params_2d, dxy, mask[isflat], device)
            prof_3d, L_3 = self.MVNProfile3D(
                L_params_3d, dxyz_3d, mask[~isflat], device
            )
            batch_size = representation.size(0)
            profile = torch.zeros(
                batch_size, mask.size(1), dtype=prof_3d.dtype, device=device
            )
            profile[isflat] = prof_2d
            profile[~isflat] = prof_3d

            L = torch.zeros(batch_size, 3, 3, dtype=L_3.dtype, device=device)
            L[isflat] = L_2.squeeze(1)
            L[~isflat] = L_3.squeeze(1)

            return q_bg, q_I, profile, L
