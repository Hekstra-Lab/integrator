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
        self.L_transform = FillScaleTriL()
        self.batch_size = batch_size

    def constraint(self, x):
        return torch.nn.functional.softplus(x, beta=self.beta) + self.eps

    def intensity_distribution(self, params):
        loc = params[..., 0]
        scale = params[..., 1]
        scale = self.constraint(scale)
        q_I = self.intensity_dist(loc, scale)
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
        batch_size = params.size(0)
        mu = torch.zeros((batch_size, 1, 3), requires_grad=False, **factory_kwargs)
        L = params[..., 4:]
        L = self.L_transform(L)
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, scale_tril=L
        )
        log_probs = mvn.log_prob(dxy)

        #max_tensor = torch.zeros_like(log_probs)
        #for k, (log_probs_shoebox, mask_shoebox) in enumerate(
        #    zip(log_probs, mask.squeeze(-1))
        #):
        #    n = mask_shoebox.sum()
        #    max_val = torch.max(log_probs_shoebox[:n])
        #    max_tensor[k, :] = max_val

        # Use broadcasting to subtract the max_tensor from log_probs
        #log_probs_ = (log_probs - max_tensor) * mask.squeeze(-1)
        profile = torch.exp(log_probs) * mask.squeeze(-1)

        # Calculate normalization factor excluding padded regions
        # norm_factor = torch.sum(
        # torch.exp(log_probs_) * mask.squeeze(-1), dim=-1, keepdim=True
        # )
        # profile = profile / norm_factor

        return profile



    def GaussianProfile(self, params, dxy, mask):
        factory_kwargs = {
            "device": dxy.device,
            "dtype": dxy.dtype,
        }

        batch_size = params.size(0)
        mu = torch.zeros((batch_size, 1, 3), requires_grad=False, **factory_kwargs)
        L = params[..., 4:]
        L = self.L_transform(L)

        precision = torch.matmul(L, L.transpose(-1, -1))

        X = dxy 

        quadratic_form = torch.sum(
            X[..., None, :] @ precision @ X[..., :, None], dim=-1
        )

        #k = X.shape[-1]

        #log_det_precision = torch.logdet(precision)

        # Compute the log of the normalization factor
        #log_normfactor = -0.5 * (
        #    k * torch.log(torch.tensor(2.0 * math.pi)) + log_det_precision
        #)

        # Compute the log of the profile
        log_profile =  - quadratic_form.squeeze(-1)

        # Apply the mask to the log profile
        masked_log_profile = log_profile * mask.squeeze(-1)

        # Compute the profile by exponentiating the masked log profile
        profile = torch.exp(masked_log_profile)

        # Compute the normalization factor
        # norm_factor = torch.sum(profile * mask.squeeze(-1), dim=-1, keepdim=True)

        # Check for zero norm factors and set them to 1 to avoid division by zero
        # zero_norm_mask = norm_factor == 0
        # norm_factor[zero_norm_mask] = 1.0

        # Normalize the profile
        # normalized_profile = profile * mask.squeeze(-1) / norm_factor

        # Set the profile to zero for samples with zero norm factor
        # normalized_profile[zero_norm_mask.squeeze(-1)] = 0.0

        return profile


    def forward(self, representation, dxy, mask):
        params = self.linear1(representation)
        profile = self.MVNProfile3D(params, dxy, mask)
        #profile = self.GaussianProfile(params,dxy,mask)

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
        self.L_transform = FillScaleTriL()
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
