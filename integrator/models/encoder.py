from pylab import *
import torch
from integrator.layers import Linear, ResidualLayer
from integrator.models import MLP


class Encoder(torch.nn.Module):
    def __init__(self, depth, dmodel, feature_dim, dropout=None):
        super().__init__()
        self.dropout = None
        self.mlp_1 = MLP(
            dmodel, depth, d_in=feature_dim, dropout=self.dropout, output_dims=dmodel
        )
        self.mean_pool = MeanPool()
        # I + bg + MVN
        # 2 + 2 + 6 + 3
        # self.linear = Linear(64, 2 + 2 + 6 + 3)

    def forward(self, shoebox_data, mask=None):
        out = self.mlp_1(shoebox_data)
        pooled_out = self.mean_pool(out, mask)
        # outputs = self.linear(pooled_out)
        return pooled_out


# Embed shoeboxes
class MeanPool(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.register_buffer(
            "dim",
            torch.tensor(dim),
        )

    def forward(self, data, mask=None):
        out = data.sum(1, keepdim=True)
        if mask is None:
            denom = data.shape[-1]
        else:
            denom = mask.sum(-2, keepdim=True)
        out = out / denom

        return out


# %%
class RotationPixelEncoder(torch.nn.Module):
    """
    Encodes pixels into (num_reflection x max_voxel_sixe x d_model)
    """

    def __init__(self, depth, dmodel, d_in_refl=4, dropout=None):
        super().__init__()
        self.dropout = dropout
        self.mlp_1 = MLP(
            dmodel, depth, d_in=d_in_refl, dropout=self.dropout, output_dims=dmodel
        )

    def forward(self, shoebox, mask=None):
        out = self.mlp_1(shoebox)
        pixel_rep = torch.relu(out)
        return pixel_rep


class MLPPij(torch.nn.Module):
    def __init__(self, width, depth, dropout=None, output_dims=None):
        super().__init__()
        layers = []
        layers.extend([ResidualLayer(width, dropout=dropout) for i in range(depth)])
        if output_dims is not None:
            layers.append(Linear(width, output_dims))
        self.main = torch.nn.Sequential(*layers)

    def forward(self, data, **kwargs):
        out = self.main(data)
        return out


class ProfilePredictor(torch.nn.Module):
    """
    Outputs p_ij matrix to scale Intensity values
    """

    def __init__(self, dmodel, depth, max_pixel, dropout=None):
        super().__init__()
        self.dropout = dropout
        self.mlp_1 = MLPPij(dmodel, depth, dropout=self.dropout, output_dims=64)
        self.mean_pool = MeanPool()
        self.linear = Linear(64, max_pixel)

    def forward(self, refl_representation, pixel_rep, mask=None):
        sum = pixel_rep + refl_representation.expand_as(pixel_rep)
        out = self.mlp_1(sum)
        pooled_out = self.mean_pool(out, mask)
        out = self.linear(pooled_out)
        out = torch.softmax(out, axis=-1)
        out = out.squeeze(-2)

        return out


# # %%
# from rs_distributions import distributions as rsd

# # Variational distributions
# intensity_dist = rsd.FoldedNormal
# background_dist = rsd.FoldedNormal

# # Prior distributions
# p_I_scale = 0.5
# prior_I = torch.distributions.log_normal.LogNormal(
# loc=torch.tensor(7.0, requires_grad=False),
# scale=torch.tensor(1.4, requires_grad=False),
# )
# prior_bg = torch.distributions.exponential.Exponential(
# torch.tensor(0.01, requires_grad=False),
# # scale=torch.tensor(1, requires_grad=False),
# )
# p_bg_scale = 0.05

# # %%
# # Use GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# data_pix = torch.randn((100, 3000, 6))
# data_refl = torch.randn((100, 3000, 7))
# pix_encoder = RotationPixelEncoder(depth=10, dmodel=64, d_in_refl=6)
# refl_encoder = Encoder(depth=10, dmodel=64, feature_dim=7)
# profile_ = ProfilePredictor(dmodel=64, depth=10, max_pixel=3000)
# profile = profile_(encoded_refls, encoded_pixels)
# q_constructor = VariationalDistributionBuilder(
# dmodel=64, intensity_dist=intensity_dist, background_dist=background_dist
# )


# encoded_pixels = pix_encoder(data_pix)
# encoded_refls = refl_encoder(data_refl)

# q_bg, q_I = q_constructor(encoded_refls)


# samples_bg = q_bg.rsample([10])
# samples_I = q_I.rsample([10])

# ressult = (samples_I.expand(10, 100, 3000) * profile) + samples_bg


# %%
# class VariationalDistributionBuilder(torch.nn.Module):
# def __init__(
# self,
# dmodel,
# intensity_dist,
# background_dist,
# eps=1e-12,
# beta=1.0,
# output_dim=4,
# batch_size=100,
# dtype=None,
# device=None,
# ):
# super().__init__()
# self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
# self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)
# self.output_dim = output_dim
# self.input_dim = dmodel
# self.linear1 = Linear(self.input_dim, self.output_dim)
# self.intensity_dist = intensity_dist
# self.background_dist = background_dist
# self.batch_size = batch_size

# def constraint(self, x):
# return torch.nn.functional.softplus(x, beta=self.beta) + self.eps

# def intensity_distribution(self, params):
# loc = params[..., 0]
# scale = params[..., 1]
# scale = self.constraint(scale)
# q_I = self.background_dist(loc, scale)
# return q_I

# def background(self, params):
# mu = params[..., 2]
# sigma = params[..., 3]
# sigma = self.constraint(sigma)
# q_bg = self.background_dist(mu, sigma)
# return q_bg

# # def get_params(self, representation):
# # return self.linear1(representation)

# def forward(self, representation):
# params = self.linear1(representation)

# # variational background distribution
# q_bg = self.background(params)

# # variational intensity distribution
# q_I = self.intensity_distribution(params)
# # print(f'loc_min:{loc.min()},loc_max{loc.max()},scale_min:{scale.min()},scale_max:{scale.max()}')

# return q_bg, q_I
