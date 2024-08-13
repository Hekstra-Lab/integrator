# from pylab import *
import torch
from rs_distributions.transforms import FillScaleTriL
from torch.distributions.transforms import ExpTransform, SoftplusTransform
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical
from integrator.layers import Linear

# We need to modify this script so that the 2D and 3D models are special cases of the
# 2D and 3D mixture models

# The dimensionality and tensor sizes should be inferred from the number of components, and the profile type.

# We can set a string attribute to specify the type of profile model that should be used.
# Profile type
# If mixture, then specify number of components, the dimensionality of the linear layers, etc.


# # Define a simple CNN
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = torch.nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.adaptive_pool = torch.nn.AdaptiveAvgPool1d(
            10
        )  # Adjust to subregion size 10x10
        self.fc = torch.nn.Linear(128 * 10, 441 * 3)  # Output size for subregion 10x10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = x.view(x.size(0), 3, 21**2)  # Reshape to subregion 10x10
        return x


class MVN3D(torch.nn.Module):
    def __init__(self, dmodel):
        super().__init__()
        self.dmodel = dmodel
        self.L_transform = FillScaleTriL(diag_transform=SoftplusTransform())
        self.mean_layer = Linear(dmodel, 3)
        self.scale_layer = Linear(dmodel, 6)

    def forward(self, representation, dxyz, num_planes=3):
        batch_size = representation.size(0)

        means = self.mean_layer(representation).view(batch_size, 1, 3)

        scales = self.scale_layer(representation).view(batch_size, 1, 6)

        L = FillScaleTriL(diag_transform=SoftplusTransform())(scales)

        mvn = MultivariateNormal(means, scale_tril=L)

        log_probs = mvn.log_prob(dxyz.view(441 * 3, batch_size, 3))

        profile = torch.exp(log_probs).view(batch_size, num_planes * 441)

        return profile, L


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


class BackgroundIndicator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = SimpleCNN()

    def forward(self, representation):
        bg_profile_params = self.cnn(representation)
        bg_profile = torch.sigmoid(bg_profile_params)
        return bg_profile


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


# dmodel = 64
# representation = torch.randn(2, 1, dmodel)
# dxyz = torch.randn(2, 441 * 3, 3)
# mask = torch.ones(2, 441 * 3)


# # Specify background and intensity distribution
# intensity_dist = torch.distributions.gamma.Gamma
# background_dist = torch.distributions.gamma.Gamma


# # bg_distribution model
# bg_distribution_model = BackgroundDistribution(dmodel, background_dist)

# isflat = torch.zeros(2, 441 * 3)

# intensity_profile = MixtureModel3D(dmodel, num_components=5)

# background = BackgroundDistribution(dmodel, background_dist)
# bg_indicator = BackgroundIndicator()

# intensity = IntensityDistribution(dmodel, intensity_dist)

# constraint = Constraint(eps=1e-12, beta=1.0)

# builder = Builder(
# intensity_distribution_model,
# bg_distribution_model,
# intensity_profile,
# bg_indicator=bg_indicator,
# )

# builder(representation, dxyz, mask, isflat, use_mixture_model=True)
# # The builder works for the


# class Builder(torch.nn.Module):
# def __init__(
# self,
# dmodel,
# intensity_dist,
# background_dist,
# eps=1e-12,
# beta=1.0,
# num_planes=3,
# num_components=5,  # number of components in the mixture model
# ):
# super().__init__()
# self.register_buffer("eps", torch.tensor(eps))
# self.register_buffer("beta", torch.tensor(beta))
# # self.linear = Linear(dmodel, 9 * 3)  # Predict params for 3 sets of 3 Gaussians

# self.dmodel = dmodel
# self.linear_bg_params = Linear(dmodel, 2)
# self.linear_intensity_params = Linear(dmodel, 2)
# self.intensity_dist = intensity_dist
# self.background_dist = background_dist
# self.L_transform = FillScaleTriL(diag_transform=SoftplusTransform())
# # self.mixture_weight_layer = Linear(dmodel, num_planes * num_components)
# self.mixture_weight_layer = Linear(dmodel, num_components)
# self.mean_layer_3d = Linear(dmodel, (num_components - 1) * 3)
# # self.mean_layer_2d = Linear(
# # dmodel, num_planes * (num_components - 1) * 2
# # )  # Adjusted for fewer parameters
# # self.scale_layer_2d = Linear(dmodel, num_planes * num_components * 3)
# self.scale_layer_3d = Linear(self.dmodel, num_components * 6)
# self.num_planes = num_planes
# self.num_components = num_components
# self.cnn = SimpleCNN()

# def constraint(self, x):
# return F.softplus(x, beta=self.beta) + self.eps

# def intensity_distribution(self, intensity_params):
# loc = self.constraint(intensity_params[..., 0])
# scale = self.constraint(intensity_params[..., 1])
# return self.intensity_dist(loc, scale)

# # def intensity_distribution(self, intensity_params):
# # loc = self.constraint(intensity_params[..., :3])
# # scale = self.constraint(intensity_params[..., 3:])
# # return self.intensity_dist(loc, scale)

# def penalty(self, L):
# diag_elements = L.diagonal(dim1=-2, dim2=-1)
# penalty = torch.sum(diag_elements**2)
# return penalty

# def distance_penalty(self, means):
# # Calculate pairwise distances between the means
# batch_size, num_planes, num_components, _ = means.shape

# means = means.view(batch_size * num_planes, num_components, 2)

# dist_matrix = torch.cdist(means, means, p=2)

# # Exclude self-distances by adding a large value
# inf_mask = (
# torch.eye(num_components, device=means.device)
# .unsqueeze(0)
# .expand(batch_size * num_planes, -1, -1)
# )

# dist_matrix = dist_matrix + inf_mask * 1e6

# # Penalize means that are too close or too far
# too_close = torch.exp(-dist_matrix)

# too_far = torch.exp(
# dist_matrix - 10
# )  # Adjust the constant based on the desired maximum distance

# penalty = torch.sum(too_close + too_far)

# return penalty

# def background(self, bgparams):
# mu = self.constraint(bgparams[..., 0])
# sigma = self.constraint(bgparams[..., 1])
# return self.background_dist(mu, sigma)

# # def background(self, bgparams):
# # mu = self.constraint(bgparams[..., :3])
# # sigma = self.constraint(bgparams[..., 3:])
# # return self.background_dist(mu, sigma)

# def MVNProfile2D_per_im(self, L_params, dxy, device, batch_size):
# mu = torch.zeros((batch_size, 1, 2), requires_grad=False, device=device).to(
# torch.float32
# )
# L_2d = self.L_transform(L_params).to(torch.float32)
# mvn = MultivariateNormal(mu, scale_tril=L_2d)
# log_probs = mvn.log_prob(dxy)
# profile = torch.exp(log_probs)
# L = torch.zeros((batch_size, 1, 3, 3), device=device)
# L[:, :, :2, :2] = L_2d
# return profile, L

# def UnnormalizedGaussianProfile2D(self, L_params, dxy, device, batch_size):
# mu = torch.zeros((1, 2), requires_grad=False, device=device).to(torch.float32)
# L_2d = self.L_transform(L_params).to(torch.float32)
# dxy_centered = dxy - mu.unsqueeze(0).unsqueeze(0)
# inv_L_2d = torch.inverse(L_2d)
# mahalanobis_distance = torch.sum(
# dxy_centered @ inv_L_2d.unsqueeze(0).unsqueeze(0) * dxy_centered, dim=-1
# )
# profile = torch.exp(-0.5 * mahalanobis_distance)
# L = torch.zeros((batch_size, 1, 3, 3), device=device)
# L[:, :, :2, :2] = L_2d
# return profile, L

# def MixtureModel2D(self, representation, dxyz, num_planes=3, num_components=5):
# batch_size = representation.size(0)

# mixture_weights = self.mixture_weight_layer(representation).view(
# batch_size, num_planes, num_components
# )
# mixture_weights = F.softmax(mixture_weights, dim=-1)

# means = self.mean_layer(representation).view(
# batch_size, num_planes, num_components - 1, 2
# )

# # Anchor one MVN per plane by setting its mean to 0
# zero_means = torch.zeros(
# (batch_size, num_planes, 1, 2), device=representation.device
# )
# means = torch.cat([zero_means, means], dim=2)

# scales = self.scale_layer(representation).view(
# batch_size, num_planes, num_components, 3
# )

# L = FillScaleTriL(diag_transform=SoftplusTransform())(scales)

# mvn = MultivariateNormal(means, scale_tril=L)

# mix = Categorical(mixture_weights)

# gmm = MixtureSameFamily(mixture_distribution=mix, component_distribution=mvn)

# dxy = dxyz[..., :2]

# log_probs = gmm.log_prob(dxy.view(441, batch_size, 3, 2))

# profile = torch.exp(log_probs).view(batch_size, num_planes * 441)

# # Calculate distance penalty

# return profile, L

# def MixtureModel3D(self, representation, dxyz, num_planes=3):
# num_components = self.num_components
# batch_size = representation.size(0)
# mixture_weights = self.mixture_weight_layer(representation).view(
# batch_size, num_components
# )
# mixture_weights = F.softmax(mixture_weights, dim=-1)

# means = self.mean_layer_3d(representation).view(
# batch_size, num_components - 1, 3
# )

# zero_means = torch.zeros((batch_size, 1, 3), device=representation.device)

# means = torch.cat([zero_means, means], dim=1)

# scale_layer = Linear(self.dmodel, num_components * 6)

# scales = self.scale_layer_3d(representation).view(batch_size, num_components, 6)

# L = FillScaleTriL(diag_transform=SoftplusTransform())(scales)

# mvn = MultivariateNormal(means, scale_tril=L)

# mix = Categorical(mixture_weights)

# gmm = MixtureSameFamily(mixture_distribution=mix, component_distribution=mvn)

# log_probs = gmm.log_prob(dxyz.view(441 * 3, batch_size, 3))

# profile = torch.exp(log_probs).view(batch_size, num_planes * 441)

# return profile, L

# def forward(self, representation, dxyz, mask, isflat, use_mixture_model=True):
# device = representation.device

# # params = self.linear(representation)

# bgparams = self.linear_bg_params(representation)

# intensity_params = self.linear_intensity_params(representation)

# q_bg = self.background(bgparams)

# bg_profile_params = self.cnn(representation)

# bg_profile = torch.sigmoid(bg_profile_params)

# q_I = self.intensity_distribution(intensity_params)

# batch_size = bgparams.size(0)

# dxy = dxyz[..., :2]

# # if use_mixture_model:
# # profile, L = self.MixtureModel2D(representation, dxyz)

# # penalty = torch.zeros(0)

# # return q_bg, q_I, profile, L, penalty, bg_profile  # Example usage:

# if use_mixture_model:
# profile, L = self.MixtureModel3D(representation, dxyz)

# penalty = torch.zeros(0)

# return q_bg, q_I, profile, L, penalty, bg_profile  # Example usage:

# else:
# profile1, L1 = self.MVNProfile2D_per_im(
# params[..., :3], dxy[..., :441, :], device, batch_size
# )
# profile2, L2 = self.MVNProfile2D_per_im(
# params[..., 3:6], dxy[..., 441 : 441 * 2, :], device, batch_size
# )
# profile3, L3 = self.MVNProfile2D_per_im(
# params[..., 6:9], dxy[..., 441 * 2 :, :], device, batch_size
# )
# profile = torch.cat((profile1, profile2, profile3), dim=-1)
# penalty = self.penalty(L1) + self.penalty(L2) + self.penalty(L3)

# return q_bg, q_I, profile, L, penalty, bg_profile  # Example usage:


if __name__ == "__main__":
    # Parameters
    batch_size = 2
    num_planes = 3
    num_components = 4
    d_model = 64

    # Define distributions
    intensity_dist = torch.distributions.Normal
    background_dist = torch.distributions.Normal

    # Create Builder
    builder = Builder(d_model, intensity_dist, background_dist)

    # Example inputs
    representation = torch.randn(batch_size, 1, d_model)
    dxyz = torch.randn(batch_size, 441 * num_planes, 3)
    mask = torch.ones(batch_size, 441 * num_planes)
    isflat = torch.zeros(batch_size, 441 * num_planes)

    # Forward pass
    q_bg, q_I, profile, L, penalty, bg_profile = builder(
        representation, dxyz, mask, isflat, use_mixture_model=True
    )

    z = q_I.rsample([100])
    bg = q_bg.rsample([100])

    z.shape
    bg.shape

    z.permute(1, 0, 2) * profile.unsqueeze(1) + bg.permute(
        1, 0, 2
    ) * bg_profile.unsqueeze(1)

    profile.shape
    bg_profile.shape

    print(f"Profile shape: {profile.shape}")
    print(profile)
    print(f"Background profile shape: {bg_profile.shape}")
    print(bg_profile)


# %%
# import torch
# import torch.nn astorch.nn
# import torch.nn.functional as F
# from torch.distributions import Categorical, Dirichlet
# from integrator.layers import Linear
# from rs_distributions.transforms import FillScaleTriL
# from torch.distributions.transforms import SoftplusTransform

# import torch
# import torch.nn astorch.nn
# import torch.nn.functional as F
# from torch.distributions import Categorical, Dirichlet
# from integrator.layers import Linear
# from rs_distributions.transforms import FillScaleTriL
# from torch.distributions.transforms import SoftplusTransform


# class Builder(torch.nn.Module):
# def __init__(
# self,
# dmodel,
# intensity_dist,
# background_dist,
# eps=1e-12,
# beta=1.0,
# num_planes=3,
# num_cubes=50,
# grid_size=21,
# ):
# super().__init__()
# self.register_buffer("eps", torch.tensor(eps))
# self.register_buffer("beta", torch.tensor(beta))

# self.linear_bg_params = Linear(dmodel, 2)
# self.linear_intensity_params = Linear(dmodel, 2)
# self.intensity_dist = intensity_dist
# self.background_dist = background_dist
# self.num_planes = num_planes
# self.num_cubes = num_cubes
# self.grid_size = grid_size
# self.max_index = grid_size * grid_size - 22
# self.num_indices = self.max_index + 1

# self.index_layer = Linear(dmodel, num_planes * num_cubes * self.num_indices)
# self.profile_layer = Linear(dmodel, num_planes * num_cubes * 4)

# def constraint(self, x):
# return F.softplus(x, beta=self.beta) + self.eps

# def intensity_distribution(self, intensity_params):
# loc = self.constraint(intensity_params[..., 0])
# scale = self.constraint(intensity_params[..., 1])
# return self.intensity_dist(loc, scale)

# def penalty(self, L):
# diag_elements = L.diagonal(dim1=-2, dim2=-1)
# penalty = torch.sum(diag_elements**2)
# return penalty

# def distance_penalty(self, means):
# batch_size, num_planes, num_components, _ = means.shape
# means = means.view(batch_size * num_planes, num_components, 2)
# dist_matrix = torch.cdist(means, means, p=2)
# inf_mask = (
# torch.eye(num_components, device=means.device)
# .unsqueeze(0)
# .expand(batch_size * num_planes, -1, -1)
# )
# dist_matrix = dist_matrix + inf_mask * 1e6
# too_close = torch.exp(-dist_matrix)
# too_far = torch.exp(dist_matrix - 10)
# penalty = torch.sum(too_close + too_far)
# return penalty

# def background(self, bgparams):
# mu = self.constraint(bgparams[..., 0])
# sigma = self.constraint(bgparams[..., 1])
# return self.background_dist(mu, sigma)

# def map_profiles(self, batch_size, sampled_indices, profile_values):
# flattened_tensor = torch.zeros(
# (batch_size, self.grid_size * self.grid_size * self.num_planes)
# )

# for b in range(batch_size):
# for p in range(self.num_planes):
# for c in range(self.num_cubes):
# idx = sampled_indices[b, p, c].item()
# max_index = (self.grid_size * self.grid_size) - 22

# if idx <= max_index:
# top_left = idx
# top_right = idx + 1
# bottom_left = idx + self.grid_size
# bottom_right = idx + self.grid_size + 1

# if (
# top_left < self.grid_size * self.grid_size
# and top_right < self.grid_size * self.grid_size
# and bottom_left < self.grid_size * self.grid_size
# and bottom_right < self.grid_size * self.grid_size
# ):
# if (
# flattened_tensor[
# b, top_left + p * self.grid_size * self.grid_size
# ]
# == 0
# ):
# flattened_tensor[
# b, top_left + p * self.grid_size * self.grid_size
# ] = profile_values[b, p, c, 0]
# else:
# flattened_tensor[
# b, top_left + p * self.grid_size * self.grid_size
# ] *= profile_values[b, p, c, 0]

# if (
# flattened_tensor[
# b, top_right + p * self.grid_size * self.grid_size
# ]
# == 0
# ):
# flattened_tensor[
# b, top_right + p * self.grid_size * self.grid_size
# ] = profile_values[b, p, c, 1]
# else:
# flattened_tensor[
# b, top_right + p * self.grid_size * self.grid_size
# ] *= profile_values[b, p, c, 1]

# if (
# flattened_tensor[
# b, bottom_left + p * self.grid_size * self.grid_size
# ]
# == 0
# ):
# flattened_tensor[
# b, bottom_left + p * self.grid_size * self.grid_size
# ] = profile_values[b, p, c, 2]
# else:
# flattened_tensor[
# b, bottom_left + p * self.grid_size * self.grid_size
# ] *= profile_values[b, p, c, 2]

# if (
# flattened_tensor[
# b,
# bottom_right + p * self.grid_size * self.grid_size,
# ]
# == 0
# ):
# flattened_tensor[
# b,
# bottom_right + p * self.grid_size * self.grid_size,
# ] = profile_values[b, p, c, 3]
# else:
# flattened_tensor[
# b,
# bottom_right + p * self.grid_size * self.grid_size,
# ] *= profile_values[b, p, c, 3]

# return flattened_tensor

# def forward(self, representation, dxyz, mask, isflat, use_mixture_model=True):
# device = representation.device

# bgparams = self.linear_bg_params(representation)
# intensity_params = self.linear_intensity_params(representation)

# q_bg = self.background(bgparams)
# q_I = self.intensity_distribution(intensity_params)

# batch_size = bgparams.size(0)

# logits = self.index_layer(representation)
# logits = logits.view(
# batch_size, self.num_planes, self.num_cubes, self.num_indices
# )
# probs = F.softmax(logits, dim=-1)

# m = Categorical(probs)
# sampled_indices = m.sample()

# profile_logits = self.profile_layer(representation)
# profile_logits = (
# F.softplus(
# profile_logits.view(batch_size, self.num_planes, self.num_cubes, 4)
# )
# + 1e-3
# )

# profile_values = Dirichlet(profile_logits).sample()

# profile = self.map_profiles(batch_size, sampled_indices, profile_values)

# L = torch.ones([1])
# return q_bg, q_I, profile, L, None


# # %%
# d_model = 64

# batch_size = 2
# num_planes = 3
# intensity_dist = torch.distributions.Normal
# background_dist = torch.distributions.Normal

# disbuilder = Builder(d_model, intensity_dist, background_dist)
# reprsentation = torch.randn(batch_size, 1, d_model)
# dxyz = torch.randn(batch_size, 441 * num_planes, 3)
# mask = torch.ones(batch_size, 441 * num_planes)
# isflat = torch.zeros(batch_size, 441 * num_planes)

# q_bg, q_I, profile, L, penalty = disbuilder(
# reprsentation, dxyz, mask, isflat, use_mixture_model=True
# )

# %%
# if __name__ == "__main__":
# # Parameters
# batch_size = 2
# num_planes = 3
# num_components = 3
# d_model = 64

# # Define distributions
# intensity_dist = torch.distributions.Normal
# background_dist = torch.distributions.Normal

# # Create Builder
# builder = Builder(d_model, intensity_dist, background_dist)

# # Example inputs
# representation = torch.randn(batch_size, 1, d_model)
# dxyz = torch.randn(batch_size, 441 * num_planes, 3)
# mask = torch.ones(batch_size, 441 * num_planes)
# isflat = torch.zeros(batch_size, 441 * num_planes)

# # Forward pass
# q_bg, q_I, profile, L, penalty = builder(
# representation, dxyz, mask, isflat, use_mixture_model=True
# )

# print(f"Profile shape: {profile.shape}")
# print(profile)
# # %%
# import torch
# import torch.nn.functional as F
# from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical
# from integrator.layers import Linear
# from rs_distributions.transforms import FillScaleTriL
# from torch.distributions.transforms import SoftplusTransform

# # Parameters
# batch_size = 2
# d_model = 64
# num_planes = 3
# num_components = 4  # Including the anchored component

# # Generate random input data
# representation = torch.randn(batch_size, 1, d_model)
# dxyz = torch.randn(batch_size, 441 * num_planes, 3)

# # Initialize linear layers

# mixture_weight_layer = Linear(d_model, num_components)  # one weight per component

# mean_layer = Linear(
# d_model, (num_components - 1) * 3
# )  # three params per component (minus anchored component)

# scale_layer = Linear(
# d_model, num_components * 6
# )  # six params per component for Lower Tril


# # Compute mixture weights, means, and scales

# mixture_weights = mixture_weight_layer(representation)

# mixture_weights = F.softmax(mixture_weights, dim=-1)

# means = mean_layer(representation)

# zero_means = torch.zeros((batch_size, 3), device=representation.device)

# means.shape, zero_means.shape

# means = means.view(batch_size, num_components - 1, 3)

# means = torch.cat([zero_means.unsqueeze(1), means], dim=1)

# scales = scale_layer(representation).view(batch_size, num_components, 6)


# L_transform = FillScaleTriL(diag_transform=SoftplusTransform())
# L = L_transform(scales)


# L = L_transform(scales.view(batch_size * num_planes * num_components, 6)).view(
# batch_size, num_planes, num_components, 3, 3
# )

# # Create multivariate normal distributions
# mvn = MultivariateNormal(means, scale_tril=L)
# mix = Categorical(mixture_weights)
# gmm = MixtureSameFamily(mixture_distribution=mix, component_distribution=mvn)

# # Prepare dxyz
# dxyz = dxyz.view(batch_size, num_planes, 441, 3)


# mvn.sample([1])

# # Calculate log probabilities and profile
# log_probs = gmm.log_prob(dxyz.view(batch_size * num_planes * num_components, 441, 3))
# profile = torch.exp(log_probs).view(batch_size, num_planes * 441)

# print("Profile shape:", profile.shape)
# print("L shape:", L.shape)
# print("Means shape:", means.shape)
# print("Mixture weights shape:", mixture_weights.shape)


# torch.distributions.dirichlet.Dirichlet(torch.tensor([ 0.5,0.5,0.5,0.8])).sample([4])

# logits =torch.randn(10,4)

# (torch.sigmoid(logits) * grid_size - small_grid).to(int)


# grid_size = 21*21
# small_grid = 2*2


# # %%
# import torch
# import torch.nn astorch.nn
# import torch.nn.functional as F

# class MapToTargetRange(nn.Module):
# def __init__(self, target_min=1, target_max=10):
# super(MapToTargetRange, self).__init__()
# self.target_min = target_min
# self.target_max = target_max

# def forward(self, x):
# x02 = torch.tanh(x) + 1  # x in range(0,2)
# scale = (self.target_max - self.target_min) / 2.0
# return x02 * scale + self.target_min

# # Example usage in a simple model
# class SimpleModel(nn.Module):
# def __init__(self, input_dim, target_min=1, target_max=10):
# super(SimpleModel, self).__init__()
# self.fc =torch.nn.Linear(input_dim, 4)
# self.activation = MapToTargetRange(target_min, target_max)

# def forward(self, x):
# x = self.fc(x)
# x = self.activation(x)
# return x

# # Create a model instance
# input_dim = 1000
# model = SimpleModel(input_dim=input_dim, target_min=0, target_max=20)

# # Testing
# a = torch.randn(10, input_dim)
# b = model(a)
# print(b.min().item(), b.max().item())


# # %%
# import torch
# import torch.nn astorch.nn
# import torch.nn.functional as F

# import torch
# import torch.nn astorch.nn
# import torch.nn.functional as F

# class MultiPlaneGridIndexModel(nn.Module):
# def __init__(self, input_dim, num_planes, num_cubes, grid_size):
# super(MultiPlaneGridIndexModel, self).__init__()
# self.num_planes = num_planes
# self.num_cubes = num_cubes
# self.grid_size = grid_size
# self.max_index = grid_size * grid_size - 22  # Valid range for top-left index of 2x2 grid
# self.num_indices = self.max_index + 1

# # Define the layers
# self.fc =torch.nn.Linear(input_dim, num_planes * num_cubes * self.num_indices)

# def forward(self, x):
# logits = self.fc(x)
# logits = logits.view(-1, self.num_planes, self.num_cubes, self.num_indices)
# probs = F.softmax(logits, dim=-1)
# return probs

# # Example usage
# input_dim = 1000
# num_planes = 3
# num_cubes = 20
# grid_size = 21

# # Create a model instance
# model = MultiPlaneGridIndexModel(input_dim, num_planes, num_cubes, grid_size)

# # Example input
# x = torch.randn(2, input_dim)

# # Get the probabilities for each index
# probs = model(x)

# # Sample indices from the categorical distribution
# m = torch.distributions.Categorical(probs)
# sampled_indices = m.sample()

# # Get the most likely indices
# max_indices = torch.argmax(probs, dim=-1)

# print("Sampled indices:", sampled_indices)
# print("Max indices:", max_indices)

# sampled_indices.shape

# lin = Linear(d_model,20*4*3)
# logits = torch.nn.functional.softplus(lin(representation).view(2,3,20,4))


# profile = torch.distributions.dirichlet.Dirichlet(logits).sample()

# profile.shape
