from pylab import *
import torch
from integrator.layers import Linear
from rs_distributions.transforms import FillScaleTriL
from torch.distributions.transforms import ExpTransform, SoftplusTransform
import rs_distributions.distributions as rsd
import torch.distributions.constraints as constraints
import torch.nn as nn
import torch.nn.functional as F


# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(
            21
        )  # Adjust the size to match the desired output
        self.fc = Linear(128 * 21, 3 * 21 * 21)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 1, 3, 21, 21)
        return x


class DistributionBuilder(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        intensity_dist,
        background_dist,
        eps=1e-12,
        beta=1.0,
    ):
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        # self.linear = Linear(dmodel, 10)  # Single layer for all params
        self.linear = Linear(dmodel, 7)  # Single layer for all params
        self.intensity_dist = intensity_dist
        self.background_dist = background_dist
        self.L_transform = FillScaleTriL(diag_transform=SoftplusTransform())
        # self.simpleCNN = SimpleCNN()

    def constraint(self, x):
        return torch.nn.functional.softplus(x, beta=self.beta) + self.eps

    def intensity_distribution(self, params):
        loc = self.constraint(params[..., 0])
        scale = self.constraint(params[..., 1])
        return self.intensity_dist(loc, scale)

    def background(self, params):
        mu = self.constraint(params[..., 2])
        sigma = self.constraint(params[..., 3])
        return self.background_dist(mu, sigma)

    def MVNProfile3D(self, L_params, dxyz, device):
        print("all3d")
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

    def MVNProfile2D(self, L_params, dxy, device, batch_size):
        mu = torch.zeros((batch_size, 1, 2), requires_grad=False, device=device).to(
            torch.float32
        )
        L_2d = self.L_transform(L_params[..., :3]).to(
            torch.float32
        )  # Use only the first three params
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
        params = self.linear(representation)
        q_bg = self.background(params)
        q_I = self.intensity_distribution(params)

        L_params = params[..., 4:]  # First three params for 2D case

        batch_size = L_params.size(0)
        L_params = params[..., 4:]  # First three params for 2D case
        dxy = dxyz[..., :2]
        profile, L = self.MVNProfile2D(L_params, dxy, device, batch_size)

        return q_bg, q_I, profile, L

        # test code 2024-07-30

        # dxy = dxyz[..., :2]

        # batch_size

        # image weights
        # output = self.simpleCNN(representation)
        # image_weights = F.softmax(output, dim=2).reshape(batch_size, 3 * 21 * 21)

        # profile, L = self.MVNProfile2D(L_params, dxy, device, batch_size)

        # return q_bg, q_I, profile, L, image_weights

        # if isflat.all():
        # L_params = params[..., 4:7]  # First three params for 2D case
        # dxy = dxyz[..., :2]
        # profile, L = self.MVNProfile2D(L_params, dxy, device, batch_size)
        # return q_bg, q_I, profile, L

        # elif not isflat.any():
        # L_params = params[..., 4:10]  # All six params for 3D case
        # profile, L = self.MVNProfile3D(L_params, dxyz, device)
        # return q_bg, q_I, profile, L

        # else:
        # # rep_2d = representation[isflat]
        # # rep_3d = representation[~isflat]
        # dxyz_3d = dxyz[~isflat]
        # dxy = dxyz[isflat][..., :2]

        # L_params_2d = params[isflat][..., 4:7]
        # L_params_3d = params[~isflat][..., 4:10]

        # prof_2d, L_2 = self.MVNProfile2D(L_params_2d, dxy, device, batch_size)
        # prof_3d, L_3 = self.MVNProfile3D(L_params_3d, dxyz_3d, device)
        # batch_size = representation.size(0)
        # profile = torch.zeros(
        # batch_size, mask.size(1), dtype=prof_3d.dtype, device=device
        # )
        # profile[isflat] = prof_2d
        # profile[~isflat] = prof_3d

        # L = torch.zeros(batch_size, 3, 3, dtype=L_3.dtype, device=device)
        # L[isflat] = L_2.squeeze(1)
        # L[~isflat] = L_3.squeeze(1)

        # return q_bg, q_I, profile, L
