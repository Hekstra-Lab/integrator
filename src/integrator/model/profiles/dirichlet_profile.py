import torch
import torch.nn.functional as F
from integrator.layers import Linear


# class DirichletProfile(torch.nn.Module):
# """
# Dirichlet profile model
# """

# def __init__(self, dmodel, rank=None, mc_samples=100, num_components=3 * 21 * 21):
# super().__init__()
# self.dmodel = dmodel
# self.mc_samples = mc_samples
# self.num_components = num_components
# self.alpha_layer = Linear(self.dmodel, self.num_components)
# self.rank = rank
# self.eps = 1e-6

# def forward(self, representation):
# alphas = self.alpha_layer(representation)
# alphas = F.softplus(alphas) + self.eps
# q_p = torch.distributions.Dirichlet(alphas)

# # profile = q_p.rsample([self.mc_samples])

# # return profile, q_p
# return q_p

import torch.nn as nn


class DirichletProfile(nn.Module):
    def __init__(self, dmodel, rank=None, mc_samples=100, num_components=3 * 21 * 21):
        super().__init__()
        self.dmodel = dmodel
        self.mc_samples = mc_samples

        # Spatial dimensions
        self.depth = 3
        self.height = 21
        self.width = 21

        # Project to a feature vector
        self.projection = nn.Linear(dmodel, 256)

        # Create initial small spatial representation
        # We'll start with a 6×6 feature map that will upsample to 21×21
        self.initial_spatial = nn.Sequential(
            nn.Linear(256, 16 * 6 * 6),  # 16 channels with 6×6 spatial dimensions
            nn.ReLU(),
        )

        # Careful calculation for transposed convolution:
        # For ConvTranspose2d with kernel_size=4, stride=3, padding=1, output_padding=0:
        # output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
        # For input 6×6: (6-1)*3 - 2*1 + 4 + 0 = 15 + 2 = 17
        # For input 7×7: (7-1)*3 - 2*1 + 4 + 0 = 18 + 2 = 20

        # First upsampling from 6×6 to 18×18
        # Then second upsampling from 18×18 to exactly 21×21
        self.upsampling = nn.Sequential(
            # First transposed convolution: 6×6 → ~18×18
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=3, padding=1),
            nn.ReLU(),
            # Second transposed convolution: 18×18 → 21×21
            # For stride=1, padding=0, and output_padding=0:
            # output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
            # = (18-1)*1 - 2*0 + 4 + 0 = 17 + 4 = 21
            nn.ConvTranspose2d(8, self.depth, kernel_size=4, stride=1, padding=0),
        )

        # Spatial bias
        self.spatial_bias = nn.Parameter(
            torch.ones(1, self.depth, self.height, self.width) * 1.0
        )

        # Initialize weights
        self._init_weights()

        self.eps = 1e-6

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, representation):
        batch_size = representation.shape[0]

        # Project to features
        x = self.projection(representation)
        x = F.relu(x)

        # Create initial spatial representation
        x = self.initial_spatial(x)
        x = x.view(batch_size, 16, 6, 6)  # Reshape to [batch, 16, 6, 6]

        # Apply transposed convolutions for upsampling
        x = self.upsampling(x)

        # Verify the output shape is exactly what we want
        if x.shape[2:] != (self.height, self.width):
            # If there's still a mismatch, we can use interpolation as a fallback
            x = F.interpolate(
                x, size=(self.height, self.width), mode="bilinear", align_corners=False
            )

        # Add bias and ensure positivity
        alphas_spatial = x + self.spatial_bias
        alphas_spatial = F.softplus(alphas_spatial) + self.eps

        # Reshape to [batch, 3*21*21] for Dirichlet
        alphas_flat = alphas_spatial.reshape(batch_size, -1)

        return torch.distributions.Dirichlet(alphas_flat)


# class Spatial25DDirichletProfile(nn.Module):
# def __init__(self, dmodel, rank=None, mc_samples=100, num_components=3 * 21 * 21):
# super().__init__()
# self.dmodel = dmodel
# self.mc_samples = mc_samples

# # Spatial dimensions
# self.depth = 3
# self.height = 21
# self.width = 21

# # Project to a feature vector that will be shared across depth slices
# self.projection = nn.Linear(dmodel, 256)

# # Create features for each depth slice
# self.depth_features = nn.Linear(256, self.depth * 64)

# # 2D upsampling for each depth slice
# self.spatial_decoder = nn.Sequential(
# nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
# nn.ReLU(),
# nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
# nn.ReLU(),
# nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)
# )

# # Spatial bias
# self.spatial_bias = nn.Parameter(torch.ones(1, self.depth, self.height, self.width) * 1.0)

# self.eps = 1e-6

# def forward(self, representation):
# batch_size = representation.shape[0]

# # Project to features
# x = self.projection(representation)
# x = F.relu(x)

# # Create depth-specific features
# depth_x = self.depth_features(x)
# depth_x = depth_x.view(batch_size, self.depth, 64)

# # Process each depth slice separately
# alphas_spatial = []
# for d in range(self.depth):
# # Get features for this depth
# slice_features = depth_x[:, d, :]  # [batch, 64]

# # Reshape for 2D processing
# slice_features = slice_features.view(batch_size, 64, 1, 1)

# # Apply 2D upsampling
# slice_output = self.spatial_decoder(slice_features)  # [batch, 1, H, W]

# # Make sure output is exactly the right size
# if slice_output.shape[2:] != (self.height, self.width):
# slice_output = F.interpolate(
# slice_output, size=(self.height, self.width),
# mode='bilinear', align_corners=False
# )

# alphas_spatial.append(slice_output)

# # Combine all depth slices
# alphas_spatial = torch.cat(alphas_spatial, dim=1)  # [batch, depth, height, width]

# # Add bias and ensure positivity
# alphas_spatial = alphas_spatial + self.spatial_bias
# alphas_spatial = F.softplus(alphas_spatial) + self.eps

# # Reshape to [batch, 3*21*21] for Dirichlet
# alphas_flat = alphas_spatial.reshape(batch_size, -1)

# return torch.distributions.Dirichlet(alphas_flat)
