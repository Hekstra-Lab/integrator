import torch
import torch.nn.functional as F
import torch.nn as nn


class BetaProfile(nn.Module):
    """
    Beta profile model (independent Beta for each p_ij in (0,1))
    """

    def __init__(self, dmodel, rank=None, mc_samples=100, num_components=3 * 21 * 21):
        super().__init__()
        self.dmodel = dmodel
        self.mc_samples = mc_samples
        self.num_components = num_components
        # Output 2 parameters (alpha, beta) per component
        self.ab_layer = nn.Linear(self.dmodel, 2 * self.num_components)
        self.rank = rank
        self.eps = 1e-6

    def forward(self, representation):
        # Output shape: [..., 2 * num_components]
        ab = self.ab_layer(representation)
        # Split into alpha (concentration1) and beta (concentration0)
        alpha, beta = torch.chunk(ab, 2, dim=-1)
        # Ensure positivity with softplus
        alpha = F.softplus(alpha) + self.eps
        beta = F.softplus(beta) + self.eps
        # Independent Beta distributions for each p_ij
        q_p = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
        return q_p


# class BetaProfile(nn.Module):
# """
# Beta profile model with spatial structure awareness
# """

# def __init__(self, dmodel, rank=None, mc_samples=100, num_components=3 * 21 * 21):
# super().__init__()
# self.dmodel = dmodel
# self.mc_samples = mc_samples

# # Explicitly store the spatial dimensions
# self.depth = 3
# self.height = 21
# self.width = 21

# # Create a feature extractor network
# self.feature_extractor = nn.Sequential(
# nn.Linear(dmodel, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
# )

# # Create a spatial decoder for alpha parameters
# self.alpha_decoder = nn.Sequential(
# nn.ConvTranspose2d(
# 128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
# ),
# nn.ReLU(),
# nn.ConvTranspose2d(
# 64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
# ),
# nn.ReLU(),
# nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
# )

# # Create a spatial decoder for beta parameters
# self.beta_decoder = nn.Sequential(
# nn.ConvTranspose2d(
# 128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
# ),
# nn.ReLU(),
# nn.ConvTranspose2d(
# 64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
# ),
# nn.ReLU(),
# nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
# )

# # Final layer for alpha parameters
# self.alpha_layer = nn.Conv2d(16, self.depth, kernel_size=3, stride=1, padding=1)

# # Final layer for beta parameters
# self.beta_layer = nn.Conv2d(16, self.depth, kernel_size=3, stride=1, padding=1)

# # Spatial biases
# self.alpha_bias = nn.Parameter(
# torch.ones(1, self.depth, self.height, self.width) * 0.5
# )
# self.beta_bias = nn.Parameter(
# torch.ones(1, self.depth, self.height, self.width) * 0.5
# )

# self.eps = 1e-6

# def forward(self, representation):
# """
# Generate a Beta profile with spatial structure from the input representation.

# Args:
# representation: Tensor of shape [..., dmodel]

# Returns:
# Beta distribution object
# """
# batch_size = representation.shape[0]

# # Extract features
# features = self.feature_extractor(representation)

# # Reshape for spatial decoders
# features = features.view(batch_size, 128, 1, 1)

# # Generate alpha and beta parameters separately
# alpha_decoded = self.alpha_decoder(features)
# beta_decoded = self.beta_decoder(features)

# # Interpolate to target size
# alpha_decoded = F.interpolate(
# alpha_decoded,
# size=(self.height, self.width),
# mode="bilinear",
# align_corners=False,
# )
# beta_decoded = F.interpolate(
# beta_decoded,
# size=(self.height, self.width),
# mode="bilinear",
# align_corners=False,
# )

# # Apply final layers
# alpha_spatial = self.alpha_layer(alpha_decoded)
# beta_spatial = self.beta_layer(beta_decoded)

# # Add learned spatial biases
# alpha_spatial = alpha_spatial + self.alpha_bias
# beta_spatial = beta_spatial + self.beta_bias

# # Ensure positivity with softplus and add small epsilon
# alpha_spatial = F.softplus(alpha_spatial) + self.eps
# beta_spatial = F.softplus(beta_spatial) + self.eps

# # Reshape to [batch, 3*21*21] for Independent Beta
# alpha_flat = alpha_spatial.reshape(batch_size, -1)
# beta_flat = beta_spatial.reshape(batch_size, -1)

# # Create and return Beta distribution
# return torch.distributions.Beta(
# concentration1=alpha_flat, concentration0=beta_flat
# )


if __name__ == "__main__":
    data = torch.randn(10, 64)
