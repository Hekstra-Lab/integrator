import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.transforms import SoftplusTransform
from integrator.layers import Linear
from rs_distributions.transforms import FillScaleTriL


class MVNProfile(torch.nn.Module):
    def __init__(self, dmodel, image_shape):
        super().__init__()
        self.dmodel = dmodel

        # Use different transformation for more flexible scale learning
        self.L_transform = FillScaleTriL(diag_transform=SoftplusTransform())

        # Create scale and mean prediction layers
        self.scale_layer = Linear(self.dmodel, 6, bias=True)

        # Initialize scale_layer to output an isotropic Gaussian by default
        with torch.no_grad():
            # Invert ELU+1: bias = value - 1 for value > 0 (inverse of elu(x) + 1)
            init_scale = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
            init_scale_raw = (
                init_scale - 1.0
            )  # elu(x) + 1 = s  => x = s - 1 (since s > 0)
            self.scale_layer.bias.copy_(init_scale_raw)
            torch.nn.init.zeros_(
                self.scale_layer.weight
            )  # prevents representation from influencing scale at init

        self.mean_layer = Linear(self.dmodel, 3)

        # Calculate image dimensions
        d, h, w = image_shape
        self.image_shape = image_shape

        # Create centered coordinate grid
        z_coords = torch.arange(d).float() - (d - 1) / 2
        y_coords = torch.arange(h).float() - (h - 1) / 2
        x_coords = torch.arange(w).float() - (w - 1) / 2

        z_coords = z_coords.view(d, 1, 1).expand(d, h, w)
        y_coords = y_coords.view(1, h, 1).expand(d, h, w)
        x_coords = x_coords.view(1, 1, w).expand(d, h, w)

        # Stack coordinates
        pixel_positions = torch.stack([x_coords, y_coords, z_coords], dim=-1)
        pixel_positions = pixel_positions.view(-1, 3)

        # Register buffer
        self.register_buffer("pixel_positions", pixel_positions)

        # Create a default scale parameter for initialization guidance
        self.register_buffer(
            "scale_init", torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]).view(1, 1, 6)
        )

    def forward(self, representation):
        batch_size = representation.size(0)

        # Predict mean offsets
        means = self.mean_layer(representation).view(batch_size, 1, 3)

        # Predict scale parameters - use ELU+1 for positive values with better gradient properties
        scales_raw = self.scale_layer(representation).view(batch_size, 1, 6)

        # Instead of sigmoid, use ELU+1 which has a better gradient flow and unbounded upper range
        # This allows the model to learn scales more freely
        scales = torch.nn.functional.elu(scales_raw) + 1.0

        # Transform scales
        L = self.L_transform(scales).to(torch.float32)

        # Create MVN distribution
        mvn = MultivariateNormal(means, scale_tril=L)

        # Compute log probabilities
        pixel_positions = self.pixel_positions.unsqueeze(0).expand(batch_size, -1, -1)
        log_probs = mvn.log_prob(pixel_positions)

        # Convert to probabilities
        # Subtract max for numerical stability (prevents overflow)
        log_probs_stable = log_probs - log_probs.max(dim=1, keepdim=True)[0]
        profile = torch.exp(log_probs_stable)

        # Normalize to sum to 1
        profile = profile / (profile.sum(dim=1, keepdim=True) + 1e-10)

        return profile


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    profile_model = MVNProfile(dmodel=64, image_shape=(3, 21, 21))

    # Create a batch of representations (assuming 10 sample with 64-dimensional representation)
    representation = torch.randn(10, 64)

    profile = profile_model(representation)
    # The output should have shape [1, 3*21*21] = [1, 1323]

    expanded = profile.unsqueeze_(1).expand(-1, 100, -1)

    plt.imshow(profile[0].detach().reshape(3, 21, 21)[1])
    plt.show()
