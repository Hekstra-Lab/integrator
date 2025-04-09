import torch
import torch.nn as nn
import torch.nn.functional as F
from integrator.layers import Linear


class DirichletProfile(torch.nn.Module):
    """
    Dirichlet profile model
    """

    def __init__(self, dmodel, rank=None, mc_samples=100, num_components=3 * 21 * 21):
        super().__init__()
        self.dmodel = dmodel
        self.mc_samples = mc_samples
        self.num_components = num_components
        self.alpha_layer = Linear(self.dmodel, self.num_components)
        self.rank = rank
        self.max_value = 5000.0
        self.eps = 1e-6

    def smooth_bound(self, x, max_val):
        return max_val * torch.sigmoid(x)

    def forward(self, representation):
        alphas = self.alpha_layer(representation)
        alphas = self.smooth_bound(alphas, self.max_value)
        alphas = F.softplus(alphas) + self.eps
        q_p = torch.distributions.Dirichlet(alphas)

        return q_p


class UnetDirichletProfile(torch.nn.Module):
    """
    Dirichlet profile model
    """

    def __init__(
        self, dmodel=64, rank=None, mc_samples=100, num_components=3 * 21 * 21
    ):
        super().__init__()
        self.dmodel = dmodel
        self.mc_samples = mc_samples
        self.num_components = num_components
        self.rank = rank
        self.eps = 1e-6
        self.max_value = 500.0

    def smooth_bound(self, x, max_val):
        return max_val * torch.sigmoid(x)

    # def forward(self, representation):
    def forward(self, alphas):
        alphas = F.softplus(alphas) + self.eps
        q_p = torch.distributions.Dirichlet(alphas)

        return q_p


# class UnetDirichletProfile(nn.Module):
# def __init__(self, input_dim=1323, output_dim=1323):
# super().__init__()
# # Projection while preserving dimensionality
# self.projection = nn.Sequential(
# nn.Linear(input_dim, input_dim),
# nn.LayerNorm(input_dim),  # Stabilizes training
# nn.GELU(),  # Smoother activation than ReLU
# nn.Linear(input_dim, input_dim),
# )
# # Skip connection if dimensions match
# self.skip_proj = None
# if input_dim != output_dim:
# self.skip_proj = nn.Linear(input_dim, output_dim)

# self.eps = 1e-6

# def forward(self, x):
# # Main transformation
# trans = self.projection(x)

# # Skip connection
# if self.skip_proj is not None:
# skip = self.skip_proj(x)
# else:
# skip = x

# # Combine with skip connection
# combined = trans + skip

# # Ensure positive values for Dirichlet
# alphas = F.softplus(combined) + self.eps
# q_p = torch.distributions.Dirichlet(alphas)

# return q_p


if __name__ == "__main__":
    # Example usage
    dmodel = 64
