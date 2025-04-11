import torch
import torch.nn as nn
import torch.nn.functional as F
from integrator.layers import Linear


class DirichletProfile(torch.nn.Module):
    """
    Dirichlet profile model
    """

    def __init__(self, dmodel=None, num_components=3 * 21 * 21):
        super().__init__()
        if dmodel is not None:
            self.alpha_layer = Linear(dmodel, num_components)
        self.dmodel = dmodel
        self.eps = 1e-6

    def forward(self, alphas):
        if self.dmodel is not None:
            alphas = self.alpha_layer(alphas)
        alphas = F.softplus(alphas) + self.eps
        q_p = torch.distributions.Dirichlet(alphas)

        return q_p


class UnetDirichletProfile(torch.nn.Module):
    """
    Dirichlet profile model
    """

    def __init__(self, dmodel=None, num_components=3 * 21 * 21):
        super().__init__()
        if dmodel is not None:
            self.alpha_layer = Linear(dmodel, num_components)
        self.dmodel = dmodel
        self.eps = 1e-6

    def forward(self, alphas):
        if self.dmodel is not None:
            alphas = self.alpha_layer(alphas)
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
