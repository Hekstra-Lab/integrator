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


if __name__ == "__main__":
    # Example usage
    dmodel = 64
