import torch
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
        self.eps = 1e-6

    def forward(self, representation):
        alphas = self.alpha_layer(representation)
        alphas = F.softplus(alphas) + self.eps
        q_p = torch.distributions.Dirichlet(alphas)

        return q_p
