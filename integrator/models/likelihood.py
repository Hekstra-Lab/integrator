from pylab import *
import torch
from integrator.layers import Linear, ResidualLayer
from integrator.models import MLP


class PoissonLikelihood(torch.nn.Module):
    def __init__(self, beta=1.0, eps=1e-8):
        super().__init__()
        self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
        self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)

    def constraint(self, x):
        # return torch.nn.functional.softplus(x, beta=self.beta) + self.eps
        return x + self.eps

    def forward(self, counts, p, bg, q, mc_samples=100):
        z = q.rsample([mc_samples])
        rate = z * p.permute(2, 0, 1) + bg[None, ...]
        # rate = z * profile[None,...] + bg[None,...]
        # rate = self.constraint(rate)
        ll = torch.distributions.Poisson(rate).log_prob(counts)
        ll = ll.mean(0)
        return ll
