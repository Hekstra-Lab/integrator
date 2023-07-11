from pylab import *
import torch
from integrator.layers import Linear,ResidualLayer
from integrator.models import MLP


class PoissonLikelihood(torch.nn.Module):
    def __init__(self, beta=1., eps=1e-8,prior_mean=5,prior_std=1):
        super().__init__()
        self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)
        self.prior = torch.distributions.LogNormal(prior_mean,prior_std)

    def constraint(self, x):
        #return torch.nn.functional.softplus(x, beta=self.beta) + self.eps
        return x + self.eps

    def forward(self, counts, profile, bg, q, mc_samples=100,vi=True):
        z = q.rsample([mc_samples])
        rate = z * profile[None,...] + bg[None,...]
        kl_term = 0
        #rate = self.constraint(rate)
        ll = torch.distributions.Poisson(rate).log_prob(counts)
        ll = ll.mean(0)

        if vi:
            q_log_prob = q.log_prob(z)
            p_log_prob = self.prior.log_prob(z)
            kl_term = (q_log_prob - p_log_prob).mean()
        else:
            kl_term = 0

        return ll,kl_term


