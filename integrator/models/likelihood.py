from pylab import *
import torch
from integrator.layers import Linear, ResidualLayer
from integrator.models import MLP


class PoissonLikelihood(torch.nn.Module):
    def __init__(self, beta=1.0, eps=1e-8, prior_bern=0.2, prior_mean=2, prior_std=1):
        super().__init__()
        self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
        self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)
        self.priorBern = torch.distributions.bernoulli.Bernoulli(
            prior_bern
        )  # Prior Bern(p) of pixel belonging to a refl
        self.priorLogNorm = torch.distributions.LogNormal(prior_mean, prior_std)

    def constraint(self, x):
        # return torch.nn.functional.softplus(x, beta=self.beta) + self.eps
        return x + self.eps

    def forward(self, norm_factor,counts, p, bg, q, mc_samples=100, vi=True):
        # Take sample from LogNormal
        z = q.rsample([mc_samples])
        kl_term = 0  # no kl divergence
        p = p.permute(2, 0, 1)

        # calculate lambda
        rate = z * (p*norm_factor[...,None]) + bg[None, ...]
        # rate = z * profile[None,...] + bg[None,...]
        # rate = self.constraint(rate)

        # counts ~ Pois(rate) = Pois(z * p + bg)
        ll = torch.distributions.Poisson(rate).log_prob(counts)

        # Expected log likelihood
        ll = ll.mean(0)

        if vi:
            q_log_prob = q.log_prob(z)
            p_log_prob = self.priorLogNorm.log_prob(z)
            bern = torch.distributions.bernoulli.Bernoulli(p)
            kl_bern = torch.distributions.kl.kl_divergence(bern, self.priorBern).mean()
            kl_term = .01*(q_log_prob - p_log_prob).mean() + kl_bern

        else:
            kl_term = 0  # set to 0 when vi false

        return ll, kl_term
