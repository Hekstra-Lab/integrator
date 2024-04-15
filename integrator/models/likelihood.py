from pylab import *
import torch
from integrator.layers import Linear, ResidualLayer
from integrator.models import MLP


# %%
class PoissonLikelihood(torch.nn.Module):
    """
    Attributes:
        beta:
        prior_std: std parameter for prior LogNormal
        prior_mean: mean parameter for prior LogNormal
        lognorm_scale: scale DKL(LogNorm||LogNorm)
        prior_bern_p: parameter for prior Bernoulli distribution
        scale_bern: scale DKL(pij||bern)
        priorLogNorm: prior LogNormal distribution
        priorBern: prior Bernoulli distribution
    """

    def __init__(
        self,
        beta=1.0,
        eps=1e-8,
        prior_bern_p=0.2,
        prior_mean=3,  # Prior mean for LogNorm
        prior_std=1,  # Prior std for LogNorm
        lognorm_scale=0.01,  # influence of DKL(LogNorm||LogNorm) term
        scale_bern=1,  # influence of DKL(bern||bern) term
    ):
        super().__init__()
        self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
        self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)
        self.prior_std = torch.nn.Parameter(
            data=torch.tensor(prior_std), requires_grad=False
        )
        self.prior_mean = torch.nn.Parameter(
            data=torch.tensor(prior_mean), requires_grad=False
        )
        self.lognorm_scale = torch.nn.Parameter(
            data=torch.tensor(lognorm_scale), requires_grad=False
        )
        self.prior_bern_p = prior_bern_p
        self.priorLogNorm = torch.distributions.LogNormal(prior_mean, prior_std)
        self.scale_bern = torch.nn.Parameter(
            data=torch.tensor(scale_bern), requires_grad=False
        )
        self.priorBern = torch.distributions.bernoulli.Bernoulli(
            prior_bern_p
        )  # Prior Bern(p) of pixel belonging to a refl

    def constraint(self, x):
        return x + self.eps

    def forward(
        self,
        counts,
        pijrep,
        bg,
        q,
        emp_bg,
        kl_lognorm_scale,
        eps=1e-8,
        bg_penalty_scaling=None,
        kl_bern_scale=None,
        mc_samples=100,
        vi=True,
        mask=None,
    ):
        """
        Args:
            counts: photon counts from original data
            pijrep: pixel representation
            bg: background
            q: distribution
            emp_bg: empirical mean background
            bg_penalty_scaling:
            profile_scale: scale
            mc_samples: number of monte carlo samples
            vi: use variational inference
            mask: mask for padded data

        Returns:
        """

        # Take sample from LogNormal
        z = q.rsample([mc_samples])

        # Set KL term
        kl_term = 0

        # rate = (z * pijrep * norm_factor.unsqueeze(-1)) + bg
        rate = (z * (pijrep)) + bg
        rate = rate + eps

        # counts ~ Pois(rate) = Pois(z * p + bg)
        counts = torch.clamp(counts, min=0)

        # Empirical background
        bg_penalty = 0
        # bg_penalty = (bg - emp_bg) ** 2
        # bg_penalty = bg_penalty * mask
        # bg_penalty = bg_penalty.mean() * bg_penalty_scaling

        if mask is not None:
            ll = torch.distributions.Poisson(rate).log_prob(counts.to(torch.int32))
            ll = ll * mask

        else:
            ll = torch.distributions.Poisson(rate).log_prob(counts.to(torch.int32))

        # Calculate KL-divergence
        if vi:
            # KL(lognorm)
            q_log_prob = q.log_prob(z)
            p_log_prob = self.priorLogNorm.log_prob(z)
            kl_lognorm = q_log_prob - p_log_prob

            # KL(Bernoulli)
            bern = torch.distributions.bernoulli.Bernoulli(pijrep)
            kl_bern = torch.distributions.kl.kl_divergence(bern, self.priorBern).mean()

            # zero out pads
            masked_kl_lognorm = kl_lognorm * mask
            # masked_kl_bern = kl_bern * mask

            # total kl
            kl_term = (
                kl_lognorm_scale
                * masked_kl_lognorm.mean()
                # + kl_bern_scale * masked_kl_bern.mean()
            )

            # kl_term = masked_kl_bern.mean()

        else:
            kl_term = 0  # set to 0 when vi false

        return ll, kl_term, bg_penalty


#############
# class PoissonLikelihood(torch.nn.Module):
# def __init__(
# self,
# beta=1.0,
# eps=1e-8,
# prior_bern_p=0.2,  # Prior p for prior Bern(p)
# prior_mean=2,  # Prior mean for LogNorm
# prior_std=1,  # Prior std for LogNorm
# lognorm_scale=0.01,  # influence of DKL(LogNorm||LogNorm) term
# scale_bern=1,  # influence of DKL(bern||bern) term
# ):
# super().__init__()
# self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
# self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)
# self.prior_std = torch.nn.Parameter(
# data=torch.tensor(prior_std), requires_grad=False
# )
# self.prior_mean = torch.nn.Parameter(
# data=torch.tensor(prior_mean), requires_grad=False
# )
# self.lognorm_scale = torch.nn.Parameter(
# data=torch.tensor(lognorm_scale), requires_grad=False
# )
# self.prior_bern_p = prior_bern_p
# self.priorLogNorm = torch.distributions.LogNormal(prior_mean, prior_std)
# self.scale_bern = torch.nn.Parameter(
# data=torch.tensor(scale_bern), requires_grad=False
# )
# self.priorBern = torch.distributions.bernoulli.Bernoulli(
# prior_bern_p
# )  # Prior Bern(p) of pixel belonging to a refl

# def constraint(self, x):
# # return torch.nn.functional.softplus(x, beta=self.beta) + self.eps
# return x + self.eps

# def forward(self, norm_factor, counts, pijrep, bg, q, mc_samples=100, vi=True):
# # Take sample from LogNormal
# z = q.rsample([mc_samples])

# # Set KL term
# kl_term = 0
# # p = pijrep.permute(2, 0, 1)

# # calculate lambda
# # rate = z * p * norm_factor + bg[None, ...]
# rate = z * pijrep * norm_factor + bg
# # rate = rate.permute(1, 2, 0)
# # rate = z * profile[None,...] + bg[None,...]
# # rate = self.constraint(rate)

# # counts ~ Pois(rate) = Pois(z * p + bg)
# ll = torch.distributions.Poisson(rate).log_prob(counts)

# # Expected log likelihood
# ll = ll.mean(0)

# # Calculate KL-divergence
# if vi:
# q_log_prob = q.log_prob(z)
# p_log_prob = self.priorLogNorm.log_prob(z)
# bern = torch.distributions.bernoulli.Bernoulli(pijrep)
# kl_bern = torch.distributions.kl.kl_divergence(bern, self.priorBern).mean()
# kl_term = (
# self.lognorm_scale * (q_log_prob - p_log_prob).mean()
# + self.scale_bern * kl_bern
# )

# else:
# kl_term = 0  # set to 0 when vi false

# return ll, kl_term


# %%
class PoissonLikelihoodV2(torch.nn.Module):
    """
    Attributes:
        beta:
        prior_std: std parameter for prior LogNormal
        prior_mean: mean parameter for prior LogNormal
        lognorm_scale: scale DKL(LogNorm||LogNorm)
        prior_bern_p: parameter for prior Bernoulli distribution
        priorLogNorm: prior LogNormal distribution
        priorBern: prior Bernoulli distribution
    """

    def __init__(
        self,
        beta=1.0,
        eps=1e-8,
        prior_bern_p=0.2,
        prior_mean=3,  # Prior mean for LogNorm
        prior_std=1,  # Prior std for LogNorm
        lognorm_scale=0.01,  # influence of DKL(LogNorm||LogNorm) term
        scale_bern=1,  # influence of DKL(bern||bern) term
    ):
        super().__init__()
        self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
        self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)
        self.prior_std = torch.nn.Parameter(
            data=torch.tensor(prior_std), requires_grad=False
        )
        self.prior_mean = torch.nn.Parameter(
            data=torch.tensor(prior_mean), requires_grad=False
        )
        self.lognorm_scale = torch.nn.Parameter(
            data=torch.tensor(lognorm_scale), requires_grad=False
        )
        self.priorLogNorm = torch.distributions.LogNormal(prior_mean, prior_std)

    def constraint(self, x):
        return x + self.eps

    def forward(
        self,
        counts,
        q_bg,
        q_I,
        profile,
        eps=1e-8,
        mc_samples=10,
        vi=True,
        mask=None,
    ):
        """
        Args:
            counts: observed photon counts
            q_bg: variational background distribution
            q_I: variational intensity distribution
            profile: MVN profile model
            mc_samples: number of monte carlo samples
            vi: use KL-term
            mask: mask for padded entries

        Returns: log-likelihood and KL(q|p)
        """

        # Sample from variational distributions
        z = q_I.rsample([mc_samples])
        bg = q_bg.rsample([mc_samples])

        # Set KL term
        kl_term = 0

        # Calculate the rate
        rate = (z * (profile)) + bg
        rate = rate + eps

        counts = torch.clamp(counts, min=0)  # do not clamp, use a mask instead

        if mask is not None:
            # Mask out padded terms
            ll = torch.distributions.Poisson(rate).log_prob(counts.to(torch.int32))
            ll = ll * mask

        else:
            ll = torch.distributions.Poisson(rate).log_prob(counts.to(torch.int32))

        # Calculate KL-divergence
        if vi:
            # KL(lognorm)
            q_log_prob = q_I.log_prob(z)
            p_log_prob = self.priorLogNorm.log_prob(z)
            kl_lognorm = q_log_prob - p_log_prob

            # zero out pads
            masked_kl_lognorm = kl_lognorm * mask

            # total kl
            kl_term = masked_kl_lognorm.mean()

        else:
            kl_term = 0  # set to 0 when vi false

        return ll, kl_term
