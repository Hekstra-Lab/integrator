from pylab import *
import torch
from integrator.layers import Linear

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
        vi=False,
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
