from pylab import *
import torch
from integrator.layers import Linear


class PoissonLikelihoodV2(torch.nn.Module):
    """
    Attributes:
        beta:
        p_I_scale: scale DKL(q_I||p_I)
        p_bg_scale: scale DKL(q_I||p_I)
        prior_I: prior distribution for intensity
        prior_bg: prior distribution for background
    """

    def __init__(
        self,
        beta=1.0,
        eps=1e-8,
        prior_I=None,
        prior_bg=None,
        prior_profile=None,
        p_I_scale=0.001,  # influence of DKL(LogNorm||LogNorm) term
        p_bg_scale=0.001,
        p_profile_scale=0.01,
    ):
        super().__init__()
        self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
        self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)
        self.p_I_scale = torch.nn.Parameter(
            data=torch.tensor(p_I_scale), requires_grad=False
        )
        self.p_bg_scale = torch.nn.Parameter(
            data=torch.tensor(p_bg_scale), requires_grad=False
        )
        self.prior_profile_scale = torch.nn.Parameter(
            data=torch.tensor(p_profile_scale), requires_grad=False
        )
        self.prior_I = prior_I
        self.prior_bg = prior_bg
        self.prior_profile = prior_profile

    def forward(
        self,
        counts,
        q_bg,
        q_I,
        profile,
        L,
        eps=1e-5,
        mc_samples=100,
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
        # rate = z.permute(1,0,2) * (profile) + bg.permute(1,0,2)
        rate = z.permute(1, 0, 2) * (profile.unsqueeze(1)) + bg.permute(1, 0, 2)

        # ll = torch.distributions.Poisson(rate).log_prob(counts)
        ll = torch.distributions.Poisson(rate + eps).log_prob(counts.unsqueeze(1))

        # ll = ll * mask if mask is not None else ll

        # Calculate KL-divergence only if the corresponding priors and distributions are available
        if q_I is not None and self.prior_I is not None:
            kl_I = q_I.log_prob(z) - self.prior_I.log_prob(z)
            kl_term += kl_I.mean() * self.p_I_scale

        if q_bg is not None and self.prior_bg is not None:
            kl_bg = q_bg.log_prob(bg) - self.prior_bg.log_prob(bg)
            kl_term += kl_bg.mean() * self.p_bg_scale

        if self.prior_profile is not None:
            profile_dist = torch.distributions.MultivariateNormal(
                torch.zeros_like(L), scale_tril=L
            )
            kl_profile = torch.distributions.kl_divergence(
                profile_dist, self.prior_profile
            )
            kl_term += kl_profile.mean() * self.prior_profile_scale

        return ll, kl_term, rate
