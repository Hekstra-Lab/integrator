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
        p_I_scale=0.01,  # influence of DKL(LogNorm||LogNorm) term
        p_bg_scale=0.01,
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
        self.prior_I = prior_I
        self.prior_bg = prior_bg

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
        z = q_I.rsample([mc_samples]) + eps
        bg = q_bg.rsample([mc_samples]) + eps

        # Set KL term
        kl_term = 0

        # Calculate the rate
        rate = (z * (profile)) + bg
        rate = rate + eps

        counts = torch.clamp(counts, min=0)  # do not clamp, use a mask instead

        if mask is not None:
            # Mask out padded terms
            ll = torch.distributions.Poisson(rate).log_prob(counts)
            ll = ll * mask

        else:
            ll = torch.distributions.Poisson(rate).log_prob(counts)

        # Calculate KL-divergence
        if vi:
            # Calculate KL-divergence only if the corresponding priors and distributions are available
            if q_I is not None and self.prior_I is not None:
                kl_I = q_I.log_prob(z) - self.prior_I.log_prob(z)
                if mask is not None:
                    kl_I = kl_I * mask
                kl_term += kl_I.mean() * self.p_I_scale

            if q_bg is not None and self.prior_bg is not None:
                kl_bg = q_bg.log_prob(bg) - self.prior_bg.log_prob(bg)
                if mask is not None:
                    kl_bg = kl_bg * mask
                kl_term += kl_bg.mean() * self.p_bg_scale

            # zero out pads

        else:
            kl_term = 0  # set to 0 when vi false

        return ll, kl_term
