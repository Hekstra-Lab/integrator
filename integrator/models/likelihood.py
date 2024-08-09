from pylab import *
import torch


class Decoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        q_bg,
        q_I,
        profile,
        bg_profile,
        mc_samples=100,
    ):
        # Sample from variational distributions
        z = q_I.rsample([mc_samples])
        bg = q_bg.rsample([mc_samples])

        rate = z.permute(1, 0, 2) * profile.unsqueeze(1) + bg.permute(
            1, 0, 2
        ) * bg_profile.unsqueeze(1)

        return rate, z, bg


class LossMixtureModel3D(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        prior_I=None,
        prior_bg=None,
        p_I_scale=0.001,
        p_bg_scale=0.001,
    ):
        super().__init__()
        self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
        self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)
        self.prior_I = prior_I
        self.prior_bg = prior_bg
        self.p_I_scale = torch.nn.Parameter(
            data=torch.tensor(p_I_scale), requires_grad=False
        )
        self.p_bg_scale = torch.nn.Parameter(
            data=torch.tensor(p_bg_scale), requires_grad=False
        )

    def forward(
        self,
        rate,
        z,
        bg,
        counts,
        q_bg,
        q_I,
        dead_pixel_mask,
        eps=1e-5,
    ):
        ll = torch.distributions.Poisson(rate + eps).log_prob(counts.unsqueeze(1))

        kl_term = 0

        # Calculate KL-divergence only if the corresponding priors and distributions are available
        if q_I is not None and self.prior_I is not None:
            kl_I = q_I.log_prob(z + eps) - self.prior_I.log_prob(z + eps)
            kl_term += kl_I.mean() * self.p_I_scale

        if q_bg is not None:
            kl_bg = q_bg.log_prob(bg + eps) - self.prior_bg.log_prob(bg + eps)
            kl_term += kl_bg.mean() * self.p_bg_scale

        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)

        nll = -(torch.sum(ll_mean) / torch.sum(dead_pixel_mask))

        return nll, kl_term


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
        eps=1e-5,
        prior_I=None,
        prior_bg=None,
        p_I_scale=0.001,  # influence of DKL(LogNorm||LogNorm) term
        p_bg_scale=0.001,
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

    def forward(
        self,
        counts,
        q_bg,
        q_I,
        profile,
        bg_profile,
        # image_weights,
        eps=1e-5,
        mc_samples=100,
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

        counts = counts

        # Sample from variational distributions
        z = q_I.rsample([mc_samples])
        bg = q_bg.rsample([mc_samples])

        # Set KL term to zero
        kl_term = 0

        # Calculate the rate
        # rate = z.permute(1, 0, 2) * (profile.unsqueeze(1)) + bg.permute(1, 0, 2)
        # rate = z.permute(1, 0, 2) * (profile.unsqueeze(1)) + bg.permute(1, 0, 2)

        # rate = z.permute(1, 0, 2) * profile.unsqueeze(1)

        # rates = torch.split(rate, 441, dim=-1)

        # For 3d mixture models

        rate = z.permute(1, 0, 2) * profile.unsqueeze(1) + bg.permute(
            1, 0, 2
        ) * bg_profile.unsqueeze(1)

        # # For 2d mixture models
        # zs = torch.split(z.permute(1, 0, 2, 3), 1, dim=-1)
        # bgs = torch.split(bg.permute(1, 0, 2, 3), 1, dim=-1)
        # profs = torch.split(profile.unsqueeze(1), 441, dim=-1)

        # bgprofs = torch.split(bg_profile, 441, dim=1)

        # rate = torch.cat(
        # (
        # (
        # zs[0].squeeze(-1) * profs[0]
        # + bgs[0].squeeze(-1) * bgprofs[0].unsqueeze(1)
        # ),
        # (
        # zs[1].squeeze(-1) * profs[1]
        # + bgs[1].squeeze(-1) * bgprofs[1].unsqueeze(1)
        # ),
        # (
        # zs[2].squeeze(-1) * profs[2]
        # + bgs[2].squeeze(-1) * bgprofs[2].unsqueeze(1)
        # ),
        # ),
        # dim=-1,
        # )

        ll = torch.distributions.Poisson(rate + eps).log_prob(counts.unsqueeze(1))

        # ll = ll * mask if mask is not None else ll

        # Calculate KL-divergence only if the corresponding priors and distributions are available
        if q_I is not None and self.prior_I is not None:
            kl_I = q_I.log_prob(z + eps) - self.prior_I.log_prob(z + eps)
            kl_term += kl_I.mean() * self.p_I_scale

        if q_bg is not None:
            kl_bg = q_bg.log_prob(bg + eps) - self.prior_bg.log_prob(bg + eps)
            kl_term += kl_bg.mean() * self.p_bg_scale

        return ll, kl_term, rate


if __name__ == "__main__":
    from integrator.models import Builder

    intensity_dist = torch.distributions.gamma.Gamma
    background_dist = torch.distributions.gamma.Gamma

    d_model = 10

    builder = Builder(d_model, intensity_dist, background_dist)

    batch_size = 2
    num_planes = 3

    # Example inputs
    representation = torch.randn(batch_size, 1, d_model)
    dxyz = torch.randn(batch_size, 441 * num_planes, 3)
    mask = torch.ones(batch_size, 441 * num_planes)
    isflat = torch.zeros(batch_size, 441 * num_planes)

    # Forward pass
    q_bg, q_I, profile, L, penalty, bg_profile = builder(
        representation, dxyz, mask, isflat, use_mixture_model=True
    )

    z = q_I.rsample([100])
    bg = q_bg.rsample([100])

    zs = torch.split(z.permute(1, 0, 2, 3), 1, dim=-1)
    bgs = torch.split(bg.permute(1, 0, 2, 3), 1, dim=-1)

    profs = torch.split(profile.unsqueeze(1), 441, dim=-1)

    bgprofs = torch.split(bg_profile, 441, dim=1)

    rate = torch.cat(
        (
            (
                zs[0].squeeze(-1) * profs[0]
                + bgs[0].squeeze(-1) * bgprofs[0].unsqueeze(1)
            ),
            (
                zs[1].squeeze(-1) * profs[1]
                + bgs[1].squeeze(-1) * bgprofs[1].unsqueeze(1)
            ),
            (
                zs[2].squeeze(-1) * profs[2]
                + bgs[2].squeeze(-1) * bgprofs[2].unsqueeze(1)
            ),
        ),
        dim=-1,
    )
