import torch


class Loss(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        p_I=torch.distributions.exponential.Exponential(1.0),
        # p_bg=torch.distributions.exponential.Exponential(1.0),
        p_bg=torch.distributions.normal.Normal(0.0, 0.5),
        # p_I=torch.distributions.log_normal.LogNormal(0.0, 1.0),
        p_I_scale=0.0001,
        p_bg_scale=0.0001,
        mc_samples=100,
    ):
        super().__init__()
        self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
        self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)
        self.p_I = p_I
        self.p_bg = p_bg
        self.p_I_scale = torch.nn.Parameter(
            data=torch.tensor(p_I_scale), requires_grad=False
        )
        self.p_bg_scale = torch.nn.Parameter(
            data=torch.tensor(p_bg_scale), requires_grad=False
        )
        self.mc_samples = mc_samples

    def forward(
        self,
        rate,
        counts,
        q_I,
        q_bg,
        dead_pixel_mask,
        eps=1e-5,
    ):
        ll = torch.distributions.Poisson(rate + eps).log_prob(counts.unsqueeze(1))

        kl_term = 0

        if q_I is not None and self.p_I is not None:
            kl_I = torch.distributions.kl.kl_divergence(q_I, self.p_I)
            kl_term += kl_I.mean() * self.p_I_scale

        if q_bg is not None and self.p_bg is not None:
            kl_bg = torch.distributions.kl.kl_divergence(q_bg, self.p_bg)
            kl_term += kl_bg.mean() * self.p_bg_scale

        # if q_I is not None and self.p_I is not None:
        # # Monte Carlo sampling from the variational distribution
        # z_samples = q_I.rsample([self.mc_samples])

        # # Compute log probabilities for the samples
        # log_qz = q_I.log_prob(z_samples)
        # log_pz = self.p_I.log_prob(z_samples)

        # # Monte Carlo estimate of KL divergence
        # kl_I = torch.mean(log_qz - log_pz, dim=0)  # Averaging over samples
        # kl_term += kl_I.mean() * self.p_I_scale

        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)
        nll = -(torch.sum(ll_mean) / torch.sum(dead_pixel_mask))

        return nll, kl_term
