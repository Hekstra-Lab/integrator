import torch


class Loss(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        prior_I=torch.distributions.exponential.Exponential(1.0),
        prior_bg=torch.distributions.exponential.Exponential(1.0),
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
