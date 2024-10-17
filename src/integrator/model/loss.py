import torch


class Loss(torch.nn.Module):
    """
    Attributes:
        device:
        eps:
        beta:
        p_I_scale:
        p_bg_scale:
        p_p_scale:
        p_I:
        p_bg:
        p_p:
    """

    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        p_I=torch.distributions.exponential.Exponential(1.0),
        p_bg=torch.distributions.normal.Normal(0.0, 0.5),
        p_p=torch.distributions.dirichlet.Dirichlet(torch.ones(3 * 21 * 21)),
        p_I_scale=0.0001,
        p_bg_scale=0.0001,
        p_p_scale=0.0001,
        mc_samples=100,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.eps = torch.nn.Parameter(
            torch.tensor(eps, device=self.device), requires_grad=False
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(beta, device=self.device), requires_grad=False
        )
        self.p_I_scale = torch.nn.Parameter(
            torch.tensor(p_I_scale, device=self.device), requires_grad=False
        )
        self.p_bg_scale = torch.nn.Parameter(
            torch.tensor(p_bg_scale, device=self.device), requires_grad=False
        )
        self.p_p_scale = torch.nn.Parameter(
            torch.tensor(p_p_scale, device=self.device), requires_grad=False
        )

        # Move distribution parameters to the correct device
        self.p_I = self._move_distribution_to_device(p_I)
        self.p_bg = self._move_distribution_to_device(p_bg)
        self.p_p = self._move_distribution_to_device(p_p)

    def _move_distribution_to_device(self, dist):
        if isinstance(dist, torch.distributions.Distribution):
            new_params = {
                param: getattr(dist, param).to(self.device)
                for param in dist.arg_constraints
            }
            return type(dist)(**new_params)
        return dist

    def forward(self, rate, counts, q_I, q_bg, q_p, dead_pixel_mask, eps=1e-5):
        rate = rate.to(self.device)

        counts = counts.to(self.device)

        dead_pixel_mask = dead_pixel_mask.to(self.device)

        ll = torch.distributions.Poisson(rate + eps).log_prob(counts.unsqueeze(1))

        p_p = self._move_distribution_to_device(
            torch.distributions.dirichlet.Dirichlet(torch.ones(3 * 21 * 21))
        )

        kl_term = 0
        if q_I is not None and self.p_I is not None:
            q_I = self._move_distribution_to_device(q_I)
            kl_I = torch.distributions.kl.kl_divergence(q_I, self.p_I)
            kl_term += kl_I.mean() * self.p_I_scale

        if q_bg is not None and self.p_bg is not None:
            q_bg = self._move_distribution_to_device(q_bg)
            kl_bg = torch.distributions.kl.kl_divergence(q_bg, self.p_bg)
            kl_term += kl_bg.mean() * self.p_bg_scale

        if q_p is not None and self.p_p is not None:
            q_p = self._move_distribution_to_device(q_p)
            kl_p = torch.distributions.kl.kl_divergence(q_p, p_p)
            kl_term += kl_p.mean() * self.p_p_scale

        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)

        nll = -(torch.sum(ll_mean) / torch.sum(dead_pixel_mask))

        return nll, kl_term
