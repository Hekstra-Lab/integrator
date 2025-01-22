import torch
from integrator.model.loss import BaseLoss


class Loss(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        p_bg=torch.distributions.gamma.Gamma(torch.tensor(1.0), torch.tensor(1.0)),
        p_I=torch.distributions.gamma.Gamma(torch.tensor(1.0), torch.tensor(1.0)),
        p_p_scale=0.0001,
        p_bg_scale=0.0001,
        p_I_scale=0.0001,
        recon_scale=0.00,
    ):
        super().__init__()

        # Don't specify device in __init__, let PyTorch handle it
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("recon_scale", torch.tensor(recon_scale))
        self.register_buffer("beta", torch.tensor(beta))

        # Scale parameters
        self.register_buffer("p_I_scale", torch.tensor(p_I_scale))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale))
        self.register_buffer("p_p_scale", torch.tensor(p_p_scale))

        # Prior distributions
        # Move tensors to appropriate device in forward pass
        self.p_bg = p_bg
        self.p_I = p_I

        # Dirichlet prior parameters will be moved to correct device in forward
        self.register_buffer("dirichlet_concentration", torch.ones(3 * 21 * 21))

    # TODO: Make to() dynamic based on distribution type
    def to(self, device):
        # Override to() to ensure all components are moved to the correct device
        super().to(device)
        self.p_bg.loc = self.p_bg.loc.to(device)
        self.p_bg.scale = self.p_bg.scale.to(device)
        self.p_I.loc = self.p_I.loc.to(device)
        self.p_I.scale = self.p_I.scale.to(device)
        return self

    def forward(self, rate, counts, q_p, q_I, q_bg, dead_pixel_mask):
        # Ensure all inputs are on the same device
        device = rate.device

        # Move priors to the correct device if needed
        if not hasattr(self, "p_p") or self.p_p.concentration.device != device:
            self.p_p = torch.distributions.dirichlet.Dirichlet(
                self.dirichlet_concentration.to(device)
            )

        # Ensure other components are on the correct device
        counts = counts.to(device)
        dead_pixel_mask = dead_pixel_mask.to(device)

        # Calculate log likelihood
        ll = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))

        # Calculate KL divergences with device-aware tensors
        kl = torch.distributions.kl.kl_divergence(q_p, self.p_p) * self.p_p_scale
        kl += torch.distributions.kl.kl_divergence(q_bg, self.p_bg) * self.p_bg_scale
        kl += torch.distributions.kl.kl_divergence(q_I, self.p_I) * self.p_I_scale

        # Calculate reconstruction loss
        recon_loss = torch.abs(rate.mean(1) - counts) / (counts + self.eps)
        recon_loss = recon_loss.mean() * self.recon_scale

        # Calculate final loss terms
        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)
        ll_sum = ll_mean.sum(dim=1)
        neg_ll_sum = -ll_sum

        total_loss = neg_ll_sum + kl + recon_loss

        return total_loss, neg_ll_sum, kl, recon_loss
