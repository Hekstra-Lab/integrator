import torch
from integrator.model.loss import BaseLoss


class LaplaceLoss(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        p_bg=torch.distributions.gamma.Gamma(torch.tensor(1.0), torch.tensor(1.0)),
        p_I=torch.distributions.gamma.Gamma(torch.tensor(1.0), torch.tensor(1.0)),
        p_p=None,
        p_p_scale=0.0001,
        p_bg_scale=0.0001,
        p_I_scale=0.0001,
        recon_scale=0.00,
        mc_samples=100,
        tv_loss_scale=0.1,
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
        self.p_p = p_p

        # Dirichlet prior parameters will be moved to correct device in forward
        self.register_buffer("dirichlet_concentration", torch.ones(3 * 21 * 21))

        self.mc_samples = mc_samples
        self.tv_loss_scale = tv_loss_scale

    def total_variation_3d(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Tensor of shape (batch, depth, height, width)
        Returns:
            tv_loss: Scalar tensor
        """
        # Differences along depth (z-axis)
        diff_depth = torch.abs(volume[:, 1:, :, :] - volume[:, :-1, :, :])
        # Differences along height (y-axis)
        diff_height = torch.abs(volume[:, :, 1:, :] - volume[:, :, :-1, :])
        # Differences along width (x-axis)
        diff_width = torch.abs(volume[:, :, :, 1:] - volume[:, :, :, :-1])

        # Sum over all spatial dimensions and batch
        tv_loss = diff_depth.sum() + diff_height.sum() + diff_width.sum()
        return tv_loss

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
        # if not hasattr(self, "p_p") or self.p_p.concentration.device != device:
        # self.p_p = torch.distributions.dirichlet.Dirichlet(
        # self.dirichlet_concentration.to(device)
        # )

        # Ensure other components are on the correct device
        counts = counts.to(device)
        dead_pixel_mask = dead_pixel_mask.to(device)

        # Calculate log likelihood
        ll = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))

        # Calculate KL divergences with device-aware tensors
        samples_p = q_p.rsample([self.mc_samples])  # Reparameterized sampling
        log_q_p = q_p.log_prob(samples_p)
        log_p_p = self.p_p.log_prob(samples_p)

        # tv loss on profile
        tv_loss = (
            self.total_variation_3d(q_p.mean.reshape(-1, 3, 21, 21))
            * self.tv_loss_scale
        )

        kl = (log_q_p - log_p_p).mean()

        kl += (
            torch.distributions.kl.kl_divergence(q_bg, self.p_bg) * self.p_bg_scale
        ).mean()

        samples = q_I.rsample([self.mc_samples])  # Reparameterized sampling
        log_q = q_I.log_prob(samples)
        log_p = self.p_I.log_prob(samples)

        kl += (log_q - log_p).mean()

        # kl += torch.distributions.kl.kl_divergence(q_I, self.p_I) * self.p_I_scale

        # Calculate reconstruction loss
        recon_loss = torch.abs(rate.mean(1) - counts) / (counts + self.eps)
        recon_loss = recon_loss.mean() * self.recon_scale

        # Calculate final loss terms
        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)
        ll_sum = ll_mean.sum(dim=1)
        neg_ll_sum = -ll_sum

        total_loss = neg_ll_sum + kl + recon_loss + tv_loss

        return total_loss, neg_ll_sum, kl, recon_loss
