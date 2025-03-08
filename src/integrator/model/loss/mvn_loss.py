import torch


class MVNLoss(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        p_I_name="gamma",
        p_I_params={"concentration": 1.0, "rate": 1.0},
        p_I_scale=0.0001,
        p_bg_name="gamma",
        p_bg_params={"concentration": 1.0, "rate": 1.0},
        p_bg_scale=0.0001,
        # Optional regularization for the profile
        tv_loss_scale=None,
        simpson_scale=None,
        smoothness_scale=None,
    ):
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("p_I_scale", torch.tensor(p_I_scale))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale))

        # Store distribution types and parameters
        self.p_I_name = p_I_name
        self.p_I_params = p_I_params
        self.p_bg_name = p_bg_name
        self.p_bg_params = p_bg_params

        # Register parameters as buffers based on distribution type
        self._register_distribution_params(p_I_name, p_I_params, prefix="p_I_")
        self._register_distribution_params(p_bg_name, p_bg_params, prefix="p_bg_")

        # Optional regularization parameters
        self.tv_loss_scale = tv_loss_scale
        self.simpson_scale = simpson_scale
        self.smoothness_scale = smoothness_scale

    def _register_distribution_params(self, name, params, prefix):
        """Register distribution parameters as buffers with appropriate prefixes"""
        if name == "gamma":
            self.register_buffer(
                f"{prefix}concentration", torch.tensor(params["concentration"])
            )
            self.register_buffer(f"{prefix}rate", torch.tensor(params["rate"]))
        elif name == "log_normal":
            self.register_buffer(f"{prefix}loc", torch.tensor(params["loc"]))
            self.register_buffer(f"{prefix}scale", torch.tensor(params["scale"]))
        elif name == "exponential":
            self.register_buffer(f"{prefix}rate", torch.tensor(params["rate"]))
        # Add more distribution types as needed

    def get_prior(self, name, params_prefix, device):
        """Create a distribution on the specified device"""
        if name == "gamma":
            concentration = getattr(self, f"{params_prefix}concentration").to(device)
            rate = getattr(self, f"{params_prefix}rate").to(device)
            return torch.distributions.gamma.Gamma(
                concentration=concentration, rate=rate
            )
        elif name == "log_normal":
            loc = getattr(self, f"{params_prefix}loc").to(device)
            scale = getattr(self, f"{params_prefix}scale").to(device)
            return torch.distributions.log_normal.LogNormal(loc=loc, scale=scale)
        elif name == "exponential":
            rate = getattr(self, f"{params_prefix}rate").to(device)
            return torch.distributions.exponential.Exponential(rate=rate)
        # Add more distribution types as needed

    def inverse_simpson_regularization(self, p, eps=1e-6):
        batch_size = p.shape[0]
        p_flat = p.view(batch_size, -1)
        simpson = torch.sum(p_flat**2, dim=1)
        inv_simpson = 1.0 / (simpson + eps)
        return inv_simpson

    def total_variation_3d(self, volume):
        batch_size = volume.shape[0]
        batch_tv_loss = torch.zeros(batch_size, device=volume.device)

        for b in range(batch_size):
            single_vol = volume[b : b + 1]
            diff_depth = torch.abs(single_vol[:, 1:, :, :] - single_vol[:, :-1, :, :])
            diff_height = torch.abs(single_vol[:, :, 1:, :] - single_vol[:, :, :-1, :])
            diff_width = torch.abs(single_vol[:, :, :, 1:] - single_vol[:, :, :, :-1])
            batch_tv_loss[b] = diff_depth.sum() + diff_height.sum() + diff_width.sum()

        return batch_tv_loss

    def gaussian_smoothness_loss(self, profile, sigma=1.0):
        """Encourages the profile to be more smooth (like a Gaussian)"""
        batch_size = profile.shape[0]
        profile_reshaped = profile.view(batch_size, 3, 21, 21)

        # Calculate gradient of profile
        diff_y = torch.abs(
            profile_reshaped[:, :, 1:, :] - profile_reshaped[:, :, :-1, :]
        )
        diff_x = torch.abs(
            profile_reshaped[:, :, :, 1:] - profile_reshaped[:, :, :, :-1]
        )

        # Penalize large gradients, encouraging smoothness
        smoothness_loss = diff_y.sum(dim=(1, 2, 3)) + diff_x.sum(dim=(1, 2, 3))
        return smoothness_loss

    def compute_kl(self, q_dist, p_dist):
        """Compute KL divergence between distributions, with fallback sampling if needed."""
        try:
            return torch.distributions.kl.kl_divergence(q_dist, p_dist)
        except NotImplementedError:
            samples = q_dist.rsample([100])
            log_q = q_dist.log_prob(samples)
            log_p = p_dist.log_prob(samples)
            return (log_q - log_p).mean(dim=0)

    def forward(self, rate, counts, profile, q_I, q_bg, dead_pixel_mask):
        device = rate.device
        batch_size = rate.shape[0]

        # Ensure other components are on the correct device
        counts = counts.to(device)
        dead_pixel_mask = dead_pixel_mask.to(device)

        # Create distributions on the correct device
        p_I = self.get_prior(self.p_I_name, "p_I_", device)
        p_bg = self.get_prior(self.p_bg_name, "p_bg_", device)

        # Calculate log likelihood
        ll = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))

        # Calculate KL terms
        kl_terms = torch.zeros(batch_size, device=device)

        # KL for intensity
        kl_I = self.compute_kl(q_I, p_I)
        kl_I = kl_I.expand(batch_size) if kl_I.dim() == 0 else kl_I
        kl_terms += kl_I * self.p_I_scale

        # KL for background
        kl_bg = self.compute_kl(q_bg, p_bg)
        kl_bg = kl_bg.expand(batch_size) if kl_bg.dim() == 0 else kl_bg
        kl_terms += kl_bg * self.p_bg_scale

        # Optional regularization for profile
        profile_reg = torch.zeros(batch_size, device=device)

        # Add profile regularization if requested
        if self.simpson_scale is not None:
            profile_reshaped = profile.view(batch_size, 3, 21, 21)
            simpson_reg = self.inverse_simpson_regularization(profile_reshaped)
            profile_reg += simpson_reg * self.simpson_scale

        if self.tv_loss_scale is not None:
            profile_reshaped = profile.view(batch_size, 3, 21, 21)
            tv_loss = self.total_variation_3d(profile_reshaped)
            profile_reg += tv_loss * self.tv_loss_scale

        if self.smoothness_scale is not None:
            smoothness_reg = self.gaussian_smoothness_loss(profile)
            profile_reg += smoothness_reg * self.smoothness_scale

        # Calculate negative log likelihood
        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)
        neg_ll_batch = -ll_mean.sum(dim=1)

        # Combine all loss terms
        batch_loss = neg_ll_batch + kl_terms + profile_reg
        total_loss = batch_loss.mean()

        # Return all components for monitoring
        return (
            total_loss,
            neg_ll_batch.mean(),
            kl_terms.mean(),
            kl_bg.mean(),
            kl_I.mean(),
            profile_reg.mean()
            if profile_reg.sum() > 0
            else torch.tensor(0.0, device=device),
        )
