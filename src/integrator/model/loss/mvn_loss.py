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
            self.register_buffer(f"{prefix}concentration", torch.tensor(params["concentration"]))
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
            return torch.distributions.gamma.Gamma(concentration=concentration, rate=rate)
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
        diff_y = torch.abs(profile_reshaped[:, :, 1:, :] - profile_reshaped[:, :, :-1, :])
        diff_x = torch.abs(profile_reshaped[:, :, :, 1:] - profile_reshaped[:, :, :, :-1])

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

    def forward(self, rate, counts, profile, q_bg, masks):
        device = rate.device
        batch_size = rate.shape[0]

        # Ensure other components are on the correct device
        counts = counts.to(device)
        masks = masks.to(device)

        # Create distributions on the correct device
        p_bg = self.get_prior(self.p_bg_name, "p_bg_", device)

        # Calculate KL terms
        kl_terms = torch.zeros(batch_size, device=device)

        # KL for background
        kl_bg = self.compute_kl(q_bg, p_bg)
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

        ll_mean = (
            (
                torch.distributions.Poisson(rate).log_prob(counts.unsqueeze(1)) * masks.unsqueeze(1)
            ).mean(1)
        ).sum(1) / masks.sum(1)
        # Calculate negative log likelihood
        neg_ll_batch = (-ll_mean).sum()

        # Combine all loss terms
        batch_loss = neg_ll_batch + kl_terms
        total_loss = batch_loss.mean()

        # Return all components for monitoring
        return (
            total_loss,
            neg_ll_batch.mean(),
            kl_terms.mean(),
            kl_bg.mean(),
            # kl_i.mean() if p_I is not None else torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            profile_reg.mean() if profile_reg.sum() > 0 else torch.tensor(0.0, device=device),
        )


class LRMVNLoss(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        p_bg_name="half_normal",
        p_bg_params={"scale": 1.0},
        p_I_name="gamma",
        p_I_params={"concentration": 1.0, "rate": 1.0},
        p_prf_mean={"loc": 0.0, "scale": 5.0},
        p_prf_diag={"scale": 0.3},
        p_prf_factor={"loc": 0.0, "scale": 0.5},
        p_bg_w=0.0001,
        p_I_w=0.00001,
        p_prf_mean_w=0.001,
        p_prf_factor_w=0.001,
        p_prf_diag_w=0.001,
        prior_shape=(3, 21, 21),
    ):
        super().__init__()

        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("p_bg_w", torch.tensor(p_bg_w))
        self.register_buffer("p_bg_w", torch.tensor(p_bg_params["scale"]))
        self.register_buffer("p_prf_mean_w", torch.tensor(p_prf_mean_w))
        self.register_buffer("p_prf_factor_w", torch.tensor(p_prf_factor_w))
        self.register_buffer("p_prf_diag_w", torch.tensor(p_prf_diag_w))

        self._register_distribution_params(p_I_name, p_I_params, prefix="p_I_")
        self.register_buffer("p_I_w", torch.tensor(p_I_w))

        self.p_prf_mean = p_prf_mean
        self.p_prf_diag = p_prf_diag
        self.p_prf_factor = p_prf_factor

        # Store distribution names and params
        self.p_bg_name = p_bg_name
        self.p_bg_params = p_bg_params
        self.p_I_name = p_I_name
        self.p_I_params = p_I_params

        # Number of elements in the profile
        self.profile_size = prior_shape[0] * prior_shape[1] * prior_shape[2]

    def get_prior(self, name, params_prefix, device):
        """Create a distribution on the specified device"""
        if name == "gamma":
            concentration = getattr(self, f"{params_prefix}concentration").to(device)
            rate = getattr(self, f"{params_prefix}rate").to(device)
            return torch.distributions.gamma.Gamma(concentration=concentration, rate=rate)
        elif name == "log_normal":
            loc = getattr(self, f"{params_prefix}loc").to(device)
            scale = getattr(self, f"{params_prefix}scale").to(device)
            return torch.distributions.log_normal.LogNormal(loc=loc, scale=scale)
        elif name == "exponential":
            rate = getattr(self, f"{params_prefix}rate").to(device)
            return torch.distributions.exponential.Exponential(rate=rate)
        # Add more distribution types as needed

    def _register_distribution_params(self, name, params, prefix):
        """Register distribution parameters as buffers with appropriate prefixes"""
        if name == "gamma":
            self.register_buffer(f"{prefix}concentration", torch.tensor(params["concentration"]))
            self.register_buffer(f"{prefix}rate", torch.tensor(params["rate"]))
        elif name == "log_normal":
            self.register_buffer(f"{prefix}loc", torch.tensor(params["loc"]))
            self.register_buffer(f"{prefix}scale", torch.tensor(params["scale"]))
        elif name == "exponential":
            self.register_buffer(f"{prefix}rate", torch.tensor(params["rate"]))

    def compute_kl(self, q, p, num_samples=100):
        try:
            return torch.distributions.kl.kl_divergence(q, p)
        except NotImplementedError:
            # Sample from q
            samples = q.rsample((num_samples,))
            log_q = q.log_prob(samples)
            log_p = p.log_prob(samples)
            kl_estimate = (log_q - log_p).mean(dim=0)
            return kl_estimate.sum(dim=-1)

    def forward(self, rate, counts, q_bg, q_i, masks, q_p_mean, q_p_diag, q_p_factor):
        device = rate.device
        batch_size = rate.shape[0]
        self.current_batch_size = batch_size

        counts = counts.to(device)
        masks = masks.to(device)

        p_bg = torch.distributions.half_normal.HalfNormal(
            scale=torch.tensor(self.p_bg_w, device=device)
        )

        p_I = self.get_prior(self.p_I_name, "p_I_", device)

        p_prf_mean = torch.distributions.normal.Normal(
            loc=torch.tensor(self.p_prf_mean["loc"], device=device),
            scale=torch.tensor(self.p_prf_mean["scale"], device=device),
        )

        p_prf_diag = torch.distributions.half_normal.HalfNormal(
            scale=torch.tensor(self.p_prf_diag["scale"], device=device)
        )

        p_prf_factor = torch.distributions.normal.Normal(
            loc=torch.tensor(self.p_prf_factor["loc"], device=device).view(1, 3, 1),
            scale=torch.tensor(self.p_prf_factor["scale"], device=device).view(1, 3, 1),
        )

        # calculate kl terms
        kl_terms = torch.zeros(batch_size, device=device)

        kl_i = self.compute_kl(q_i, p_I)
        kl_terms += kl_i * self.p_I_w

        # calculate background and intensity kl divergence
        kl_bg = torch.distributions.kl.kl_divergence(q_bg, p_bg)
        kl_terms += kl_bg * self.p_bg_w

        kl_p_prf_mean = torch.distributions.kl.kl_divergence(q_p_mean, p_prf_mean)
        kl_terms += kl_p_prf_mean.sum() * self.p_prf_mean_w

        kl_p_prf_diag = torch.distributions.kl.kl_divergence(q_p_diag, p_prf_diag)
        kl_terms += kl_p_prf_diag.sum() * self.p_prf_diag_w

        kl_p_factor = torch.distributions.kl.kl_divergence(q_p_factor, p_prf_factor)
        kl_terms += kl_p_factor.sum() * self.p_prf_factor_w

        ll = torch.distributions.Poisson(rate + self.eps).log_prob(
            counts.unsqueeze(1)
        )  # [B, MC, pixels]
        ll_mean = torch.mean(ll, dim=1) * masks.squeeze(-1)

        # calculate negative log likelihood
        neg_ll_batch = (-ll_mean).sum(1)

        # combine all loss terms
        batch_loss = neg_ll_batch + kl_terms  # [batch_size, 1]

        # final scalar loss
        total_loss = batch_loss.mean()

        # return all components for monitoring
        return (
            total_loss,
            neg_ll_batch.mean(),
            kl_terms.mean(),
            kl_bg.mean() * self.p_bg_w,
            kl_i.mean() * self.p_I_w,
            kl_p_prf_mean.mean() * self.p_prf_mean_w,
            kl_p_prf_diag.mean() * self.p_prf_diag_w,
            kl_p_factor.mean() * self.p_prf_factor_w,
        )
