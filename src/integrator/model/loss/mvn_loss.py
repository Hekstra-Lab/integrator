import torch


class tempMVNLoss(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        p_I=torch.distributions.exponential.Exponential(1.0),
        p_bg=torch.distributions.exponential.Exponential(1.0),
        p_I_scale=0.0001,
        p_bg_scale=0.0001,
        # Optional regularization for the profile
        tv_loss_scale=None,
        simpson_scale=None,
        smoothness_scale=None,
    ):
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        self.p_I = p_I
        self.p_bg = p_bg
        self.register_buffer("p_I_scale", torch.tensor(p_I_scale))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale))

        # Optional regularization parameters
        self.tv_loss_scale = tv_loss_scale
        self.simpson_scale = simpson_scale
        self.smoothness_scale = smoothness_scale

    def inverse_simpson_regularization(self, p, eps=1e-6):
        batch_size = p.shape[0]
        p_flat = p.view(batch_size, -1)
        simpson = torch.sum(p_flat**2, dim=1)
        inv_simpson = 1.0 / (simpson + eps)
        return inv_simpson

    def total_variation_3d(self, volume):
        # Implementation from your paste.txt
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

        # Calculate log likelihood
        ll = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))

        # Calculate KL terms
        kl_terms = torch.zeros(batch_size, device=device)

        # KL for intensity
        kl_I = self.compute_kl(q_I, self.p_I)
        kl_I = kl_I.expand(batch_size) if kl_I.dim() == 0 else kl_I
        kl_terms += kl_I * self.p_I_scale

        # KL for background
        kl_bg = self.compute_kl(q_bg, self.p_bg)
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

        # Add in the forward method of MVNLoss:
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



class MVNLoss(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        p_I_scale=0.0001,
        p_bg_scale=0.0001,
        # Optional regularization for the profile
        tv_loss_scale=None,
        simpson_scale=None,
        smoothness_scale=None,
        device=None,  # Add device parameter
    ):
        super().__init__()
        # Use the provided device or default to the current device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.register_buffer("eps", torch.tensor(eps, device=self.device))
        self.register_buffer("beta", torch.tensor(beta, device=self.device))
        self.register_buffer("p_I_scale", torch.tensor(p_I_scale, device=self.device))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale, device=self.device))
        
        # Create the distributions in to() method instead of here
        # This ensures they will be properly moved to the right device later
        self.p_I_lambda = 1.0  # Store the parameter, not the distribution
        self.p_bg_lambda = 1.0
        
        # Optional regularization parameters
        self.tv_loss_scale = tv_loss_scale
        self.simpson_scale = simpson_scale
        self.smoothness_scale = smoothness_scale
    
    def to(self, device):
        # Override to() to ensure distributions are also moved to the device
        super().to(device)
        self.device = device
        return self

    def _get_prior_distributions(self):
        # Create the distributions on the correct device when needed
        p_I = torch.distributions.exponential.Exponential(
            torch.tensor(self.p_I_lambda, device=self.device)
        )
        p_bg = torch.distributions.exponential.Exponential(
            torch.tensor(self.p_bg_lambda, device=self.device)
        )
        return p_I, p_bg

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
            # Make sure samples are on the correct device
            device = next(self.parameters()).device
            samples = q_dist.rsample([100])
            log_q = q_dist.log_prob(samples)
            log_p = p_dist.log_prob(samples)
            return (log_q - log_p).mean(dim=0)

    def forward(self, rate, counts, profile, q_I, q_bg, dead_pixel_mask):
        # Get the current device from one of the inputs
        device = rate.device
        self.device = device  # Update the module's device
        
        batch_size = rate.shape[0]

        # Ensure other components are on the correct device
        counts = counts.to(device)
        dead_pixel_mask = dead_pixel_mask.to(device)
        
        # Get prior distributions on the correct device
        p_I, p_bg = self._get_prior_distributions()

        # Calculate log likelihood
        ll = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))

        # Calculate KL terms
        kl_terms = torch.zeros(batch_size, device=device)

        # KL for intensity 
        # Make sure q_I is on the correct device
        if hasattr(q_I, 'loc') and q_I.loc.device != device:
            q_I_params = {k: v.to(device) if torch.is_tensor(v) else v for k, v in q_I.__dict__.items() if k != '_validate_args'}
            q_I = type(q_I)(**q_I_params)
            
        kl_I = self.compute_kl(q_I, p_I)
        kl_I = kl_I.expand(batch_size) if kl_I.dim() == 0 else kl_I
        kl_terms += kl_I * self.p_I_scale

        # KL for background
        # Make sure q_bg is on the correct device
        if hasattr(q_bg, 'loc') and q_bg.loc.device != device:
            q_bg_params = {k: v.to(device) if torch.is_tensor(v) else v for k, v in q_bg.__dict__.items() if k != '_validate_args'}
            q_bg = type(q_bg)(**q_bg_params)
            
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
