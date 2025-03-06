import torch
from integrator.model.loss import BaseLoss


def create_center_focused_dirichlet_prior(
    shape=(3, 21, 21),
    base_alpha=5.0,
    center_alpha=0.01,
    decay_factor=0.5,
    peak_percentage=0.05,
):
    """
    Create a Dirichlet prior concentration vector with lower values (higher concentration)
    near the center of the image.

    Parameters:
    -----------
    shape : tuple
        Shape of the 3D image (channels, height, width)
    base_alpha : float
        Base concentration parameter value for most elements (higher = more uniform)
    center_alpha : float
        Minimum concentration value at the center (lower = more concentrated)
    decay_factor : float
        Controls how quickly the concentration values increase with distance from center
    peak_percentage : float
        Approximate percentage of elements that should have high concentration (low alpha)

    Returns:
    --------
    alpha_vector : torch.Tensor
        Flattened concentration vector for Dirichlet prior as a PyTorch tensor
    """
    channels, height, width = shape
    total_elements = channels * height * width

    # Create a 3D array filled with the base alpha value
    alpha_3d = np.ones(shape) * base_alpha

    # Calculate center coordinates
    center_c = channels // 2
    center_h = height // 2
    center_w = width // 2

    # Calculate distance from center for each position
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                # Calculate normalized distance from center (0 to 1 scale)
                dist_c = abs(c - center_c) / (channels / 2)
                dist_h = abs(h - center_h) / (height / 2)
                dist_w = abs(w - center_w) / (width / 2)

                # Euclidean distance in normalized space
                distance = np.sqrt(dist_c**2 + dist_h**2 + dist_w**2) / np.sqrt(3)

                # Apply exponential increase based on distance
                # For elements close to center: use low alpha (high concentration)
                # For elements far from center: use high alpha (low concentration)
                if (
                    distance < peak_percentage * 5
                ):  # Adjust this multiplier to control the size of high concentration region
                    alpha_value = (
                        center_alpha
                        + (base_alpha - center_alpha)
                        * (distance / (peak_percentage * 5)) ** decay_factor
                    )
                    alpha_3d[c, h, w] = alpha_value

    # Flatten the 3D array to get the concentration vector and convert to torch tensor
    alpha_vector = torch.tensor(alpha_3d.flatten(), dtype=torch.float32)

    return alpha_vector


class Loss(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        p_bg: torch.distributions.Distribution | None = None,
        p_I: torch.distributions.Distribution | None = None,
        p_p: torch.distributions.Distribution | None = None,
        p_p_scale: float = 0.0001,
        p_bg_scale: float = 0.0001,
        p_I_scale: float = 0.0001,
        simpson_scale: float | None = None,
        tv_loss_scale: float | None = None,
        entropy_scale: float | None = 0.5,
        use_center_focused_prior: bool = True,  # Add this parameter
        prior_shape: tuple = (3, 21, 21),  # Matches your default shape
        prior_base_alpha: float = 5.0,  # Center-focused prior parameters
        prior_center_alpha: float = 0.01,
        prior_decay_factor: float = 0.5,
        prior_peak_percentage: float = 0.05,
    ):
        super().__init__()

        # Original initialization code
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

        # Scale parameters
        self.register_buffer("p_I_scale", torch.tensor(p_I_scale))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale))
        self.register_buffer("p_p_scale", torch.tensor(p_p_scale))

        # Prior distributions
        self.p_bg = p_bg
        self.p_I = p_I

        # Create center-focused Dirichlet prior if requested
        if use_center_focused_prior and p_p is None:
            # Create the center-focused concentration vector
            alpha_vector = create_center_focused_dirichlet_prior(
                shape=prior_shape,
                base_alpha=prior_base_alpha,
                center_alpha=prior_center_alpha,
                decay_factor=prior_decay_factor,
                peak_percentage=prior_peak_percentage,
            )
            # Create Dirichlet distribution with this concentration vector
            self.p_p = torch.distributions.dirichlet.Dirichlet(alpha_vector)
            # Store the concentration for device moving
            self.register_buffer("dirichlet_concentration", alpha_vector)
        else:
            # Use the provided p_p or default
            self.p_p = p_p
            if self.p_p.__class__.__name__ == "Dirichlet":
                self.register_buffer(
                    "dirichlet_concentration",
                    self.p_p.concentration
                    if hasattr(self.p_p, "concentration")
                    else torch.ones(3 * 21 * 21) * 0.1,
                )

        self.simpson_scale = simpson_scale
        self.tv_loss_scale = tv_loss_scale
        self.entropy_scale = entropy_scale

    def to(self, device):
        """
        Override to() to ensure all distribution parameters are moved to the correct device.
        """
        super().to(device)

        # Move prior distributions to device
        if self.p_bg is not None:
            self._move_distribution_to_device(self.p_bg, device)

        if self.p_I is not None:
            self._move_distribution_to_device(self.p_I, device)

        if self.p_p is not None:
            self._move_distribution_to_device(self.p_p, device)

        return self

    def _move_distribution_to_device(self, dist, device):
        """Helper method to move distribution parameters to the specified device"""
        # Check which type of distribution and move accordingly
        if isinstance(dist, torch.distributions.gamma.Gamma):
            dist.concentration = dist.concentration.to(device)
            dist.rate = dist.rate.to(device)
        elif isinstance(dist, torch.distributions.log_normal.LogNormal):
            dist.loc = dist.loc.to(device)
            dist.scale = dist.scale.to(device)
        elif isinstance(dist, torch.distributions.dirichlet.Dirichlet):
            dist.concentration = dist.concentration.to(device)
        elif isinstance(dist, torch.distributions.beta.Beta):
            dist.concentration1 = dist.concentration1.to(device)
            dist.concentration0 = dist.concentration0.to(device)
        elif isinstance(dist, torch.distributions.laplace.Laplace):
            dist.loc = dist.loc.to(device)
            dist.scale = dist.scale.to(device)

    def inverse_simpson_regularization(
        self, p: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Args:
            p: Tensor of shape (batch, depth, height, width)
        Returns:
            inv_simpson_reg: Scalar regularization term to minimize.
        """
        batch_size = p.shape[0]
        p_flat = p.view(batch_size, -1)  # Shape: (batch, num_components)
        simpson = torch.sum(p_flat**2, dim=1)  # Shape: (batch,)
        inv_simpson = 1.0 / (simpson + eps)  # Add eps to avoid division by zero
        return inv_simpson  # Return per-batch values, don't reduce yet

    def concentration_entropy_loss(self, profile_mean):
        """
        Calculate entropy of profile - lower entropy means more concentrated

        Args:
            profile_mean: Mean of the Dirichlet distribution, shape [batch, pixels]

        Returns:
            Entropy loss (batch-wise)
        """
        # Reshape if needed
        if profile_mean.dim() == 4:  # [batch, D, H, W]
            profile_mean = profile_mean.view(profile_mean.size(0), -1)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        entropy = -(profile_mean * torch.log(profile_mean + eps)).sum(dim=1)

        return entropy

    def total_variation_3d(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Tensor of shape (batch, depth, height, width)
        Returns:
            tv_loss: Tensor of shape (batch,) with per-batch TV loss
        """
        batch_size = volume.shape[0]

        # Initialize per-batch TV loss
        batch_tv_loss = torch.zeros(batch_size, device=volume.device)

        for b in range(batch_size):
            # Get single example volume
            single_vol = volume[b : b + 1]

            # Calculate TV components for this example
            diff_depth = torch.abs(single_vol[:, 1:, :, :] - single_vol[:, :-1, :, :])
            diff_height = torch.abs(single_vol[:, :, 1:, :] - single_vol[:, :, :-1, :])
            diff_width = torch.abs(single_vol[:, :, :, 1:] - single_vol[:, :, :, :-1])

            # Sum for this example and store
            batch_tv_loss[b] = diff_depth.sum() + diff_height.sum() + diff_width.sum()

        return batch_tv_loss

    def _kl_sampling(self, q_dist, p_dist, num_samples=100):
        """
        Calculate KL divergence using sampling when analytic form is not available.

        Args:
            q_dist: The variational (posterior) distribution
            p_dist: The prior distribution
            num_samples: Number of samples for Monte Carlo estimation

        Returns:
            KL divergence estimate, preserving batch dimension
        """
        # Monte Carlo approximation: E_q[log q(x) - log p(x)]
        samples = q_dist.rsample((num_samples,))
        log_q = q_dist.log_prob(samples)
        log_p = p_dist.log_prob(samples)

        # Average across sample dimension only, preserve batch dimension
        return (log_q - log_p).mean(dim=0)

    def compute_kl(self, q_dist, p_dist):
        """Compute KL divergence between distributions, with fallback sampling if needed."""
        try:
            return torch.distributions.kl.kl_divergence(q_dist, p_dist)
        except NotImplementedError:
            samples = q_dist.rsample([100])
            log_q = q_dist.log_prob(samples)
            log_p = p_dist.log_prob(samples)
            return (log_q - log_p).mean(dim=0)

    def _ensure_batch_dim(self, tensor, batch_size, device):
        """
        Ensure tensor has batch dimension, broadcasting if needed.

        Args:
            tensor: Input tensor to check
            batch_size: Expected batch size
            device: Device for the tensor

        Returns:
            Tensor with batch dimension of size batch_size
        """
        if tensor.dim() == 0:  # Scalar tensor
            return tensor.expand(batch_size).to(device)

        if tensor.dim() == 1 and tensor.size(0) == batch_size:
            return tensor  # Already correct shape

        # If tensor has complex shape, reduce all but batch dimension
        if tensor.dim() > 1 and tensor.size(0) == batch_size:
            non_batch_dims = list(range(1, tensor.dim()))
            if non_batch_dims:
                return tensor.mean(dim=non_batch_dims)

        # Default case: something unexpected, return broadcasted tensor
        return tensor.expand(batch_size).to(device)

    def forward(self, rate, counts, q_p, q_I, q_bg, dead_pixel_mask):
        """
        Forward pass that preserves batch dimension until final reduction.
        """
        # Ensure all inputs are on the same device
        device = rate.device
        batch_size = rate.shape[0]

        # Initialize Dirichlet prior if needed
        if hasattr(self.p_p, "concentration"):
            self.p_p = torch.distributions.dirichlet.Dirichlet(
                self.dirichlet_concentration.to(device)
            )

        # Ensure other components are on the correct device
        counts = counts.to(device)
        dead_pixel_mask = dead_pixel_mask.to(device)

        # Calculate log likelihood
        ll = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))

        # Calculate all KL terms, keeping batch dimension intact
        kl_terms = torch.zeros(batch_size, device=device)

        # Only calculate profile KL if we have both the surrogate and prior distributions
        if self.p_p is not None:
            kl_p = self.compute_kl(q_p, self.p_p)
            # Ensure kl_p has batch dimension
            kl_p = self._ensure_batch_dim(kl_p, batch_size, device)
            kl_terms += kl_p * self.p_p_scale

        # Calculate background KL divergence
        kl_bg = self.compute_kl(q_bg, self.p_bg)
        kl_bg = self._ensure_batch_dim(kl_bg, batch_size, device)
        kl_terms += kl_bg * self.p_bg_scale

        # Calculate intensity KL divergence
        kl_I = self.compute_kl(q_I, self.p_I)
        kl_I = self._ensure_batch_dim(kl_I, batch_size, device)
        kl_terms += kl_I * self.p_I_scale

        # Initialize additional loss terms (per batch)
        profile_simpson_batch = torch.zeros(batch_size, device=device)
        tv_loss_batch = torch.zeros(batch_size, device=device)
        entropy_loss_batch = torch.zeros(batch_size, device=device)

        if (
            self.simpson_scale is not None
            and self.p_p.__class__.__name__ == "Dirichlet"
        ):
            # Profile simpson regularization (per batch)
            profile_simpson_batch = (
                self.inverse_simpson_regularization(q_p.mean.reshape(-1, 3, 21, 21))
                * self.simpson_scale
            )

        if self.tv_loss_scale is not None:
            # TV loss (per batch)
            tv_loss_batch = (
                self.total_variation_3d(q_p.mean.reshape(-1, 3, 21, 21))
                * self.tv_loss_scale
            )

        if self.entropy_scale is not None:
            # Entropy loss (per batch)
            entropy_loss = self.concentration_entropy_loss(q_p.mean)
            entropy_loss_batch = entropy_loss * self.entropy_scale

        # Calculate negative log likelihood (per batch)
        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)
        # Sum over pixels but keep batch dimension
        neg_ll_batch = -ll_mean.sum(dim=1)

        # Combine all loss terms (per batch)
        batch_loss = (
            neg_ll_batch
            + kl_terms
            + tv_loss_batch
            + profile_simpson_batch
            + entropy_loss_batch
        )

        # Now reduce to scalar for final loss
        total_loss = batch_loss.mean()

        # Match your original return format, but use batch-preserved values where possible
        return (
            total_loss,
            neg_ll_batch.mean(),
            kl_terms.mean(),
            kl_bg.mean(),
            kl_I.mean(),
            kl_p.mean() if self.p_p.__class__.__name__ == "Dirichlet" else 0.0,
            tv_loss_batch.mean(),
            profile_simpson_batch.mean(),
            entropy_loss_batch.mean(),
        )


class tempLoss(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        p_bg: torch.distributions.Distribution | None = None,
        p_I: torch.distributions.Distribution | None = None,
        p_p: torch.distributions.Distribution | None = None,
        p_p_scale: float = 0.0001,
        p_bg_scale: float = 0.0001,
        p_I_scale: float = 0.0001,
        simpson_scale: float | None = None,
        tv_loss_scale: float | None = None,
        entropy_scale: float | None = 0.5,
    ):
        super().__init__()

        # Don't specify device in __init__, let PyTorch handle it
        self.register_buffer("eps", torch.tensor(eps))
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
        if self.p_p.__class__.__name__ == "Dirichlet":
            self.register_buffer(
                "dirichlet_concentration", torch.ones(3 * 21 * 21) * 0.1
            )

        self.simpson_scale = simpson_scale
        self.tv_loss_scale = tv_loss_scale
        self.entropy_scale = entropy_scale

    def to(self, device):
        """
        Override to() to ensure all distribution parameters are moved to the correct device.
        """
        super().to(device)

        # Move prior distributions to device
        if self.p_bg is not None:
            self._move_distribution_to_device(self.p_bg, device)

        if self.p_I is not None:
            self._move_distribution_to_device(self.p_I, device)

        if self.p_p is not None:
            self._move_distribution_to_device(self.p_p, device)

        return self

    def _move_distribution_to_device(self, dist, device):
        """Helper method to move distribution parameters to the specified device"""
        # Check which type of distribution and move accordingly
        if isinstance(dist, torch.distributions.gamma.Gamma):
            dist.concentration = dist.concentration.to(device)
            dist.rate = dist.rate.to(device)
        elif isinstance(dist, torch.distributions.log_normal.LogNormal):
            dist.loc = dist.loc.to(device)
            dist.scale = dist.scale.to(device)
        elif isinstance(dist, torch.distributions.dirichlet.Dirichlet):
            dist.concentration = dist.concentration.to(device)
        elif isinstance(dist, torch.distributions.beta.Beta):
            dist.concentration1 = dist.concentration1.to(device)
            dist.concentration0 = dist.concentration0.to(device)
        elif isinstance(dist, torch.distributions.laplace.Laplace):
            dist.loc = dist.loc.to(device)
            dist.scale = dist.scale.to(device)

    def inverse_simpson_regularization(
        self, p: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Args:
            p: Tensor of shape (batch, depth, height, width)
        Returns:
            inv_simpson_reg: Scalar regularization term to minimize.
        """
        batch_size = p.shape[0]
        p_flat = p.view(batch_size, -1)  # Shape: (batch, num_components)
        simpson = torch.sum(p_flat**2, dim=1)  # Shape: (batch,)
        inv_simpson = 1.0 / (simpson + eps)  # Add eps to avoid division by zero
        return inv_simpson  # Return per-batch values, don't reduce yet

    def concentration_entropy_loss(self, profile_mean):
        """
        Calculate entropy of profile - lower entropy means more concentrated

        Args:
            profile_mean: Mean of the Dirichlet distribution, shape [batch, pixels]

        Returns:
            Entropy loss (batch-wise)
        """
        # Reshape if needed
        if profile_mean.dim() == 4:  # [batch, D, H, W]
            profile_mean = profile_mean.view(profile_mean.size(0), -1)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        entropy = -(profile_mean * torch.log(profile_mean + eps)).sum(dim=1)

        return entropy

    def total_variation_3d(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Tensor of shape (batch, depth, height, width)
        Returns:
            tv_loss: Tensor of shape (batch,) with per-batch TV loss
        """
        batch_size = volume.shape[0]

        # Initialize per-batch TV loss
        batch_tv_loss = torch.zeros(batch_size, device=volume.device)

        for b in range(batch_size):
            # Get single example volume
            single_vol = volume[b : b + 1]

            # Calculate TV components for this example
            diff_depth = torch.abs(single_vol[:, 1:, :, :] - single_vol[:, :-1, :, :])
            diff_height = torch.abs(single_vol[:, :, 1:, :] - single_vol[:, :, :-1, :])
            diff_width = torch.abs(single_vol[:, :, :, 1:] - single_vol[:, :, :, :-1])

            # Sum for this example and store
            batch_tv_loss[b] = diff_depth.sum() + diff_height.sum() + diff_width.sum()

        return batch_tv_loss

    def _kl_sampling(self, q_dist, p_dist, num_samples=100):
        """
        Calculate KL divergence using sampling when analytic form is not available.

        Args:
            q_dist: The variational (posterior) distribution
            p_dist: The prior distribution
            num_samples: Number of samples for Monte Carlo estimation

        Returns:
            KL divergence estimate, preserving batch dimension
        """
        # Monte Carlo approximation: E_q[log q(x) - log p(x)]
        samples = q_dist.rsample((num_samples,))
        log_q = q_dist.log_prob(samples)
        log_p = p_dist.log_prob(samples)

        # Average across sample dimension only, preserve batch dimension
        return (log_q - log_p).mean(dim=0)

    def compute_kl(self, q_dist, p_dist):
        """Compute KL divergence between distributions, with fallback sampling if needed."""
        try:
            return torch.distributions.kl.kl_divergence(q_dist, p_dist)
        except NotImplementedError:
            samples = q_dist.rsample([100])
            log_q = q_dist.log_prob(samples)
            log_p = p_dist.log_prob(samples)
            return (log_q - log_p).mean(dim=0)

    def _ensure_batch_dim(self, tensor, batch_size, device):
        """
        Ensure tensor has batch dimension, broadcasting if needed.

        Args:
            tensor: Input tensor to check
            batch_size: Expected batch size
            device: Device for the tensor

        Returns:
            Tensor with batch dimension of size batch_size
        """
        if tensor.dim() == 0:  # Scalar tensor
            return tensor.expand(batch_size).to(device)

        if tensor.dim() == 1 and tensor.size(0) == batch_size:
            return tensor  # Already correct shape

        # If tensor has complex shape, reduce all but batch dimension
        if tensor.dim() > 1 and tensor.size(0) == batch_size:
            non_batch_dims = list(range(1, tensor.dim()))
            if non_batch_dims:
                return tensor.mean(dim=non_batch_dims)

        # Default case: something unexpected, return broadcasted tensor
        return tensor.expand(batch_size).to(device)

    def forward(self, rate, counts, q_p, q_I, q_bg, dead_pixel_mask):
        """
        Forward pass that preserves batch dimension until final reduction.
        """
        # Ensure all inputs are on the same device
        device = rate.device
        batch_size = rate.shape[0]

        # Initialize Dirichlet prior if needed
        if hasattr(self.p_p, "concentration"):
            self.p_p = torch.distributions.dirichlet.Dirichlet(
                self.dirichlet_concentration.to(device)
            )

        # Ensure other components are on the correct device
        counts = counts.to(device)
        dead_pixel_mask = dead_pixel_mask.to(device)

        # Calculate log likelihood
        ll = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))

        # Calculate all KL terms, keeping batch dimension intact
        kl_terms = torch.zeros(batch_size, device=device)

        # Only calculate profile KL if we have both the surrogate and prior distributions
        if self.p_p is not None:
            kl_p = self.compute_kl(q_p, self.p_p)
            # Ensure kl_p has batch dimension
            kl_p = self._ensure_batch_dim(kl_p, batch_size, device)
            kl_terms += kl_p * self.p_p_scale

        # Calculate background KL divergence
        kl_bg = self.compute_kl(q_bg, self.p_bg)
        kl_bg = self._ensure_batch_dim(kl_bg, batch_size, device)
        kl_terms += kl_bg * self.p_bg_scale

        # Calculate intensity KL divergence
        kl_I = self.compute_kl(q_I, self.p_I)
        kl_I = self._ensure_batch_dim(kl_I, batch_size, device)
        kl_terms += kl_I * self.p_I_scale

        # Initialize additional loss terms (per batch)
        profile_simpson_batch = torch.zeros(batch_size, device=device)
        tv_loss_batch = torch.zeros(batch_size, device=device)
        entropy_loss_batch = torch.zeros(batch_size, device=device)

        if (
            self.simpson_scale is not None
            and self.p_p.__class__.__name__ == "Dirichlet"
        ):
            # Profile simpson regularization (per batch)
            profile_simpson_batch = (
                self.inverse_simpson_regularization(q_p.mean.reshape(-1, 3, 21, 21))
                * self.simpson_scale
            )

        if self.tv_loss_scale is not None:
            # TV loss (per batch)
            tv_loss_batch = (
                self.total_variation_3d(q_p.mean.reshape(-1, 3, 21, 21))
                * self.tv_loss_scale
            )

        if self.entropy_scale is not None:
            # Entropy loss (per batch)
            entropy_loss = self.concentration_entropy_loss(q_p.mean)
            entropy_loss_batch = entropy_loss * self.entropy_scale

        # Calculate negative log likelihood (per batch)
        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)
        # Sum over pixels but keep batch dimension
        neg_ll_batch = -ll_mean.sum(dim=1)

        # Combine all loss terms (per batch)
        batch_loss = (
            neg_ll_batch
            + kl_terms
            + tv_loss_batch
            + profile_simpson_batch
            + entropy_loss_batch
        )

        # Now reduce to scalar for final loss
        total_loss = batch_loss.mean()

        # Match your original return format, but use batch-preserved values where possible
        return (
            total_loss,
            neg_ll_batch.mean(),
            kl_terms.mean(),
            kl_bg.mean(),
            kl_I.mean(),
            kl_p.mean() if self.p_p.__class__.__name__ == "Dirichlet" else 0.0,
            tv_loss_batch.mean(),
            profile_simpson_batch.mean(),
            entropy_loss_batch.mean(),
        )
