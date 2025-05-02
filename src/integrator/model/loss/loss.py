import torch
from integrator.model.loss import BaseLoss
from rs_distributions import modules as rsm
import numpy as np


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
        # Profile prior
        p_p_name=None,  # Type: "dirichlet", "beta", or None
        p_p_params=None,  # Parameters for the distribution
        p_p_scale=0.0001,
        # Background prior
        p_bg_name="gamma",
        p_bg_params={"concentration": 1.0, "rate": 1.0},
        p_bg_scale=0.0001,
        # Intensity prior
        p_I_name="gamma",
        p_I_params={"concentration": 1.0, "rate": 1.0},
        p_I_scale=0.0001,
        # Optional regularization
        simpson_scale=None,
        tv_loss_scale=None,
        entropy_scale=None,
        # Center-focused prior parameters (for Dirichlet)
        use_center_focused_prior=True,
        prior_shape=(3, 21, 21),
        prior_base_alpha=5.0,
        prior_center_alpha=0.01,
        prior_decay_factor=0.5,
        prior_peak_percentage=0.05,
        prior_tensor=None,
    ):
        super().__init__()

        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("p_I_scale", torch.tensor(p_I_scale))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale))
        self.register_buffer("p_p_scale", torch.tensor(p_p_scale))

        # Store distribution names and params
        self.p_I_name = p_I_name
        self.p_I_params = p_I_params
        self.p_bg_name = p_bg_name
        self.p_bg_params = p_bg_params
        self.p_p_name = p_p_name
        self.p_p_params = p_p_params
        if prior_tensor is not None:
            self.concentration = torch.load(prior_tensor, weights_only=False)
            self.concentration[self.concentration > 2] *= 40
            self.concentration /= self.concentration.sum()
        else:
            self.concentration = torch.ones(1323) * p_p_params["concentration"]

        # Register parameters for I and bg distributions
        self._register_distribution_params(p_I_name, p_I_params, prefix="p_I_")
        self._register_distribution_params(p_bg_name, p_bg_params, prefix="p_bg_")

        # Number of elements in the profile
        self.profile_size = prior_shape[0] * prior_shape[1] * prior_shape[2]

        # Handle profile prior (p_p) - special handling for Dirichlet
        if p_p_name == "dirichlet":
            # Check if concentration is provided
            if p_p_params and "concentration" in p_p_params:
                # If concentration is provided, create uniform Dirichlet with that concentration
                alpha_vector = (
                    torch.ones(self.profile_size) * p_p_params["concentration"]
                )
                self.register_buffer("dirichlet_concentration", alpha_vector)
            elif use_center_focused_prior:
                # Create center-focused Dirichlet prior
                alpha_vector = create_center_focused_dirichlet_prior(
                    shape=prior_shape,
                    base_alpha=prior_base_alpha,
                    center_alpha=prior_center_alpha,
                    decay_factor=prior_decay_factor,
                    peak_percentage=prior_peak_percentage,
                )
                self.register_buffer("dirichlet_concentration", alpha_vector)
            else:
                # Default uniform Dirichlet with concentration=1.0
                alpha_vector = torch.ones(self.profile_size)
                self.register_buffer("dirichlet_concentration", alpha_vector)
        elif p_p_name is not None:
            # Register parameters for other distribution types
            self._register_distribution_params(p_p_name, p_p_params, prefix="p_p_")

        # Optional regularization parameters
        self.simpson_scale = simpson_scale
        self.tv_loss_scale = tv_loss_scale
        self.entropy_scale = entropy_scale

        # Store shape for profile reshaping
        self.prior_shape = prior_shape

    def _register_distribution_params(self, name, params, prefix):
        """Register distribution parameters as buffers with appropriate prefixes"""
        if name is None or params is None:
            return

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
        elif name == "beta":
            self.register_buffer(
                f"{prefix}concentration1", torch.tensor(params["concentration1"])
            )
            self.register_buffer(
                f"{prefix}concentration0", torch.tensor(params["concentration0"])
            )
        elif name == "half_normal":
            self.register_buffer(f"{prefix}scale", torch.tensor(params["scale"]))
        elif name == "laplace":
            self.register_buffer(f"{prefix}loc", torch.tensor(params["loc"]))
            self.register_buffer(f"{prefix}scale", torch.tensor(params["scale"]))

    def get_prior(self, name, params_prefix, device, default_return=None):
        """Create a distribution on the specified device"""
        if name is None:
            return default_return

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
        elif name == "half_normal":
            scale = getattr(self, f"{params_prefix}scale").to(device)
            return torch.distributions.half_normal.HalfNormal(scale=scale)

        elif name == "beta":
            concentration1 = getattr(self, f"{params_prefix}concentration1").to(device)
            concentration0 = getattr(self, f"{params_prefix}concentration0").to(device)
            return torch.distributions.beta.Beta(
                concentration1=concentration1, concentration0=concentration0
            )
        elif name == "laplace":
            loc = getattr(self, f"{params_prefix}loc").to(device)
            scale = getattr(self, f"{params_prefix}scale").to(device)
            return torch.distributions.laplace.Laplace(loc=loc, scale=scale)
        elif name == "dirichlet":
            # For Dirichlet, use the dirichlet_concentration buffer
            if hasattr(self, "dirichlet_concentration"):
                return torch.distributions.dirichlet.Dirichlet(
                    self.dirichlet_concentration.to(device)
                )

        # Default case: return None or provided default
        return default_return

    def inverse_simpson_regularization(self, p, eps=1e-6):
        batch_size = p.shape[0]
        p_flat = p.view(batch_size, -1)
        simpson = torch.sum(p_flat**2, dim=1)
        inv_simpson = 1.0 / (simpson + eps)
        return inv_simpson

    def concentration_entropy_loss(self, profile_mean):
        """Calculate entropy of profile - lower entropy means more concentrated"""
        # Reshape if needed
        if profile_mean.dim() == 4:  # [batch, D, H, W]
            profile_mean = profile_mean.view(profile_mean.size(0), -1)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        entropy = -(profile_mean * torch.log(profile_mean + eps)).sum(dim=1)
        return entropy

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
        """Ensure tensor has batch dimension, broadcasting if needed."""
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
        # Get device and batch size
        device = rate.device
        batch_size = rate.shape[0]

        # Ensure inputs are on the correct device
        counts = counts.to(device)
        dead_pixel_mask = dead_pixel_mask.to(device)

        # Create distributions on the correct device
        p_I = self.get_prior(self.p_I_name, "p_I_", device)
        p_bg = self.get_prior(self.p_bg_name, "p_bg_", device)
        p_p = torch.distributions.dirichlet.Dirichlet(self.concentration.to(device))

        # Calculate log likelihood
        ll = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))

        # Calculate KL terms
        kl_terms = torch.zeros(batch_size, device=device)
        kl_p = torch.tensor(0.0, device=device)  # Default value

        # Only calculate profile KL if we have both distributions
        kl_p = self.compute_kl(q_p, p_p)
        kl_terms += kl_p * self.p_p_scale

        # Calculate background and intensity KL divergence
        kl_bg = self.compute_kl(q_bg, p_bg)
        kl_terms += kl_bg * self.p_bg_scale

        kl_I = self.compute_kl(q_I, p_I)
        kl_terms += kl_I * self.p_I_scale

        # Initialize regularization terms
        profile_simpson_batch = torch.zeros(batch_size, device=device)
        tv_loss_batch = torch.zeros(batch_size, device=device)
        entropy_loss_batch = torch.zeros(batch_size, device=device)

        # Apply regularization if profile exists and has mean attribute
        if q_p is not None and hasattr(q_p, "mean"):
            # Reshape profile if needed
            profile_shape = (-1,) + self.prior_shape

            if self.simpson_scale is not None:
                profile_reshaped = q_p.mean.reshape(profile_shape)
                profile_simpson_batch = (
                    self.inverse_simpson_regularization(profile_reshaped)
                    * self.simpson_scale
                )

            if self.tv_loss_scale is not None:
                profile_reshaped = q_p.mean.reshape(profile_shape)
                tv_loss_batch = (
                    self.total_variation_3d(profile_reshaped) * self.tv_loss_scale
                )

            if self.entropy_scale is not None:
                entropy_loss = self.concentration_entropy_loss(q_p.mean)
                entropy_loss_batch = entropy_loss * self.entropy_scale

        # Calculate negative log likelihood
        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)
        neg_ll_batch = -ll_mean.sum(dim=1)

        # Combine all loss terms
        batch_loss = (
            neg_ll_batch
            + kl_terms
            + tv_loss_batch
            + profile_simpson_batch
            + entropy_loss_batch
        )

        # Final scalar loss
        total_loss = batch_loss.mean()

        # Return all components for monitoring
        return (
            total_loss,
            neg_ll_batch.mean(),
            kl_terms.mean(),
            kl_bg.mean(),
            kl_I.mean(),
            kl_p.mean() if p_p is not None else torch.tensor(0.0, device=device),
            tv_loss_batch.mean(),
            profile_simpson_batch.mean(),
            entropy_loss_batch.mean(),
        )
