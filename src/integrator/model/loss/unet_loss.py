import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt


def create_center_focused_dirichlet_prior(
    shape=(3, 21, 21),
    base_alpha=0.001,  # outer region
    center_alpha=50.0,  # high alpha at the center => center gets more mass
    decay_factor=0.2,
    peak_percentage=0.1,
):
    channels, height, width = shape
    alpha_3d = np.ones(shape) * base_alpha

    center_c = channels // 2
    center_h = height // 2
    center_w = width // 2

    for c in range(channels):
        for h in range(height):
            for w in range(width):
                # Normalized distance from center
                dist_c = abs(c - center_c) / (channels / 2)
                dist_h = abs(h - center_h) / (height / 2)
                dist_w = abs(w - center_w) / (width / 2)
                distance = np.sqrt(dist_c**2 + dist_h**2 + dist_w**2) / np.sqrt(3)

                if distance < peak_percentage * 5:
                    alpha_value = (
                        center_alpha
                        - (center_alpha - base_alpha)
                        * (distance / (peak_percentage * 5)) ** decay_factor
                    )
                    alpha_3d[c, h, w] = alpha_value

    alpha_vector = torch.tensor(alpha_3d.flatten(), dtype=torch.float32)
    return alpha_vector


# %%
class UnetLoss(torch.nn.Module):
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
        use_center_focused_prior=True,
        prior_shape=(3, 21, 21),
        prior_base_alpha=0.001,
        prior_center_alpha=50.0,
        prior_decay_factor=0.2,
        prior_peak_percentage=0.1,
        p_I_name=None,
        p_I_params=None,
        p_I_scale=0.0001,
    ):
        super().__init__()

        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale))
        self.register_buffer("p_p_scale", torch.tensor(p_p_scale))
        self.register_buffer("p_I_scale", torch.tensor(p_I_scale))

        # Store distribution names and params
        self.p_bg_name = p_bg_name
        self.p_bg_params = p_bg_params
        self.p_p_name = p_p_name
        self.p_p_params = p_p_params

        # Register parameters for I and bg distributions
        self._register_distribution_params(p_bg_name, p_bg_params, prefix="p_bg_")

        if p_I_name is not None:
            self.p_I_name = p_I_name
            self._register_distribution_params(p_I_name, p_I_params, prefix="p_I_")
        else:
            pass

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

        # Store shape for profile reshaping
        self.prior_shape = prior_shape
        self.tv_scale = 0.1

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

    def total_variation_loss(self, alpha):
        """
        alpha: shape (N, C, H, W)
        Returns a scalar that is the total variation penalty (anisotropic TV).
        """
        # Shifted differences in the horizontal (x) direction
        diff_x = alpha[:, :, :, 1:] - alpha[:, :, :, :-1]
        # Shifted differences in the vertical (y) direction
        diff_y = alpha[:, :, 1:, :] - alpha[:, :, :-1, :]

        # L1 norm of these differences
        tv_x = diff_x.abs().sum()
        tv_y = diff_y.abs().sum()

        # total TV is sum of horizontal and vertical
        tv = tv_x + tv_y
        return tv

    def compute_kl(self, q_dist, p_dist):
        """Compute KL divergence between distributions with more robust fallback."""

        if q_dist is None or p_dist is None:
            return torch.tensor(0.0, device=self.eps.device)

        try:
            # Try analytical KL first
            return torch.distributions.kl.kl_divergence(q_dist, p_dist)
        except NotImplementedError:
            # More stable sampling approach with more samples
            num_samples = 1000  # Increase sample count for stability
            samples = q_dist.rsample([num_samples])
            log_q = q_dist.log_prob(samples)
            log_p = p_dist.log_prob(samples)
            return (log_q - log_p).mean(dim=0)
        except Exception as e:
            # Log the error for debugging but don't silence it completely
            print(f"Error in KL computation: {e}")
            # Return a small value but consider raising a warning
            return torch.tensor(0.0, device=self.eps.device)

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

    def forward(self, rate, counts, q_p, q_bg, masks, q_I=None):
        # Get device and batch size
        device = rate.device
        batch_size = rate.shape[0]

        # Ensure inputs are on the correct device
        counts = counts.to(device)
        masks = masks.to(device)

        # Create distributions on the correct device
        p_bg = self.get_prior(self.p_bg_name, "p_bg_", device)
        p_p = self.get_prior(self.p_p_name, "p_p_", device)

        if q_I is not None:
            p_I = self.get_prior(self.p_I_name, "p_I_", device)
        else:
            p_I = None

        # Calculate KL terms
        kl_terms = torch.zeros(batch_size, device=device)
        kl_p = torch.tensor(0.0, device=device)  # Default value

        # Only calculate profile KL if we have both distributions
        if p_p is not None and q_p is not None:
            kl_p = self.compute_kl(q_p, p_p)
            # kl_p = self._ensure_batch_dim(kl_p, batch_size, device)
            kl_terms += kl_p * self.p_p_scale

        if p_I is not None:
            kl_I = self.compute_kl(q_I, p_I)
            kl_terms += kl_I * self.p_I_scale

        # Calculate background and intensity KL divergence
        kl_bg = self.compute_kl(q_bg, p_bg)
        # kl_bg = self._ensure_batch_dim(kl_bg, batch_size, device)
        kl_terms += kl_bg * self.p_bg_scale

        # Initialize regularization terms
        # Apply regularization if profile exists and has mean attribute
        if q_p is not None and hasattr(q_p, "mean"):
            # Reshape profile if needed
            profile_shape = (-1,) + self.prior_shape

        ll_mean = (
            (
                torch.distributions.Poisson(rate).log_prob(counts.unsqueeze(1))
                * masks.unsqueeze(1)
            ).mean(1)
        ).sum(1) / masks.sum(1)

        # Calculate negative log likelihood
        neg_ll_batch = (-ll_mean).sum()

        tv_loss = (
            self.total_variation_loss(q_p.mean.view(-1, 3, 21, 21)) * self.tv_scale
        )

        # Combine all loss terms
        batch_loss = neg_ll_batch + kl_terms
        total_loss = batch_loss.mean() + tv_loss

        # Return all components for monitoring
        return (
            total_loss,
            neg_ll_batch.mean(),
            kl_terms.mean(),
            kl_bg.mean(),
            kl_p.mean() if p_p is not None else torch.tensor(0.0, device=device),
            tv_loss
            # kl_I.mean() if p_I is not None else torch.tensor(0.0, device=device),
        )


if __name__ == "__main__":
    plt.imshow(
        torch.distributions.dirichlet.Dirichlet(
            create_center_focused_dirichlet_prior()
        ).mean.reshape(3, 21, 21)[1]
    )
    plt.colorbar()
    plt.show()
