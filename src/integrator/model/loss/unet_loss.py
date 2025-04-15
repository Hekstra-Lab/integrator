import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt


def create_center_focused_dirichlet_prior(
    shape=(3, 21, 21),
    base_alpha=0.1,  # outer region
    center_alpha=100.0,  # high alpha at the center => center gets more mass
    decay_factor=1,
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
        # Intensity prior
        use_center_focused_prior=True,
        prior_shape=(3, 21, 21),
        prior_base_alpha=0.1,
        prior_center_alpha=5.0,
        prior_decay_factor=0.2,
        prior_peak_percentage=0.05,
        p_I_name="gamma",
        p_I_params={"concentration": 1.0, "rate": 1.0},
        p_I_scale=0.001,
    ):
        super().__init__()

        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale))
        self.register_buffer("p_p_scale", torch.tensor(p_p_scale))

        # Store distribution names and params
        self.p_bg_name = p_bg_name
        self.p_bg_params = p_bg_params
        self.p_p_name = p_p_name
        self.p_p_params = p_p_params

        # Register parameters for bg distribution
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

        elif name == "half_normal":
            self.register_buffer(f"{prefix}scale", torch.tensor(params["scale"]))

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
        elif name == "half_normal":
            scale = getattr(self, f"{params_prefix}scale").to(device)
            return torch.distributions.half_normal.HalfNormal(scale=scale)

        elif name == "log_normal":
            loc = getattr(self, f"{params_prefix}loc").to(device)
            scale = getattr(self, f"{params_prefix}scale").to(device)
            return torch.distributions.log_normal.LogNormal(loc=loc, scale=scale)
        elif name == "exponential":
            rate = getattr(self, f"{params_prefix}rate").to(device)
            return torch.distributions.exponential.Exponential(rate=rate)
        elif name == "dirichlet":
            # For Dirichlet, use the dirichlet_concentration buffer
            if hasattr(self, "dirichlet_concentration"):
                concentration = self.dirichlet_concentration.to(device)
                # Get batch size from the q distribution (you'll need to pass it in)
                if hasattr(self, "current_batch_size") and self.current_batch_size > 1:
                    # Create a batch of identical priors
                    # This still constrains all samples to the same prior shape
                    # but allows proper batch-wise KL calculation
                    concentration = concentration.unsqueeze(0).expand(
                        self.current_batch_size, -1
                    )
                return torch.distributions.dirichlet.Dirichlet(concentration)

        return default_return

    def forward(self, rate, counts, q_p, q_bg, masks):
        # Get device and batch size
        device = rate.device
        batch_size = rate.shape[0]
        self.current_batch_size = batch_size

        # Ensure inputs are on the correct device
        counts = counts.to(device)
        masks = masks.to(device)

        # Create distributions on the correct device
        p_bg = self.get_prior(self.p_bg_name, "p_bg_", device)
        p_p = self.get_prior(self.p_p_name, "p_p_", device)

        # Calculate KL terms
        kl_terms = torch.zeros(batch_size, device=device)
        kl_p = torch.tensor(0.0, device=device)  # Default value

        # Only calculate profile KL if we have both distributions
        kl_p = torch.distributions.kl.kl_divergence(q_p, p_p)
        kl_terms += kl_p * self.p_p_scale

        # Calculate background and intensity KL divergence
        kl_bg = torch.distributions.kl.kl_divergence(q_bg, p_bg)
        kl_bg = kl_bg.sum(-1)
        kl_terms += kl_bg * self.p_bg_scale

        ll = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * masks.squeeze(-1)
        # ll_mean = (
        # (
        # torch.distributions.Poisson(rate).log_prob(counts.unsqueeze(1))
        # * masks.unsqueeze(1)
        # ).mean(1)
        # ).sum(1) / masks.sum(1)

        # Calculate negative log likelihood
        neg_ll_batch = (-ll_mean).sum(1)

        # Combine all loss terms
        batch_loss = neg_ll_batch + kl_terms

        # Final scalar loss
        total_loss = batch_loss.mean()

        # Return all components for monitoring
        return (
            total_loss,
            neg_ll_batch.mean(),
            kl_terms.mean(),
            kl_bg.mean() * self.p_bg_scale,
            kl_p.mean() * self.p_p_scale,
        )


if __name__ == "__main__":
    plt.imshow(
        torch.distributions.dirichlet.Dirichlet(
            create_center_focused_dirichlet_prior(peak_percentage=0.08)
        ).concentration.reshape(3, 21, 21)[1]
    )
    plt.colorbar()
    plt.show()
