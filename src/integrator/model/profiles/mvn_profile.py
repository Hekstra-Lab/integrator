import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.transforms import SoftplusTransform
from integrator.layers import Linear
from rs_distributions.transforms import FillScaleTriL


class tempMVNProfile(nn.Module):
    def __init__(self, dmodel, image_shape):
        super().__init__()
        self.dmodel = dmodel

        # Use your custom FillScaleTriL
        self.L_transform = FillScaleTriL()

        # Create scale and mean prediction layers
        self.scale_layer = Linear(self.dmodel, 6)  # for a 3×3 TriL, we need 6 params
        self.mean_layer = Linear(self.dmodel, 3)

        # Build coordinate grid for shape (d, h, w)
        d, h, w = image_shape
        z_coords = torch.arange(d).float() - (d - 1) / 2
        y_coords = torch.arange(h).float() - (h - 1) / 2
        x_coords = torch.arange(w).float() - (w - 1) / 2
        z_coords = z_coords.view(d, 1, 1).expand(d, h, w)
        y_coords = y_coords.view(1, h, 1).expand(d, h, w)
        x_coords = x_coords.view(1, 1, w).expand(d, h, w)

        pixel_positions = torch.stack([x_coords, y_coords, z_coords], dim=-1)
        pixel_positions = pixel_positions.view(-1, 3)

        # Register pixel_positions for device management
        self.register_buffer("pixel_positions", pixel_positions)

    def forward(self, representation):
        """
        representation: [batch_size, dmodel]

        Returns:
            means: [batch_size, 3]
            L:     [batch_size, 3, 3]   (cholesky factor)
            pixel_positions: [num_pixels, 3]
        """
        batch_size = representation.size(0)

        # 1) Predict means => shape [batch_size, 3]
        means = self.mean_layer(representation)  # [b, 3]

        # 2) Predict scale params => shape [batch_size, 6]
        scales_raw = self.scale_layer(representation)  # [b, 6]
        # For positivity, you might do e.g. elu + 1
        scales_positivized = torch.nn.functional.elu(scales_raw) + 1.0

        # 3) Transform to lower-triangular L => [b, 3, 3]
        L = self.L_transform(scales_positivized)

        # return means, L, self.pixel_positions
        return means, L, self.pixel_positions


###############################################################################
# SignalAwareMVNProfile: blend predicted covariance with an isotropic fallback
###############################################################################
class SignalAwareMVNProfile(nn.Module):
    """
    Penalize low signal_prob by 'pulling' covariance toward an isotropic fallback.
    """

    def __init__(self, dmodel, image_shape=(3, 21, 21), base_sigma=4.0, power=3.0):
        super().__init__()
        self.profile_model = MVNProfile(dmodel, image_shape=image_shape)
        self.base_sigma = base_sigma
        self.power = power
        self.log_base_sigma = nn.Parameter(torch.log(torch.tensor(base_sigma)))

    def forward(self, representation, signal_prob):
        """
        representation: [batch_size, dmodel]
        signal_prob:    [batch_size], in [0..1]

        Returns:
            profile: [batch_size, num_pixels]
        """
        batch_size = representation.size(0)

        # 1) Get predicted means, L, and pixel_positions
        means, L, pixel_positions = self.profile_model(representation)
        # shapes:
        #   means            => [batch_size, 3]
        #   L                => [batch_size, 3, 3]
        #   pixel_positions  => [num_pixels, 3]

        # 2) predicted_cov = L @ L^T => shape [b, 3, 3]
        predicted_cov = torch.bmm(L, L.transpose(-1, -2))

        # 3) Fallback isotropic = base_sigma^2 * I
        eye_3 = torch.eye(3, device=L.device, dtype=L.dtype)
        base_sigma = torch.exp(self.log_base_sigma)
        fallback_cov = eye_3.unsqueeze(0) * (base_sigma**2)  # shape [1, 3, 3]
        fallback_cov = fallback_cov.expand(batch_size, -1, -1)  # shape [b, 3, 3]

        # 4) alpha = (signal_prob^power)
        alpha = signal_prob.clamp(min=0.0, max=1.0).pow(self.power)  # [b]
        alpha = alpha.view(batch_size, 1, 1)  # for broadcasting

        # 5) Blend covariances => [b, 3, 3]
        blended_cov = alpha * predicted_cov + (1 - alpha) * fallback_cov

        ######################################################################
        # 6) Compute log_prob for each batch item in a loop, to avoid broadcast issues:
        #
        #    We have a separate MVN for each i in [0..batch_size-1].
        #    Each one sees pixel_positions of shape [num_pixels, 3].
        ######################################################################
        profiles = []
        for i in range(batch_size):
            mvn_i = MultivariateNormal(loc=means[i], covariance_matrix=blended_cov[i])
            # Evaluate log_prob at each pixel => shape [num_pixels]
            log_probs_i = mvn_i.log_prob(pixel_positions)

            # Exponentiate stably
            log_probs_i_stable = log_probs_i - log_probs_i.max()
            profile_i = torch.exp(log_probs_i_stable)

            # Normalize
            profile_i = profile_i / (profile_i.sum() + 1e-10)

            profiles.append(profile_i)

        # stack => shape [batch_size, num_pixels]
        profiles = torch.stack(profiles, dim=0)
        return profiles


class MVNProfile(torch.nn.Module):
    def __init__(self, dmodel, image_shape):
        super().__init__()
        self.dmodel = dmodel

        # Use different transformation for more flexible scale learning
        self.L_transform = FillScaleTriL(diag_transform=SoftplusTransform())

        # Create scale and mean prediction layers
        self.scale_layer = Linear(self.dmodel, 6)
        self.mean_layer = Linear(self.dmodel, 3)

        # Calculate image dimensions
        d, h, w = image_shape
        self.image_shape = image_shape

        # Create centered coordinate grid
        z_coords = torch.arange(d).float() - (d - 1) / 2
        y_coords = torch.arange(h).float() - (h - 1) / 2
        x_coords = torch.arange(w).float() - (w - 1) / 2

        z_coords = z_coords.view(d, 1, 1).expand(d, h, w)
        y_coords = y_coords.view(1, h, 1).expand(d, h, w)
        x_coords = x_coords.view(1, 1, w).expand(d, h, w)

        # Stack coordinates
        pixel_positions = torch.stack([x_coords, y_coords, z_coords], dim=-1)
        pixel_positions = pixel_positions.view(-1, 3)

        # Register buffer
        self.register_buffer("pixel_positions", pixel_positions)

        # Create a default scale parameter for initialization guidance
        self.register_buffer(
            "scale_init", torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]).view(1, 1, 6)
        )

    def forward(self, representation):
        batch_size = representation.size(0)

        # Predict mean offsets
        means = self.mean_layer(representation).view(batch_size, 1, 3)

        # Predict scale parameters - use ELU+1 for positive values with better gradient properties
        scales_raw = self.scale_layer(representation).view(batch_size, 1, 6)

        # Instead of sigmoid, use ELU+1 which has a better gradient flow and unbounded upper range
        # This allows the model to learn scales more freely
        scales = torch.nn.functional.elu(scales_raw) + 1.0

        # Transform scales
        L = self.L_transform(scales).to(torch.float32)

        # Create MVN distribution
        mvn = MultivariateNormal(means, scale_tril=L)

        # Compute log probabilities
        pixel_positions = self.pixel_positions.unsqueeze(0).expand(batch_size, -1, -1)
        log_probs = mvn.log_prob(pixel_positions)

        # Convert to probabilities
        # Subtract max for numerical stability (prevents overflow)
        log_probs_stable = log_probs - log_probs.max(dim=1, keepdim=True)[0]
        profile = torch.exp(log_probs_stable)

        # Normalize to sum to 1
        profile = profile / (profile.sum(dim=1, keepdim=True) + 1e-10)

        return profile


class tempSignalAwareMVNProfile(nn.Module):
    def __init__(self, dmodel, image_shape=(3, 21, 21), base_sigma=5.0, power=3.0):
        super().__init__()
        self.profile_model = MVNProfile(dmodel, image_shape=image_shape)
        self.base_sigma = base_sigma  # Higher base sigma for more spread
        self.power = power  # Power for non-linear response

    def forward(self, representation, signal_prob):
        batch_size = representation.size(0)

        # Make response more aggressive with power function
        # signal_prob=1 → alpha=1
        # signal_prob=0.5 → alpha=0.125 (with power=3)
        # signal_prob=0 → alpha=0
        alpha = torch.pow(signal_prob.clamp(0, 1), self.power)
        alpha = alpha.view(batch_size, 1, 1)

        # Calculate means and scales as before
        means = self.profile_model.mean_layer(representation).view(batch_size, 1, 3)
        scales_raw = self.profile_model.scale_layer(representation).view(
            batch_size, 1, 6
        )
        scales = torch.nn.functional.elu(scales_raw) + 1.0

        # Create isotropic scales with high variance for low probabilities
        isotropic_scales = torch.zeros_like(scales)
        isotropic_scales[..., 0] = self.base_sigma  # L[0,0]
        isotropic_scales[..., 2] = self.base_sigma  # L[1,1]
        isotropic_scales[..., 5] = self.base_sigma  # L[2,2]

        # Blend scales with aggressive response
        blended_scales = alpha * scales + (1 - alpha) * isotropic_scales

        # Process each item in batch to avoid shape mismatches
        profiles = []
        for i in range(batch_size):
            mean_i = means[i].squeeze(0)
            scale_i = blended_scales[i].squeeze(0)
            L_i = self.profile_model.L_transform(scale_i.unsqueeze(0)).squeeze(0)
            mvn_i = MultivariateNormal(mean_i, scale_tril=L_i)

            pixel_positions = self.profile_model.pixel_positions
            log_probs_i = mvn_i.log_prob(pixel_positions)
            log_probs_i_stable = log_probs_i - log_probs_i.max()
            profile_i = torch.exp(log_probs_i_stable)
            profile_i = profile_i / (profile_i.sum() + 1e-10)

            profiles.append(profile_i)

        profiles = torch.stack(profiles)
        return profiles


if __name__ == "__main__":
    profile_model = MVNProfile(dmodel=64, image_shape=(3, 21, 21))

    # Create a batch of representations (assuming 10 sample with 64-dimensional representation)
    representation = torch.randn(10, 64)

    profile = profile_model(representation)
    # The output should have shape [1, 3*21*21] = [1, 1323]

    expanded = profile.unsqueeze_(1).expand(-1, 100, -1)
