import torch
import numpy as np
import torch.nn as nn
from integrator.model.integrators import BaseIntegrator
from integrator.layers import Linear
from integrator.model.encoders import UNetDirichletConcentration
import torch.nn.functional as F
from integrator.model.loss import BernoulliLoss
from integrator.model.decoders import BernoulliDecoder

import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np
import torch


class StatisticalSignificanceFilter(nn.Module):
    def __init__(self, initial_alpha=0.05, steepness=2.0):
        """
        initial_alpha: Initial significance level (e.g., 0.05 for 5% significance)
        steepness: Controls how sharply the sigmoid transitions around the threshold
        """
        super().__init__()
        # Learnable significance level
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        self.steepness = steepness

    def forward(self, x, mask=None):
        """
        x: Tensor of standardized data (assumed mean 0, std 1)
        mask: Optional mask to restrict the operation to certain pixels/regions
        """
        if mask is not None:
            x = x * mask

        # Clamp alpha to a reasonable range to avoid numerical issues.
        alpha = torch.clamp(self.alpha, min=0.001, max=0.1)
        # Compute the two-tailed z-score threshold:
        # For a significance level alpha, the threshold is:
        #    z = norm.ppf(1 - alpha/2)
        # and using the relation: norm.ppf(p) = sqrt(2)*erfinv(2*p - 1)
        p = 1 - alpha / 2
        z_score = math.sqrt(2) * torch.erfinv(2 * p - 1)

        # Soft thresholding using a sigmoid: values well below z_score get suppressed,
        # values above z_score are preserved.
        significance_weight = torch.sigmoid(self.steepness * (torch.abs(x) - z_score))
        denoised = x * significance_weight

        if mask is not None:
            denoised = denoised * mask

        return denoised


class ResidualDenoiser(nn.Module):
    def __init__(self, in_channels=1, num_filters=16, num_layers=3):
        """
        A simple residual denoiser that preserves the input shape.

        Args:
            in_channels (int): Number of channels in the input.
            num_filters (int): Number of filters for the intermediate layers.
            num_layers (int): Total number of convolutional layers.
                               Must be at least 2.
        """
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2."

        # First layer: project from in_channels to num_filters.
        self.conv1 = nn.Conv3d(
            in_channels, num_filters, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.relu = nn.ReLU(inplace=True)

        # Intermediate layers.
        self.convs = nn.ModuleList(
            [
                nn.Conv3d(
                    num_filters,
                    num_filters,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                )
                for _ in range(num_layers - 2)
            ]
        )

        # Final layer: project back to in_channels.
        self.conv_last = nn.Conv3d(
            num_filters, in_channels, kernel_size=3, padding=1, padding_mode="reflect"
        )

    def forward(self, x, mask=None):
        residual = x  # Save input for residual connection

        # Apply mask if provided
        if mask is not None:
            x = x * mask

        out = self.relu(self.conv1(x))
        for conv in self.convs:
            out = self.relu(conv(out))
        out = self.conv_last(out)
        # Add residual to ensure output shape is same as input.

        # Apply mask again
        if mask is not None:
            out = out * mask

        return out + residual


class AnisotropicDiffusionPreprocessor(nn.Module):
    def __init__(self, num_iter=5, kappa=50, gamma=0.1):
        super().__init__()
        self.num_iter = num_iter
        self.kappa = nn.Parameter(torch.tensor(float(kappa)))  # Edge sensitivity
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))  # Step size

    def forward(self, x, mask=None):
        # Apply mask if provided
        if mask is not None:
            x = x * mask

        filtered = x.clone()
        kappa = F.softplus(self.kappa)  # Ensure positive
        gamma = torch.sigmoid(self.gamma) * 0.2  # Limit step size

        for _ in range(self.num_iter):
            # Calculate gradients in 6 directions
            nabla_pos_z = F.pad(
                filtered[:, :, 1:] - filtered[:, :, :-1], (0, 0, 0, 0, 1, 0)
            )
            nabla_neg_z = F.pad(
                filtered[:, :, :-1] - filtered[:, :, 1:], (0, 0, 0, 0, 0, 1)
            )
            nabla_pos_y = F.pad(
                filtered[:, :, :, 1:] - filtered[:, :, :, :-1], (0, 0, 1, 0)
            )
            nabla_neg_y = F.pad(
                filtered[:, :, :, :-1] - filtered[:, :, :, 1:], (0, 0, 0, 1)
            )
            nabla_pos_x = F.pad(
                filtered[:, :, :, :, 1:] - filtered[:, :, :, :, :-1], (1, 0)
            )
            nabla_neg_x = F.pad(
                filtered[:, :, :, :, :-1] - filtered[:, :, :, :, 1:], (0, 1)
            )

            # Calculate diffusion coefficients (preserve edges)
            c_pos_z = torch.exp(-((nabla_pos_z / kappa) ** 2))
            c_neg_z = torch.exp(-((nabla_neg_z / kappa) ** 2))
            c_pos_y = torch.exp(-((nabla_pos_y / kappa) ** 2))
            c_neg_y = torch.exp(-((nabla_neg_y / kappa) ** 2))
            c_pos_x = torch.exp(-((nabla_pos_x / kappa) ** 2))
            c_neg_x = torch.exp(-((nabla_neg_x / kappa) ** 2))

            # Update image
            update = (
                c_pos_z * nabla_pos_z
                + c_neg_z * nabla_neg_z
                + c_pos_y * nabla_pos_y
                + c_neg_y * nabla_neg_y
                + c_pos_x * nabla_pos_x
                + c_neg_x * nabla_neg_x
            )

            filtered = filtered + gamma * update

        # Apply mask again
        if mask is not None:
            filtered = filtered * mask

        return filtered


class PoissonDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoise_strength = nn.Parameter(torch.tensor(1.0))

    def _create_gaussian_kernel(self, kernel_size, sigma):
        # Create coordinate grids
        coords = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        x, y, z = torch.meshgrid(coords, coords, coords)

        # Calculate Gaussian
        kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        # Reshape for Conv3D (channels=1, preserving groups in convolution)
        return kernel.view(1, 1, kernel_size, kernel_size, kernel_size)

    def forward(self, x, mask=None):
        # Apply mask
        if mask is not None:
            x = x * mask

        # Anscombe transform (variance stabilizing for Poisson)
        transformed = 2 * torch.sqrt(x + 3 / 8)

        # Apply Gaussian denoising to stabilized data
        denoise_strength = torch.sigmoid(self.denoise_strength) * 2.0
        padding = 2
        kernel_size = 5

        # Create Gaussian kernel
        sigma = denoise_strength
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(x.device)

        # Apply filtering
        filtered = F.conv3d(
            transformed.view(-1, 1, *transformed.shape[2:]), kernel, padding=padding
        )

        # Inverse Anscombe transform
        filtered = filtered.view_as(transformed)
        denoised = (filtered / 2) ** 2 - 0.1  # Simplified inverse
        denoised = torch.clamp(denoised, min=0)

        # Apply mask again
        if mask is not None:
            denoised = denoised * mask

        return denoised


class SignalPreprocessor(nn.Module):
    def __init__(self, in_channels=1, Z=3, H=21, W=21):
        super().__init__()
        self.Z, self.H, self.W = Z, H, W

        # Create center-focused weight mask
        center_mask = self._create_center_focus_mask(Z, H, W)
        self.register_buffer("center_mask", center_mask)

        # Adaptive Gaussian blur with learnable sigma
        self.init_sigma = nn.Parameter(torch.tensor(0.8))

        # Feature extraction with smaller kernel
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1, padding_mode="zeros"),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, in_channels, kernel_size=3, padding=1, padding_mode="zeros"),
        )

        self.orig_weight = nn.Parameter(torch.tensor(1.0))

        # SNR estimation layer
        self.snr_estimator = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1),
            nn.Sigmoid(),
        )

    def _create_center_focus_mask(self, Z, H, W):
        """Creates a mask that emphasizes the center and de-emphasizes edges"""
        z_coords = torch.linspace(-1, 1, Z)
        y_coords = torch.linspace(-1, 1, H)
        x_coords = torch.linspace(-1, 1, W)
        z, y, x = torch.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
        dist_sq = x**2 + y**2 + z**2
        mask = 1.0 - 0.8 * (dist_sq / dist_sq.max())
        return mask.view(1, 1, Z, H, W)

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """
        Create a Gaussian kernel for each sample in the batch.
        sigma: Tensor of shape [B, 1]
        Returns: kernel of shape [B, 1, kernel_size, kernel_size, kernel_size]
        """
        # Create coordinate grid of shape [kernel_size, kernel_size, kernel_size]
        coords = torch.linspace(
            -(kernel_size // 2),
            kernel_size // 2,
            steps=kernel_size,
            device=sigma.device,
        )
        x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
        # Reshape to [1, 1, k, k, k] so they can broadcast with sigma
        x = x.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(0).unsqueeze(0)
        z = z.unsqueeze(0).unsqueeze(0)
        # Expand sigma to [B, 1, 1, 1, 1]
        B = sigma.size(0)
        sigma = sigma.view(B, 1, 1, 1, 1)
        # Compute the Gaussian kernel for each sample
        kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
        # Normalize each kernel
        kernel = kernel / kernel.sum(dim=[2, 3, 4], keepdim=True)
        return kernel

    def _gaussian_blur(self, x, sigma):
        """Apply Gaussian blur with a per-sample (adaptive) sigma."""
        kernel_size = 5
        # Create a kernel for each sample; sigma has shape [B,1]
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(x.device)
        padding = kernel_size // 2

        # x has shape [B, C, Z, H, W] with C==1
        B, C, Z, H, W = x.shape
        # Reshape x to [1, B, Z, H, W] so that each sample is a separate group.
        x_reshaped = x.view(1, B, Z, H, W)
        # Apply convolution with groups equal to B
        blurred = F.conv3d(x_reshaped, kernel, padding=padding, groups=B)
        # Reshape back to [B, C, Z, H, W]
        blurred = blurred.view(B, C, Z, H, W)
        return blurred

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        if mask is not None:
            x = x * mask

        # Estimate SNR per sample; shape: [B, 1]
        estimated_snr = self.snr_estimator(x)
        # Compute adaptive sigma per sample (shape: [B, 1])
        adaptive_sigma = self.init_sigma * (2.0 - estimated_snr)
        # Apply adaptive Gaussian blur
        blurred = self._gaussian_blur(x, adaptive_sigma)
        # Process blurred signal
        processed = self.feature_extractor(blurred)
        processed = processed * self.center_mask

        # Edge handling: reduce edge values explicitly
        edge_mask = F.pad(
            torch.ones((1, 1, self.Z - 2, self.H - 2, self.W - 2), device=x.device),
            (1, 1, 1, 1, 1, 1),
            "constant",
            0.2,
        ).expand(batch_size, -1, -1, -1, -1)
        processed = processed * edge_mask

        # Balance original and processed signals
        alpha = torch.sigmoid(self.orig_weight)
        preservation_factor = alpha * (0.5 + 0.5 * estimated_snr)
        output = (
            preservation_factor.view(-1, 1, 1, 1, 1) * x
            + (1 - preservation_factor.view(-1, 1, 1, 1, 1)) * processed
        )

        if mask is not None:
            output = output * mask

        return output


def make_3d_gaussian_kernel(kernel_size=3, sigma=1.0):
    """
    Creates a 3D Gaussian kernel (fixed, non-learnable) for smoothing.
    kernel_size: Odd integer, e.g. 3 or 5
    sigma: Standard deviation for the Gaussian
    Returns: Tensor of shape [1, 1, k, k, k]
    """
    coords = torch.arange(kernel_size) - (kernel_size // 2)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid = x**2 + y**2 + z**2
    kernel_3d = torch.exp(-grid / (2 * sigma**2))
    kernel_3d = kernel_3d / kernel_3d.sum()
    return kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)


class SimpleBraggPreprocessor(nn.Module):
    """
    A simple 3D Gaussian smoother with reflect padding and
    a residual mixing parameter alpha.
    """

    def __init__(self, in_channels=1, kernel_size=3, sigma=1.0, alpha=0.5):
        """
        in_channels: Number of input channels (often 1 for a single shoebox).
        kernel_size: Size of the Gaussian kernel, e.g. 3 or 5.
        sigma: Standard deviation for the Gaussian blur.
        alpha: Mixing ratio. alpha=1.0 -> fully blurred, alpha=0.0 -> no smoothing.
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

        # Create a fixed 3D Gaussian kernel
        kernel_3d = make_3d_gaussian_kernel(kernel_size, sigma)
        # Register as a buffer so it stays on the correct device but is not learnable
        self.register_buffer("kernel_3d", kernel_3d)

    def forward(self, x, mask=None):
        """
        x: [batch_size, in_channels, Z, H, W] shoebox data
        Returns: A smoothed version of x mixed with the original x.
        """
        B, C, Z, H, W = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        if mask is not None:
            x = x * mask

        # Reflect-pad before convolution to avoid zero edges
        pad = self.kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad, pad, pad), mode="reflect")

        # Convolution for each channel separately (groups=C)
        # Expand kernel to [C, 1, k, k, k] so each channel is filtered identically
        kernel_expanded = self.kernel_3d.expand(C, 1, -1, -1, -1)
        # Convolution shape: input [B, C, Z+2*pad, H+2*pad, W+2*pad]
        # kernel [C, 1, k, k, k], groups=C
        blurred = F.conv3d(x_padded, kernel_expanded, groups=C)

        # Weighted sum of original and blurred
        alpha_val = torch.sigmoid(self.alpha)  # Ensure alpha is in [0,1]
        out = (1.0 - alpha_val) * x + alpha_val * blurred
        return out


class tempSignalPreprocessor(nn.Module):
    def __init__(self, in_channels=1, Z=3, H=21, W=21):
        super().__init__()
        self.Z, self.H, self.W = Z, H, W

        # Create center-focused weight mask
        center_mask = self._create_center_focus_mask(Z, H, W)
        self.register_buffer("center_mask", center_mask)

        # Adaptive Gaussian blur with learnable sigma
        self.init_sigma = nn.Parameter(torch.tensor(0.8))  # Start with less blur

        # Feature extraction with focus on center
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=5, padding=2, padding_mode="zeros"),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, in_channels, kernel_size=3, padding=1, padding_mode="zeros"),
        )

        # Balance between original and processed
        self.orig_weight = nn.Parameter(torch.tensor(0.7))  # More original signal

        # SNR estimation layer
        self.snr_estimator = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1),
            nn.Sigmoid(),  # Output between 0-1 representing estimated SNR
        )

    def _create_center_focus_mask(self, Z, H, W):
        """Creates a mask that emphasizes the center and de-emphasizes edges"""
        # Create coordinate grids
        z_coords = torch.linspace(-1, 1, Z)
        y_coords = torch.linspace(-1, 1, H)
        x_coords = torch.linspace(-1, 1, W)

        # Create meshgrid
        z, y, x = torch.meshgrid(z_coords, y_coords, x_coords)

        # Calculate distance from center (squared)
        dist_sq = x**2 + y**2 + z**2

        # Create center-focused mask (1 at center, falling off to 0.2 at edges)
        mask = 1.0 - 0.8 * (dist_sq / dist_sq.max())

        # Reshape for convolution
        return mask.view(1, 1, Z, H, W)

    def _gaussian_blur(self, x, sigma):
        """Apply Gaussian blur with dynamic sigma"""
        # Create kernel
        kernel_size = 5
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(x.device)

        # Apply convolution for blurring
        padding = kernel_size // 2
        return F.conv3d(x, kernel, padding=padding, groups=x.size(1))

    def _create_gaussian_kernel(self, kernel_size, sigma):
        # Create coordinate grids
        coords = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        x, y, z = torch.meshgrid(coords, coords, coords)

        # Calculate Gaussian
        kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        # Reshape for Conv3D (channels=1, preserving groups in convolution)
        return kernel.view(1, 1, kernel_size, kernel_size, kernel_size)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # Apply mask if provided
        if mask is not None:
            x = x * mask

        # Estimate SNR
        estimated_snr = self.snr_estimator(x)

        # Adjust sigma based on estimated SNR (lower SNR = more blur)
        # Range: base_sigma for high SNR to 2*base_sigma for low SNR
        adaptive_sigma = self.init_sigma * (2.0 - estimated_snr)

        # Apply adaptive Gaussian blur
        blurred = self._gaussian_blur(x, adaptive_sigma)

        # Apply center-focused processing
        processed = self.feature_extractor(blurred)
        processed = processed * self.center_mask

        # Edge handling - explicitly reduce edge values
        edge_mask = F.pad(
            torch.ones((1, 1, self.Z - 2, self.H - 2, self.W - 2), device=x.device),
            (1, 1, 1, 1, 1, 1),
            "constant",
            0.2,  # Edge values reduced to 20%
        ).expand(batch_size, -1, -1, -1, -1)

        processed = processed * edge_mask

        # Balance original and processed signals
        alpha = torch.sigmoid(self.orig_weight)

        # For higher SNR inputs, preserve more of the original
        preservation_factor = alpha * (0.5 + 0.5 * estimated_snr)
        output = (
            preservation_factor.view(-1, 1, 1, 1, 1) * x
            + (1 - preservation_factor.view(-1, 1, 1, 1, 1)) * processed
        )

        # Final mask application
        if mask is not None:
            output = output * mask

        return output


class SimpleConcentrationRefinement(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # Use small 3x3x3 kernels with reflect padding to preserve spatial dimensions
        self.conv1 = nn.Conv3d(
            in_channels, 8, kernel_size=3, padding=1, padding_mode="zeros"
        )
        self.conv2 = nn.Conv3d(8, 1, kernel_size=3, padding=1, padding_mode="zeros")
        # Learnable global concentration scaling parameter

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, 1, Z, H, W]
        Returns a refined concentration map of the same shape.
        """

        # Reshape if needed
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 1, 3, 21, 21)

        # Simple two-layer conv network
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        # Apply softplus to ensure positivity and add a small constant for stability
        refined = F.softplus(out) + 1e-6
        return refined.view(-1, 3 * 21 * 21)


class PCAReconstructionLayer(nn.Module):
    def __init__(self, input_shape, bottleneck_dim=200, use_residual=True):
        """
        A linear autoencoder that reconstructs the input from a lower-dimensional bottleneck,
        approximating PCA reconstruction.

        Args:
          input_shape (tuple): Shape of one sample, e.g. (C, D, H, W).
          bottleneck_dim (int): The dimension of the bottleneck (should be lower than the flattened dimension).
          use_residual (bool): If True, add the original input to the reconstructed output.
        """
        super().__init__()
        self.input_shape = input_shape  # e.g. (1, 3, 21, 21)
        self.use_residual = use_residual
        flat_dim = 1
        for d in input_shape:
            flat_dim *= d

        # Create a linear encoder and decoder with no nonlinearity.
        # For true PCA equivalence, you might tie weights (decoder weight = encoder weight^T).
        # Here, we let them be free parameters.
        self.encoder = nn.Linear(flat_dim, bottleneck_dim, bias=False)
        self.decoder = nn.Linear(bottleneck_dim, flat_dim, bias=False)

    def forward(self, x, mask=None):
        """
        Args:
          x: Input tensor of shape [B, C, D, H, W].
        Returns:
          Reconstructed tensor of the same shape.
        """

        if mask is not None:
            x = x * mask

        B = x.shape[0]
        x_flat = x.view(B, -1)
        code = self.encoder(x_flat)
        recon_flat = self.decoder(code)
        recon = recon_flat.view(x.shape)
        if self.use_residual:
            if mask is not None:
                return (recon + x) * mask
            else:
                return recon + x
        else:
            if mask is not None:
                return (recon + x) * mask
            else:
                return recon + x


class ConcentrationRefinementNetwork(nn.Module):
    def __init__(self, Z=3, H=21, W=21):
        super().__init__()

        # Simple refinement with larger kernels and bottleneck
        self.refine = nn.Sequential(
            # Down to bottleneck
            nn.Conv3d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 4, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # Bottleneck layer (force information compression)
            nn.Conv3d(4, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            # Back up to output
            nn.Conv3d(2, 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(4, 1, kernel_size=5, padding=2),
        )

        # Global concentration scaling
        self.concentration_scale = nn.Parameter(
            torch.tensor(0.2)
        )  # Start with low concentration

    def forward(self, x):
        # Reshape if needed
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 1, 3, 21, 21)

        # Apply refinement
        refined = self.refine(x)

        # Flatten and ensure positive
        params_flat = F.softplus(refined.view(x.size(0), -1)) + 1e-6

        # Apply concentration scaling (lower values = more uniform)
        scale = F.softplus(self.concentration_scale)
        return params_flat * scale


class DirichletProfile(torch.nn.Module):
    """
    Dirichlet profile model
    """

    def __init__(self, dmodel, rank=None, mc_samples=100, num_components=3 * 21 * 21):
        super().__init__()
        self.dmodel = dmodel
        self.mc_samples = mc_samples
        self.num_components = num_components
        self.alpha_layer = Linear(self.dmodel, self.num_components)
        self.rank = rank
        self.eps = 1e-6

    # def forward(self, representation):
    def forward(self, alphas):
        # alphas = self.alpha_layer(representation)
        alphas = F.softplus(alphas) + self.eps
        q_p = torch.distributions.Dirichlet(alphas)

        return q_p


class SignalAwareProfile(nn.Module):
    def __init__(self, dmodel, min_concentration=0.1):
        super().__init__()
        self.profile_model = DirichletProfile(dmodel)
        self.min_concentration = min_concentration

    def forward(self, concentration, signal_prob):
        # Get base profile distribution
        q_p = self.profile_model(concentration)

        # Modulate concentration based on signal probability
        batch_size = concentration.shape[0]
        signal_prob = signal_prob.view(batch_size, 1)

        # Linear interpolation between min_concentration and original concentration
        modulated_concentration = q_p.concentration
        modulated_concentration = (
            signal_prob * modulated_concentration
            + (1 - signal_prob)
            * torch.ones_like(modulated_concentration)
            * self.min_concentration
        )

        # Create new Dirichlet with modulated concentration
        q_p = torch.distributions.dirichlet.Dirichlet(modulated_concentration)

        return q_p


class SpectralBraggPreprocessor(nn.Module):
    """
    A Fourier-based preprocessor for Bragg peak data.
    This module performs:
      1. A 3D FFT on the input shoebox.
      2. Application of a bandpass filter in the Fourier domain,
         with smooth (cosine) tapering at the boundaries.
      3. An inverse FFT to return to spatial domain.
      4. A residual mixture with the original input controlled
         by a learnable parameter.
    """

    def __init__(
        self,
        in_channels=1,
        Z=3,
        H=21,
        W=21,
        low_cut=0.1,
        high_cut=0.3,
        smooth_width=0.05,
        mix=0.5,
    ):
        """
        Args:
          in_channels: Number of input channels (typically 1 for a shoebox).
          Z, H, W: Spatial dimensions of the input.
          low_cut: Lower cutoff frequency (normalized, in cycles/pixel) for the bandpass.
          high_cut: Upper cutoff frequency.
          smooth_width: Width of the cosine taper at the band edges.
          mix: Initial mixing coefficient between original and filtered signal.
        """
        super().__init__()
        self.in_channels = in_channels
        self.Z, self.H, self.W = Z, H, W
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.smooth_width = smooth_width

        # Learnable mixing parameter (will be squashed into [0,1] via sigmoid)
        self.mix = nn.Parameter(torch.tensor(mix), requires_grad=True)

        # Precompute the frequency-domain filter mask for the given dimensions.
        self.register_buffer("filter_mask", self.create_filter_mask())

    def create_filter_mask(self):
        # Create frequency grids for each dimension using torch.fft.fftfreq,
        # which returns frequencies in the range (-0.5, 0.5]
        fz = torch.fft.fftfreq(self.Z)  # shape: [Z]
        fy = torch.fft.fftfreq(self.H)  # shape: [H]
        fx = torch.fft.fftfreq(self.W)  # shape: [W]
        # Create a 3D meshgrid (with indexing "ij")
        fz, fy, fx = torch.meshgrid(fz, fy, fx, indexing="ij")
        # Compute the radial frequency (Euclidean norm)
        r = torch.sqrt(fx**2 + fy**2 + fz**2)
        # Create a smooth bandpass filter using cosine tapers.
        # For frequencies below low_cut, ramp up from 0 to 1;
        # for frequencies above high_cut, ramp down from 1 to 0.
        # Within the passband, the filter is 1.
        mask = torch.ones_like(r)

        # Low-frequency ramp (r < low_cut + smooth_width)
        low_ramp = torch.clamp((r - self.low_cut) / self.smooth_width, min=0.0, max=1.0)
        low_taper = 0.5 * (1 - torch.cos(torch.pi * low_ramp))
        mask[r < self.low_cut + self.smooth_width] = low_taper[
            r < self.low_cut + self.smooth_width
        ]

        # High-frequency ramp (r > high_cut - smooth_width)
        high_ramp = torch.clamp(
            (self.high_cut - r) / self.smooth_width, min=0.0, max=1.0
        )
        high_taper = 0.5 * (1 - torch.cos(torch.pi * high_ramp))
        mask[r > self.high_cut - self.smooth_width] = high_taper[
            r > self.high_cut - self.smooth_width
        ]

        # Outside the band, set to 0.
        mask[r < self.low_cut] = 0.0
        mask[r > self.high_cut] = 0.0

        return mask  # shape: [Z, H, W]

    def forward(self, x, mask=None):
        """
        Args:
          x: Input tensor of shape [B, in_channels, Z, H, W]
        Returns:
          The processed tensor of the same shape.
        """
        B, C, Z, H, W = x.shape

        if mask is not None:
            x = x * mask

        # Compute FFT along spatial dimensions.
        x_fft = fft.fftn(x, dim=(-3, -2, -1))
        # Shift zero frequency to the center.
        x_fft_shift = fft.fftshift(x_fft, dim=(-3, -2, -1))

        # Apply filter mask (expand mask to [1, 1, Z, H, W] to broadcast over batch and channels)
        mask = self.filter_mask.unsqueeze(0).unsqueeze(0)
        x_fft_filtered = x_fft_shift * mask

        # Inverse shift and inverse FFT to get back to spatial domain.
        x_fft_filtered = fft.ifftshift(x_fft_filtered, dim=(-3, -2, -1))
        x_filtered = fft.ifftn(x_fft_filtered, dim=(-3, -2, -1)).real

        # Mix original and filtered using a learnable mixing parameter.
        mix_val = torch.sigmoid(self.mix)  # ensures mix in [0, 1]
        out = (1 - mix_val) * x + mix_val * x_filtered

        if mask is not None:
            out = out * mask

        return out


# %%


class SparseProfileLoss(nn.Module):
    def __init__(self, base_loss_fn, l1_lambda=1):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.l1_lambda = l1_lambda

    def forward(self, rate, counts, q_p, q_z, q_I, q_bg, masks):
        # Get base loss components
        base_loss_results = self.base_loss_fn(rate, counts, q_p, q_z, q_I, q_bg, masks)
        total_loss = base_loss_results[0]

        # L1 norm of concentration parameters
        l1_penalty = self.l1_lambda * torch.mean(torch.abs(q_p.concentration))

        # Add to loss
        sparse_loss = total_loss + l1_penalty

        return (
            sparse_loss,
            base_loss_results[1],  # neg_ll
            base_loss_results[2],  # kl
            base_loss_results[3],  # kl_z
            base_loss_results[4],  # kl_I
            base_loss_results[5],  # kl_bg
            base_loss_results[6],  # kl_p
            l1_penalty,
        )


class L1SignalComponentLoss(nn.Module):
    """
    Applies L1 regularization to the full signal component (Z*I*P)
    to encourage sparsity in the signal contribution to the rate.
    """

    def __init__(self, base_loss_fn, l1_lambda=0.01):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.l1_lambda = l1_lambda

    def forward(self, rate, counts, q_p, q_z, q_I, q_bg, masks):
        # Get base loss components
        base_loss_results = self.base_loss_fn(rate, counts, q_p, q_z, q_I, q_bg, masks)
        total_loss = base_loss_results[0]

        # Calculate the full signal component: Z*I*P
        profile = q_p.mean  # Expected profile distribution
        intensity = q_I.mean.unsqueeze(-1)  # Expected intensity
        signal_prob = q_z.probs.unsqueeze(-1)  # Signal existence probability

        # Full signal component: Z*I*P
        full_signal = signal_prob * intensity * profile

        # Apply L1 regularization - sum of absolute values
        l1_penalty = self.l1_lambda * torch.mean(torch.abs(full_signal))

        # Add to total loss
        sparse_loss = total_loss + l1_penalty

        # Return all components, including the new penalty
        return (
            sparse_loss,
            base_loss_results[1],  # neg_ll
            base_loss_results[2],  # kl
            base_loss_results[3],  # kl_z
            base_loss_results[4],  # kl_I
            base_loss_results[5],  # kl_bg
            base_loss_results[6],  # kl_p
            l1_penalty,
        )


# %%


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with residual connection"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual connection if dimensions don't match
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = self.relu(x)

        return x


class Encoder(nn.Module):
    """Encoder with progressive downsampling that handles small depth dimensions"""

    def __init__(self, in_channels, latent_dim=64, depth=3):
        super().__init__()
        self.depth = depth
        channels = [in_channels] + [
            min(latent_dim * (2**i), 256) for i in range(depth)
        ]

        # Initial convolution
        self.init_conv = ConvBlock(channels[0], channels[1])

        # Encoder blocks with downsampling
        self.enc_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        for i in range(1, depth):
            self.enc_blocks.append(ConvBlock(channels[i], channels[i + 1]))
            # Use careful pooling that preserves dimensions smaller than kernel size
            self.down_samples.append(
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            )

    def forward(self, x):
        features = []

        # Initial convolution
        x = self.init_conv(x)
        features.append(x)

        # Check dimensions to ensure we don't try to pool too small tensors
        # Encoder blocks with skip connections
        for i in range(self.depth - 1):
            # Only apply pooling if dimensions allow it
            if x.size(2) >= 2 and x.size(3) >= 2 and x.size(4) >= 2:
                x = self.down_samples[i](x)
            elif x.size(3) >= 2 and x.size(4) >= 2:
                # If depth is too small but H,W are ok, use 2D pooling instead
                x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

            x = self.enc_blocks[i](x)
            features.append(x)

        return x, features


class Decoder(nn.Module):
    """Decoder with progressive upsampling and skip connections that handles small depth dimensions"""

    def __init__(self, latent_dim=64, out_channels=1, depth=3):
        super().__init__()
        self.depth = depth
        channels = [min(latent_dim * (2**i), 256) for i in range(depth, -1, -1)]

        # Decoder blocks with upsampling
        self.dec_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i in range(depth):
            # Use transpose conv that only upsamples height and width, not depth
            self.up_samples.append(
                nn.ConvTranspose3d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                )
            )
            self.dec_blocks.append(
                ConvBlock(
                    channels[i + 1] * 2, channels[i + 1]
                )  # *2 for skip connection
            )

        # Final convolution to output channels
        self.final_conv = nn.Conv3d(channels[-1], out_channels, kernel_size=1)

    def forward(self, x, enc_features):
        # Reverse encoder features for skip connections
        enc_features = enc_features[::-1]

        # Decoder blocks with skip connections
        for i in range(self.depth):
            # Only upsample if dimensions were actually reduced
            if x.size(3) < enc_features[i].size(3) or x.size(4) < enc_features[i].size(
                4
            ):
                x = self.up_samples[i](x)

            # Handle potential size mismatch
            if x.size()[2:] != enc_features[i].size()[2:]:
                x = F.interpolate(
                    x,
                    size=enc_features[i].size()[2:],
                    mode="trilinear",
                    align_corners=False,
                )

            # Concatenate skip connection
            x = torch.cat([x, enc_features[i]], dim=1)
            x = self.dec_blocks[i](x)

        # Final convolution
        x = self.final_conv(x)
        return x


class UNetBlock(nn.Module):
    """A mini-UNet for feature refinement that handles small depth dimensions"""

    def __init__(self, channels, depth=2):
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        # Use a pooling operation that won't reduce depth
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.depth = depth

        # Down blocks
        in_channels = channels
        for i in range(depth):
            out_channels = in_channels * 2
            self.down_blocks.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels

        # Bottleneck
        self.bottleneck = ConvBlock(in_channels, in_channels * 2)

        # Up blocks
        in_channels = in_channels * 2
        for i in range(depth):
            out_channels = in_channels // 2
            self.up_blocks.append(
                nn.Sequential(
                    # Use transpose conv that won't increase depth if it's 1
                    nn.ConvTranspose3d(
                        in_channels,
                        out_channels,
                        kernel_size=(1, 2, 2),
                        stride=(1, 2, 2),
                    ),
                    ConvBlock(out_channels * 2, out_channels),  # *2 for skip connection
                )
            )
            in_channels = out_channels

    def forward(self, x):
        # Store skip connections
        skip_connections = []

        # Store input shape to know what dimensions we're working with
        _, _, depth, height, width = x.shape

        # Down pass
        for i in range(self.depth):
            x = self.down_blocks[i](x)
            skip_connections.append(x)
            # Only pool if dimensions allow
            if x.size(3) >= 2 and x.size(4) >= 2:  # If H and W are big enough
                x = self.pool(x)
            else:
                # If dimensions too small, just pass through
                pass

        # Bottleneck
        x = self.bottleneck(x)

        # Up pass with skip connections
        for i in range(self.depth):
            skip = skip_connections.pop()

            # Only upsample if needed (if we actually downsampled)
            if x.size(3) < skip.size(3) or x.size(4) < skip.size(4):
                x = self.up_blocks[i][0](x)  # Upsampling

            # Handle potential size mismatch
            if x.size()[2:] != skip.size()[2:]:
                x = F.interpolate(
                    x, size=skip.size()[2:], mode="trilinear", align_corners=False
                )

            x = torch.cat([x, skip], dim=1)  # Skip connection
            x = self.up_blocks[i][1](x)  # Conv block

        return x


class SignalPurificationNetwork(nn.Module):
    """
    Advanced signal purification network with autoencoder-UNet cascade
    for background removal and profile generation - specialized for small depth dimension
    """

    def __init__(
        self, in_channels=1, latent_dim=64, depth=3, num_components=3 * 21 * 21
    ):
        super().__init__()
        self.num_components = num_components

        # First autoencoder for background removal
        self.encoder1 = Encoder(in_channels, latent_dim, depth)
        self.decoder1 = Decoder(latent_dim, in_channels, depth)

        # First UNet for feature enhancement
        self.unet1 = UNetBlock(in_channels, depth=2)

        # Second autoencoder for profile feature extraction
        self.encoder2 = Encoder(in_channels, latent_dim, depth)
        self.decoder2 = Decoder(latent_dim, in_channels, depth)

        # Second UNet for final refinement
        self.unet2 = UNetBlock(in_channels, depth=2)

        # Concentrate to Dirichlet parameters (flattened output)
        # Use adaptive pooling for safer handling of various input sizes
        self.concentrate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(
                in_channels, 512
            ),  # Use input channels instead of calculated size
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(512, num_components),
        )

        # Attention gates for signal focus
        self.attention1 = AttentionGate(in_channels, in_channels)
        self.attention2 = AttentionGate(in_channels, in_channels)

        # Add direct pathway for very small inputs
        self.direct_pathway = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_components),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [batch, channels, depth, height, width]
            mask: Optional mask [batch, 1, depth, height, width]

        Returns:
            concentration: Flattened concentration parameters for Dirichlet
        """
        batch_size = x.shape[0]

        # Check input dimensions to choose appropriate pathway
        _, _, depth, height, width = x.shape
        use_direct_pathway = depth <= 1 or height <= 4 or width <= 4

        # Apply mask if provided
        if mask is not None:
            x = x * mask

        if use_direct_pathway:
            # Use simpler direct pathway for very small inputs
            concentration = F.softplus(self.direct_pathway(x)) + 1e-4
            return concentration

        try:
            # First autoencoder for background removal
            latent1, features1 = self.encoder1(x)
            cleaned = self.decoder1(latent1, features1)

            # Apply attention after first autoencoder
            cleaned = self.attention1(cleaned, mask) if mask is not None else cleaned

            # First UNet for feature enhancement
            enhanced = self.unet1(cleaned)

            # Second autoencoder for profile feature extraction
            latent2, features2 = self.encoder2(enhanced)
            profile_features = self.decoder2(latent2, features2)

            # Apply attention after second autoencoder
            profile_features = (
                self.attention2(profile_features, mask)
                if mask is not None
                else profile_features
            )

            # Second UNet for final refinement
            refined = self.unet2(profile_features)

            # Apply mask to refined features if provided
            if mask is not None:
                refined = refined * mask

            # Generate concentration parameters
            concentration = F.softplus(self.concentrate(refined)) + 1e-4

        except RuntimeError as e:
            # Fallback to direct pathway if we encounter issues
            print(f"Warning: Using direct pathway due to error: {str(e)}")
            concentration = F.softplus(self.direct_pathway(x)) + 1e-4

        return concentration


class AttentionGate(nn.Module):
    """Attention gate for focusing on signal regions"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_attention = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x, mask=None):
        # Generate attention weights
        attention = self.conv_attention(x)

        # Apply mask to attention if provided
        if mask is not None:
            attention = attention * mask

        # Apply attention to input
        return x * attention


# %%
class tempUNetIntegrator(BaseIntegrator):
    def __init__(
        self,
        image_encoder,
        metadata_encoder,
        # profile_model,
        unet,
        loss,
        q_bg,
        q_I,
        decoder,
        dmodel=64,
        mc_samples=100,
        learning_rate=1e-3,
        profile_threshold=0.001,
        # signal_preprocessor=SignalPreprocessor(),
        base_alpha=0.1,
        center_alpha=10.0,
        decay_factor=2.0,
        peak_percentage=0.01,
        latent_dim=64,
        autoencoder_depth=3,
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "mlp_encoder",
                "profile_model",
                "unet",
                "signal_preprocessor",
            ]
        )
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples
        self.profile_threshold = profile_threshold

        # Model components
        self.image_encoder = image_encoder
        self.metadata_encoder = metadata_encoder
        self.profile_model = DirichletProfile(dmodel)
        # self.profile_model = SignalAwareProfile(dmodel)

        # self.signal_preprocessor = signal_preprocessor
        # self.unet = unet
        # self.concentration_refinement = SimpleConcentrationRefinement()
        self.background_distribution = q_bg
        self.intensity_distribution = q_I
        # self.q_z = q_z

        self.decoder = decoder

        # Replace the downscale_upscale_preprocessor and UNet with Signal Purification Network
        self.signal_purification_network = SignalPurificationNetwork(
            in_channels=1,  # Shoebox typically has 1 channel
            latent_dim=latent_dim,
            depth=2,
            num_components=3
            * 21
            * 21,  # Assuming 3x21x21 is your Dirichlet profile size
        )

        self.loss_fn = loss  # Additional layers
        # self.fc_representation = Linear(dmodel * 2, dmodel)
        self.fc_representation = Linear(dmodel, dmodel)
        self.norm = nn.LayerNorm(dmodel)

        # Enable automatic optimization
        self.automatic_optimization = True

    def calculate_intensities(self, counts, qbg, qp, dead_pixel_mask):
        with torch.no_grad():
            counts = counts * dead_pixel_mask
            batch_counts = counts.unsqueeze(1)  # [batch_size x 1 x pixels]

            batch_bg_samples = (qbg.rsample([self.mc_samples]).unsqueeze(-1)).permute(
                1, 0, 2
            )
            batch_profile_samples = qp.rsample([self.mc_samples]).permute(
                1, 0, 2
            )  # [batch_size x mc_samples x pixels]

            batch_profile_samples = batch_profile_samples * dead_pixel_mask.unsqueeze(1)

            weighted_sum_intensity = (
                batch_counts - batch_bg_samples
            ) * batch_profile_samples

            weighted_sum_intensity_sum = weighted_sum_intensity.sum(-1)

            summed_squared_prf = torch.norm(batch_profile_samples, p=2, dim=-1).pow(2)

            division = weighted_sum_intensity_sum / summed_squared_prf

            weighted_sum_intensity_mean = division.mean(-1)

            weighted_sum_intensity_var = division.var(-1)

            profile_masks = batch_profile_samples > self.profile_threshold

            N_used = profile_masks.sum(-1).float()  # [batch_size × mc_samples]

            masked_counts = batch_counts * profile_masks

            thresholded_intensity = (
                masked_counts - batch_bg_samples * profile_masks
            ).sum(-1)

            thresholded_mean = thresholded_intensity.mean(-1)

            centered_thresh = thresholded_intensity - thresholded_mean.unsqueeze(-1)

            thresholded_var = (centered_thresh**2).sum(-1) / (N_used.mean(-1) + 1e-6)

            intensities = {
                "thresholded_mean": thresholded_mean,
                "thresholded_var": thresholded_var,
                "weighted_sum_intensity_mean": weighted_sum_intensity_mean,
                "weighted_sum_intensity_var": weighted_sum_intensity_var,
            }

            return intensities

    def forward(self, shoebox, dials, masks, metadata, counts):
        # Preprocess input data
        counts = torch.clamp(counts, min=0)

        # Extract representations from the shoebox and metadata
        shoebox_representation = self.image_encoder(shoebox, masks)
        meta_representation = self.metadata_encoder(metadata)

        # Combine representations and normalize
        representation = shoebox_representation + meta_representation
        representation = self.fc_representation(representation)
        representation = self.norm(representation)

        # Get distributions from confidence model
        q_I = self.intensity_distribution(representation)
        q_bg = self.background_distribution(representation)

        # Preprocess signal for profile estimation
        # preprocessed_signal = self.signal_preprocessor(
        # shoebox[:, :, -1].reshape(-1, 3, 21, 21).unsqueeze(1),
        # mask=masks.reshape(-1, 3, 21, 21).unsqueeze(1),
        # )

        # Generate concentration parameters - EXACTLY AS IN YOUR CODE
        # concentration = self.unet(
        # preprocessed_signal,
        # masks,
        # )

        # Prepare shoebox and mask data for processing
        shoebox_data = shoebox[:, :, -1].reshape(-1, 3, 21, 21).unsqueeze(1)
        mask_data = masks.reshape(-1, 3, 21, 21).unsqueeze(1)

        # The key change: Replace previous preprocessing pipeline with Signal Purification Network
        # This combines shoebox → autoencoder → UNet → autoencoder → UNet → Dirichlet
        concentration = self.signal_purification_network(shoebox_data, mask=mask_data)

        qp = self.profile_model(concentration)  # dirichlet

        # Calculate rate using the decoder
        rate = self.decoder(q_I, q_bg, qp)
        intensity_mean = q_I.mean.unsqueeze(-1)

        # Calculate expected values for reporting
        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "q_I": q_I,
            "qbg": q_bg,
            "qp": qp,
            "intensity_mean": intensity_mean,
            "dials_I_sum_value": dials[:, 0],
            "dials_I_sum_var": dials[:, 1],
            "dials_I_prf_value": dials[:, 2],
            "dials_I_prf_var": dials[:, 3],
            "refl_ids": dials[:, 4],
        }

    def training_step(self, batch, batch_idx):
        # Unpack batch
        shoebox, dials, masks, metadata, counts = batch

        # Get model outputs
        outputs = self(shoebox, dials, masks, metadata, counts)

        # Calculate loss.
        # Updated call: note we no longer pass a separate q_I_nosignal.
        (
            loss,
            neg_ll,
            kl,
            kl_I,
            kl_bg,
            kl_p,
            tv_loss,
            simpson_loss,
            entropy_loss,
        ) = self.loss_fn(
            outputs["rates"],
            outputs["counts"],
            outputs["qp"],
            outputs["q_I"],
            outputs["qbg"],
            outputs["masks"],
        )

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # Log metrics
        self.log("train_loss", loss.mean())
        self.log("train_nll", neg_ll.mean())
        self.log("train_kl", kl.mean())
        self.log("train_kl_I", kl_I.mean())
        self.log("train_kl_bg", kl_bg.mean())
        self.log("train_kl_p", kl_p.mean())

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        shoebox, dials, masks, metadata, counts = batch

        # Get model outputs
        outputs = self(shoebox, dials, masks, metadata, counts)

        (
            loss,
            neg_ll,
            kl,
            kl_I,
            kl_bg,
            kl_p,
            tv_loss,
            simpson_loss,
            entropy_loss,
        ) = self.loss_fn(
            outputs["rates"],
            outputs["counts"],
            outputs["qp"],
            outputs["q_I"],
            outputs["qbg"],
            outputs["masks"],
        )

        # Log metrics
        self.log("val_loss", loss.mean())
        self.log("val_nll", neg_ll.mean())
        self.log("val_kl", kl.mean())
        self.log("val_kl_I", kl_I.mean())
        self.log("val_kl_bg", kl_bg.mean())
        self.log("val_kl_p", kl_p.mean())
        self.log("val_tv_loss", tv_loss.mean())
        self.log("val_simpson_loss", simpson_loss.mean())
        self.log("val_entropy_loss", entropy_loss.mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)

        return {
            "intensity_mean": outputs["intensity_mean"],
            "intensity_var": outputs["intensity_var"],
            "signal_prob": outputs["signal_prob"],
            "refl_ids": outputs["refl_ids"],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# %%

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and ReLU activation"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class UNet3D(nn.Module):
    """
    Simplified UNet that operates primarily in 2D when depth dimension is 1,
    specialized for inputs with shape (batch, channels, 1, H, W)
    """

    def __init__(
        self, in_channels, out_channels, depth=2, base_filters=32, max_filters=256
    ):
        super().__init__()
        self.depth = depth

        # Calculate filter sizes for each level (limit depth to prevent over-reduction)
        if depth > 2 and base_filters >= 32:
            # For 10x10 inputs, more than 2 levels of pooling is risky
            depth = 2
            print(f"Warning: Reducing depth to {depth} to avoid dimension issues")

        filter_sizes = [
            min(base_filters * (2**i), max_filters) for i in range(depth + 1)
        ]

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()

        # First block (no downsampling before this)
        self.encoder_blocks.append(ConvBlock(in_channels, filter_sizes[0]))

        # Remaining encoder blocks
        for i in range(1, depth + 1):
            self.encoder_blocks.append(ConvBlock(filter_sizes[i - 1], filter_sizes[i]))

        # 2D pooling - treat as a 2D UNet when depth=1
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Decoder blocks and upsampling operations
        self.decoder_blocks = nn.ModuleList()
        self.upsampling = nn.ModuleList()

        # Create decoder path
        for i in range(depth, 0, -1):
            # 2D upsampling with transposed convolution
            self.upsampling.append(
                nn.ConvTranspose3d(
                    filter_sizes[i],
                    filter_sizes[i - 1],
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                )
            )

            # Decoder block after concatenation (features from encoder + upsampled)
            self.decoder_blocks.append(
                ConvBlock(filter_sizes[i - 1] * 2, filter_sizes[i - 1])
            )

        # Final convolution to produce the desired number of output channels
        self.final_conv = nn.Conv3d(filter_sizes[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Track encoder outputs for skip connections
        encoder_outputs = []

        # Safety check - if input dimensions are too small, use direct pathway
        _, _, depth, height, width = x.shape
        if height < 4 or width < 4:
            raise RuntimeError(
                f"Input spatial dimensions {height}x{width} too small for UNet"
            )

        # Encoder path
        for i, encoder_block in enumerate(self.encoder_blocks):
            # Apply the encoder block first
            x = encoder_block(x)

            # Store output for skip connection
            if i < self.depth:
                encoder_outputs.append(x)

                # Apply pooling after storing the skip connection
                # Only pool if height and width are big enough
                if (
                    i < self.depth
                    and height // (2 ** (i + 1)) >= 2
                    and width // (2 ** (i + 1)) >= 2
                ):
                    x = self.pool(x)
                else:
                    # If dimensions would be too small after pooling, stop the encoder path
                    break

        # Decoder path
        for i in range(len(encoder_outputs) - 1):
            # Get corresponding encoder output for skip connection
            skip_connection = encoder_outputs[-(i + 1)]

            # Check if upsampling is needed
            if x.size(3) < skip_connection.size(3) or x.size(4) < skip_connection.size(
                4
            ):
                # Upsample current features with careful handling of dimensions
                try:
                    x = self.upsampling[i](x)
                except RuntimeError:
                    # Fallback to interpolation if transposed conv fails
                    x = F.interpolate(
                        x,
                        size=(depth, skip_connection.size(3), skip_connection.size(4)),
                        mode="nearest",
                    )

            # Ensure dimensions match for concatenation
            if x.size()[2:] != skip_connection.size()[2:]:
                x = F.interpolate(x, size=skip_connection.size()[2:], mode="nearest")

            # Concatenate skip connection
            x = torch.cat([x, skip_connection], dim=1)

            # Apply decoder block
            x = self.decoder_blocks[i](x)

        # Final convolution
        x = self.final_conv(x)

        return x


class Lambda(nn.Module):
    """Lambda layer for applying arbitrary functions during forward pass"""

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class SignalProfileNetwork(nn.Module):
    """
    Network for extracting signal profile features and transforming them into
    Dirichlet concentration parameters. Specialized for inputs with depth=1.
    """

    def __init__(self, in_channels=1, hidden_channels=32, num_components=3 * 21 * 21):
        super().__init__()
        self.num_components = num_components

        # Feature extraction - keeping depth dimension intact
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(
                in_channels, hidden_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            ),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Use a simplified UNet with careful dimension handling
        self.unet = UNet3D(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            depth=1,  # Just one level of depth for 10x10 inputs
            base_filters=hidden_channels,
        )

        # More direct approach for concentration parameters that avoids dimension issues
        self.to_dirichlet = nn.Sequential(
            # Process each spatial location with shared weights
            nn.Conv3d(hidden_channels, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            # Global pooling to aggregate information
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            # Transform to concentration parameters
            nn.Linear(16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_components),
            nn.Softplus(),
            Lambda(lambda x: x + 1e-4),  # Add small epsilon
        )

        # Very robust direct pathway that will always work
        self.direct_pathway = nn.Sequential(
            # Use separable convolutions to prevent dimension issues
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            # Global pooling immediately to avoid dimension issues
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            # Transform to concentration parameters with a deep MLP
            nn.Linear(hidden_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_components),
            nn.Softplus(),
            Lambda(lambda x: x + 1e-4),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [batch, channels, depth, height, width]
            mask: Optional mask [batch, channels, depth, height, width]

        Returns:
            concentration: Parameters for Dirichlet distribution [batch, num_components]
        """
        batch_size = x.shape[0]
        _, _, depth, height, width = x.shape

        # For very small inputs, use direct pathway immediately
        if depth < 1 or height < 4 or width < 4:
            return self.direct_pathway(x)

        # Apply mask if provided
        if mask is not None:
            x = x * mask

        # Try the main pathway, fall back to direct pathway if there's an error
        try:
            # Extract features
            features = self.feature_extractor(x)

            # Process through UNet
            unet_features = self.unet(features)

            # Apply mask again if provided
            if mask is not None:
                unet_features = unet_features * mask

            # Convert to Dirichlet parameters
            concentration = self.to_dirichlet(unet_features)

        except RuntimeError as e:
            print(f"Warning: Using direct pathway due to error: {str(e)}")
            concentration = self.direct_pathway(x)

        return concentration


class UltraSimpleSignalProfileNetwork(nn.Module):
    """
    An extremely simplified network that avoids all dimension issues
    by using a 2D CNN approach on inputs with depth=1.
    """

    def __init__(self, in_channels=1, hidden_channels=32, num_components=3 * 21 * 21):
        super().__init__()
        self.num_components = num_components

        # Simple feature extraction with 2D convs wrapped as 3D
        self.features = nn.Sequential(
            # First layer - preserve depth dimension completely
            nn.Conv3d(
                in_channels, hidden_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            # Second layer - increase feature complexity
            nn.Conv3d(
                hidden_channels,
                hidden_channels * 2,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            ),
            nn.BatchNorm3d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            # Pooling only in H,W dimensions
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # Third layer - more features
            nn.Conv3d(
                hidden_channels * 2,
                hidden_channels * 4,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            ),
            nn.BatchNorm3d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            # Feature aggregation
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

        # MLP to produce concentration parameters
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_components),
            nn.Softplus(),
            # Add small epsilon to ensure valid concentration parameters
            nn.Hardtanh(min_val=1e-4, max_val=100.0),  # More stable than adding epsilon
        )

    def forward(self, x, mask=None):
        # Apply mask if provided
        if mask is not None:
            x = x * mask

        # Extract features
        x = self.features(x)

        # Generate concentration parameters
        concentration = self.mlp(x)

        return concentration


class tempppUNetIntegrator(BaseIntegrator):
    def __init__(
        self,
        image_encoder,
        metadata_encoder,
        # profile_model,
        unet,
        loss,
        q_bg,
        q_I,
        decoder,
        q_z=None,
        dmodel=64,
        mc_samples=100,
        learning_rate=1e-3,
        profile_threshold=0.001,
        signal_preprocessor=SignalPreprocessor(),
        base_alpha=0.1,
        center_alpha=10.0,
        decay_factor=2.0,
        peak_percentage=0.01,
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "mlp_encoder",
                "profile_model",
                "unet",
                "signal_preprocessor",
            ]
        )
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples
        self.profile_threshold = profile_threshold

        # Model components
        self.image_encoder = image_encoder
        self.metadata_encoder = metadata_encoder
        self.profile_model = DirichletProfile(dmodel)
        # self.profile_model = SignalAwareProfile(dmodel)

        # self.signal_preprocessor = signal_preprocessor
        # self.unet = unet
        # self.concentration_refinement = SimpleConcentrationRefinement()
        # self.unet = SignalProfileNetwork(in_channels=1)

        self.unet = UltraSimpleSignalProfileNetwork(
            in_channels=1,  # Shoebox typically has 1 channel
            num_components=3
            * 21
            * 21,  # Assuming 3x21x21 is your Dirichlet profile size
        )

        self.background_distribution = q_bg
        self.intensity_distribution = q_I
        self.q_z = q_z

        self.decoder = decoder

        self.loss_fn = loss  # Additional layers
        # self.fc_representation = Linear(dmodel * 2, dmodel)
        self.fc_representation = Linear(dmodel, dmodel)
        self.norm = nn.LayerNorm(dmodel)

        # Enable automatic optimization
        self.automatic_optimization = True

    def calculate_intensities(self, counts, qbg, qp, dead_pixel_mask):
        with torch.no_grad():
            counts = counts * dead_pixel_mask
            batch_counts = counts.unsqueeze(1)  # [batch_size x 1 x pixels]

            batch_bg_samples = (qbg.rsample([self.mc_samples]).unsqueeze(-1)).permute(
                1, 0, 2
            )
            batch_profile_samples = qp.rsample([self.mc_samples]).permute(
                1, 0, 2
            )  # [batch_size x mc_samples x pixels]

            batch_profile_samples = batch_profile_samples * dead_pixel_mask.unsqueeze(1)

            weighted_sum_intensity = (
                batch_counts - batch_bg_samples
            ) * batch_profile_samples

            weighted_sum_intensity_sum = weighted_sum_intensity.sum(-1)

            summed_squared_prf = torch.norm(batch_profile_samples, p=2, dim=-1).pow(2)

            division = weighted_sum_intensity_sum / summed_squared_prf

            weighted_sum_intensity_mean = division.mean(-1)

            weighted_sum_intensity_var = division.var(-1)

            profile_masks = batch_profile_samples > self.profile_threshold

            N_used = profile_masks.sum(-1).float()  # [batch_size × mc_samples]

            masked_counts = batch_counts * profile_masks

            thresholded_intensity = (
                masked_counts - batch_bg_samples * profile_masks
            ).sum(-1)

            thresholded_mean = thresholded_intensity.mean(-1)

            centered_thresh = thresholded_intensity - thresholded_mean.unsqueeze(-1)

            thresholded_var = (centered_thresh**2).sum(-1) / (N_used.mean(-1) + 1e-6)

            intensities = {
                "thresholded_mean": thresholded_mean,
                "thresholded_var": thresholded_var,
                "weighted_sum_intensity_mean": weighted_sum_intensity_mean,
                "weighted_sum_intensity_var": weighted_sum_intensity_var,
            }

            return intensities

    def forward(self, shoebox, dials, masks, metadata, counts):
        # Preprocess input data
        counts = torch.clamp(counts, min=0)

        # Extract representations from the shoebox and metadata
        shoebox_representation = self.image_encoder(shoebox, masks)
        meta_representation = self.metadata_encoder(metadata)

        # Combine representations and normalize
        representation = shoebox_representation + meta_representation
        representation = self.fc_representation(representation)
        representation = self.norm(representation)
        # Get distributions from confidence model
        q_I = self.intensity_distribution(representation)
        q_bg = self.background_distribution(representation)

        # Prepare shoebox and mask data for processing
        shoebox_data = shoebox[:, :, -1].reshape(-1, 3, 21, 21).unsqueeze(1)
        mask_data = masks.reshape(-1, 3, 21, 21).unsqueeze(1)

        # The key change: Replace previous preprocessing pipeline with Signal Purification Network
        # This combines shoebox → autoencoder → UNet → autoencoder → UNet → Dirichlet
        concentration = self.unet(shoebox_data, mask=mask_data)

        qp = self.profile_model(concentration)  # dirichlet

        # Get profile distribution conditioned on signal probability

        if self.q_z is not None:
            q_z = self.q_z(representation)
            qp = self.profile_model(concentration, signal_prob=q_z.probs)

            # Calculate rate using the decoder
            rate, z_samples = self.decoder(q_z, q_I, q_bg, qp)

            # Calculate expected values for reporting
            signal_prob = q_z.probs

            intensity_mean = q_I.mean.unsqueeze(-1) * signal_prob

            intensity_var = signal_prob * q_I.variance.unsqueeze(-1) + signal_prob * (
                1 - signal_prob
            ) * (q_I.mean.unsqueeze(-1) ** 2)

            return {
                "rates": rate,
                "counts": counts,
                "masks": masks,
                "qz": q_z,
                "q_I": q_I,
                "qbg": q_bg,
                "qp": qp,
                "z_samples": z_samples,
                "signal_prob": signal_prob,
                "intensity_mean": intensity_mean,
                "intensity_var": intensity_var,
                "dials_I_sum_value": dials[:, 0],
                "dials_I_sum_var": dials[:, 1],
                "dials_I_prf_value": dials[:, 2],
                "dials_I_prf_var": dials[:, 3],
                "refl_ids": dials[:, 4],
            }

        else:
            qp = self.profile_model(concentration)  # dirichlet

            # Calculate rate using the decoder
            rate = self.decoder(q_I, q_bg, qp)
            intensity_mean = q_I.mean.unsqueeze(-1)

            # Calculate expected values for reporting
            return {
                "rates": rate,
                "counts": counts,
                "masks": masks,
                "q_I": q_I,
                "qbg": q_bg,
                "qp": qp,
                "intensity_mean": intensity_mean,
                "dials_I_sum_value": dials[:, 0],
                "dials_I_sum_var": dials[:, 1],
                "dials_I_prf_value": dials[:, 2],
                "dials_I_prf_var": dials[:, 3],
                "refl_ids": dials[:, 4],
            }

    def training_step(self, batch, batch_idx):
        # Unpack batch
        shoebox, dials, masks, metadata, counts = batch

        # Get model outputs
        outputs = self(shoebox, dials, masks, metadata, counts)

        if self.q_z is not None:
            # Calculate loss.
            # Updated call: note we no longer pass a separate q_I_nosignal.
            (
                loss,
                neg_ll,
                kl,
                kl_z,
                kl_I,
                kl_bg,
                kl_p,
            ) = self.loss_fn(
                outputs["rates"],
                outputs["counts"],
                outputs["qp"],
                outputs["qz"],
                outputs["q_I"],
                outputs["qbg"],
                outputs["masks"],
            )

            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            # Log metrics
            self.log("train_loss", loss.mean())
            self.log("train_nll", neg_ll.mean())
            self.log("train_kl", kl.mean())
            self.log("train_kl_z", kl_z.mean())
            self.log("train_kl_I", kl_I.mean())
            self.log("train_kl_bg", kl_bg.mean())
            self.log("train_kl_p", kl_p.mean())
            self.log("train_signal_prob", outputs["signal_prob"].mean())

            return loss.mean()

        else:
            # Calculate loss.
            # Updated call: note we no longer pass a separate q_I_nosignal.
            (
                loss,
                neg_ll,
                kl,
                kl_I,
                kl_bg,
                kl_p,
                tv_loss,
                simpson_loss,
                entropy_loss,
            ) = self.loss_fn(
                outputs["rates"],
                outputs["counts"],
                outputs["qp"],
                outputs["q_I"],
                outputs["qbg"],
                outputs["masks"],
            )

            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            # Log metrics
            self.log("train_loss", loss.mean())
            self.log("train_nll", neg_ll.mean())
            self.log("train_kl", kl.mean())
            self.log("train_kl_I", kl_I.mean())
            self.log("train_kl_bg", kl_bg.mean())
            self.log("train_kl_p", kl_p.mean())

            return loss.mean()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        shoebox, dials, masks, metadata, counts = batch

        # Get model outputs
        outputs = self(shoebox, dials, masks, metadata, counts)

        if self.q_z is not None:
            (
                loss,
                neg_ll,
                kl,
                kl_z,
                kl_I,
                kl_bg,
                kl_p,
            ) = self.loss_fn(
                outputs["rates"],
                outputs["counts"],
                outputs["qp"],
                outputs["qz"],
                outputs["q_I"],
                outputs["qbg"],
                outputs["masks"],
            )

            # Log metrics
            self.log("val_loss", loss.mean())
            self.log("val_nll", neg_ll.mean())
            self.log("val_kl", kl.mean())
            self.log("val_kl_z", kl_z.mean())
            self.log("val_kl_I", kl_I.mean())
            self.log("val_kl_bg", kl_bg.mean())
            self.log("val_kl_p", kl_p.mean())
            self.log("val_signal_prob", outputs["signal_prob"].mean())

            return outputs

        else:
            (
                loss,
                neg_ll,
                kl,
                kl_I,
                kl_bg,
                kl_p,
                tv_loss,
                simpson_loss,
                entropy_loss,
            ) = self.loss_fn(
                outputs["rates"],
                outputs["counts"],
                outputs["qp"],
                outputs["q_I"],
                outputs["qbg"],
                outputs["masks"],
            )

            # Log metrics
            self.log("val_loss", loss.mean())
            self.log("val_nll", neg_ll.mean())
            self.log("val_kl", kl.mean())
            self.log("val_kl_I", kl_I.mean())
            self.log("val_kl_bg", kl_bg.mean())
            self.log("val_kl_p", kl_p.mean())
            self.log("val_tv_loss", tv_loss.mean())
            self.log("val_simpson_loss", simpson_loss.mean())
            self.log("val_entropy_loss", entropy_loss.mean())

            return outputs

    def predict_step(self, batch, batch_idx):
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)

        return {
            "intensity_mean": outputs["intensity_mean"],
            "intensity_var": outputs["intensity_var"],
            "signal_prob": outputs["signal_prob"],
            "refl_ids": outputs["refl_ids"],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer3D(nn.Module):
    """
    Spatial Transformer Network for 3D shoebox data, specially designed
    to handle data with a small depth dimension.
    """

    def __init__(self, input_channels=1, num_components=3 * 21 * 21):
        super(SpatialTransformer3D, self).__init__()
        self.num_components = num_components

        # Localization network - designed for inputs with depth=1-3
        self.localization = nn.Sequential(
            # First layer - preserve depth dimension
            nn.Conv3d(input_channels, 8, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU(True),
            # Second layer
            nn.Conv3d(8, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU(True),
        )

        # For 3×21×21 inputs, after two pooling layers, we get features with size 16×3×5×5
        self.loc_flatten = nn.Flatten()

        # Regressor for the transformation parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(
                16 * 3 * 5 * 5, 64
            ),  # Adjust based on your exact input dimensions
            nn.ReLU(True),
            nn.Linear(64, 12),  # 3D affine transform has 12 parameters (3×4 matrix)
        )

        # Initialize with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # 3D identity transformation matrix (3×4)
        self.fc_loc[2].bias.data.copy_(
            torch.tensor(
                [
                    1,
                    0,
                    0,
                    0,  # First row
                    0,
                    1,
                    0,
                    0,  # Second row
                    0,
                    0,
                    1,
                    0,
                ],  # Third row
                dtype=torch.float,
            )
        )

        # Feature extraction network - processes the transformed input
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv3d(16, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
        )

        # Global pooling and concentration parameters generation
        self.to_concentration = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, num_components),
            nn.Softplus(),
            nn.Hardtanh(
                min_val=1e-4, max_val=100.0
            ),  # Ensure valid concentration range
        )

    def stn(self, x):
        """Apply spatial transformer to input x"""
        # Run localization network
        xs = self.localization(x)
        xs = self.loc_flatten(xs)

        # Predict transformation parameters
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 4)

        # Generate sampling grid and apply transformation
        try:
            grid = F.affine_grid(theta, x.size(), align_corners=True)
            x_transformed = F.grid_sample(x, grid, align_corners=True)
            return x_transformed
        except RuntimeError as e:
            print(f"Warning: Error in spatial transform: {str(e)}")
            # Return original input if transformation fails
            return x

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # Apply mask if provided
        if mask is not None:
            x = x * mask

        # Apply spatial transformation
        x_transformed = self.stn(x)

        # Apply mask again after transformation if needed
        if mask is not None:
            x_transformed = x_transformed * mask

        # Extract features
        features = self.feature_extractor(x_transformed)

        # Generate concentration parameters
        concentration = self.to_concentration(features)

        return concentration


class UNetIntegrator(BaseIntegrator):
    def __init__(
        self,
        image_encoder,
        metadata_encoder,
        # profile_model,
        unet,
        loss,
        q_bg,
        q_I,
        decoder,
        q_z=None,
        dmodel=64,
        mc_samples=100,
        learning_rate=1e-3,
        profile_threshold=0.001,
        signal_preprocessor=SignalPreprocessor(),
        base_alpha=0.1,
        center_alpha=10.0,
        decay_factor=2.0,
        peak_percentage=0.01,
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "mlp_encoder",
                "profile_model",
                "unet",
                "signal_preprocessor",
            ]
        )
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples
        self.profile_threshold = profile_threshold

        # Model components
        self.image_encoder = image_encoder
        self.metadata_encoder = metadata_encoder
        self.profile_model = DirichletProfile(dmodel)
        # self.profile_model = SignalAwareProfile(dmodel)

        self.signal_preprocessor = signal_preprocessor
        self.unet = unet
        self.concentration_refinement = SimpleConcentrationRefinement()
        self.background_distribution = q_bg
        self.intensity_distribution = q_I
        self.q_z = q_z

        self.decoder = decoder

        self.loss_fn = loss  # Additional layers
        # self.fc_representation = Linear(dmodel * 2, dmodel)
        self.fc_representation = Linear(dmodel, dmodel)
        self.norm = nn.LayerNorm(dmodel)

        # Enable automatic optimization
        self.automatic_optimization = True
        self.pre_unet_norm = nn.InstanceNorm3d(1, affine=True)
        self.post_unet_norm = nn.BatchNorm3d(1)

    def calculate_intensities(self, counts, qbg, qp, dead_pixel_mask):
        with torch.no_grad():
            counts = counts * dead_pixel_mask
            batch_counts = counts.unsqueeze(1)  # [batch_size x 1 x pixels]

            batch_bg_samples = (qbg.rsample([self.mc_samples]).unsqueeze(-1)).permute(
                1, 0, 2
            )
            batch_profile_samples = qp.rsample([self.mc_samples]).permute(
                1, 0, 2
            )  # [batch_size x mc_samples x pixels]

            batch_profile_samples = batch_profile_samples * dead_pixel_mask.unsqueeze(1)

            weighted_sum_intensity = (
                batch_counts - batch_bg_samples
            ) * batch_profile_samples

            weighted_sum_intensity_sum = weighted_sum_intensity.sum(-1)

            summed_squared_prf = torch.norm(batch_profile_samples, p=2, dim=-1).pow(2)

            division = weighted_sum_intensity_sum / summed_squared_prf

            weighted_sum_intensity_mean = division.mean(-1)

            weighted_sum_intensity_var = division.var(-1)

            profile_masks = batch_profile_samples > self.profile_threshold

            N_used = profile_masks.sum(-1).float()  # [batch_size × mc_samples]

            masked_counts = batch_counts * profile_masks

            thresholded_intensity = (
                masked_counts - batch_bg_samples * profile_masks
            ).sum(-1)

            thresholded_mean = thresholded_intensity.mean(-1)

            centered_thresh = thresholded_intensity - thresholded_mean.unsqueeze(-1)

            thresholded_var = (centered_thresh**2).sum(-1) / (N_used.mean(-1) + 1e-6)

            intensities = {
                "thresholded_mean": thresholded_mean,
                "thresholded_var": thresholded_var,
                "weighted_sum_intensity_mean": weighted_sum_intensity_mean,
                "weighted_sum_intensity_var": weighted_sum_intensity_var,
            }

            return intensities

    def forward(self, shoebox, dials, masks, metadata, counts):
        # Preprocess input data
        counts = torch.clamp(counts, min=0)

        # Extract representations from the shoebox and metadata
        shoebox_representation = self.image_encoder(shoebox, masks)
        meta_representation = self.metadata_encoder(metadata)

        # Combine representations and normalize
        representation = shoebox_representation + meta_representation
        representation = self.fc_representation(representation)
        representation = self.norm(representation)

        # Get distributions from confidence model
        q_I = self.intensity_distribution(representation)
        q_bg = self.background_distribution(representation)

        # Preprocess signal for profile estimation
        preprocessed_signal = self.signal_preprocessor(
            shoebox[:, :, -1].reshape(-1, 3, 21, 21).unsqueeze(1),
            mask=masks.reshape(-1, 3, 21, 21).unsqueeze(1),
        )

        # preprocessed_signal = self.pre_unet_norm(preprocessed_signal)

        # Generate concentration parameters - EXACTLY AS IN YOUR CODE
        concentration = self.unet(
            preprocessed_signal,
            masks,
        )

        concentration = self.concentration_refinement(concentration)
        # Get profile distribution conditioned on signal probability

        if self.q_z is not None:
            q_z = self.q_z(representation)
            qp = self.profile_model(concentration, signal_prob=q_z.probs)

            # Calculate rate using the decoder
            rate, z_samples = self.decoder(q_z, q_I, q_bg, qp)

            # Calculate expected values for reporting
            signal_prob = q_z.probs

            intensity_mean = q_I.mean.unsqueeze(-1) * signal_prob

            intensity_var = signal_prob * q_I.variance.unsqueeze(-1) + signal_prob * (
                1 - signal_prob
            ) * (q_I.mean.unsqueeze(-1) ** 2)

            return {
                "rates": rate,
                "counts": counts,
                "masks": masks,
                "qz": q_z,
                "q_I": q_I,
                "qbg": q_bg,
                "qp": qp,
                "z_samples": z_samples,
                "signal_prob": signal_prob,
                "intensity_mean": intensity_mean,
                "intensity_var": intensity_var,
                "dials_I_sum_value": dials[:, 0],
                "dials_I_sum_var": dials[:, 1],
                "dials_I_prf_value": dials[:, 2],
                "dials_I_prf_var": dials[:, 3],
                "refl_ids": dials[:, 4],
            }

        else:
            qp = self.profile_model(concentration)  # dirichlet

            # Calculate rate using the decoder
            rate = self.decoder(q_I, q_bg, qp)
            intensity_mean = q_I.mean.unsqueeze(-1)

            # Calculate expected values for reporting
            return {
                "rates": rate,
                "counts": counts,
                "masks": masks,
                "q_I": q_I,
                "qbg": q_bg,
                "qp": qp,
                "intensity_mean": intensity_mean,
                "dials_I_sum_value": dials[:, 0],
                "dials_I_sum_var": dials[:, 1],
                "dials_I_prf_value": dials[:, 2],
                "dials_I_prf_var": dials[:, 3],
                "refl_ids": dials[:, 4],
            }

    def training_step(self, batch, batch_idx):
        # Unpack batch
        shoebox, dials, masks, metadata, counts = batch

        # Get model outputs
        outputs = self(shoebox, dials, masks, metadata, counts)

        if self.q_z is not None:
            # Calculate loss.
            # Updated call: note we no longer pass a separate q_I_nosignal.
            (
                loss,
                neg_ll,
                kl,
                kl_z,
                kl_I,
                kl_bg,
                kl_p,
            ) = self.loss_fn(
                outputs["rates"],
                outputs["counts"],
                outputs["qp"],
                outputs["qz"],
                outputs["q_I"],
                outputs["qbg"],
                outputs["masks"],
            )

            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            # Log metrics
            self.log("train_loss", loss.mean())
            self.log("train_nll", neg_ll.mean())
            self.log("train_kl", kl.mean())
            self.log("train_kl_z", kl_z.mean())
            self.log("train_kl_I", kl_I.mean())
            self.log("train_kl_bg", kl_bg.mean())
            self.log("train_kl_p", kl_p.mean())
            self.log("train_signal_prob", outputs["signal_prob"].mean())

            return loss.mean()

        else:
            # Calculate loss.
            # Updated call: note we no longer pass a separate q_I_nosignal.
            (
                loss,
                neg_ll,
                kl,
                kl_I,
                kl_bg,
                kl_p,
                tv_loss,
                simpson_loss,
                entropy_loss,
            ) = self.loss_fn(
                outputs["rates"],
                outputs["counts"],
                outputs["qp"],
                outputs["q_I"],
                outputs["qbg"],
                outputs["masks"],
            )

            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            # Log metrics
            self.log("train_loss", loss.mean())
            self.log("train_nll", neg_ll.mean())
            self.log("train_kl", kl.mean())
            self.log("train_kl_I", kl_I.mean())
            self.log("train_kl_bg", kl_bg.mean())
            self.log("train_kl_p", kl_p.mean())

            return loss.mean()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        shoebox, dials, masks, metadata, counts = batch

        # Get model outputs
        outputs = self(shoebox, dials, masks, metadata, counts)

        if self.q_z is not None:
            (
                loss,
                neg_ll,
                kl,
                kl_z,
                kl_I,
                kl_bg,
                kl_p,
            ) = self.loss_fn(
                outputs["rates"],
                outputs["counts"],
                outputs["qp"],
                outputs["qz"],
                outputs["q_I"],
                outputs["qbg"],
                outputs["masks"],
            )

            # Log metrics
            self.log("val_loss", loss.mean())
            self.log("val_nll", neg_ll.mean())
            self.log("val_kl", kl.mean())
            self.log("val_kl_z", kl_z.mean())
            self.log("val_kl_I", kl_I.mean())
            self.log("val_kl_bg", kl_bg.mean())
            self.log("val_kl_p", kl_p.mean())
            self.log("val_signal_prob", outputs["signal_prob"].mean())

            return outputs

        else:
            (
                loss,
                neg_ll,
                kl,
                kl_I,
                kl_bg,
                kl_p,
                tv_loss,
                simpson_loss,
                entropy_loss,
            ) = self.loss_fn(
                outputs["rates"],
                outputs["counts"],
                outputs["qp"],
                outputs["q_I"],
                outputs["qbg"],
                outputs["masks"],
            )

            # Log metrics
            self.log("val_loss", loss.mean())
            self.log("val_nll", neg_ll.mean())
            self.log("val_kl", kl.mean())
            self.log("val_kl_I", kl_I.mean())
            self.log("val_kl_bg", kl_bg.mean())
            self.log("val_kl_p", kl_p.mean())
            self.log("val_tv_loss", tv_loss.mean())
            self.log("val_simpson_loss", simpson_loss.mean())
            self.log("val_entropy_loss", entropy_loss.mean())

            return outputs

    def predict_step(self, batch, batch_idx):
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)

        return {
            "intensity_mean": outputs["intensity_mean"],
            # "intensity_var": outputs["intensity_var"],
            "refl_ids": outputs["refl_ids"],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class tempUNetIntegrator(BaseIntegrator):
    def __init__(
        self,
        cnn_encoder,
        metadata_encoder,
        # profile_model,
        confidence_intensity,
        unet,
        dmodel=64,
        mc_samples=100,
        learning_rate=1e-3,
        profile_threshold=0.001,
        signal_preprocessor=SignalPreprocessor(),
        shape=(3, 21, 21),
        base_alpha=0.1,
        center_alpha=10.0,
        decay_factor=2.0,
        peak_percentage=0.01,
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters(
            ignore=[
                "cnn_encoder",
                "mlp_encoder",
                "profile_model",
                "confidence_intensity",
                "unet",
                "signal_preprocessor",
            ]
        )
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples
        self.profile_threshold = profile_threshold

        # Model components
        self.cnn_encoder = cnn_encoder
        self.metadata_encoder = metadata_encoder
        # self.profile_model = DirichletProfile(dmodel)
        self.profile_model = SignalAwareProfile(dmodel)

        self.signal_preprocessor = signal_preprocessor
        self.unet = unet
        self.concentration_refinement = SimpleConcentrationRefinement()

        self.confidence_intensity = confidence_intensity

        self.decoder = BernoulliDecoder(mc_samples)

        self.loss_fn = L1SignalComponentLoss(
            BernoulliLoss(
                p_p=torch.distributions.dirichlet.Dirichlet(
                    create_center_focused_dirichlet_prior(
                        shape=shape,
                        base_alpha=base_alpha,
                        center_alpha=center_alpha,
                        decay_factor=decay_factor,
                        peak_percentage=peak_percentage,
                    )
                )
            ),
            l1_lambda=0.01,  # Start with a small value like 0.01
        )

        # self.loss_fn = BernoulliLoss(
        # p_p=torch.distributions.dirichlet.Dirichlet(
        # create_center_focused_dirichlet_prior(
        # shape=(3, 21, 21),
        # base_alpha=base_alpha,  # Low concentration for most elements
        # center_alpha=center_alpha,  # High concentration at center
        # decay_factor=decay_factor,  # Controls how quickly concentration falls off
        # peak_percentage=peak_percentage,  # Target ~5% of elements to have high concentration
        # )
        # )
        # )

        # Additional layers
        # self.fc_representation = Linear(dmodel * 2, dmodel)
        self.fc_representation = Linear(dmodel, dmodel)
        self.norm = nn.LayerNorm(dmodel)

        # Enable automatic optimization
        self.automatic_optimization = True

    def calculate_intensities(self, counts, qbg, qp, dead_pixel_mask):
        with torch.no_grad():
            counts = counts * dead_pixel_mask
            batch_counts = counts.unsqueeze(1)  # [batch_size x 1 x pixels]

            batch_bg_samples = (qbg.rsample([self.mc_samples]).unsqueeze(-1)).permute(
                1, 0, 2
            )
            batch_profile_samples = qp.rsample([self.mc_samples]).permute(
                1, 0, 2
            )  # [batch_size x mc_samples x pixels]

            batch_profile_samples = batch_profile_samples * dead_pixel_mask.unsqueeze(1)

            weighted_sum_intensity = (
                batch_counts - batch_bg_samples
            ) * batch_profile_samples

            weighted_sum_intensity_sum = weighted_sum_intensity.sum(-1)

            summed_squared_prf = torch.norm(batch_profile_samples, p=2, dim=-1).pow(2)

            division = weighted_sum_intensity_sum / summed_squared_prf

            weighted_sum_intensity_mean = division.mean(-1)

            weighted_sum_intensity_var = division.var(-1)

            profile_masks = batch_profile_samples > self.profile_threshold

            N_used = profile_masks.sum(-1).float()  # [batch_size × mc_samples]

            masked_counts = batch_counts * profile_masks

            thresholded_intensity = (
                masked_counts - batch_bg_samples * profile_masks
            ).sum(-1)

            thresholded_mean = thresholded_intensity.mean(-1)

            centered_thresh = thresholded_intensity - thresholded_mean.unsqueeze(-1)

            thresholded_var = (centered_thresh**2).sum(-1) / (N_used.mean(-1) + 1e-6)

            intensities = {
                "thresholded_mean": thresholded_mean,
                "thresholded_var": thresholded_var,
                "weighted_sum_intensity_mean": weighted_sum_intensity_mean,
                "weighted_sum_intensity_var": weighted_sum_intensity_var,
            }

            return intensities

    def forward(self, shoebox, dials, masks, metadata, counts):
        # Preprocess input data
        counts = torch.clamp(counts, min=0)

        # Extract representations from the shoebox and metadata
        shoebox_representation = self.cnn_encoder(shoebox, masks)
        meta_representation = self.metadata_encoder(metadata)

        # Preprocess signal for profile estimation
        preprocessed_signal = self.signal_preprocessor(
            shoebox[:, :, -1].reshape(-1, 3, 21, 21).unsqueeze(1),
            mask=masks.reshape(-1, 3, 21, 21).unsqueeze(1),
        )

        # Generate concentration parameters using UNet and refine them
        concentration = self.unet(
            preprocessed_signal,
            # shoebox[:, :, -1].reshape(-1, 3, 21, 21).unsqueeze(1),
            masks.reshape(-1, 3, 21, 21).unsqueeze(1),
        )
        # concentration = self.concentration_refinement(concentration)

        # Combine representations and normalize
        # representation = torch.cat([shoebox_representation, meta_representation], dim=1)
        representation = shoebox_representation + meta_representation
        representation = self.fc_representation(representation)
        representation = self.norm(representation)

        # Get variational distributions from the Bayesian confidence network
        # NOTE: The spike-and-slab formulation now returns:
        # q_z, q_I, q_bg, effective_signal_intensity
        q_z, q_I, q_bg, _ = self.confidence_intensity(representation)
        # qp = self.profile_model(concentration)
        qp = self.profile_model(concentration, signal_prob=q_z.probs)

        # Calculate rate using the decoder. New decoder expects (q_z, q_I, q_bg, qp)
        rate, z_samples = self.decoder(q_z, q_I, q_bg, qp)

        # Calculate expected intensity values for reporting.
        # Under spike-and-slab, if signal is absent the contribution is zero.

        signal_prob = q_z.probs  # Relaxed mean of signal existence

        intensity_mean = q_I.mean.unsqueeze(-1) * signal_prob

        intensity_var = signal_prob * q_I.variance.unsqueeze(-1) + signal_prob * (
            1 - signal_prob
        ) * (q_I.mean.unsqueeze(-1) ** 2)

        # Return everything needed for loss calculation and evaluation
        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qz": q_z,
            "q_I": q_I,
            "qbg": q_bg,
            "qp": qp,
            "z_samples": z_samples,
            "signal_prob": signal_prob,
            "intensity_mean": intensity_mean,
            "intensity_var": intensity_var,
            "dials_I_sum_value": dials[:, 0],
            "dials_I_sum_var": dials[:, 1],
            "dials_I_prf_value": dials[:, 2],
            "dials_I_prf_var": dials[:, 3],
            "refl_ids": dials[:, 4],
        }

    def training_step(self, batch, batch_idx):
        # Unpack batch
        shoebox, dials, masks, metadata, counts = batch

        # Get model outputs
        outputs = self(shoebox, dials, masks, metadata, counts)

        # Calculate loss.
        # Updated call: note we no longer pass a separate q_I_nosignal.
        (loss, neg_ll, kl, kl_z, kl_I, kl_bg, kl_p, sparsity_penalty) = self.loss_fn(
            outputs["rates"],
            outputs["counts"],
            outputs["qp"],
            outputs["qz"],
            outputs["q_I"],
            outputs["qbg"],
            outputs["masks"],
        )

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # Log metrics
        self.log("train_loss", loss.mean())
        self.log("train_nll", neg_ll.mean())
        self.log("train_kl", kl.mean())
        self.log("train_kl_z", kl_z.mean())
        self.log("train_kl_I", kl_I.mean())
        self.log("train_kl_bg", kl_bg.mean())
        self.log("train_kl_p", kl_p.mean())
        self.log("train_signal_prob", outputs["signal_prob"].mean())

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        shoebox, dials, masks, metadata, counts = batch

        # Get model outputs
        outputs = self(shoebox, dials, masks, metadata, counts)

        # Calculate validation metrics
        (loss, neg_ll, kl, kl_z, kl_I, kl_bg, kl_p, sparsity_penalty) = self.loss_fn(
            outputs["rates"],
            outputs["counts"],
            outputs["qp"],
            outputs["qz"],
            outputs["q_I"],
            outputs["qbg"],
            outputs["masks"],
        )

        # Log metrics
        self.log("val_loss", loss.mean())
        self.log("val_nll", neg_ll.mean())
        self.log("val_kl", kl.mean())
        self.log("val_kl_z", kl_z.mean())
        self.log("val_kl_I", kl_I.mean())
        self.log("val_kl_bg", kl_bg.mean())
        self.log("val_kl_p", kl_p.mean())
        self.log("val_signal_prob", outputs["signal_prob"].mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)

        return {
            "intensity_mean": outputs["intensity_mean"],
            "intensity_var": outputs["intensity_var"],
            "signal_prob": outputs["signal_prob"],
            "refl_ids": outputs["refl_ids"],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
