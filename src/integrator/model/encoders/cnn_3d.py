import torch
from integrator.layers import Residual, MLP, MeanPool
from integrator.model.encoders import BaseEncoder


class CNN_3d(torch.nn.Module):
    def __init__(self, Z=3, H=21, W=21, conv_channels=64, use_norm=True):
        super().__init__()
        self.Z = Z
        self.H = H
        self.W = W

        # Simple counts pathway only
        self.features = torch.nn.Sequential(
            # Initial 3D convolution
            torch.nn.Conv3d(
                in_channels=1,  # Just counts
                out_channels=conv_channels,
                kernel_size=3,
                padding=1,
            ),
            # Optional normalization
            torch.nn.BatchNorm3d(conv_channels) if use_norm else torch.nn.Identity(),
            torch.nn.ReLU(inplace=True),
            # Second conv block
            torch.nn.Conv3d(
                in_channels=conv_channels,
                out_channels=conv_channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.BatchNorm3d(conv_channels) if use_norm else torch.nn.Identity(),
            torch.nn.ReLU(inplace=True),
        )

        # Global pooling
        self.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))

    def reshape_input(self, x, mask=None):
        # Extract counts and reshape to 3D volume
        counts = x[..., -1]  # Shape: [batch_size, Z*H*W]
        counts = counts.view(-1, self.Z, self.H, self.W)
        counts = counts.unsqueeze(1)  # Add channel dim: [batch_size, 1, Z, H, W]

        if mask is not None:
            mask = mask.view(-1, 1, self.Z, self.H, self.W)
            counts = counts * mask

        return counts

    def forward(self, x, mask=None):
        # Reshape input
        x = self.reshape_input(x, mask)

        # Process through CNN
        x = self.features(x)

        # Global pooling and flatten
        x = self.pool(x)
        x = torch.flatten(x, 1)

        return x
