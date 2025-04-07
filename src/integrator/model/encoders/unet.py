import torch
import torch.nn as nn
import torch.nn.functional as F


class tempUNetDirichletConcentration(torch.nn.Module):
    def __init__(self, in_channels=1, Z=3, H=21, W=21, concentration_scale=1.0):
        super().__init__()
        self.Z = Z
        self.H = H
        self.W = W
        self.concentration_scale = concentration_scale

        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # Decoder path
        self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)

        self.dec1 = nn.Sequential(
            nn.Conv3d(
                32, 16, kernel_size=3, padding=1
            ),  # 32 channels due to skip connection
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        # Final layer - outputs Dirichlet concentration parameters
        self.final = nn.Conv3d(16, 1, kernel_size=1)
        self.softplus = nn.Softplus()

    def reshape_input(self, x, mask=None):
        # Extract counts and reshape to 3D volume
        # counts = x[..., -1]  # Shape: [batch_size, Z*H*W]
        counts = x.view(-1, self.Z, self.H, self.W)
        counts = counts.unsqueeze(1)  # Add channel dim: [batch_size, 1, Z, H, W]

        if mask is not None:
            mask = mask.view(-1, 1, self.Z, self.H, self.W)
            counts = counts * mask

        return counts

    def forward(self, x, mask=None):
        # Reshape input if needed
        if x.dim() < 5:  # If not already [batch, channel, Z, H, W]
            x = self.reshape_input(x, mask)

        # Encoder path
        e1 = self.enc1(x)
        e1_pool = self.pool1(e1)
        e2 = self.enc2(e1_pool)

        # Bottleneck
        b = self.bottleneck(e2)

        # Decoder path with skip connections
        d1 = self.upconv1(b)
        # Ensure dimensions match for skip connection
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(
                d1, size=e1.shape[2:], mode="trilinear", align_corners=False
            )
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Final layer to generate concentration parameters
        alphas_raw = self.final(d1)

        # Ensure positive concentration parameters
        alphas = self.softplus(alphas_raw.view(x.shape[0], -1)) + 1e-6

        # Apply concentration scale (higher values = sharper distributions)
        alphas = alphas * self.concentration_scale


import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm3d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attention_map = self.attention(x)
        return x * attention_map


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
        )
        self.identity_map = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.identity_map(x)
        out = self.conv_block(x)
        out += identity
        return self.relu(out)


# %%
class UNetDirichletConcentration(nn.Module):
    def __init__(
        self, in_channels=1, Z=3, H=21, W=21, concentration_scale=1.0, dropout_rate=0.5
    ):
        super().__init__()
        self.Z = Z
        self.H = H
        self.W = W
        self.concentration_scale = concentration_scale

        # Encoder path with residual blocks - single pooling for small Z dimension
        self.enc1 = ResidualBlock(in_channels, 32)
        self.attn1 = AttentionBlock(32)
        # Use careful pooling that won't reduce Z dimension too much
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout1 = nn.Dropout3d(dropout_rate)

        self.enc2 = ResidualBlock(32, 64)
        self.attn2 = AttentionBlock(64)

        # Bottleneck with spatial and channel attention
        self.bottleneck = ResidualBlock(64, 64)
        self.bottleneck_attn = AttentionBlock(64)

        # Decoder path
        self.upconv1 = nn.ConvTranspose3d(
            64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec1 = ResidualBlock(64, 32)  # 64 from concat
        self.dec1_attn = AttentionBlock(32)

        # Final layer with refined localization
        self.final = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=1),
        )

        self.softplus = nn.Softplus()

        # Noise reduction branch - learns to identify and suppress noise
        self.noise_filter = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=1),
            nn.Sigmoid(),  # Output is a noise suppression mask
        )

    def reshape_input(self, x, mask=None):
        # Handle the flattened input format
        if x.dim() == 2:  # [batch_size, Z*H*W]
            counts = x.view(-1, self.Z, self.H, self.W)
        else:
            counts = x.view(-1, self.Z, self.H, self.W)

        counts = counts.unsqueeze(1)  # Add channel dim: [batch_size, 1, Z, H, W]

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.view(-1, self.Z, self.H, self.W)
            mask = mask.unsqueeze(1)  # Add channel dim
            counts = counts * mask

        return counts

    def forward(self, x, mask=None):
        # Reshape input if needed
        if x.dim() < 5:  # If not already [batch, channel, Z, H, W]
            x = self.reshape_input(x, mask)

        # Store original input dimensions
        input_shape = x.shape[2:]

        # Encoder path with attention and residual connections
        e1 = self.enc1(x)
        e1_attn = self.attn1(e1)
        e1_pool = self.pool1(e1_attn)  # Only pools H and W dimensions
        e1_pool = self.dropout1(e1_pool)

        e2 = self.enc2(e1_pool)
        e2_attn = self.attn2(e2)

        # Bottleneck with attention
        b = self.bottleneck(e2_attn)
        b_attn = self.bottleneck_attn(b)

        # Decoder path with skip connections
        d1 = self.upconv1(b_attn)

        # Handle any dimension mismatch for the skip connection
        if d1.shape[2:] != e1_attn.shape[2:]:
            d1 = F.interpolate(
                d1, size=e1_attn.shape[2:], mode="trilinear", align_corners=False
            )

        d1 = torch.cat([d1, e1_attn], dim=1)
        d1 = self.dec1(d1)
        d1_attn = self.dec1_attn(d1)

        # Create noise suppression mask
        noise_mask = self.noise_filter(d1_attn)

        # Final layer to generate concentration parameters
        alphas_raw = self.final(d1_attn)

        # Apply the noise suppression mask
        alphas_raw = alphas_raw * noise_mask

        # Ensure output matches input dimensions
        if alphas_raw.shape[2:] != input_shape:
            alphas_raw = F.interpolate(
                alphas_raw, size=input_shape, mode="trilinear", align_corners=False
            )

        # Ensure positive concentration parameters with higher minimum value
        alphas = self.softplus(alphas_raw.view(x.shape[0], -1)) + 1e-4

        # Apply concentration scale (higher values = sharper distributions)
        alphas = alphas * self.concentration_scale

        return alphas


class FocalDirichletLoss(nn.Module):
    """
    Focal loss variant for Dirichlet distribution with noise penalty
    """

    def __init__(self, gamma=2.0, eps=1e-6, noise_penalty=0.1):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.noise_penalty = noise_penalty

    def forward(self, alphas, targets, peak_mask=None):
        # Convert targets to one-hot encoding if needed
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=alphas.shape[1])

        # Calculate probabilities (expectations of Dirichlet distribution)
        alpha_0 = torch.sum(alphas, dim=1, keepdim=True)
        probs = alphas / alpha_0

        # Apply focal weighting
        pt = torch.sum(probs * targets, dim=1)
        focal_weight = (1 - pt + self.eps) ** self.gamma

        # Calculate negative log likelihood of Dirichlet distribution
        loss = -torch.sum(targets * torch.log(probs + self.eps), dim=1)

        # Apply focal weighting to the loss
        weighted_loss = focal_weight * loss

        # Add noise penalty if peak mask is provided
        if peak_mask is not None:
            # Penalize high probabilities in non-peak regions
            noise_regions = 1.0 - peak_mask
            noise_loss = torch.sum(probs * noise_regions, dim=1)
            weighted_loss = weighted_loss + self.noise_penalty * noise_loss

        return weighted_loss.mean()
