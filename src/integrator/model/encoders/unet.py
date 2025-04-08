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
class BasicResBlock3D(nn.Module):
    """
    A simple 3D residual block.
    By default, it keeps the same spatial resolution unless `stride>1`.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Convert 'stride' to a tuple if it's just an integer
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,  # "same" padding for 3D
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # If in/out channels or stride differ, we project the identity
        # to match shape for the residual addition.
        self.downsample = None
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class UNetDirichletConcentration(nn.Module):
    """
    A 3D ResNet-style encoder-decoder that:
      - Keeps the depth dimension (Z=3) unchanged (stride=(1,2,2) in encoder).
      - Uses interpolation to ensure exact output size matching.
      - Ensures final shape == input shape (except for channel count).
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=16, eps=1e-6):
        super().__init__()
        self.base_channels = base_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # -----------------------
        # Encoder
        # -----------------------
        # Stage 1: stride=1
        self.enc1 = BasicResBlock3D(in_channels, base_channels, stride=(1, 1, 1))
        # Downsample 1 in H,W only (stride=(1,2,2)):
        self.enc2 = BasicResBlock3D(base_channels, base_channels * 2, stride=(1, 2, 2))

        # Another block at this scale (stride=1)
        self.enc3 = BasicResBlock3D(
            base_channels * 2, base_channels * 2, stride=(1, 1, 1)
        )
        # Downsample 2 in H,W only:
        self.enc4 = BasicResBlock3D(
            base_channels * 2, base_channels * 4, stride=(1, 2, 2)
        )

        # Deeper features (stride=1)
        self.enc5 = BasicResBlock3D(
            base_channels * 4, base_channels * 4, stride=(1, 1, 1)
        )

        # -----------------------
        # Decoder - Using interpolation instead of transposed convolutions
        # -----------------------

        # Upsampling block 1
        self.up_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            nn.Conv3d(
                base_channels * 4,
                base_channels * 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.dec1 = BasicResBlock3D(
            base_channels * 2, base_channels * 2, stride=(1, 1, 1)
        )

        # Upsampling block 2
        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            nn.Conv3d(
                base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.dec2 = BasicResBlock3D(base_channels, base_channels, stride=(1, 1, 1))

        # Final projection to out_channels
        self.final_conv = nn.Conv3d(
            in_channels=base_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.eps = eps

    def forward(self, x, mask=None):
        """
        Input: [B, in_channels, Z=3, H=21, W=21]
        Output: [B, out_channels, 3, 21, 21]
        """
        # Store original input size for final resize
        if mask is not None:
            x = (x[:, :, -1] * mask).view(-1, 1, 3, 21, 21)
        else:
            x = x[:, :, -1].view(-1, 1, 3, 21, 21)
        input_shape = x.shape

        # 1) Encoder
        x1 = self.enc1(x)  # [B, base_channels, 3, 21, 21]
        x2 = self.enc2(x1)  # [B, base_channels*2, 3, 10/11, 10/11]
        x3 = self.enc3(x2)  # same shape
        x4 = self.enc4(x3)  # [B, base_channels*4, 3, 5/6, 5/6]
        x5 = self.enc5(x4)  # same shape

        # 2) Decoder with interpolation-based upsampling and skip connections
        # First upsampling + skip connection from encoder
        x = self.up_conv1(x5)
        # Add skip connection from x3 (matching feature maps)
        if x.shape[2:] != x3.shape[2:]:
            x3_resized = F.interpolate(
                x3, size=x.shape[2:], mode="trilinear", align_corners=False
            )
        else:
            x3_resized = x3
        x = x + x3_resized  # Skip connection
        x = self.dec1(x)

        # Second upsampling + skip connection from encoder
        x = self.up_conv2(x)
        # Add skip connection from x1 (matching feature maps)
        if x.shape[2:] != x1.shape[2:]:
            x1_resized = F.interpolate(
                x1, size=x.shape[2:], mode="trilinear", align_corners=False
            )
        else:
            x1_resized = x1
        x = x + x1_resized  # Skip connection
        x = self.dec2(x)

        # Final projection
        x = self.final_conv(x)

        # Final resize to guarantee exact match with input dimensions
        if x.shape[2:] != input_shape[2:]:
            x = F.interpolate(
                x, size=input_shape[2:], mode="trilinear", align_corners=False
            )

        if mask is not None:
            x = x.view(-1, 3 * 21 * 21) * mask
        else:
            x = x.view(-1, 3 * 21 * 21)

        return x


# Test the fixed model
def test_model():
    model = UNetDirichletConcentration(in_channels=1, out_channels=1, base_channels=8)
    x = torch.randn(10, 1323, 7)
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)  # Should be [2, 1, 3, 21, 21]
    return out.shape == x.shape


# x = x.view(-1, 1, 3, 21, 21)
# x1 = model.enc1(x)
# x2 = model.enc2(x1)
# x3 = model.enc3(x2)
# x4 = model.enc4(x3)
# x5 = model.enc5(x4)

# x = model.up_conv1(x5)
# x = model.dec1(x)

# x = model.up_conv2(x)

# x = model.dec2(x)

# x = model.final_conv(x)


# x.shape[2:]
