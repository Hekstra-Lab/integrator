import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear


class tempShoeboxEncoder(nn.Module):
    def __init__(
        self,
        out_dim=64,
        conv1_in_channels=1,
        conv1_out_channels=16,
        norm1_num_groups=4,
        norm1_num_channels=16,
        conv2_in_channels=16,
        conv2_out_channels=32,
        norm2_num_groups=4,
        norm2_num_channels=32,
    ):
        """
        Args:
            out_dim: Output dimension of the encoded representation.
        """
        super(ShoeboxEncoder, self).__init__()
        # The input shape is  (B, 1, 3, 21, 21).
        self.conv1 = nn.Conv3d(
            in_channels=conv1_in_channels,
            out_channels=conv1_out_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        )
        self.norm1 = nn.GroupNorm(
            num_groups=norm1_num_groups, num_channels=norm1_num_channels
        )

        # Pooling applied only across height and width.
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2), ceil_mode=True
        )

        # Convolution layer #2: Use a kernel that spans depth
        self.conv2 = nn.Conv3d(
            in_channels=conv2_in_channels,
            out_channels=conv2_out_channels,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=0,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=norm2_num_groups, num_channels=norm2_num_channels
        )

        # After conv1, input shape is: (B, 16, 3, 21, 21);
        # after pooling: (B, 16, 3, approx. ceil(21/2)=11, ceil(21/2)=11);
        # after conv2: depth: 3-3+1=1; spatial dims: 11-3+1=9 (assuming exact arithmetic).

        flattened_size = 32 * 1 * 9 * 9
        self.fc = nn.Linear(flattened_size, out_dim)

    def forward(self, x, mask=None):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        rep = F.relu(self.fc(x))
        return rep


class temp2ShoeboxEncoder(nn.Module):
    def __init__(
        self,
        input_channels=1,
        out_dim=64,
        conv1_out_channels=16,
        conv2_out_channels=32,
        norm1_num_groups=4,
        norm2_num_groups=4,
        input_shape=(3, 21, 21),  # (D, H, W) — needed for output shape inference
        kernel_size=(1, 3, 3),
        padding=(0, 1, 1),
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=input_channels,
            out_channels=conv1_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.norm1 = nn.GroupNorm(
            num_groups=norm1_num_groups,
            num_channels=conv1_out_channels,
        )

        self.pool = nn.MaxPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            ceil_mode=True,
        )

        self.conv2 = nn.Conv3d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=0,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=norm2_num_groups,
            num_channels=conv2_out_channels,
        )

        # Dynamically calculate flattened size
        self.flattened_size = self._get_flattened_size(input_channels, input_shape)
        self.fc = nn.Linear(self.flattened_size, out_dim)

    def _get_flattened_size(self, in_channels, input_shape):
        # Simulate a forward pass to infer the output shape
        with torch.no_grad():
            x = torch.zeros(1, in_channels, *input_shape)
            x = self.pool(F.relu(self.norm1(self.conv1(x))))
            x = F.relu(self.norm2(self.conv2(x)))
            return x.numel()

    def forward(self, x, mask=None):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        rep = F.relu(self.fc(x))
        return rep


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride
        )
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(4, out_channels)

        self.skip = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(4, out_channels),
            )
            if downsample or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class ShoeboxEncoder(nn.Module):
    def __init__(self, in_channels=21, out_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )

        self.layer1 = ResidualBlock3D(32, 32)
        self.layer2 = ResidualBlock3D(32, 64, stride=2, downsample=True)
        self.layer3 = ResidualBlock3D(64, 64)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = Linear(64, out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
