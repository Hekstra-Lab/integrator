import torch
import torch.nn as nn
import torch.nn.functional as F


class ShoeboxEncoder(torch.nn.Module):
    def __init__(
        self,
        input_shape=(3, 21, 21),
        in_channels=1,
        out_dim=64,
        conv1_out_channels=16,
        conv1_kernel=(1, 3, 3),
        conv1_padding=(0, 1, 1),
        norm1_num_groups=4,
        pool_kernel=(1, 2, 2),
        pool_stride=(1, 2, 2),
        conv2_out_channels=32,
        conv2_kernel=(3, 3, 3),
        conv2_padding=(0, 0, 0),
        norm2_num_groups=4,
    ):
        """
        Args:
            input_shape ():
            in_channels ():
            out_dim ():
            conv1_out_channels ():
            conv1_kernel ():
            conv1_padding ():
            norm1_num_groups ():
            pool_kernel ():
            pool_stride ():
            conv2_out_channels ():
            conv2_kernel ():
            conv2_padding ():
            norm2_num_groups ():
        """
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=conv1_out_channels,
            kernel_size=conv1_kernel,
            padding=conv1_padding,
        )
        self.norm1 = nn.GroupNorm(norm1_num_groups, conv1_out_channels)
        self.pool = nn.MaxPool3d(
            kernel_size=pool_kernel, stride=pool_stride, ceil_mode=True
        )
        self.conv2 = nn.Conv3d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=conv2_kernel,
            padding=conv2_padding,
        )
        self.norm2 = nn.GroupNorm(norm2_num_groups, conv2_out_channels)

        # Dynamically calculate flattened size
        self.flattened_size = self._infer_flattened_size(
            input_shape=input_shape, in_channels=in_channels
        )
        self.fc = nn.Linear(self.flattened_size, out_dim)

    def _infer_flattened_size(self, input_shape, in_channels):
        # input_shape: (H, W, D)
        with torch.no_grad():
            dummy = torch.zeros(
                1, in_channels, input_shape[0], input_shape[1], input_shape[2]
            )  # (B, C, D, H, W)
            x = self.pool(F.relu(self.norm1(self.conv1(dummy))))
            x = F.relu(self.norm2(self.conv2(x)))
            return x.numel()

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.tensor):
            mask (torch.tensor):

        Returns:

        """
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))


class IntensityEncoder(torch.nn.Module):
    def __init__(
        self,
        input_shape=(3, 21, 21),  # (D,H,W)
        in_channels=1,
        out_dim=64,
        conv1_out_channels=16,
        conv1_kernel=(1, 3, 3),
        conv1_padding=(0, 1, 1),
        norm1_num_groups=4,
        pool_kernel=(1, 2, 2),
        pool_stride=(1, 2, 2),
        conv2_out_channels=32,
        conv2_kernel=(3, 3, 3),
        conv2_padding=(0, 0, 0),
        norm2_num_groups=4,
        conv3_out_channels=64,
        conv3_kernel=(3, 3, 3),
        conv3_padding=(1, 1, 1),
        norm3_num_groups=8,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=conv1_out_channels,
            kernel_size=conv1_kernel,
            padding=conv1_padding,
        )
        self.norm1 = nn.GroupNorm(norm1_num_groups, conv1_out_channels)

        # Pooling layer
        self.pool = nn.MaxPool3d(
            kernel_size=pool_kernel, stride=pool_stride, ceil_mode=True
        )

        self.conv2 = nn.Conv3d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=conv2_kernel,
            padding=conv2_padding,
        )
        self.norm2 = nn.GroupNorm(norm2_num_groups, conv2_out_channels)

        self.conv3 = nn.Conv3d(
            in_channels=conv2_out_channels,
            out_channels=conv3_out_channels,
            kernel_size=conv3_kernel,
            padding=conv3_padding,
        )
        self.norm3 = nn.GroupNorm(norm3_num_groups, conv3_out_channels)

        self.adaptive_pool = nn.AdaptiveAvgPool3d(
            1
        )  # Output: (batch, channels, 1, 1, 1)

        self.fc = nn.Linear(conv3_out_channels, out_dim)

    def forward(self, x, mask=None):
        # First conv + norm + activation
        x = F.relu(self.norm1(self.conv1(x)))

        # Pooling
        x = self.pool(x)

        # Second conv + norm + activation
        x = F.relu(self.norm2(self.conv2(x)))

        # Third conv + norm + activation
        x = F.relu(self.norm3(self.conv3(x)))

        # Adaptive average pooling - reduces to (batch, channels, 1, 1, 1)
        x = self.adaptive_pool(x)

        # Squeeze and apply final linear layer
        x = x.squeeze()  # Remove dimensions of size 1
        return F.relu(self.fc(x))
