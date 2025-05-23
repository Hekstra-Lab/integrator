import torch
import torch.nn.functional as F
import torch.nn as nn


class ShoeboxEncoder(nn.Module):
    def __init__(
        self,
        input_shape=(3, 21, 21),  # (D,H,W)
        in_channels=21,
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
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))
