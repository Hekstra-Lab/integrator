import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear


class ShoeboxEncoder(nn.Module):
    def __init__(
        self,
        out_dim=64,
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
            in_channels=1,
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
        # assuming input is shape (B, 3*21*21, 7) and last dim is photons
        # x = x[:, :, -1].reshape(x.shape[0], 1, 3, 21, 21)
        x = x.reshape(x.shape[0], 1, 3, 21, 21)

        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        rep = F.relu(self.fc(x))
        return rep
