import torch
import torch.nn as nn
import torch.nn.functional as F


class NormFreeNet(nn.Module):
    """
    Minimal MLP for 1323-D inputs.
    d_in → 512 → 128 → out_dim
    """

    def __init__(
        self,
        out_dim: int,
        dropout_rate: float,
        d_in=1323,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 512)  # layer 1
        self.fc2 = nn.Linear(512, 128)  # layer 2
        self.fc3 = nn.Linear(128, out_dim)  # output layer
        self.dropout = nn.Dropout(dropout_rate)  # optional; set p=0.0 to disable

    def forward(self, x, mask=None):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # no effect if p=0
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NormFreeConv3D(nn.Module):
    def __init__(
        self,
        out_dim=64,
        in_channels=1,
        input_shape=(21, 21, 3),  # (H, W, D)
    ):
        super().__init__()

        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.in_channels = in_channels

        # Conv layers
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )

        self.pool = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2), ceil_mode=True
        )

        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding=(0, 0, 0),
        )

        # Calculate flattened size
        self.flattened_size = self._infer_flattened_size(
            input_shape=input_shape, in_channels=in_channels
        )

        # Final linear layer
        self.fc = nn.Linear(self.flattened_size, out_dim)

    def _infer_flattened_size(self, input_shape, in_channels):
        # input_shape: (H, W, D)
        with torch.no_grad():
            # (B, C, D, H, W)
            dummy = torch.zeros(
                1, in_channels, input_shape[2], input_shape[0], input_shape[1]
            )
            x = self.pool(self.act(self.conv1(dummy)))
            x = self.act(self.conv2(x))
            return x.numel()

    def forward(self, x, mask=None):
        # Reshape input to [B, C, D, H, W]
        if x.ndim != 5:
            x = x.view(x.shape[0], self.in_channels, 3, 21, 21)

        # Apply first conv and pooling
        x = self.act(self.conv1(x))
        x = self.pool(x)

        # Apply second conv
        x = self.act(self.conv2(x))

        # Flatten and apply fully connected layer
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)

        return x
