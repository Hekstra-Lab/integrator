import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

OPERATIONS = {
    "2d": {
        "conv": nn.Conv2d,
        "max_pool": nn.MaxPool2d,
        "adaptive_pool": nn.AdaptiveAvgPool2d,
        "dropout": nn.Dropout2d,
    },
    "3d": {
        "conv": nn.Conv3d,
        "max_pool": nn.MaxPool3d,
        "adaptive_pool": nn.AdaptiveAvgPool3d,
        "dropout": nn.Dropout3d,
    },
}


class ShoeboxEncoder(nn.Module):
    """CNN encoder producing a fixed-length embedding from a shoebox volume.

    This module applies two Conv3d + GroupNorm + relu blocks with an
    intermediate MaxPool3d, then flattens and projects to `encoder_out`.

    Args:
        use_coord_channels: If True, concatenate normalized (z, y, x)
            coordinate channels to the input before the first conv. This
            gives the encoder explicit spatial-position information that
            pure convolutions can only approximate via boundary effects.
            For Bragg shoeboxes (peaks centered by DIALS), absolute
            position carries real information.
        dropout: Probability of zeroing entire channels after each
            conv+norm+relu block (using ``DropoutNd``, not ``Dropout``).
            Channel-wise dropout for CNNs breaks the encoder's ability
            to rely on any single feature map, which targets the
            encoder-decoder co-adaptation form of overfitting where the
            encoder routes training-reflection features to a W-friendly
            subspace. Default 0.0 disables dropout.
    """

    def __init__(
        self,
        data_dim: str,
        input_shape: tuple[int, ...] = (3, 21, 21),
        in_channels: int = 1,
        encoder_out: int = 64,
        conv1_out_channels: int = 16,
        conv1_kernel_size: tuple[int, int, int] = (1, 3, 3),
        conv1_padding: tuple[int, int, int] = (0, 1, 1),
        norm1_num_groups: int = 4,
        pool_kernel_size: tuple[int, int, int] = (1, 2, 2),
        pool_stride: tuple[int, ...] = (1, 2, 2),
        conv2_out_channels: int = 32,
        conv2_kernel_size: tuple[int, int, int] = (3, 3, 3),
        conv2_padding: tuple[int, int, int] = (0, 0, 0),
        norm2_num_groups: int = 4,
        use_coord_channels: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder_out = encoder_out
        self.use_coord_channels = use_coord_channels
        self.dropout_p = float(dropout)
        if self.dropout_p > 0:
            # Channel-wise dropout: zero entire feature maps. More
            # effective than element-wise dropout for CNNs because
            # adjacent pixels share the same feature channel anyway.
            dropout_cls = OPERATIONS[data_dim]["dropout"]
            self.dropout1 = dropout_cls(self.dropout_p)
            self.dropout2 = dropout_cls(self.dropout_p)
        else:
            self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()

        # Precompute coord channels as a buffer so they move with the
        # module (.to(device), .cuda(), etc.) without being parameters.
        effective_in_channels = in_channels

        self.conv1 = OPERATIONS[data_dim]["conv"](
            in_channels=effective_in_channels,
            out_channels=conv1_out_channels,
            kernel_size=conv1_kernel_size,
            padding=conv1_padding,
        )
        self.norm1 = nn.GroupNorm(
            num_groups=norm1_num_groups,
            num_channels=conv1_out_channels,
        )

        self.pool = OPERATIONS[data_dim]["max_pool"](
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            ceil_mode=True,
        )
        self.conv2 = OPERATIONS[data_dim]["conv"](
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=conv2_kernel_size,
            padding=conv2_padding,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=norm2_num_groups,
            num_channels=conv2_out_channels,
        )

        # Dynamically calculate flattened size
        self.flattened_size = self._infer_flattened_size(
            input_shape=input_shape,
            in_channels=effective_in_channels,
        )
        self.fc = torch.nn.Linear(
            in_features=self.flattened_size,
            out_features=encoder_out,
        )

    def _infer_flattened_size(self, input_shape, in_channels):
        # input_shape: (H, W, D)
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                in_channels,
                *input_shape,
            )  # (B, C, D, H, W)
            x = self.pool(F.relu(self.norm1(self.conv1(dummy))))
            x = F.relu(self.norm2(self.conv2(x)))
            return x.numel()

    def forward(self, x):
        if self.use_coord_channels:
            # Broadcast the precomputed coord grid across the batch and
            # concatenate on the channel dim.
            coords: Tensor = self.coord_channels  # type: ignore[assignment]
            coords = coords.unsqueeze(0).expand(
                x.shape[0], *([-1] * coords.ndim)
            )
            x = torch.cat([x, coords], dim=1)
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = F.relu(x)

        return x


class IntensityEncoder(nn.Module):
    def __init__(
        self,
        data_dim: str,
        in_channels=1,
        encoder_out=64,
        conv1_out_channels=16,
        conv1_kernel_size=(3, 3),
        conv1_padding=(1, 1),
        norm1_num_groups=4,
        pool_kernel_size=(2, 2),
        pool_stride=(2, 2),
        conv2_out_channels=32,
        conv2_kernel_size=(3, 3),
        conv2_padding=(0, 0),
        norm2_num_groups=4,
        conv3_out_channels=64,
        conv3_kernel_size=(3, 3),
        conv3_padding=(1, 1),
        norm3_num_groups=8,
    ):
        super().__init__()

        self.conv1 = OPERATIONS[data_dim]["conv"](
            in_channels=in_channels,
            out_channels=conv1_out_channels,
            kernel_size=conv1_kernel_size,
            padding=conv1_padding,
        )
        self.norm1 = nn.GroupNorm(
            num_groups=norm1_num_groups,
            num_channels=conv1_out_channels,
        )

        # self.pool = nn.MaxPool2d(
        self.pool = OPERATIONS[data_dim]["max_pool"](
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            ceil_mode=True,
        )

        # self.conv2 = nn.Conv2d(
        self.conv2 = OPERATIONS[data_dim]["conv"](
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=conv2_kernel_size,
            padding=conv2_padding,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=norm2_num_groups,
            num_channels=conv2_out_channels,
        )

        self.conv3 = OPERATIONS[data_dim]["conv"](
            in_channels=conv2_out_channels,
            out_channels=conv3_out_channels,
            kernel_size=conv3_kernel_size,
            padding=conv3_padding,
        )
        self.norm3 = nn.GroupNorm(
            num_groups=norm3_num_groups,
            num_channels=conv3_out_channels,
        )

        self.adaptive_pool = OPERATIONS[data_dim]["adaptive_pool"](
            1
        )  # Output: (batch, channels, 1, 1)

        self.fc = torch.nn.Linear(
            in_features=conv3_out_channels,
            out_features=encoder_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.flatten(1)  # From (B, C, 1, ...) to (B, C)
        x = self.fc(x)
        x = F.relu(x)

        return x
