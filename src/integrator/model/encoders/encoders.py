import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integrator.layers import Linear

OPERATIONS = {
    "2d": {
        "conv": nn.Conv2d,
        "max_pool": nn.MaxPool2d,
        "adaptive_pool": nn.AdaptiveAvgPool2d,
    },
    "3d": {
        "conv": nn.Conv3d,
        "max_pool": nn.MaxPool3d,
        "adaptive_pool": nn.AdaptiveAvgPool3d,
    },
}


class ShoeboxEncoder(nn.Module):
    """CNN encoder producing a fixed-length embedding from a shoebox volume.

    This module applies two Conv3d + GroupNorm + relu blocks with an
    intermediate MaxPool3d, then flattens and projects to `encoder_out`.
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
    ):
        super().__init__()

        self.encoder_out = encoder_out
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
            in_channels=in_channels,
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
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))

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
        self.mish = nn.Mish()

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = self.adaptive_pool(x)
        # x = x.squeeze(-1).squeeze(-1)  # From (B, C, 1, 1) to (B, C)
        x = x.squeeze()  # From (B, C, 1, 1) to (B, C)
        x = self.fc(x)
        x = F.relu(x)

        return x


# %%
class IntensityEncoder2DMinimal(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        encoder_out: int = 64,
        conv1_out: int = 16,
        conv2_out: int = 32,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv1_out,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)

        self.fc = torch.nn.Linear(conv2_out, encoder_out)

        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.activation(self.conv1(x))  # (B, conv1_out, H, W)
        x = self.pool(x)  # (B, conv1_out, H/2, W/2)
        x = self.activation(self.conv2(x))  # (B, conv2_out, H/2, W/2)
        x = self.global_pool(x)  # (B, conv2_out, 1, 1)
        x = x.view(x.size(0), -1)  # (B, conv2_out)
        x = self.fc(x)  # (B, encoder_out)

        return x


# %%
class ProfileEncoder2DMinimal(nn.Module):
    def __init__(
        self,
        in_channels=1,
        encoder_out=64,
        conv1_out=16,
        conv2_out=32,
        pool_kernel_size=2,
        input_shape=(21, 21),
    ):
        """
        input_shape: (H, W) of shoebox
        """
        super().__init__()
        self.in_channels = in_channels
        self.encoder_out = encoder_out

        # 1st conv block
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv1_out,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size)

        # 2nd conv block
        self.conv2 = nn.Conv2d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        # nonlinearity
        self.activation = nn.SiLU()

        # compute flattened size dynamically
        self.flattened_size = self._infer_flattened_size(input_shape)

        # linear projection to embedding
        self.fc = Linear(
            in_features=self.flattened_size,
            out_features=encoder_out,
            bias=False,
        )

    def _infer_flattened_size(self, input_shape):
        with torch.no_grad():
            H, W = input_shape
            dummy = torch.zeros(1, self.in_channels, H, W)
            x = self.activation(self.conv1(dummy))
            x = self.pool(x)  # reduces to ~ (H/2, W/2)
            x = self.activation(self.conv2(x))
            return x.numel()

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))

        x = x.view(x.size(0), -1)  # (B, flattened_size)

        x = self.fc(x)
        x = self.activation(x)

        return x


if __name__ == "__main__":
    import torch

    A = torch.ones(100, 21)
    B = torch.ones(100, 21)
    C = torch.ones(100, 3)

    concentration = torch.einsum("bi,bj,bl -> blij", A, B, C).reshape(
        -1, 3 * 21 * 21
    )
