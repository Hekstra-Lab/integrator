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


class ProfileEncoder(nn.Module):
    """CNN encoder producing a fixed-length embedding from a shoebox volume.

    This module applies two Conv3d + GroupNorm + relu blocks with an
    intermediate MaxPool3d, then flattens and projects to `encoder_out`.

    When `position_dim > 0`, detector position features are concatenated
    to the flattened CNN output before the MLP head, giving the encoder
    spatial information about the shoebox.

    Args:
        dropout: Probability of zeroing entire channels after each
            conv+norm+relu block (using `DropoutNd`, not `Dropout`).
        position_dim: Number of position features to concatenate (0=off,
            2=raw (x,y), or higher for Fourier features).
        position_fourier_order: If > 0, expand (x,y) into sinusoidal
            Fourier features: [sin(πkx), cos(πkx), ...] for k=1..order.
            Gives 4*order features. position_dim is ignored when set.
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
        dropout: float = 0.0,
        position_dim: int = 0,
        position_fourier_order: int = 0,
    ):
        super().__init__()

        self.encoder_out = encoder_out
        self.dropout_p = float(dropout)
        if self.dropout_p > 0:
            dropout_cls = OPERATIONS[data_dim]["dropout"]
            self.dropout1 = dropout_cls(self.dropout_p)
            self.dropout2 = dropout_cls(self.dropout_p)
        else:
            self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()

        self.position_fourier_order = position_fourier_order
        if position_fourier_order > 0:
            self.n_pos_features = 4 * position_fourier_order
        else:
            self.n_pos_features = position_dim

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
            in_features=self.flattened_size + self.n_pos_features,
            out_features=encoder_out,
        )

    def _infer_flattened_size(self, input_shape, in_channels):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_shape)
            x = self.pool(F.relu(self.norm1(self.conv1(dummy))))
            x = F.relu(self.norm2(self.conv2(x)))
            return x.numel()

    def _position_features(self, pos: Tensor) -> Tensor:
        """(B, 2) normalized position → (B, n_pos_features)."""
        if self.position_fourier_order > 0:
            freqs = torch.arange(
                1,
                self.position_fourier_order + 1,
                device=pos.device,
                dtype=pos.dtype,
            )
            # pos: (B, 2), freqs: (K,) → angles: (B, 2, K)
            angles = pos.unsqueeze(-1) * freqs * torch.pi
            return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(1)
        return pos

    def forward(self, x: Tensor, position: Tensor | None = None) -> Tensor:
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)

        if self.n_pos_features > 0:
            if position is not None:
                pos_feat = self._position_features(position)
            else:
                pos_feat = x.new_zeros(x.size(0), self.n_pos_features)
            x = torch.cat([x, pos_feat], dim=-1)

        x = self.fc(x)
        x = F.relu(x)

        return x


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: Tensor) -> Tensor:
        b, c = x.shape[:2]
        s = x.view(b, c, -1).mean(dim=-1)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(b, c, *([1] * (x.dim() - 2)))


class _ResBlock2d(nn.Module):
    """Pre-activation residual block with optional SE and channel change."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        groups: int = 8,
        se_reduction: int = 4,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.se = _SEBlock(out_ch, se_reduction)

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = F.silu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        return F.silu(out + self.skip(x))


class ResidualProfileEncoder(nn.Module):
    """Mini-ResNet encoder for 2D shoeboxes.

    Architecture:
      5×5 stem → ResBlock(C1→C1) → ResBlock(C1→C2, stride=2) →
      ResBlock(C2→C2) → GlobalAvgPool → Linear → SiLU

    Compared to ProfileEncoder:
      - 5×5 first kernel for larger initial receptive field
      - Residual connections + SE attention
      - Global average pooling instead of flatten (eliminates huge FC layer)
      - SiLU activation throughout
    """

    def __init__(
        self,
        encoder_out: int = 64,
        in_channels: int = 1,
        stem_channels: int = 32,
        block_channels: int = 64,
        groups: int = 8,
        se_reduction: int = 4,
        dropout: float = 0.0,
        # unused — accepted for compatibility with factory/presets
        data_dim: str = "2d",
        input_shape: tuple[int, ...] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.encoder_out = encoder_out

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 5, padding=2, bias=False),
            nn.GroupNorm(min(groups, stem_channels), stem_channels),
            nn.SiLU(),
        )

        self.block1 = _ResBlock2d(
            stem_channels, stem_channels, stride=1,
            groups=groups, se_reduction=se_reduction,
        )
        self.block2 = _ResBlock2d(
            stem_channels, block_channels, stride=2,
            groups=groups, se_reduction=se_reduction,
        )
        self.block3 = _ResBlock2d(
            block_channels, block_channels, stride=1,
            groups=groups, se_reduction=se_reduction,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(block_channels, encoder_out)

    def forward(self, x: Tensor, position: Tensor | None = None) -> Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = F.silu(self.fc(x))
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
