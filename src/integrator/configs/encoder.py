from dataclasses import dataclass
from typing import Literal


@dataclass
class ProfileEncoderArgs:
    data_dim: Literal["2d", "3d"]
    in_channels: int
    input_shape: tuple[int, ...]
    encoder_out: int
    conv1_out_channels: int
    conv1_kernel_size: tuple[int, ...]
    conv1_padding: tuple[int, ...]
    norm1_num_groups: int
    pool_kernel_size: tuple[int, ...]
    pool_stride: tuple[int, ...]
    conv2_out_channels: int
    conv2_kernel_size: tuple[int, ...]
    conv2_padding: tuple[int, ...]
    norm2_num_groups: int
    dropout: float = 0.0
    position_dim: int = 0
    position_fourier_order: int = 0
    # ResidualProfileEncoder fields (ignored by ProfileEncoder)
    stem_channels: int | None = None
    block_channels: int | None = None
    groups: int = 8
    se_reduction: int = 4

    def __post_init__(self):
        if self.in_channels < 1:
            raise ValueError(
                f"in_channels must be >= 1, got {self.in_channels}"
            )
        if self.encoder_out < 1:
            raise ValueError(
                f"encoder_out must be >= 1, got {self.encoder_out}"
            )


@dataclass
class IntensityEncoderArgs:
    data_dim: Literal["2d", "3d"]
    in_channels: int
    encoder_out: int
    conv1_out_channels: int
    conv1_kernel_size: tuple[int, ...]
    conv1_padding: tuple[int, ...]
    norm1_num_groups: int
    pool_kernel_size: tuple[int, ...]
    pool_stride: tuple[int, ...]
    conv2_out_channels: int
    conv2_kernel_size: tuple[int, ...]
    conv2_padding: tuple[int, ...]
    norm2_num_groups: int
    conv3_out_channels: int
    conv3_kernel_size: tuple[int, ...]
    conv3_padding: tuple[int, ...]
    norm3_num_groups: int

    def __post_init__(self):
        if self.in_channels < 1:
            raise ValueError(
                f"in_channels must be >= 1, got {self.in_channels}"
            )
        if self.encoder_out < 1:
            raise ValueError(
                f"encoder_out must be >= 1, got {self.encoder_out}"
            )


@dataclass
class ResidualProfileEncoderArgs:
    encoder_out: int = 64
    in_channels: int = 1
    stem_channels: int = 32
    block_channels: int = 64
    groups: int = 8
    se_reduction: int = 4
    dropout: float = 0.0
    data_dim: str = "2d"


@dataclass
class EncoderConfig:
    name: str
    args: ProfileEncoderArgs | IntensityEncoderArgs | ResidualProfileEncoderArgs
