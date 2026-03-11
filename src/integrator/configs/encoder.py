from dataclasses import dataclass
from typing import Literal


@dataclass
class ShoeboxEncoderArgs:
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

    def __post_init__(self):
        pass


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
        pass


@dataclass
class BorderStatsEncoderArgs:
    D: int
    H: int
    W: int
    encoder_out: int
    hidden_dim: int = 32

    def __post_init__(self):
        pass


@dataclass
class BorderPixelMLPEncoderArgs:
    D: int
    H: int
    W: int
    encoder_out: int
    hidden_dim: int = 128
    depth: int = 2

    def __post_init__(self):
        pass


@dataclass
class EncoderConfig:
    name: str
    args: ShoeboxEncoderArgs | IntensityEncoderArgs

    def __post_init__(self):
        pass


@dataclass
class Encoders:
    encoders: list[EncoderConfig]

    def __post_init__(self):
        pass
