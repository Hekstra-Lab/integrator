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
    use_coord_channels: bool = False
    dropout: float = 0.0

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
class EncoderConfig:
    name: str
    args: ShoeboxEncoderArgs | IntensityEncoderArgs


@dataclass
class Encoders:
    encoders: list[EncoderConfig]
