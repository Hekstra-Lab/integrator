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
class MetadataEncoderArgs:
    encoder_in: int
    encoder_out: int
    hidden_dim: int = 128
    depth: int = 3
    dropout: float = 0.0

    def __post_init__(self):
        if self.encoder_in < 1:
            raise ValueError(f"encoder_in must be >= 1, got {self.encoder_in}")
        if self.encoder_out < 1:
            raise ValueError(
                f"encoder_out must be >= 1, got {self.encoder_out}"
            )
        if self.depth < 0:
            raise ValueError(f"depth must be >= 0, got {self.depth}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout}"
            )


@dataclass
class EncoderConfig:
    name: str
    args: ShoeboxEncoderArgs | IntensityEncoderArgs | MetadataEncoderArgs


@dataclass
class Encoders:
    encoders: list[EncoderConfig]
