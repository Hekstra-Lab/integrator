from dataclasses import dataclass, fields
from typing import Literal


def _check_spatial_ndims(args) -> None:
    """Every tuple/list arg (kernels, padding, input_shape) must match data_dim."""
    ndim = 2 if args.data_dim == "2d" else 3
    for f in fields(args):
        v = getattr(args, f.name)
        if isinstance(v, (list, tuple)) and len(v) != ndim:
            raise ValueError(
                f"{f.name} must have {ndim} elements for data_dim="
                f"{args.data_dim!r}, got {v}"
            )


@dataclass
class ProfileEncoderArgs:
    """Constructor arguments for the convolutional profile encoder.

    The `conv{1,2}_*`, `norm{1,2}_*`, and `pool_*` fields configure a two-block
    CNN stack (conv then group-norm) with a single pooling stage between blocks.

    Attributes:
        data_dim: Encoder dimensionality, `2d` (Conv2d) or `3d` (Conv3d).
        in_channels: Number of input channels; must be `>= 1`.
        input_shape: Spatial shoebox shape `(height, width)` or `(depth, height, width)`.
        encoder_out: Width of the output embedding; must be `>= 1`.
        conv1_out_channels: Output channels of the first conv block.
        conv1_kernel_size: Kernel size of the first conv block.
        conv1_padding: Padding of the first conv block.
        norm1_num_groups: Number of groups in the first group-norm.
        pool_kernel_size: Pooling kernel size between the two conv blocks.
        pool_stride: Pooling stride between the two conv blocks.
        conv2_out_channels: Output channels of the second conv block.
        conv2_kernel_size: Kernel size of the second conv block.
        conv2_padding: Padding of the second conv block.
        norm2_num_groups: Number of groups in the second group-norm.
        dropout: Dropout probability applied before the output projection.
    """

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

    def __post_init__(self):
        if self.in_channels < 1:
            raise ValueError(
                f"in_channels must be >= 1, got {self.in_channels}"
            )
        if self.encoder_out < 1:
            raise ValueError(
                f"encoder_out must be >= 1, got {self.encoder_out}"
            )
        _check_spatial_ndims(self)


@dataclass
class IntensityEncoderArgs:
    """Constructor arguments for the convolutional intensity/background encoder.

    The `conv{1,2,3}_*` and `norm{1,2,3}_*` fields configure a three-block CNN
    stack (conv then group-norm) with a single pooling stage.

    Attributes:
        data_dim: Encoder dimensionality, `2d` (Conv2d) or `3d` (Conv3d).
        in_channels: Number of input channels; must be `>= 1`.
        encoder_out: Width of the output embedding; must be `>= 1`.
        conv1_out_channels: Output channels of the first conv block.
        conv1_kernel_size: Kernel size of the first conv block.
        conv1_padding: Padding of the first conv block.
        norm1_num_groups: Number of groups in the first group-norm.
        pool_kernel_size: Pooling kernel size.
        pool_stride: Pooling stride.
        conv2_out_channels: Output channels of the second conv block.
        conv2_kernel_size: Kernel size of the second conv block.
        conv2_padding: Padding of the second conv block.
        norm2_num_groups: Number of groups in the second group-norm.
        conv3_out_channels: Output channels of the third conv block.
        conv3_kernel_size: Kernel size of the third conv block.
        conv3_padding: Padding of the third conv block.
        norm3_num_groups: Number of groups in the third group-norm.
    """

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
        _check_spatial_ndims(self)


@dataclass
class EncoderConfig:
    """Registry selection for a single encoder: a `name` plus its typed `args`.

    Attributes:
        name: Registry key naming the encoder class to construct.
        args: Constructor arguments, a `ProfileEncoderArgs` or `IntensityEncoderArgs`.
    """

    name: str
    args: ProfileEncoderArgs | IntensityEncoderArgs
