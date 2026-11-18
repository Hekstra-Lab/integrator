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
        output_activation: `relu` (default, the original encoder), `none` or
            `layernorm` after the output projection; see `ProfileEncoder`.
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
    output_activation: Literal["relu", "none", "layernorm"] = "relu"
    head: Literal["flatten", "pool", "moments"] = "flatten"
    """How the conv feature map is reduced to `encoder_out`.

    `flatten` (default) is the original: flatten the whole map and apply one
    dense layer. On a 41x41 shoebox that layer is 32 x 23,104 = 739,328
    parameters, 97.5% of the encoder and 77% of the model, against 18,944 for
    the conv stack that does the vision; it is also the layer whose units all
    go negative when the profile encoder dies.

    `pool` adds one more pooling stage before the flatten, cutting the dense
    layer by roughly an order of magnitude while keeping position.

    `moments` replaces the dense layer with the spatial moments of each
    feature map -- mass, centroid and second moments -- which are the
    quantities a profile is made of, at `6 * conv2_out_channels` inputs
    instead of 23,104. Plain global average pooling is deliberately not an
    option: it discards position, and the profile has to know where in the
    box the spot sits.
    """

    def __post_init__(self):
        if self.head not in ("flatten", "pool", "moments"):
            raise ValueError(
                f"head must be flatten, pool or moments, got {self.head!r}"
            )
        if self.output_activation not in ("relu", "none", "layernorm"):
            raise ValueError(
                "output_activation must be relu, none or layernorm, "
                f"got {self.output_activation!r}"
            )
        if self.in_channels < 1:
            raise ValueError(
                f"in_channels must be >= 1, got {self.in_channels}"
            )
        if self.encoder_out < 1:
            raise ValueError(
                f"encoder_out must be >= 1, got {self.encoder_out}"
            )
        _check_spatial_ndims(self)


def _check_positive(args) -> None:
    """Shared validation for the shape-agnostic ragged encoder args."""
    if args.in_channels < 1:
        raise ValueError(f"in_channels must be >= 1, got {args.in_channels}")
    if args.encoder_out < 1:
        raise ValueError(f"encoder_out must be >= 1, got {args.encoder_out}")


@dataclass
class RaggedShoeboxEncoderArgs:
    """Constructor arguments for `RaggedShoeboxEncoder`.

    Shape-agnostic: unlike `ProfileEncoderArgs` there is no `input_shape` or
    `data_dim` - the encoder handles variable-size (D, H, W) shoeboxes via a
    mask propagated through pooling, so every field mirrors the encoder's own
    3-D conv defaults and nothing needs a spatial-dim check.
    """

    in_channels: int = 1
    encoder_out: int = 64
    conv1_out_channels: int = 16
    conv1_kernel_size: tuple[int, ...] = (1, 3, 3)
    conv1_padding: tuple[int, ...] = (0, 1, 1)
    norm1_num_groups: int = 4
    pool_kernel_size: tuple[int, ...] = (1, 2, 2)
    pool_stride: tuple[int, ...] = (1, 2, 2)
    conv2_out_channels: int = 32
    conv2_kernel_size: tuple[int, ...] = (3, 3, 3)
    conv2_padding: tuple[int, ...] = (1, 1, 1)
    norm2_num_groups: int = 4
    dropout: float = 0.0

    def __post_init__(self):
        _check_positive(self)


@dataclass
class RaggedIntensityEncoderArgs:
    """Constructor arguments for `RaggedIntensityEncoder`.

    Shape-agnostic three-block CNN (see `RaggedShoeboxEncoderArgs`); mirrors
    the encoder's own defaults with one extra conv/norm block over the
    profile encoder.
    """

    in_channels: int = 1
    encoder_out: int = 64
    conv1_out_channels: int = 16
    conv1_kernel_size: tuple[int, ...] = (3, 3, 3)
    conv1_padding: tuple[int, ...] = (1, 1, 1)
    norm1_num_groups: int = 4
    pool_kernel_size: tuple[int, ...] = (1, 2, 2)
    pool_stride: tuple[int, ...] = (1, 2, 2)
    conv2_out_channels: int = 32
    conv2_kernel_size: tuple[int, ...] = (3, 3, 3)
    conv2_padding: tuple[int, ...] = (1, 1, 1)
    norm2_num_groups: int = 4
    conv3_out_channels: int = 64
    conv3_kernel_size: tuple[int, ...] = (3, 3, 3)
    conv3_padding: tuple[int, ...] = (1, 1, 1)
    norm3_num_groups: int = 8

    def __post_init__(self):
        _check_positive(self)


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
class SummaryStatsEncoderArgs:
    """Arguments for `SummaryStatsEncoder`.

    The statistics are scale-free by construction, so there is nothing to fit
    per dataset except `log_total_loc` and `log_total_scale`, which standardize
    the one feature that is a scale.
    """

    data_dim: str | None = None
    input_shape: tuple[int, ...] = (3, 21, 21)
    encoder_out: int = 32
    hidden: int = 16
    log_total_loc: float = 7.0
    log_total_scale: float = 3.0
    beam_center: tuple[float, float] | None = None
    quantiles: tuple[float, ...] | None = None
    tail_multiples: tuple[float, ...] | None = None
