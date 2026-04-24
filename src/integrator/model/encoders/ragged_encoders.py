"""Variable-size shoebox encoders for ragged batches.

Mirrors `ShoeboxEncoder` and `IntensityEncoder` from `encoders.py`, but:
  - Inputs are padded-to-batch-max 5D tensors with an explicit mask.
  - The `flatten + Linear` head is replaced with masked global average pool
    followed by a projection, since flattened size varies per batch.
  - conv2 uses padding=1 (vs the fixed encoder's padding=0) so that any
    shoebox >=(3, 3, 3) can flow through without producing zero-sized
    spatial dims. On dataset 140 the smallest reflection is (2, 8, 9), so
    conv2 with a (3,3,3) kernel and no padding would yield D=0.

Nothing here imports from `encoders.py`; the two encoder families can be used
side by side without interfering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _downsample_mask_pool3d(
    mask: Tensor, kernel_size, stride, ceil_mode=True
) -> Tensor:
    """Downsample a bool mask with max-pool semantics: a spatial output cell
    is considered valid if ANY of its input cells was valid. Matches the
    receptive field of an unmasked MaxPool3d with the same kernel/stride."""
    m = mask.to(torch.float32).unsqueeze(1)  # (B, 1, D, H, W)
    m = F.max_pool3d(m, kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode)
    return m.squeeze(1).bool()  # (B, d, h, w)


def _masked_global_avg_pool(x: Tensor, mask: Tensor) -> Tensor:
    """Average feature-map channels over the valid voxels only.

    x:    (B, C, D, H, W)
    mask: (B, D, H, W)   bool
    returns: (B, C)
    """
    m = mask.unsqueeze(1).to(x.dtype)  # (B, 1, D, H, W)
    num = (x * m).sum(dim=(2, 3, 4))  # (B, C)
    den = m.sum(dim=(2, 3, 4)).clamp(min=1.0)  # (B, 1), but broadcasts
    return num / den


class RaggedShoeboxEncoder(nn.Module):
    """Same block structure as `ShoeboxEncoder`, shape-agnostic.

    Two conv blocks (conv+GroupNorm+ReLU) with an intermediate MaxPool3d,
    then a *masked* global avg pool and a linear projection to `encoder_out`.

    Differences from `ShoeboxEncoder`:
      - Drops `input_shape`, `use_coord_channels`, `_infer_flattened_size`.
      - conv2 defaults to padding=(1,1,1) so the network handles small
        shoeboxes without shrinking to zero.
      - `forward(x, mask)` — mask is propagated through the pool.
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_out: int = 64,
        conv1_out_channels: int = 16,
        conv1_kernel_size=(1, 3, 3),
        conv1_padding=(0, 1, 1),
        norm1_num_groups: int = 4,
        pool_kernel_size=(1, 2, 2),
        pool_stride=(1, 2, 2),
        conv2_out_channels: int = 32,
        conv2_kernel_size=(3, 3, 3),
        conv2_padding=(1, 1, 1),  # changed from (0,0,0) for variable-size safety
        norm2_num_groups: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder_out = encoder_out
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride

        self.conv1 = nn.Conv3d(
            in_channels,
            conv1_out_channels,
            kernel_size=conv1_kernel_size,
            padding=conv1_padding,
        )
        self.norm1 = nn.GroupNorm(norm1_num_groups, conv1_out_channels)

        self.pool = nn.MaxPool3d(
            kernel_size=pool_kernel_size, stride=pool_stride, ceil_mode=True
        )

        self.conv2 = nn.Conv3d(
            conv1_out_channels,
            conv2_out_channels,
            kernel_size=conv2_kernel_size,
            padding=conv2_padding,
        )
        self.norm2 = nn.GroupNorm(norm2_num_groups, conv2_out_channels)

        self.dropout_p = float(dropout)
        if self.dropout_p > 0:
            self.dropout1 = nn.Dropout3d(self.dropout_p)
            self.dropout2 = nn.Dropout3d(self.dropout_p)
        else:
            self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()

        self.fc = nn.Linear(conv2_out_channels, encoder_out)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        x:    (B, C_in, D, H, W)  — padded shoebox
        mask: (B, D, H, W)         — bool, True at real+valid voxels
        """
        # Zero padded voxels before conv so they don't inject non-zero
        # activations through the bias/weight path. (Conv bias alone would
        # already break this if we didn't.)
        x = x * mask.unsqueeze(1).to(x.dtype)

        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout1(x)

        # Pool features AND the mask so they stay aligned downstream
        x = self.pool(x)
        mask = _downsample_mask_pool3d(
            mask, kernel_size=self.pool_kernel_size, stride=self.pool_stride, ceil_mode=True
        )

        x = F.relu(self.norm2(self.conv2(x)))
        x = self.dropout2(x)

        # Masked global average pool — only average valid voxels
        pooled = _masked_global_avg_pool(x, mask)  # (B, C)
        return F.relu(self.fc(pooled))


class RaggedIntensityEncoder(nn.Module):
    """Same block structure as `IntensityEncoder`, shape-agnostic.

    Three conv blocks (conv+GroupNorm+ReLU) with one intermediate MaxPool3d,
    then a *masked* global avg pool and a linear projection. The original
    IntensityEncoder used AdaptiveAvgPool3d(1) at the end, which averages
    over padding for ragged inputs; we swap that for a masked pool.
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_out: int = 64,
        conv1_out_channels: int = 16,
        conv1_kernel_size=(3, 3, 3),
        conv1_padding=(1, 1, 1),
        norm1_num_groups: int = 4,
        pool_kernel_size=(1, 2, 2),
        pool_stride=(1, 2, 2),
        conv2_out_channels: int = 32,
        conv2_kernel_size=(3, 3, 3),
        conv2_padding=(1, 1, 1),  # was (0,0,0) in IntensityEncoder
        norm2_num_groups: int = 4,
        conv3_out_channels: int = 64,
        conv3_kernel_size=(3, 3, 3),
        conv3_padding=(1, 1, 1),
        norm3_num_groups: int = 8,
    ):
        super().__init__()
        self.encoder_out = encoder_out
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride

        self.conv1 = nn.Conv3d(
            in_channels,
            conv1_out_channels,
            kernel_size=conv1_kernel_size,
            padding=conv1_padding,
        )
        self.norm1 = nn.GroupNorm(norm1_num_groups, conv1_out_channels)

        self.pool = nn.MaxPool3d(
            kernel_size=pool_kernel_size, stride=pool_stride, ceil_mode=True
        )

        self.conv2 = nn.Conv3d(
            conv1_out_channels,
            conv2_out_channels,
            kernel_size=conv2_kernel_size,
            padding=conv2_padding,
        )
        self.norm2 = nn.GroupNorm(norm2_num_groups, conv2_out_channels)

        self.conv3 = nn.Conv3d(
            conv2_out_channels,
            conv3_out_channels,
            kernel_size=conv3_kernel_size,
            padding=conv3_padding,
        )
        self.norm3 = nn.GroupNorm(norm3_num_groups, conv3_out_channels)

        self.fc = nn.Linear(conv3_out_channels, encoder_out)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        x:    (B, C_in, D, H, W)
        mask: (B, D, H, W)  bool
        """
        x = x * mask.unsqueeze(1).to(x.dtype)

        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        mask = _downsample_mask_pool3d(
            mask, kernel_size=self.pool_kernel_size, stride=self.pool_stride, ceil_mode=True
        )

        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))

        pooled = _masked_global_avg_pool(x, mask)
        return F.relu(self.fc(pooled))
