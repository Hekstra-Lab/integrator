import torch.nn as nn
from torch import Tensor
from torch.nn import Linear

from integrator.layers import ResidualLayer


class MLPMetadataEncoder(nn.Module):
    """Legacy metadata encoder.

    Kept for back-compat with older configs. Prefer `MetadataEncoder` below
    for new work — it has sensible hidden-dim defaults, shallower depth, and
    GELU activations.
    """

    def __init__(
        self,
        encoder_in: int,
        encoder_out: int,
        depth: int = 10,
        dropout: bool | float = 0.0,
    ):
        super().__init__()
        layers = []
        hidden_dim = encoder_in * 2

        # Input layer
        layers.append(Linear(encoder_in, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        # shape: [B x hidden_dim]

        for _ in range(depth):
            layers.append(ResidualLayer(hidden_dim, dropout_rate=dropout))
        # shape: [B x hidden_dim]

        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(Linear(hidden_dim, encoder_out))
        # shape: [B x output_dims]

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x


class _ResidualMLPBlock(nn.Module):
    """Pre-norm residual block for a standard MLP."""

    def __init__(self, width: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class MetadataEncoder(nn.Module):
    """MLP encoder for low-dimensional metadata features.

    Cleaner replacement for ``MLPMetadataEncoder``:
      - ``hidden_dim`` decoupled from ``encoder_in`` (no more 16-unit hidden
        layers when the metadata is 8-dim)
      - Default depth of 3 (was 10 — overkill for low-dim input)
      - Pre-norm residual blocks with GELU activations
      - Standard biased ``nn.Linear`` (old version disabled biases)

    Callers are expected to pass already-standardized inputs (roughly
    zero mean, unit variance per feature). See
    ``_extract_meta_features`` in ``hierarchical_integrator.py`` for the
    canonical pre-normalization using fixed dataset-level statistics.
    Adding an internal normalization layer on top is redundant and risks
    fighting the pre-normalization; trust the pre-normalization instead.
    """

    def __init__(
        self,
        encoder_in: int,
        encoder_out: int,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(encoder_in, hidden_dim)]
        for _ in range(depth):
            layers.append(_ResidualMLPBlock(hidden_dim, dropout=dropout))
        layers.extend(
            [
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, encoder_out),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
