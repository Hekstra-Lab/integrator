import torch.nn as nn
from torch import Tensor
from torch.nn import Linear

from integrator.layers import ResidualLayer


class MLPMetadataEncoder(nn.Module):
    """MLP encoder for metadata features.

    Used by legacy 2-encoder and 3-encoder (non-hierarchical) integrator
    variants that take metadata as an auxiliary input stream.
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
