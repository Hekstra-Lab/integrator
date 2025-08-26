import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, Sequential

from integrator.layers import ResidualLayer


class MLPMetadataEncoder(nn.Module):
    """Encoder to be used with experimental metadata associated with each shoebox, such as predicted centroid coordinates,
    Miller indices, average shoebox intensity, etc.
    The architecture is based off a Residual Network.
    """

    model: Sequential
    """A `torch.nn.Sequential` container containing the sequence of layers applied to the input shoebox."""

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


# -
if __name__ == "__main__":
    import torch

    x = torch.randn(10, 10)
    encoder = MLPMetadataEncoder(
        encoder_in=10,
        encoder_out=64,
        depth=5,
        dropout=0.0,
    )

    out = encoder(x)
