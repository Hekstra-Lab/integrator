import torch.nn as nn
from torch.nn import Linear

from integrator.layers import ResidualLayer


class MLPMetadataEncoder(nn.Module):
    def __init__(self, feature_dim, depth=10, dropout=0.0, output_dims=None):
        super().__init__()
        layers = []
        hidden_dim = feature_dim * 2

        # Input layer
        layers.append(Linear(feature_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))  #
        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth):
            layers.append(ResidualLayer(hidden_dim, dropout_rate=dropout))

        if output_dims is not None:
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(Linear(hidden_dim, output_dims))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x
