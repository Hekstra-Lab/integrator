import torch
import torch.nn as nn
from torch.nn import Linear
from integrator.layers import MLP, MeanPool, ResidualLayer
from integrator.model.encoders import BaseEncoder


class MLPMetadataEncoder(BaseEncoder):
    def __init__(self, feature_dim, depth=10, dropout=0.0, output_dims=None):
        super().__init__()
        layers = []
        hidden_dim = feature_dim * 2

        # Input projection layer
        layers.append(Linear(feature_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))  #
        layers.append(nn.ReLU(inplace=True))

        # Residual blocks
        for _ in range(depth):
            layers.append(ResidualLayer(hidden_dim, dropout_rate=dropout))

        # Output layer if needed
        if output_dims is not None:
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(Linear(hidden_dim, output_dims))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Process through the model
        x = self.model(x)
        return x


class temp(nn.Module):
    def __init__(
        self, depth=10, dmodel=64, feature_dim=7, output_dims=64, dropout=None
    ):
        super().__init__()
        self.linear = Linear(feature_dim, dmodel)
        self.relu = nn.ReLU()
        # self.layer_norm = torch.nn.LayerNorm(dmodel)
        self.mlp_1 = MLP(dmodel, depth, dropout=dropout, output_dims=output_dims)

    def forward(self, shoebox_data):
        # shoebox_data shape: [batch_size, num_pixels, feature_dim]
        batch_size, features = shoebox_data.shape

        # Initial linear transformation
        out = self.linear(shoebox_data)
        out = self.relu(out)
        # out = self.layer_norm(out)
        out = self.mlp_1(out)
        return out
