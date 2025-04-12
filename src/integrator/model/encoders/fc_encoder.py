import torch
import torch.nn as nn
from torch.nn import Linear
from integrator.layers import MLP, MeanPool
from integrator.model.encoders import BaseEncoder


class DynamicTanh(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class MLPMetadataEncoder(BaseEncoder):
    def __init__(
        self, depth=10, dmodel=64, feature_dim=7, output_dims=64, dropout=None
    ):
        super().__init__()
        self.linear = Linear(feature_dim, dmodel)
        # self.relu = torch.nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        # self.batch_norm = nn.BatchNorm1d(dmodel)
        self.layer_norm = torch.nn.LayerNorm(dmodel)
        self.mlp_1 = MLP(dmodel, depth, dropout=dropout, output_dims=output_dims)

    def forward(self, shoebox_data):
        # shoebox_data shape: [batch_size, num_pixels, feature_dim]
        batch_size, features = shoebox_data.shape

        # Initial linear transformation
        out = self.linear(shoebox_data)
        out = self.relu(out)
        out = self.layer_norm(out)
        out = self.layer_norm(out)
        out = self.mlp_1(out)
        return out
