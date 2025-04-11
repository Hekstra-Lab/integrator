import torch
from torch.nn import Linear

from integrator.layers import Residual, MLP, MeanPool
from integrator.model.encoders import BaseEncoder
import torch.nn as nn


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Apply tanh with learnable alpha parameter
        x = torch.tanh(self.alpha * x)

        # Reshape weight and bias for proper broadcasting with 4D tensors (N,C,H,W)
        if len(x.shape) == 4:
            weight = self.weight.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
        else:
            weight = self.weight
            bias = self.bias

        return x * weight + bias


class tempMLPImageEncoder(BaseEncoder):
    def __init__(self, depth=10, dmodel=64, feature_dim=7, dropout=None):
        super().__init__()
        self.DyT = DyT(feature_dim)
        self.linear = Linear(feature_dim, dmodel)
        self.relu = nn.ReLU()
        self.mlp_1 = MLP(dmodel, depth, dropout=dropout, output_dims=dmodel)

    def forward(self, shoebox_data, masks):
        out = self.linear(shoebox_data)
        out = self.DyT(out)
        out = self.relu(out)
        out = self.mlp_1(out)
        return out


class MLPImageEncoder(BaseEncoder):
    def __init__(self, depth=10, dmodel=64, feature_dim=7, dropout=None):
        super().__init__()
        self.linear = nn.Linear(feature_dim, dmodel)
        self.dyt = DyT(dmodel)  # Apply DyT after dimension change
        self.relu = nn.ReLU()
        self.mlp_1 = MLP(dmodel, depth, dropout=dropout, output_dims=dmodel)

    def forward(self, shoebox_data, masks):
        out = self.linear(shoebox_data)
        out = self.dyt(out)  # Apply DyT before activation
        out = self.relu(out)
        out = self.mlp_1(out)
        return out


class MeanPool(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.register_buffer(
            "dim",
            torch.tensor(dim),
        )

    def forward(self, data, mask=None):
        data = data * mask
        out = torch.sum(data, dim=1, keepdim=True)
        if mask is None:
            denom = data.shape[-1]
        else:
            denom = torch.sum(mask, dim=-2, keepdim=True)
        out = out / denom
        return out.squeeze(1)
