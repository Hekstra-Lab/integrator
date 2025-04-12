import torch
from torch.nn import Linear
from integrator.layers import MLP, MeanPool
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


class tempMLPImageEncoder(BaseEncoder):
    def __init__(self, depth=10, dmodel=64, feature_dim=7, dropout=None):
        super().__init__()
        self.linear = nn.Linear(feature_dim, dmodel)
        self.relu = nn.ReLU()
        self.mlp_1 = MLP(dmodel, depth, dropout=dropout, output_dims=dmodel)

    def forward(self, shoebox_data, masks):
        out = self.linear(shoebox_data)
        out = self.relu(out)
        out = self.mlp_1(out)
        return out


class MLPImageEncoder(torch.nn.Module):
    def __init__(self, depth=10, dmodel=64, feature_dim=7, dropout=None):
        super().__init__()
        self.linear = Linear(feature_dim, dmodel)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.batch_norm = torch.nn.BatchNorm1d(dmodel)
        self.mlp_1 = MLP(dmodel, depth, dropout=dropout, output_dims=dmodel)
        self.mean_pool = MeanPool()

    def forward(self, shoebox_data, mask):
        batch_size, num_pixels, _ = shoebox_data.shape

        # Initial transformations
        out = self.linear(shoebox_data)
        out = self.relu(out)

        # Reshape for BatchNorm1d, apply it, then reshape back
        out = out.view(batch_size * num_pixels, -1)
        # out = self.batch_norm(out)
        out = out.view(batch_size, num_pixels, -1)

        # Pass through residual blocks
        out = self.mlp_1(out)
        pooled_out = self.mean_pool(out, mask.unsqueeze(-1))

        return pooled_out


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
