import torch
from torch.nn import Linear

from integrator.layers import Residual, MLP, MeanPool
from integrator.model.encoders import BaseEncoder
import torch.nn as nn


class MLPImageEncoder(BaseEncoder):
    def __init__(self, depth=10, dmodel=64, feature_dim=7, dropout=None):
        super().__init__()
        self.linear = Linear(feature_dim, dmodel)
        self.relu = nn.ReLU()
        self.mlp_1 = MLP(dmodel, depth, dropout=dropout, output_dims=dmodel)

    def forward(self, shoebox_data, masks):
        out = self.linear(shoebox_data)
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
