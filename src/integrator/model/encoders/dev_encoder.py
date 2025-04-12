import torch
from torch.nn import Linear
from integrator.layers import MLP, MeanPool
from integrator.model.encoders import BaseEncoder


class DevEncoder(BaseEncoder):
    def __init__(self, depth=10, dmodel=64, feature_dim=7, dropout=None):
        super().__init__()
        self.linear = Linear(feature_dim, dmodel)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.batch_norm = nn.BatchNorm1d(dmodel)
        self.layer_norm = torch.nn.LayerNorm(dmodel)
        self.mlp_1 = MLP(dmodel, depth, dropout=dropout, output_dims=dmodel)
        self.mean_pool = MeanPool()

    def forward(self, shoebox_data, masks):
        # shoebox_data shape: [batch_size, num_pixels, feature_dim]
        # batch_size, features = shoebox_data.shape

        # Initial linear transformation
        out = self.linear(shoebox_data)
        out = self.relu(out)
        out = self.layer_norm(out)
        out = self.mlp_1(out)
        pooled_out = self.mean_pool(out, masks.unsqueeze(-1))
        return pooled_out


class MeanPool(torch.nn.Module):
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
