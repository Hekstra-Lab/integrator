from pylab import *
import torch
from integrator.layers import Linear, ResidualLayer
from integrator.models import MLP


class Encoder(torch.nn.Module):
    def __init__(self, depth, dmodel, feature_dim, dropout=None):
        super().__init__()
        self.dropout = None
        self.mlp_1 = MLP(
            dmodel, depth, d_in=feature_dim, dropout=self.dropout, output_dims=dmodel
        )
        self.mean_pool = MeanPool()
        # I + bg + MVN
        # 2 + 2 + 6 + 3
        self.linear = Linear(64, 2 + 2 + 6 + 3)

    def forward(self, shoebox_data, mask=None):
        out = self.mlp_1(shoebox_data)
        pooled_out = self.mean_pool(out, mask)
        # outputs = self.linear(pooled_out)
        return pooled_out


# Embed shoeboxes
class MeanPool(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.register_buffer(
            "dim",
            torch.tensor(dim),
        )

    def forward(self, data, mask=None):
        out = data.sum(1, keepdim=True)
        if mask is None:
            denom = data.shape[-1]
        else:
            denom = mask.sum(-2, keepdim=True)
        out = out / denom

        return out
