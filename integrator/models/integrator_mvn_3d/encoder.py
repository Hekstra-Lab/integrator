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
        # self.linear = Linear(64, 2 + 2 + 6 + 3)

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
        data = data * mask
        out = torch.sum(data, dim=1, keepdim=True)
        if mask is None:
            denom = data.shape[-1]
        else:
            denom = torch.sum(mask, dim=-2, keepdim=True)
        out = out / denom

        return out


# %%

# %%


# %%
class RotationPixelEncoder(torch.nn.Module):
    """
    Encodes pixels into (num_reflection x max_voxel_sixe x d_model)
    """

    def __init__(self, depth, dmodel, d_in_refl=4, dropout=None):
        super().__init__()
        self.dropout = dropout
        self.mlp_1 = MLP(
            dmodel, depth, d_in=d_in_refl, dropout=self.dropout, output_dims=dmodel
        )

    def forward(self, shoebox, mask=None):
        out = self.mlp_1(shoebox)
        pixel_rep = torch.relu(out)
        return pixel_rep


class MLPPij(torch.nn.Module):
    def __init__(self, width, depth, dropout=None, output_dims=None):
        super().__init__()
        layers = []
        layers.extend([ResidualLayer(width, dropout=dropout) for i in range(depth)])
        if output_dims is not None:
            layers.append(Linear(width, output_dims))
        self.main = torch.nn.Sequential(*layers)

    def forward(self, data, **kwargs):
        out = self.main(data)
        return out


class ProfilePredictor(torch.nn.Module):
    """
    Outputs p_ij matrix to scale Intensity values
    """

    def __init__(self, dmodel, depth, max_pixel, dropout=None):
        super().__init__()
        self.dropout = dropout
        self.mlp_1 = MLPPij(dmodel, depth, dropout=self.dropout, output_dims=64)
        self.mean_pool = MeanPool()
        self.linear = Linear(64, max_pixel)

    def forward(self, refl_representation, pixel_rep, mask=None):
        sum = pixel_rep + refl_representation.expand_as(pixel_rep)
        out = self.mlp_1(sum)
        pooled_out = self.mean_pool(out, mask)
        out = self.linear(pooled_out)
        out = torch.softmax(out, axis=-1)
        out = out.squeeze(-2)

        return out
