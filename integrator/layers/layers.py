from pylab import *
import torch
import math
from .util import weight_initializer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Linear(torch.nn.Linear):
    def reset_parameters(self) -> None:
        self.weight = weight_initializer(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class ResidualLayer(torch.nn.Module):
    def __init__(self, dims, dropout=None):
        super().__init__()
        self.linear_1 = Linear(dims, 2 * dims)
        self.linear_2 = Linear(2 * dims, dims)
        self.dropout = (
            torch.nn.Dropout(dropout) if dropout is not None else torch.nn.Identity()
        )

    def activation(self, data):
        return torch.relu(data)

    def forward(self, data):
        out = self.activation(data)
        out = self.linear_1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        return out + data


class Transformer(torch.nn.Module):
    def __init__(
        self, d_model: int, d_hid: int, nhead: int, nlayers: int, batch_first=True
    ):
        super().__init__()

        # Layers ,
        self.d_model = d_model
        encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_hid
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layers, nlayers)

    def forward(self, data, training=None, **kwargs):
        out = data
        out = self.transformer(out)
        return out


class DynamicLinear(torch.nn.Module):
    def __init__(self, input_dim, output_dim_2D, output_dim_3D):
        super().__init__()
        self.output_dim_2D = output_dim_2D
        self.output_dim_3D = output_dim_3D

        self.linear1 = Linear(input_dim, self.output_dim_2D)
        self.linear2 = Linear(input_dim, self.output_dim_3D)

    def forward(self, x, d_flag):
        out1 = self.linear1(x)
        out2 = self.linear2(x)
        return out1, out2
