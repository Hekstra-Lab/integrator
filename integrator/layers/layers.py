from pylab import *
import torch
import math
from .util import weight_initializer


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
