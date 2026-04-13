import math

import torch
from torch import nn


# Trunacated Normal
def weight_initializer(weight):
    fan_avg = 0.5 * (weight.shape[-1] + weight.shape[-2])
    std = math.sqrt(1.0 / fan_avg / 10.0)
    a = -2.0 * std
    b = 2.0 * std
    torch.nn.init.trunc_normal_(weight, 0.0, std, a, b)
    return weight


class Linear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias=False):
        super().__init__(
            in_features, out_features, bias=bias
        )  # Set bias=False

    def reset_parameters(self) -> None:
        self.weight = weight_initializer(self.weight)


class ResidualLayer(nn.Module):
    def __init__(
        self,
        width: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        # First layer
        self.fc1 = Linear(width, width)
        self.norm1 = nn.LayerNorm(width)

        # Second layer
        self.fc2 = Linear(width, width)
        self.norm2 = nn.LayerNorm(width)

        # Activation and dropout
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x

        # First layer
        out = self.norm1(x)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.dropout(out)

        # Second layer
        out = self.norm2(out)
        out = self.relu(out)
        out = self.fc2(out)

        # Residual connection
        out = out + residual
        out = self.relu(out)

        return out
