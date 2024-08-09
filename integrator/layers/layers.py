from pylab import *
import torch.nn.functional as F
import torch
from .util import weight_initializer
import torch.nn as nn


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


class Residual(nn.Module):  # @save
    """The Residual block of ResNet models."""

    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(
            num_channels, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        # self.bn1 = nn.LazyBatchNorm2d()
        # self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        # Y = F.relu(self.bn1(self.conv1(X)))
        # Y = self.bn2(self.conv2(Y))
        Y = F.relu(self.conv1(X))
        Y = self.conv2(Y)

        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
