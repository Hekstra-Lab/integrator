import torch
import math
from torch import nn
import torch.nn.functional as F


# Trunacated Normal
def weight_initializer(weight):
    fan_avg = 0.5 * (weight.shape[-1] + weight.shape[-2])
    std = math.sqrt(1.0 / fan_avg / 10.0)
    a = -2.0 * std
    b = 2.0 * std
    torch.nn.init.trunc_normal_(weight, 0.0, std, a, b)
    return weight


class Linear(torch.nn.Linear):
    def reset_parameters(self) -> None:
        self.weight = weight_initializer(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class Residual(nn.Module):

    """The Residual block of ResNet models."""

    def __init__(self, in_channels, out_channels, strides=1, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels or strides != 1:
            self.conv3 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=strides
            )
        else:
            self.conv3 = None

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    #            if self.conv3:
    #                self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = self.conv1(X)
        if self.use_bn:
            Y = self.bn1(Y)
        Y = F.relu(Y)
        Y = self.conv2(Y)
        if self.use_bn:
            Y = self.bn2(Y)

        if self.conv3:
            X = self.conv3(X)
            # if self.use_bn:
            #    X = self.bn3(X)
        Y += X
        return F.relu(Y)


class ResidualLayer(nn.Module):
    def __init__(self, width, dropout=None, use_bn=False):
        super().__init__()
        self.use_bn = use_bn

        self.fc1 = nn.Linear(width, width)

        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(width)

        self.fc2 = nn.Linear(width, width)

        if self.use_bn:
            self.bn2 = nn.BatchNorm1d(width)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.fc2(out)
        if self.use_bn:
            out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class MLP(torch.nn.Module):
    def __init__(self, width, depth, dropout=None, output_dims=None, user_bn=True):
        super().__init__()
        layers = [
            ResidualLayer(width, dropout=dropout, use_bn=user_bn) for _ in range(depth)
        ]
        if output_dims is not None:
            layers.append(Linear(width, output_dims))
        self.main = torch.nn.Sequential(*layers)

    def forward(self, data):
        batch_size, num_pixels, features = data.shape
        data = data.view(-1, features)
        out = self.main(data)
        return out.view(batch_size, num_pixels, -1)
