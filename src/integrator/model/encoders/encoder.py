import torch
import torch.nn as nn
from torch.nn import Linear
from integrator.layers import Residual, MLP, MeanPool
from integrator.model.encoders import BaseEncoder


class CNNResNet(BaseEncoder):
    def __init__(
        self,
        use_bn=True,
        conv1_in_channel=6,
        conv1_out_channel=64,
        conv1_kernel_size=7,
        conv1_stride=2,
        conv1_padding=3,
        layer1_num_blocks=3,
        conv2_out_channel=128,
        layer2_stride=1,
        layer2_num_blocks=3,
        maxpool_in_channel=64,
        maxpool_out_channel=64,
        maxpool_kernel_size=3,
        maxpool_stride=2,
        maxpool_padding=1,
        dmodel=None,
        dropout=None,
        feature_dim=None,
        depth=None,
        Z=3,  # Add Z, H, W parameters here
        H=21,
        W=21,
    ):
        super(CNNResNet, self).__init__()
        self.use_bn = use_bn
        self.Z = Z
        self.H = H
        self.W = W

        self.conv1 = torch.nn.Conv2d(
            conv1_in_channel,
            conv1_out_channel,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=conv1_padding,
            bias=False,
        )
        if self.use_bn:
            self.bn1 = torch.nn.BatchNorm2d(conv1_out_channel)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=0)

        self.layer1 = self._make_layer(
            conv1_out_channel, conv1_out_channel, layer1_num_blocks, use_bn=self.use_bn
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1, use_bn=True):
        layers = []
        layers.append(
            Residual(in_channels, out_channels, strides=stride, use_bn=use_bn)
        )
        for _ in range(1, num_blocks):
            layers.append(Residual(out_channels, out_channels, use_bn=use_bn))
        return torch.nn.Sequential(*layers)

    def reshape(self, x, mask):
        counts_ = x[..., -1].reshape(x.size(0), self.Z, self.H, self.W)
        counts_ = counts_ * mask.reshape(mask.size(0), self.Z, self.H, self.W)

        X_coords = x[..., 0][..., : self.H * self.W].view(x.size(0), 1, self.H, self.W)
        Y_coords = x[..., 1][..., : self.H * self.W].view(x.size(0), 1, self.H, self.W)
        Z_coords = x[..., 2][..., self.H * self.W : self.H * self.W * 2].view(
            x.size(0), 1, self.H, self.W
        )

        x = torch.cat([X_coords, Y_coords, Z_coords, counts_], dim=1)

        return x

    def forward(self, x, mask):
        x = self.reshape(x, mask)
        x = self.conv1(x)

        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class DynamicTanh(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Save original shape for later
        original_shape = x.shape

        # Apply tanh with learnable scaling factor
        x = torch.tanh(self.alpha * x)

        # Handle different tensor dimensions
        if x.dim() == 5:  # [B, C, Z, H, W] - 3D CNN
            weight = self.weight.view(1, -1, 1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1, 1)
        elif x.dim() == 4:  # [B, C, H, W] - 2D CNN
            weight = self.weight.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
        elif x.dim() == 3:  # [B, N, C] - MLP with pixel dimension
            weight = self.weight.view(1, 1, -1)
            bias = self.bias.view(1, 1, -1)
        else:  # [B, C] - MLP without pixel dimension
            weight = self.weight.view(1, -1)
            bias = self.bias.view(1, -1)

        return x * weight + bias


class tempMLPImageEncoder(torch.nn.Module):
    def __init__(self, depth=10, dmodel=64, feature_dim=7, dropout=None):
        super().__init__()
        self.linear = Linear(feature_dim, dmodel)
        self.relu = torch.nn.ReLU(inplace=True)
        self.batch_norm = torch.nn.BatchNorm1d(dmodel)
        self.dyt = DynamicTanh(dmodel)
        self.mlp_1 = MLP(dmodel, depth, dropout=dropout, output_dims=dmodel)
        self.mean_pool = MeanPool()

    def forward(self, shoebox_data, mask):
        batch_size, num_pixels, _ = shoebox_data.shape

        # Initial transformations
        out = self.linear(shoebox_data)
        out = self.relu(out)

        # Reshape for BatchNorm1d, apply it, then reshape back
        out = out.view(batch_size * num_pixels, -1)
        out = self.batch_norm(out)
        # out = self.dyt(out)
        out = out.view(batch_size, num_pixels, -1)

        # Pass through residual blocks
        out = self.mlp_1(out)
        pooled_out = self.mean_pool(out, mask.unsqueeze(-1))

        return pooled_out
