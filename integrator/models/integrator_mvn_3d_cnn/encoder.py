import torch
import torch.nn as nn
import torch.nn.functional as F


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
            if self.conv3:
                self.bn3 = nn.BatchNorm2d(out_channels)

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
            if self.use_bn:
                X = self.bn3(X)
        Y += X
        return F.relu(Y)


class Encoder(nn.Module):
    def __init__(self, use_bn=True):
        super(Encoder, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define the layers in the sequence they are applied
        self.layer1 = self._make_layer(64, 64, 4, use_bn=self.use_bn)
#        self.layer2 = self._make_layer(64, 128, 3, stride=2, use_bn=self.use_bn)
        # self.layer3 = self._make_layer(128, 256, 3, stride=2, use_bn=self.use_bn)
        # self.layer4 = self._make_layer(256, 512, 2, stride=2, use_bn=self.use_bn)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1, use_bn=True):
        layers = []
        layers.append(
            Residual(in_channels, out_channels, strides=stride, use_bn=use_bn)
        )
        for _ in range(1, num_blocks):
            layers.append(Residual(out_channels, out_channels, use_bn=use_bn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if  self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        #x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x
