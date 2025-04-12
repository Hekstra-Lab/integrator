import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock2D(nn.Module):
    """
    A basic residual block for 2D images:
    - Conv2D(in_ch -> out_ch), BN, ReLU
    - Conv2D(out_ch -> out_ch), BN
    + skip connection
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # If in_ch != out_ch, use a 1x1 conv to match channels in the skip
        self.skip_proj = None
        if in_ch != out_ch:
            self.skip_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        skip = x
        if self.skip_proj is not None:
            skip = self.skip_proj(skip)

        # Add and ReLU
        out = out + skip
        out = F.relu(out, inplace=True)
        return out


class tempDirichletConcentration(nn.Module):
    """
    A fully convolutional 2D ResNet that preserves input spatial dims.
    For example, if input is (N, 3, 21, 21), output is (N, out_channels, 21, 21).

    - Initial conv -> some number of residual blocks -> final 1x1 conv
    """

    def __init__(self, in_channels=3, base_ch=32, num_blocks=4, out_channels=1):
        super().__init__()

        # 1) Initial convolution to go from in_channels -> base_ch
        self.conv_in = nn.Conv2d(
            in_channels, base_ch, kernel_size=3, padding=1, stride=1
        )
        self.bn_in = nn.BatchNorm2d(base_ch)

        # 2) Stack of residual blocks
        self.blocks = nn.ModuleList()
        current_channels = base_ch
        for _ in range(num_blocks):
            block = ResidualBlock2D(current_channels, current_channels)
            self.blocks.append(block)

        # 3) Optional final conv to produce 'out_channels'
        self.conv_out = nn.Conv2d(
            current_channels, out_channels, kernel_size=3, padding=1, stride=1
        )

    def forward(self, x, mask=None):
        """
        x: (N, in_channels, H, W)
        returns: (N, out_channels, H, W)
        """
        # Initial conv + BN + ReLU
        x = x[:, :, -1].view(-1, 3, 21, 21)
        x = self.conv_in(x)  # => (N, base_ch, H, W)
        x = self.bn_in(x)
        x = F.relu(x, inplace=True)

        # Residual blocks
        for block in self.blocks:
            x = block(x)  # => (N, base_ch, H, W)

        # Final output conv
        x = self.conv_out(x)  # => (N, out_channels, H, W)

        return x.view(-1, 3 * 21 * 21)


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Apply tanh with learnable alpha parameter
        x = torch.tanh(self.alpha * x)

        # Reshape weight and bias for proper broadcasting with 4D tensors (N,C,H,W)
        if len(x.shape) == 4:
            weight = self.weight.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
        else:
            weight = self.weight
            bias = self.bias

        return x * weight + bias


class ResidualBlock2D(nn.Module):
    """
    A basic residual block for 2D images:
    - Conv2D(in_ch -> out_ch), DyT, ReLU
    - Conv2D(out_ch -> out_ch), DyT
    + skip connection
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1)
        self.dyt1 = DyT(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, stride=1)
        self.dyt2 = DyT(out_ch)
        # If in_ch != out_ch, use a 1x1 conv to match channels in the skip
        self.skip_proj = None
        if in_ch != out_ch:
            self.skip_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.dyt1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.dyt2(out)
        # Skip connection
        skip = x
        if self.skip_proj is not None:
            skip = self.skip_proj(skip)
        # Add and ReLU
        out = out + skip
        out = F.relu(out, inplace=True)
        return out


class DirichletConcentration(nn.Module):
    """
    A fully convolutional 2D ResNet that preserves input spatial dims.
    For example, if input is (N, 3, 21, 21), output is (N, out_channels, 21, 21).
    - Initial conv -> some number of residual blocks -> final 1x1 conv
    """

    def __init__(self, in_channels=3, base_ch=32, num_blocks=4, out_channels=1):
        super().__init__()
        # 1) Initial convolution to go from in_channels -> base_ch
        self.conv_in = nn.Conv2d(
            in_channels, base_ch, kernel_size=3, padding=1, stride=1
        )
        self.dyt_in = DyT(base_ch)  # Replacing BatchNorm with DyT
        # 2) Stack of residual blocks
        self.blocks = nn.ModuleList()
        current_channels = base_ch
        for _ in range(num_blocks):
            block = ResidualBlock2D(current_channels, current_channels)
            self.blocks.append(block)
        # 3) Optional final conv to produce 'out_channels'
        self.conv_out = nn.Conv2d(
            current_channels, out_channels, kernel_size=3, padding=1, stride=1
        )

    def forward(self, x, mask=None):
        """
        x: (N, in_channels, H, W)
        returns: (N, out_channels, H, W)
        """
        # Initial conv + DyT + ReLU
        x = x[:, :, -1].view(-1, 3, 21, 21)
        x = self.conv_in(x)  # => (N, base_ch, H, W)
        x = self.dyt_in(x)
        x = F.relu(x, inplace=True)
        # Residual blocks
        for block in self.blocks:
            x = block(x)  # => (N, base_ch, H, W)
        # Final output conv
        x = self.conv_out(x)  # => (N, out_channels, H, W)
        return x.view(-1, 3 * 21 * 21)


class ResNetEncoder(nn.Module):
    def __init__(
        self, in_channels=3, base_ch=32, num_blocks=4, out_channels=1, vector_dim=64
    ):
        super().__init__()
        # 1) Initial convolution to go from in_channels -> base_ch
        self.conv_in = nn.Conv2d(
            in_channels, base_ch, kernel_size=3, padding=1, stride=1
        )
        self.dyt_in = DyT(base_ch)

        # 2) Stack of residual blocks
        self.blocks = nn.ModuleList()
        current_channels = base_ch
        for _ in range(num_blocks):
            block = ResidualBlock2D(current_channels, current_channels)
            self.blocks.append(block)

        # 3) Global pooling and projection to vector
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(current_channels, vector_dim)

    def forward(self, x, mask=None):
        """
        x: (N, in_channels, H, W)
        returns: (N, vector_dim)
        """
        # Initial conv + DyT + ReLU
        x = x[:, :, -1].view(-1, 3, 21, 21)
        x = self.conv_in(x)
        x = self.dyt_in(x)
        x = F.relu(x, inplace=True)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Global pooling and projection to vector
        x = self.global_pool(x)  # -> (N, current_channels, 1, 1)
        x = x.view(x.size(0), -1)  # -> (N, current_channels)
        x = self.fc(x)  # -> (N, vector_dim)

        return x


if __name__ == "__main__":
    # shoebox shape: (N, 3*21*21)

    model = DirichletConcentration(
        in_channels=3, base_ch=32, num_blocks=4, out_channels=3
    )

    # Make a dummy shoebox
    dummy_input = torch.randn(10, 1323, 7)

    output = model(dummy_input)

    print("Output shape:", output.shape)

    resnet_model = ResNetEncoder()
    resnet_output = resnet_model(dummy_input)
# %%
