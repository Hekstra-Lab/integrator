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
    def __init__(self, in_features: int, out_features: int, bias=False):
        super().__init__(in_features, out_features, bias=bias)  # Set bias=False

    def reset_parameters(self) -> None:
        self.weight = weight_initializer(self.weight)


class ResidualLayer(nn.Module):
    def __init__(self, width, dropout=None):
        super().__init__()

        self.fc1 = Linear(width, width)
        self.fc2 = Linear(width, width)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out


class ResidualLayer(nn.Module):
    def __init__(self, width, dropout=None, use_bn=True):
        super().__init__()
        self.use_bn = use_bn

        self.fc1 = Linear(width, width)

        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(width)

        self.fc2 = Linear(width, width)

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


class MLP(nn.Module):
    def __init__(self, width, depth, dropout=None, output_dims=None):
        super().__init__()
        layers = [ResidualLayer(width, dropout=dropout) for _ in range(depth)]
        if output_dims is not None:
            layers.append(Linear(width, output_dims))
        self.main = nn.Sequential(*layers)

    def forward(self, data):
        # Check if the input has 2 or 3 dimensions
        if len(data.shape) == 3:
            batch_size, num_pixels, features = data.shape
            data = data.view(
                -1, features
            )  # Flatten to [batch_size * num_pixels, features]
        elif len(data.shape) == 2:
            batch_size, features = data.shape
            num_pixels = None  # No pixels in this case

        out = self.main(data)

        if num_pixels is not None:
            out = out.view(batch_size, num_pixels, -1)  # Reshape back if needed
        return out


class tempMLP(nn.Module):
    def __init__(self, width, depth, dropout=None, output_dims=None):
        super().__init__()
        layers = [ResidualLayer(width, dropout=dropout) for _ in range(depth)]
        if output_dims is not None:
            layers.append(Linear(width, output_dims))
        self.main = nn.Sequential(*layers)

    def forward(self, data):
        # Check if the input has 2 or 3 dimensions
        if len(data.shape) == 3:
            batch_size, num_pixels, features = data.shape
            data = data.view(
                -1, features
            )  # Flatten to [batch_size * num_pixels, features]
        elif len(data.shape) == 2:
            batch_size, features = data.shape
            num_pixels = None  # No pixels in this case

        out = self.main(data)

        if num_pixels is not None:
            out = out.view(batch_size, num_pixels, -1)  # Reshape back if needed
        return out


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


class ResidualLayer(nn.Module):
    def __init__(self, width, dropout_rate=0.0):
        super().__init__()
        # First layer
        self.fc1 = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)  # 

        # Second layer
        self.fc2 = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)  # 

        # Activation and dropout
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        residual = x

        # First layer
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.relu(out)
        if self.dropout is not None:
            out = self.dropout(out)

        # Second layer
        out = self.fc2(out)
        out = self.norm2(out)

        # Residual connection
        out = out + residual
        out = self.relu(out)

        return out


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=120, depth=10, dropout_rate=0.1, output_dim=None
    ):
        super().__init__()
        layers = []

        # Input projection layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))  # 
        layers.append(nn.ReLU(inplace=True))

        # Residual blocks
        for _ in range(depth):
            layers.append(ResidualLayer(hidden_dim, dropout_rate=dropout_rate))

        # Output layer if needed
        if output_dim is not None:
            layers.append(
                nn.LayerNorm(hidden_dim)
            )   
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

        # Apply proper initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Process through the model
        x = self.model(x)
        return x
