import torch
import math
from torch.nn import Linear
from integrator.layers import Residual, MLP, MeanPool
from integrator.model.encoders import BaseEncoder


class CNNResNet2(BaseEncoder):
    def __init__(
        self,
        use_ln=True,
        conv1_out_channel=64,
        conv1_kernel_size=7,
        conv1_stride=2,
        conv1_padding=3,
        layer1_num_blocks=3,
        Z=3,
        H=21,
        W=21,
        dmodel=32,
    ):
        super().__init__()
        self.use_ln = use_ln
        self.Z = Z
        self.H = H
        self.W = W

        # Calculate output dimensions after conv1
        self.Z_out = (Z + 2 * conv1_padding - conv1_kernel_size) // conv1_stride + 1
        self.H_out = (H + 2 * conv1_padding - conv1_kernel_size) // conv1_stride + 1
        self.W_out = (W + 2 * conv1_padding - conv1_kernel_size) // conv1_stride + 1

        # Improved position encoding
        self.pos_encoding = ImprovedPositionalEncoding3D(Z, H, W, dmodel)

        # Counts pathway
        self.conv1_counts = torch.nn.Conv3d(
            1,
            conv1_out_channel // 2,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=conv1_padding,
        )
        self.norm_counts = torch.nn.GroupNorm(
            8, conv1_out_channel // 2
        )  # Use GroupNorm instead of LayerNorm

        # Position pathway
        self.conv1_pos = torch.nn.Conv3d(
            dmodel,
            conv1_out_channel // 2,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=conv1_padding,
        )
        self.norm_pos = torch.nn.GroupNorm(
            8, conv1_out_channel // 2
        )  # Use GroupNorm instead of LayerNorm

        # Feature scaling parameters
        self.counts_scale = torch.nn.Parameter(torch.ones(1))
        self.pos_scale = torch.nn.Parameter(torch.ones(1))

        # ReLU activation
        self.relu = torch.nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(
                conv1_out_channel, conv1_out_channel, kernel_size=3, padding=1
            ),
            torch.nn.GroupNorm(8, conv1_out_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(
                conv1_out_channel, conv1_out_channel, kernel_size=3, padding=1
            ),
            torch.nn.GroupNorm(8, conv1_out_channel),
        )

        # Global pooling
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.debug = False

    def reshape_counts(self, x, mask=None):
        # Extract counts (last feature)
        counts = x[..., -1]  # Shape: [batch_size, 1323]

        # Reshape counts to 3D volume [batch_size, Z, H, W]
        counts_volume = counts.view(-1, self.Z, self.H, self.W)

        # Add channel dimension
        counts_volume = counts_volume.unsqueeze(1)  # Shape: [batch_size, 1, Z, H, W]

        if mask is not None:
            mask = mask.view(-1, 1, self.Z, self.H, self.W)
            counts_volume = counts_volume * mask

        return counts_volume

    def forward(self, x, mask=None):
        # Process counts
        counts_volume = self.reshape_counts(x, mask)
        counts_feat = self.conv1_counts(counts_volume)
        counts_feat = self.norm_counts(counts_feat)
        counts_feat = self.relu(counts_feat)
        counts_feat = counts_feat * self.counts_scale

        # Process positions
        pos_encoding = self.pos_encoding(x)
        pos_feat = self.conv1_pos(pos_encoding)
        pos_feat = self.norm_pos(pos_feat)
        pos_feat = self.relu(pos_feat)
        pos_feat = pos_feat * self.pos_scale

        # Combine features
        x = torch.cat([counts_feat, pos_feat], dim=1)

        # Residual connection
        identity = x
        x = self.layer1(x)
        x = x + identity
        x = self.relu(x)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class SinusoidalPositionalEncoding3D(torch.nn.Module):
    def __init__(self, Z, H, W, dmodel=32):
        super().__init__()
        self.dmodel = dmodel

        # Create normalized position indices (0 to 1)
        pos_z = torch.linspace(0, 1, Z).float()
        pos_h = torch.linspace(0, 1, H).float()
        pos_w = torch.linspace(0, 1, W).float()

        # Calculate sinusoidal encoding frequencies
        div_term = torch.exp(
            torch.arange(0, dmodel, 2).float() * -(math.log(10000.0) / dmodel)
        )

        # Create encodings for each dimension
        pe_z = torch.zeros(Z, 1, dmodel)
        pe_h = torch.zeros(H, 1, dmodel)
        pe_w = torch.zeros(W, 1, dmodel)

        # Apply sinusoidal encoding
        pe_z[:, 0, 0::2] = torch.sin(pos_z[:, None] * div_term * math.pi)
        pe_z[:, 0, 1::2] = torch.cos(pos_z[:, None] * div_term * math.pi)

        pe_h[:, 0, 0::2] = torch.sin(pos_h[:, None] * div_term * math.pi)
        pe_h[:, 0, 1::2] = torch.cos(pos_h[:, None] * div_term * math.pi)

        pe_w[:, 0, 0::2] = torch.sin(pos_w[:, None] * div_term * math.pi)
        pe_w[:, 0, 1::2] = torch.cos(pos_w[:, None] * div_term * math.pi)

        # Combine encodings into 3D grid
        pe = torch.zeros(1, dmodel * 3, Z, H, W)
        for z in range(Z):
            for h in range(H):
                for w in range(W):
                    pe[0, :dmodel, z, h, w] = pe_z[z, 0]
                    pe[0, dmodel : 2 * dmodel, z, h, w] = pe_h[h, 0]
                    pe[0, 2 * dmodel :, z, h, w] = pe_w[w, 0]

        self.register_buffer("pe", pe)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.expand(batch_size, -1, -1, -1, -1)


class ImprovedDetectorPositionalEncoding3D(torch.nn.Module):
    def __init__(self, Z, H, W):
        super().__init__()
        self.Z = Z
        self.H = H
        self.W = W

        # Register coordinate ranges as buffers
        self.register_buffer(
            "coord_ranges",
            torch.tensor(
                [
                    [0.0, 2500.0],  # X range
                    [0.0, 2500.0],  # Y range
                    [0.0, 2.0],  # Z range
                ]
            ),
        )

        # Add learnable scale factors
        self.scale_factors = torch.nn.Parameter(torch.ones(3))
        self.norm = torch.nn.LayerNorm([3, Z, H, W])

    def forward(self, x):
        B = x.size(0)

        # Extract and normalize coordinates
        coords = []
        for i in range(3):
            coord = x[..., i]
            coord_norm = (
                2
                * (coord - self.coord_ranges[i, 0])
                / (self.coord_ranges[i, 1] - self.coord_ranges[i, 0])
                - 1
            )
            coord_norm = coord_norm.view(B, 1, self.Z, self.H, self.W)
            coords.append(coord_norm * self.scale_factors[i])

        # Combine coordinates
        pos = torch.cat(coords, dim=1)
        return self.norm(pos)


class ImprovedPositionalEncoding3D(torch.nn.Module):
    def __init__(self, Z, H, W, dmodel=32):
        super().__init__()
        self.detector_pe = ImprovedDetectorPositionalEncoding3D(Z, H, W)
        self.sinusoidal_pe = SinusoidalPositionalEncoding3D(Z, H, W, dmodel)

        # Balanced projection layers
        self.proj_detector = torch.nn.Sequential(
            torch.nn.Conv3d(3, dmodel, 1),
            torch.nn.LayerNorm([dmodel, Z, H, W]),
            torch.nn.ReLU(),
            torch.nn.Conv3d(dmodel, dmodel, 1),
        )

        self.proj_sinusoidal = torch.nn.Sequential(
            torch.nn.Conv3d(dmodel * 3, dmodel, 1),
            torch.nn.LayerNorm([dmodel, Z, H, W]),
            torch.nn.ReLU(),
            torch.nn.Conv3d(dmodel, dmodel, 1),
        )

        # Learnable combination weights
        self.combine_weights = torch.nn.Parameter(torch.ones(2))
        self.final_norm = torch.nn.LayerNorm([dmodel, Z, H, W])

    def forward(self, x):
        # Get both encodings
        detector_pos = self.detector_pe(x)
        sin_pos = self.sinusoidal_pe(x)

        # Project both
        detector_feat = self.proj_detector(detector_pos)
        sin_feat = self.proj_sinusoidal(sin_pos)

        # Weighted combination
        weights = torch.nn.functional.softmax(self.combine_weights, dim=0)
        combined = weights[0] * detector_feat + weights[1] * sin_feat

        return self.final_norm(combined)
