import pytorch_lightning as pl
import pytorch_lightning as pl
from pathlib import Path
import subprocess
import torch
import numpy as np
from dials.array_family import flex
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset
from pathlib import Path
import os
import numpy as np
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from integrator.layers import Constraint, MLP, Residual
from integrator import ShoeboxDataModule


import pytorch_lightning as pl
from pathlib import Path
import subprocess
import torch
import numpy as np
from dials.array_family import flex
from integrator.model import DirichletProfile

torch.set_float32_matmul_precision("medium")


# %%
class CrystalNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # Each GroupNorm needs weight/bias size matching its input channels
        self.coord_norm = nn.GroupNorm(
            1, num_channels // 4 * 3
        )  # 3/4 of channels for coords
        self.intensity_norm = nn.GroupNorm(
            1, num_channels // 4
        )  # 1/4 of channels for intensity

    def forward(self, x):
        # Calculate split point based on input channels
        n_coord_channels = (x.shape[1] * 3) // 4

        # Split channels
        coords = x[:, :n_coord_channels]
        counts = x[:, n_coord_channels:]

        # Normalize
        coords_norm = self.coord_norm(coords)
        counts_norm = self.intensity_norm(counts)

        # Concatenate back
        return torch.cat([coords_norm, counts_norm], dim=1)


class OutputHandler:
    @staticmethod
    def get_output_value(outputs, key):
        output_map = {
            "q_I_mean": lambda x: x["qI"].mean,
            "q_I_var": lambda x: x["qI"].variance,
            "refl_ids": lambda x: x["refl_ids"],
        }
        return output_map.get(key, lambda x: x[key])(outputs)


class DIALSCallback(pl.Callback):
    def __init__(
        self,
        every_n_epochs=5,
        output_dir="./dials_output/",
        dials_files_path=None,
        posterior_refls="posterior.refl",
        weighted_sum_refls="weighted_sum.refl",
        thresholded_refls="thresholded.refl",
        integrated_expt="/n/holylabs/LABS/hekstra_lab/Users/laldama/integratorv2/integrator/logs/DIALS_/CNNResNetSoftmax_08_045/integrated.expt",
        dials_env_path="/n/hekstra_lab/people/aldama/software/dials-v3-16-1/dials_env.sh",
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        if dials_files_path is None:
            raise ValueError("dials_files_path must be provided")
        self.dials_files_path = dials_files_path
        self.posterior_refls = posterior_refls
        self.weighted_sum_refls = weighted_sum_refls
        self.thresholded_refls = thresholded_refls
        self.integrated_expt = integrated_expt
        self.dials_env_path = dials_env_path

    def run_dials_command(self, command, wait=False):
        """Run a DIALS command asynchronously or synchronously with proper environment sourcing."""
        full_command = f"source {self.dials_env_path} && {command}"
        print(f"Executing command: {full_command}")

        try:
            process = subprocess.Popen(
                full_command,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # If wait is True, block until the process completes
            if wait:
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        returncode=process.returncode,
                        cmd=full_command,
                        output=stdout,
                        stderr=stderr,
                    )
                return stdout, stderr

            # Return the process to allow asynchronous execution
            return process

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

            raise

    def process_with_dials(self, epoch_dir):
        """Process reflection files with DIALS scaling and merge asynchronously."""
        if not Path(self.integrated_expt).exists():
            raise FileNotFoundError(f"Experiment file not found: {self.integrated_expt}")

        methods = {
            "posterior": self.posterior_refls,
            "weighted_sum": self.weighted_sum_refls,
            "thresholded": self.thresholded_refls,
        }

        for method_name, refl_file in methods.items():
            method_dir = epoch_dir / method_name
            method_dir.mkdir(exist_ok=True)

            input_refl = epoch_dir / refl_file
            scaled_refl = method_dir / "scaled.refl"
            scaled_expt = method_dir / "scaled.expt"

            if not input_refl.exists():
                print(f"Warning: Reflection file {input_refl} not found")
                continue

            try:
                print(f"\nProcessing {method_name} reflections...")
                print(f"Input reflection file: {input_refl}")
                print(f"Experiment file: {self.integrated_expt}")

                # Run dials.scale asynchronously
                scale_command = (
                    f"dials.scale {input_refl} {self.integrated_expt} "
                    f"output.reflections={scaled_refl} "
                    f"output.experiments={scaled_expt} "
                    f"output.html={method_dir}/scaling.html "
                    f"output.log={method_dir}/scaling.log"
                )
                scale_process = self.run_dials_command(scale_command)

                # Define dials.merge command to run after scaling completes
                def run_merge():
                    merge_command = (
                        f"dials.merge {scaled_refl} {scaled_expt}"
                        f"output.log={method_dir}/merged.log"
                        f"output.html={method_dir}/merged.html"
                        f"output.mtz={method_dir}/merged.mtz"
                    )
                    try:
                        self.run_dials_command(merge_command, wait=True)
                        print(f"Successfully merged {method_name} reflections")
                    except Exception as e:
                        print(f"Error merging {method_name} reflections: {str(e)}")

                # Use a background thread to monitor scaling and trigger merging
                import threading

                def monitor_scale():
                    scale_process.wait()
                    if scale_process.returncode == 0:
                        print(f"Successfully scaled {method_name} reflections")
                        run_merge()
                    else:
                        print(
                            f"Scaling failed for {method_name} with return code {scale_process.returncode}"
                        )

                threading.Thread(target=monitor_scale, daemon=True).start()

            except Exception as e:
                print(f"Error processing {method_name} reflections: {str(e)}")


    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        # Create epoch directory
        epoch_dir = self.output_dir / f"epoch_{trainer.current_epoch}"
        epoch_dir.mkdir(exist_ok=True)

        # Save checkpoint
        checkpoint_path = epoch_dir / "checkpoint.ckpt"
        trainer.save_checkpoint(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Load model from checkpoint with same architecture
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_model = type(pl_module).load_from_checkpoint(
            checkpoint_path,
            cnn_encoder=pl_module.cnn_encoder,
            fc_encoder=pl_module.fc_encoder,
            q_bg=pl_module.background_distribution,
            q_I=pl_module.intensity_distribution,
            profile_model=pl_module.profile_model,
            num_pixels=3 * 21 * 21,
            dataloader_length=len(trainer.datamodule.train_dataloader()),
            batch_size=trainer.datamodule.batch_size,
            dmodel=64,
            mc_samples=1000,
            profile_threshold=0.005,
        )
        loaded_model = loaded_model.to(device)
        loaded_model.eval()

        # Initialize output accumulators
        accumulated_outputs = defaultdict(list)

        # Process predictions
        with torch.no_grad():
            for batch in tqdm(
                trainer.datamodule.predict_dataloader(),
                desc=f"Processing epoch {trainer.current_epoch}",
            ):
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Move batch to device
                shoebox, dials, masks, metadata, counts = [x.to(device) for x in batch]

                # Get predictions
                try:
                    outputs = loaded_model(shoebox, dials, masks, metadata, counts)
                except RuntimeError as e:
                    print(f"Error during prediction: {str(e)}")
                    print(f"Model device: {next(loaded_model.parameters()).device}")
                    print(
                        f"Input devices: {shoebox.device}, {dials.device}, {masks.device}, {metadata.device}, {counts.device}"
                    )
                    raise

                # Store outputs (moving to CPU immediately)
                accumulated_outputs["qI_mean"].append(outputs["qI"].mean.cpu())
                accumulated_outputs["qI_variance"].append(outputs["qI"].variance.cpu())
                accumulated_outputs["weighted_sum_mean"].append(
                    outputs["weighted_sum_mean"].cpu()
                )
                accumulated_outputs["weighted_sum_var"].append(
                    outputs["weighted_sum_var"].cpu()
                )
                accumulated_outputs["thresholded_mean"].append(
                    outputs["thresholded_mean"].cpu()
                )
                accumulated_outputs["thresholded_var"].append(
                    outputs["thresholded_var"].cpu()
                )
                accumulated_outputs["refl_ids"].append(outputs["refl_ids"].cpu())

                # Clear GPU cache again
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Concatenate results
        final_outputs = {
            k: torch.cat(v).numpy() for k, v in accumulated_outputs.items()
        }

        # Load reflection table

        refl_file = "/n/holylabs/LABS/hekstra_lab/Users/laldama/integratorv2/integrator/data/pass1/reflections_.refl"
        tbl = flex.reflection_table.from_file(str(refl_file))

        # Create and apply selection mask
        sel = np.zeros(len(tbl), dtype=bool)
        reflection_ids = final_outputs["refl_ids"].astype(np.int32)
        sel[reflection_ids] = True
        temp = tbl.select(flex.bool(sel))

        # Write reflection files
        for method, means_key, vars_key, output_file in [
            ("posterior", "qI_mean", "qI_variance", self.posterior_refls),
            (
                "weighted_sum",
                "weighted_sum_mean",
                "weighted_sum_var",
                self.weighted_sum_refls,
            ),
            (
                "thresholded",
                "thresholded_mean",
                "thresholded_var",
                self.thresholded_refls,
            ),
        ]:
            temp_copy = temp.copy()
            temp_copy["intensity.prf.value"] = flex.double(final_outputs[means_key])
            temp_copy["intensity.prf.variance"] = flex.double(final_outputs[vars_key])
            temp_copy["intensity.sum.value"] = flex.double(final_outputs[means_key])
            temp_copy["intensity.sum.variance"] = flex.double(final_outputs[vars_key])
            temp_copy.as_file(str(epoch_dir / output_file))

        # Process with DIALS
        try:
            self.process_with_dials(epoch_dir)
            print(f"Successfully processed epoch {trainer.current_epoch} with DIALS")
        except Exception as e:
            print(f"Error processing with DIALS: {str(e)}")

        # Clean up
        del loaded_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# %%


class ResidualLayer(nn.Module):
    def __init__(self, width, dropout=None, use_norm=True):  # Changed parameter name
        super().__init__()
        self.use_norm = use_norm
        self.fc1 = Linear(width, width)
        if self.use_norm:
            # Change LayerNorm to GroupNorm
            self.norm1 = nn.GroupNorm(1, width)  # num_groups=1
        self.fc2 = Linear(width, width)
        if self.use_norm:
            self.norm2 = nn.GroupNorm(1, width)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = (
            nn.Dropout(dropout) if dropout else None
        )  # class Residual(nn.Module):

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        if self.use_norm:
            out = self.norm1(out)
        out = self.relu(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.fc2(out)
        if self.use_norm:
            out = self.norm2(out)
        out += residual
        out = self.relu(out)
        return out


def init_weights(m):
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class SinusoidalPositionalEncoding3D(nn.Module):
    def __init__(self, Z, H, W, d_model=32):
        super().__init__()
        self.d_model = d_model

        # Create normalized position indices (0 to 1)
        pos_z = torch.linspace(0, 1, Z).float()
        pos_h = torch.linspace(0, 1, H).float()
        pos_w = torch.linspace(0, 1, W).float()

        # Calculate sinusoidal encoding frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        # Create encodings for each dimension
        pe_z = torch.zeros(Z, 1, d_model)
        pe_h = torch.zeros(H, 1, d_model)
        pe_w = torch.zeros(W, 1, d_model)

        # Apply sinusoidal encoding
        pe_z[:, 0, 0::2] = torch.sin(pos_z[:, None] * div_term * math.pi)
        pe_z[:, 0, 1::2] = torch.cos(pos_z[:, None] * div_term * math.pi)

        pe_h[:, 0, 0::2] = torch.sin(pos_h[:, None] * div_term * math.pi)
        pe_h[:, 0, 1::2] = torch.cos(pos_h[:, None] * div_term * math.pi)

        pe_w[:, 0, 0::2] = torch.sin(pos_w[:, None] * div_term * math.pi)
        pe_w[:, 0, 1::2] = torch.cos(pos_w[:, None] * div_term * math.pi)

        # Combine encodings into 3D grid
        pe = torch.zeros(1, d_model * 3, Z, H, W)
        for z in range(Z):
            for h in range(H):
                for w in range(W):
                    pe[0, :d_model, z, h, w] = pe_z[z, 0]
                    pe[0, d_model : 2 * d_model, z, h, w] = pe_h[h, 0]
                    pe[0, 2 * d_model :, z, h, w] = pe_w[w, 0]

        self.register_buffer("pe", pe)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.expand(batch_size, -1, -1, -1, -1)


class ImprovedDetectorPositionalEncoding3D(nn.Module):
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
        self.scale_factors = nn.Parameter(torch.ones(3))
        self.norm = nn.LayerNorm([3, Z, H, W])

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


class ImprovedPositionalEncoding3D(nn.Module):
    def __init__(self, Z, H, W, d_model=32):
        super().__init__()
        self.detector_pe = ImprovedDetectorPositionalEncoding3D(Z, H, W)
        self.sinusoidal_pe = SinusoidalPositionalEncoding3D(Z, H, W, d_model)

        # Balanced projection layers
        self.proj_detector = nn.Sequential(
            nn.Conv3d(3, d_model, 1),
            nn.LayerNorm([d_model, Z, H, W]),
            nn.ReLU(),
            nn.Conv3d(d_model, d_model, 1),
        )

        self.proj_sinusoidal = nn.Sequential(
            nn.Conv3d(d_model * 3, d_model, 1),
            nn.LayerNorm([d_model, Z, H, W]),
            nn.ReLU(),
            nn.Conv3d(d_model, d_model, 1),
        )

        # Learnable combination weights
        self.combine_weights = nn.Parameter(torch.ones(2))
        self.final_norm = nn.LayerNorm([d_model, Z, H, W])

    def forward(self, x):
        # Get both encodings
        detector_pos = self.detector_pe(x)
        sin_pos = self.sinusoidal_pe(x)

        # Project both
        detector_feat = self.proj_detector(detector_pos)
        sin_feat = self.proj_sinusoidal(sin_pos)

        # Weighted combination
        weights = F.softmax(self.combine_weights, dim=0)
        combined = weights[0] * detector_feat + weights[1] * sin_feat

        return self.final_norm(combined)


class CNNResNet(torch.nn.Module):
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
        d_model=32,
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
        self.pos_encoding = ImprovedPositionalEncoding3D(Z, H, W, d_model)

        # Counts pathway
        self.conv1_counts = nn.Conv3d(
            1,
            conv1_out_channel // 2,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=conv1_padding,
        )
        self.norm_counts = nn.GroupNorm(
            8, conv1_out_channel // 2
        )  # Use GroupNorm instead of LayerNorm

        # Position pathway
        self.conv1_pos = nn.Conv3d(
            d_model,
            conv1_out_channel // 2,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=conv1_padding,
        )
        self.norm_pos = nn.GroupNorm(
            8, conv1_out_channel // 2
        )  # Use GroupNorm instead of LayerNorm

        # Feature scaling parameters
        self.counts_scale = nn.Parameter(torch.ones(1))
        self.pos_scale = nn.Parameter(torch.ones(1))

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = nn.Sequential(
            nn.Conv3d(conv1_out_channel, conv1_out_channel, kernel_size=3, padding=1),
            nn.GroupNorm(8, conv1_out_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(conv1_out_channel, conv1_out_channel, kernel_size=3, padding=1),
            nn.GroupNorm(8, conv1_out_channel),
        )

        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
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


def weight_initializer(weight):
    fan_avg = 0.5 * (weight.shape[-1] + weight.shape[-2])
    std = math.sqrt(1.0 / fan_avg / 10.0)
    a = -2.0 * std
    b = 2.0 * std
    torch.nn.init.trunc_normal_(weight, 0.0, std, a, b)
    return weight


class Linear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)  # Set bias=False

    def reset_parameters(self) -> None:
        self.weight = weight_initializer(self.weight)


class Decoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        q_I,
        q_bg,
        q_p,
        mc_samples=100,
    ):
        # Sample from variational distributions
        z = q_I.rsample([mc_samples]).unsqueeze(-1)
        bg = q_bg.rsample([mc_samples]).unsqueeze(-1)
        qp = q_p.rsample([mc_samples])

        rate = z.permute(1, 0, 2) * qp.permute(1, 0, 2) + bg.permute(1, 0, 2)

        return rate


class BackgroundDistribution(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        q_bg,
        constraint=Constraint(),
    ):
        super().__init__()
        self.fc = Linear(dmodel, 2)
        self.q_bg = q_bg
        self.constraint = constraint

    def background(self, params):
        loc = self.constraint(params[..., 0])
        scale = self.constraint(params[..., 1])
        return self.q_bg(loc, scale)

    def forward(self, representation):
        params = self.fc(representation)
        q_bg = self.background(params)
        return q_bg


class IntensityDistribution(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        q_I,
        # constraint=Constraint(),
    ):
        super().__init__()
        self.fc = Linear(dmodel, 2)
        self.q_I = q_I
        # self.constraint = constraint

    def intensity(self, params):
        # loc = self.constraint(params[..., 0])
        # scale = self.constraint(params[..., 1])
        loc = torch.exp(params[..., 0])
        scale = torch.exp(params[..., 1])
        return self.q_I(loc, scale)

    def forward(self, representation):
        params = self.fc(representation)
        q_I = self.intensity(params)
        return q_I


class FcMeta(torch.nn.Module):
    def __init__(self, depth=10, dmodel=64, feature_dim=7, dropout=None):
        super().__init__()
        self.linear = Linear(feature_dim, dmodel)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.batch_norm = nn.BatchNorm1d(dmodel)
        self.layer_norm = nn.LayerNorm(dmodel)
        self.mlp_1 = MLP(dmodel, depth, dropout=dropout, output_dims=dmodel)

    def forward(self, shoebox_data):
        # shoebox_data shape: [batch_size, num_pixels, feature_dim]
        batch_size, features = shoebox_data.shape

        # Initial linear transformation
        out = self.linear(shoebox_data)
        out = self.relu(out)
        out = self.layer_norm(out)
        out = self.mlp_1(out)
        return out


class Loss(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        p_bg=torch.distributions.gamma.Gamma(torch.tensor(1.0), torch.tensor(1.0)),
        p_I=torch.distributions.gamma.Gamma(torch.tensor(1.0), torch.tensor(1.0)),
        p_p_scale=0.0001,
        p_bg_scale=0.0001,
        p_I_scale=0.0001,
        mc_samples=100,
        recon_scale=0.00,
    ):
        super().__init__()

        # Don't specify device in __init__, let PyTorch handle it
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("recon_scale", torch.tensor(recon_scale))
        self.register_buffer("beta", torch.tensor(beta))

        # Scale parameters
        self.register_buffer("p_I_scale", torch.tensor(p_I_scale))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale))
        self.register_buffer("p_p_scale", torch.tensor(p_p_scale))

        # Prior distributions
        # Move tensors to appropriate device in forward pass
        self.p_bg = p_bg
        self.p_I = p_I

        # Dirichlet prior parameters will be moved to correct device in forward
        self.register_buffer("dirichlet_concentration", torch.ones(3 * 21 * 21))

    def to(self, device):
        # Override to() to ensure all components are moved to the correct device
        super().to(device)
        self.p_bg.loc = self.p_bg.loc.to(device)
        self.p_bg.scale = self.p_bg.scale.to(device)
        self.p_I.loc = self.p_I.loc.to(device)
        self.p_I.scale = self.p_I.scale.to(device)
        return self

    def forward(self, rate, counts, q_p, q_I, q_bg, dead_pixel_mask, eps=1e-5):
        # Ensure all inputs are on the same device
        device = rate.device

        # Move priors to the correct device if needed
        if not hasattr(self, "p_p") or self.p_p.concentration.device != device:
            self.p_p = torch.distributions.dirichlet.Dirichlet(
                self.dirichlet_concentration.to(device)
            )

        # Ensure other components are on the correct device
        counts = counts.to(device)
        dead_pixel_mask = dead_pixel_mask.to(device)

        # Calculate log likelihood
        ll = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))

        # Calculate KL divergences with device-aware tensors
        kl = torch.distributions.kl.kl_divergence(q_p, self.p_p) * self.p_p_scale
        kl += torch.distributions.kl.kl_divergence(q_bg, self.p_bg) * self.p_bg_scale
        kl += torch.distributions.kl.kl_divergence(q_I, self.p_I) * self.p_I_scale

        # Calculate reconstruction loss
        recon_loss = torch.abs(rate.mean(1) - counts) / (counts + self.eps)
        recon_loss = recon_loss.mean() * self.recon_scale

        # Calculate final loss terms
        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)
        ll_sum = ll_mean.sum(dim=1)
        neg_ll_sum = -ll_sum

        total_loss = neg_ll_sum + kl + recon_loss

        return total_loss, neg_ll_sum, kl, recon_loss


class Integrator(pl.LightningModule):
    def __init__(
        self,
        cnn_encoder,
        fc_encoder,
        q_bg,
        q_I,
        profile_model,
        num_pixels,
        dataloader_length,
        batch_size,
        dmodel,
        mc_samples=100,
        learning_rate=1e-3,
        profile_threshold=0.001,
    ):
        super().__init__()
        self.learning_rate = learning_rate

        # Model components
        self.cnn_encoder = cnn_encoder
        self.fc_encoder = fc_encoder
        self.profile_model = profile_model

        # Additional layers
        self.fc_representation = Linear(dmodel * 2, dmodel)
        self.decoder = Decoder()

        # Loss function
        self.loss_fn = Loss()
        self.background_distribution = q_bg
        self.intensity_distribution = q_I
        self.norm = nn.LayerNorm(dmodel)
        self.mc_samples = mc_samples
        self.profile_threshold = profile_threshold
        self.automatic_optimization = True

    #    def calculate_intensities(self, counts, qbg, qp):
    #        with torch.no_grad():
    #            batch_counts = counts.unsqueeze(1)
    #
    #            batch_bg_samples = (qbg.rsample([self.mc_samples]).unsqueeze(-1)).permute(1,0,2)
    #
    #            batch_profile_samples = qp.rsample([self.mc_samples]).permute(1,0,2)
    #
    #            weighted_sum_intensity = (
    #                    batch_counts - batch_bg_samples
    #                    ) * batch_profile_samples
    #            weighted_sum_intensity_mean = weighted_sum_intensity.sum(-1).mean(-1)
    #            weighted_sum_intensity_var = weighted_sum_intensity.sum(-1).var(-1)
    #
    #            profile_masks = (
    #                    batch_profile_samples > self.profile_threshold
    #                    )
    #            masked_counts = batch_counts * profile_masks
    #            thresholded_intensity = (
    #                    masked_counts - batch_bg_samples * profile_masks
    #                    ).sum(-1)
    #            thresholded_mean = thresholded_intensity.mean(-1)
    #            thresholded_var = thresholded_intensity.var(-1)
    #
    #            intensities = {
    #                    "thresholded_mean": thresholded_mean,
    #                    "thresholded_var": thresholded_var,
    #                    "weighted_sum_intensity_mean":weighted_sum_intensity_mean,
    #                    "weighted_sum_intensity_var":weighted_sum_intensity_var,
    #                    }
    #
    #
    #            return intensities,batch_profile_samples

    def calculate_intensities(self, counts, qbg, qp, dead_pixel_mask):
        with torch.no_grad():
            counts = counts * dead_pixel_mask
            batch_counts = counts.unsqueeze(1)  # [batch_size x 1 x pixels]

            batch_bg_samples = (qbg.rsample([self.mc_samples]).unsqueeze(-1)).permute(
                1, 0, 2
            )
            batch_profile_samples = qp.rsample([self.mc_samples]).permute(
                1, 0, 2
            )  # [batch_size x mc_samples x pixels]

            batch_profile_samples = batch_profile_samples * dead_pixel_mask.unsqueeze(1)

            weighted_sum_intensity = (
                batch_counts - batch_bg_samples
            ) * batch_profile_samples

            weighted_sum_intensity_sum = weighted_sum_intensity.sum(-1)

            summed_squared_prf = torch.norm(batch_profile_samples, p=2, dim=-1).pow(2)

            division = weighted_sum_intensity_sum / summed_squared_prf

            weighted_sum_intensity_mean = division.mean(-1)

            centered_w_ = (
                weighted_sum_intensity_sum - weighted_sum_intensity_mean.unsqueeze(-1)
            )

            weighted_sum_intensity_var = division.var(-1)

            profile_masks = batch_profile_samples > self.profile_threshold

            N_used = profile_masks.sum(-1).float()  # [batch_size × mc_samples]

            masked_counts = batch_counts * profile_masks

            thresholded_intensity = (
                masked_counts - batch_bg_samples * profile_masks
            ).sum(-1)

            thresholded_mean = thresholded_intensity.mean(-1)

            # thresholded_var = thresholded_intensity.var(-1)

            centered_thresh = thresholded_intensity - thresholded_mean.unsqueeze(-1)

            # thresholded_var = (centered_thresh ** 2).mean(-1)

            thresholded_var = (centered_thresh**2).sum(-1) / (N_used.mean(-1) + 1e-6)

            intensities = {
                "thresholded_mean": thresholded_mean,
                "thresholded_var": thresholded_var,
                "weighted_sum_intensity_mean": weighted_sum_intensity_mean,
                "weighted_sum_intensity_var": weighted_sum_intensity_var,
            }

            return intensities

    # def forward(self, shoebox, dials, masks, metadata,counts,samples):
    def forward(self, shoebox, dials, masks, metadata, counts):
        # Original forward pass
        counts = torch.clamp(counts, min=0)
        coords = metadata[..., :3]
        # dxyz = samples[...,3:6]

        batch_size, num_pixels, features = shoebox.shape

        # Get representations and distributions
        shoebox_representation = self.cnn_encoder(shoebox, masks)
        meta_representation = self.fc_encoder(metadata)

        representation = torch.cat([shoebox_representation, meta_representation], dim=1)
        representation = self.fc_representation(representation)
        representation = self.norm(representation)

        qbg = self.background_distribution(representation)
        qI = self.intensity_distribution(representation)
        qp = self.profile_model(representation)

        # Calculate intensities
        rate = self.decoder(qI, qbg, qp)
        #        dispersion = (dxyz*qp.mean.unsqueeze(-1)).sum(1).sum(-1)
        intensities = self.calculate_intensities(counts, qbg, qp, masks)

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            #            "dispersion":dispersion,
            "qI": qI,
            "qbg": qbg,
            "qp": qp,
            "dials_I_sum_value": dials[:, 0],
            "dials_I_sum_var": dials[:, 1],
            "dials_I_prf_value": dials[:, 2],
            "dials_I_prf_var": dials[:, 3],
            "refl_ids": dials[:, 4],
            "weighted_sum_mean": intensities["weighted_sum_intensity_mean"],
            "weighted_sum_var": intensities["weighted_sum_intensity_var"],
            "thresholded_mean": intensities["thresholded_mean"],
            "thresholded_var": intensities[
                "thresholded_var"
            ],  # Match the key from calculate_intensities
        }

    # def training_step(self, batch, batch_idx):
    # shoebox, dials, masks, metadata = batch
    # outputs = self(shoebox, metadata, masks, dials)

    # loss, neg_ll, kl, recon_loss = self.loss_fn(
    # outputs["rate"],
    # outputs["counts"],
    # outputs["qp"],
    # outputs["masks"]
    # )

    # # Log all components
    # self.log("train_loss", loss.mean())
    # self.log("train_nll", neg_ll.mean())
    # self.log("train_kl", kl.mean())
    # self.log("train_recon", recon_loss)

    # return loss.mean()

    # def training_step(self, batch, batch_idx):
    # shoebox, dials, masks, metadata = batch
    # outputs = self(shoebox, dials, masks, metadata)

    # neg_ll, kl = self.loss_fn(
    # outputs["rate"],
    # outputs["counts"],
    # outputs["qp"],
    # outputs["qI"],
    # outputs["qbg"],
    # outputs["masks"],
    # )
    # loss = (neg_ll + kl).mean()

    # # Log metrics
    # self.log("train_loss", loss)
    # self.log("train_nll", neg_ll.mean())
    # self.log("train_kl", kl.mean())

    # return loss

    def training_step(self, batch, batch_idx):
        # shoebox, dials, masks, metadata,counts,samples = batch
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)

        # neg_ll, kl = self.loss_fn(
        loss, neg_ll, kl, recon_loss = self.loss_fn(
            outputs["rates"],
            outputs["counts"],
            outputs["qp"],
            outputs["qI"],
            outputs["qbg"],
            outputs["masks"],
        )

        # Log metrics
        self.log("train_loss", loss.mean())
        self.log("train_nll", neg_ll.mean())
        self.log("train_kl", kl.mean())
        self.log("train_recon", recon_loss)

        return loss.mean()

    # def validation_step(self, batch, batch_idx):
    # shoebox, dials, masks, metadata = batch
    # outputs = self(shoebox, dials, masks, metadata)

    # loss,neg_ll, kl,recon_loss = self.loss_fn(
    # outputs["rate"],
    # outputs["counts"],
    # outputs["qp"],
    # outputs["qI"],
    # outputs["qbg"],
    # outputs["masks"]
    # )
    # loss = (neg_ll + kl).mean()

    # self.log("val_loss", loss)
    # self.log("val_nll", neg_ll.mean())
    # self.log("val_kl", kl.mean())
    # self.log("val_recon", recon_loss)

    # return loss.mean()

    def validation_step(self, batch, batch_idx):
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)

        # Calculate validation metrics
        loss, neg_ll, kl, recon_loss = self.loss_fn(
            outputs["rates"],
            outputs["counts"],
            outputs["qp"],
            outputs["qI"],
            outputs["qbg"],
            outputs["masks"],
        )

        # Log metrics
        self.log("val_loss", loss.mean())
        self.log("val_nll", neg_ll.mean())
        self.log("val_kl", kl.mean())
        self.log("val_recon", recon_loss)

        # Return the complete outputs dictionary
        return outputs

    def predict_step(self, batch, batch_idx):
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)
        return {
            "qI_mean": outputs["qI"].mean,
            "qI_variance": outputs["qI"].variance,
            "weighted_sum_mean": outputs["weighted_sum_mean"],
            "weighted_sum_var": outputs["weighted_sum_var"],
            "thresholded_mean": outputs["thresholded_mean"],
            "thresholded_var": outputs["thresholded_var"],
            "refl_ids": outputs["refl_ids"],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class Standardize(torch.nn.Module):
    def __init__(
        self, center=True, feature_dim=7, epsilon=1e-6, coord_indices=(0, 1, 2)
    ):
        super().__init__()
        self.epsilon = epsilon
        self.center = center
        self.coord_indices = set(coord_indices)

        # Register buffers for statistics
        self.register_buffer("mean", torch.zeros((1, 1, feature_dim)))
        self.register_buffer("m2", torch.zeros((1, 1, feature_dim)))
        self.register_buffer("pixel_count", torch.tensor(0.0))

        # Add buffers for tracking update status
        self.register_buffer("total_pixels_seen", torch.tensor(0.0))
        self.register_buffer("should_update", torch.tensor(True))
        self.register_buffer("max_pixels", torch.tensor(float("inf")))

        # Register coordinate ranges
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

    def set_max_pixels(self, num_samples, pixels_per_sample):
        self.max_pixels = torch.tensor(float(num_samples * pixels_per_sample))
        print(
            f"Standardization will stop updating after seeing {self.max_pixels} pixels"
        )

    @property
    def var(self):
        m2 = torch.clamp(self.m2, min=self.epsilon)
        return m2 / (self.pixel_count - 1).clamp(min=1)

    @property
    def std(self):
        return torch.sqrt(self.var)

    def normalize_coordinates(self, im):
        normalized = im.clone()
        for idx, (min_val, max_val) in enumerate(self.coord_ranges):
            if idx in self.coord_indices:
                normalized[..., idx] = (
                    2 * (im[..., idx] - min_val) / (max_val - min_val) - 1
                )
        return normalized

    def update(self, im, mask=None):
        if not self.should_update:
            return

        # Input shape is [batch_size, pixels, features]
        B, P, F = im.shape
        if mask is None:
            mask = torch.ones((B, P), device=im.device)

        batch_count = mask.sum()
        if batch_count == 0:
            return

        self.total_pixels_seen += batch_count

        if self.total_pixels_seen >= self.max_pixels:
            self.should_update = torch.tensor(False)
            print(
                f"\nStandardization updates stopped at {self.total_pixels_seen} pixels"
            )

        # Create feature mask for non-coordinate features
        feature_mask = torch.ones(F, dtype=torch.bool, device=im.device)
        for idx in self.coord_indices:
            feature_mask[idx] = False

        # Previous values
        old_pixel_count = self.pixel_count
        self.pixel_count += batch_count

        # Compute masked sum and mean for non-coordinate features
        masked_im = im * mask.unsqueeze(-1)
        masked_sum = masked_im[..., feature_mask].sum(dim=(0, 1))
        batch_mean = masked_sum / batch_count

        # Update mean
        delta = torch.zeros_like(self.mean.squeeze())
        delta[feature_mask] = batch_mean - self.mean.squeeze()[feature_mask]
        self.mean = (
            self.mean.squeeze() + delta * (batch_count / self.pixel_count)
        ).view(1, 1, -1)

        # Update M2
        if old_pixel_count > 0:
            delta_old = torch.zeros_like(self.mean.squeeze())
            delta_old[feature_mask] = batch_mean - self.mean.squeeze()[feature_mask]

            # Compute updates only for non-coordinate features
            feature_indices = torch.where(feature_mask)[0]
            for idx in feature_indices:
                diff1 = (
                    im[..., idx : idx + 1]
                    - batch_mean[idx - sum(i < idx for i in self.coord_indices)]
                )
                diff2 = im[..., idx : idx + 1] - self.mean[..., idx]
                self.m2[..., idx] += (diff1 * mask.unsqueeze(-1) * diff2).sum()

            correction = (
                delta_old * delta * (old_pixel_count * batch_count / self.pixel_count)
            )
            self.m2 += correction.view(1, 1, -1)
        else:
            # First batch
            feature_indices = torch.where(feature_mask)[0]
            for idx in feature_indices:
                diff = im[..., idx : idx + 1] - self.mean[..., idx]
                self.m2[..., idx] = (diff * mask.unsqueeze(-1) * diff).sum()

    def standardize(self, im, mask=None):
        # First normalize coordinates
        normalized = self.normalize_coordinates(im)

        # Create feature mask for non-coordinate features
        feature_mask = torch.ones(im.shape[-1], dtype=torch.bool, device=im.device)
        for idx in self.coord_indices:
            feature_mask[idx] = False

        # Standardize non-coordinate features
        if self.center:
            if mask is None:
                normalized[..., feature_mask] = (
                    im[..., feature_mask] - self.mean[..., feature_mask]
                ) / self.std[..., feature_mask]
            else:
                normalized[..., feature_mask] = (
                    (im[..., feature_mask] - self.mean[..., feature_mask])
                    * mask.unsqueeze(-1)
                    / self.std[..., feature_mask]
                )
        else:
            normalized[..., feature_mask] = (
                im[..., feature_mask] / self.std[..., feature_mask]
            )

        return normalized

    def forward(self, im, mask=None, training=True):
        if training and self.should_update:
            self.update(im, mask)
        return self.standardize(im, mask)


def main():
    # Data setup
    batch_size = 25

    data_module = ShoeboxDataModule(
        # data_dir="/Users/luis/integratorv3/integrator/data/hewl_816/",
        data_dir="/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/data/pass1_sub/",
        batch_size=batch_size,
        val_split=0.3,
        test_split=0.0,
        num_workers=4,
        include_test=False,
        subset_size=1000,
        cutoff=None,
        shoebox_features=True,
    )

    data_module.setup()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model components
    Z, H, W = 3, 21, 21
    dmodel = 64
    num_epochs = 20

    # Initialize model components
    cnn_encoder = CNNResNet(Z=Z, H=H, W=W)
    fc_encoder = FcMeta(dmodel=dmodel)
    profile_model = DirichletProfile(dmodel=dmodel, num_components=Z * H * W)

    # Initialize the full model
    model = Integrator(
        cnn_encoder=cnn_encoder,
        fc_encoder=fc_encoder,
        q_bg=BackgroundDistribution(dmodel=64, q_bg=torch.distributions.gamma.Gamma),
        q_I=IntensityDistribution(dmodel=64, q_I=torch.distributions.gamma.Gamma),
        profile_model=profile_model,
        num_pixels=Z * H * W,
        dataloader_length=len(data_module.train_dataloader()),
        batch_size=batch_size,
        dmodel=dmodel,
    )

    # Move model to GPU before creating optimizer
    model = model.to(device)

    dials_callback = DIALSCallback(
        every_n_epochs=5,
        output_dir="./dials_output/",
        # dials_files_path="/Users/luis/integratorv3/integrator/data/hewl_816/",
        dials_files_path="/n/holylabs/LABS/hekstra_lab/Users/laldama/pass1",
    )

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        #       accelerator="cpu",
        devices=1,
        logger=True,
        precision="16-mixed",
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        log_every_n_steps=1,
        deterministic=False,
        callbacks=[
            dials_callback,
        ],
        enable_checkpointing=True,
    )

    trainer.datamodule = data_module

    # Train the model
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    # Save checkpoint
    trainer.save_checkpoint("integrator.ckpt")

    # After training, explicitly move model to GPU again for prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Ensure model is on GPU after training
    model.eval()

    print("Starting predictions...")

    # Lists to store outputs
    all_outputs = {
        "q_I_mean": [],
        "q_I_variance": [],
        "q_bg_mean": [],
        "q_bg_variance": [],
        "dials_I_sum_value": [],
        "dials_I_sum_var": [],
        "dials_I_prf_value": [],
        "dials_I_prf_var": [],
        "refl_ids": [],
        "thresholded_mean": [],
        "thresholded_var": [],
        "weighted_sum_mean": [],
        "weighted_sum_var": [],
    }

    # Run predictions
    with torch.no_grad():
        for batch in tqdm(data_module.predict_dataloader(), desc="Processing batches"):
            # Explicitly move each tensor in the batch to the same device as the model

            shoebox, dials, masks, metadata, counts = [x.to(device) for x in batch]

            shoebox = shoebox.to(device)
            dials = dials.to(device)
            masks = masks.to(device)
            metadata = metadata.to(device)
            counts = counts.to(device)

            # Double-check model and its submodules are on the correct device
            for param in model.parameters():
                if param.device != device:
                    print(
                        f"Warning: Found parameter on {param.device} instead of {device}"
                    )
                    param.data = param.data.to(device)

            # Get predictions
            try:
                outputs = model(shoebox, dials, masks, metadata, counts)
            except RuntimeError as e:
                print(f"Error during prediction: {str(e)}")
                print(f"Model device: {next(model.parameters()).device}")
                print(
                    f"Input devices: {shoebox.device}, {dials.device}, {masks.device}, {metadata.device}, {counts.device}"
                )
                raise

            # Store outputs (move to CPU for storage)
            all_outputs["q_I_mean"].append(outputs["qI"].mean.cpu())
            all_outputs["q_I_variance"].append(outputs["qI"].variance.cpu())
            all_outputs["q_bg_mean"].append(outputs["qbg"].mean.cpu())
            all_outputs["q_bg_variance"].append(outputs["qbg"].variance.cpu())
            all_outputs["dials_I_sum_value"].append(outputs["dials_I_sum_value"].cpu())
            all_outputs["dials_I_sum_var"].append(outputs["dials_I_sum_var"].cpu())
            all_outputs["dials_I_prf_value"].append(outputs["dials_I_prf_value"].cpu())
            all_outputs["dials_I_prf_var"].append(outputs["dials_I_prf_var"].cpu())
            all_outputs["refl_ids"].append(outputs["refl_ids"].cpu())
            all_outputs["thresholded_mean"].append(outputs["thresholded_mean"].cpu())
            all_outputs["thresholded_var"].append(outputs["thresholded_var"].cpu())
            all_outputs["weighted_sum_mean"].append(outputs["weighted_sum_mean"].cpu())
            all_outputs["weighted_sum_var"].append(outputs["weighted_sum_var"].cpu())
    #            all_outputs['dispersion'].append(outputs['dispersion'].cpu())

    # Concatenate all batches
    final_outputs = {k: torch.cat(v) for k, v in all_outputs.items()}

    # Save outputs
    print("Saving predictions...")
    torch.save(final_outputs, "predictions.pt")
    print("Done!")


if __name__ == "__main__":
    main()
