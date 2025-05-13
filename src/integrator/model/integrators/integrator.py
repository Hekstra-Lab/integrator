import torch
import numpy as np
import math
import torch.nn as nn
from integrator.model.integrators import BaseIntegrator
from integrator.layers import Linear
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import torch
from integrator.model.distribution import BaseDistribution
from integrator.model.encoders import CNNResNet2
from integrator.layers import Linear, Constraint, MLP, ResidualLayer
from torch.distributions import Dirichlet, Gamma, LogNormal
from integrator.model.encoders import MLPImageEncoder, MLPMetadataEncoder
from lightning.pytorch.utilities import grad_norm


class PixelEncoder(nn.Module):
    def __init__(self, encoding_dim=64, depth=10, dmodel=64, dropout=0.0):
        super().__init__()

        # Initial projection to match the dimension we want to process
        self.input_projection = Linear(encoding_dim, dmodel)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(dmodel)

        # MLP to process each pixel's features (shared across all pixels)
        layers = []
        for _ in range(depth):
            layers.append(ResidualLayer(dmodel, dropout_rate=dropout))
        self.pixel_mlp = nn.Sequential(*layers)

        # Final layer norm
        self.final_norm = nn.LayerNorm(dmodel)

        # Optional: Add another projection after pooling
        self.output_projection = Linear(dmodel, dmodel)

    def forward(self, x):
        """
        Args:
            x: Encoded pixel data with shape [batch_size, num_pixels, encoding_dim]

        Returns:
            Representation with shape [batch_size, dmodel]
        """
        batch_size, num_pixels, encoding_dim = x.shape

        # Initial projection
        x = self.input_projection(x)  # [batch_size, num_pixels, dmodel]
        x = self.relu(x)
        x = self.norm(x)

        # Process each pixel with shared MLP
        x = self.pixel_mlp(x)  # [batch_size, num_pixels, dmodel]
        x = self.final_norm(x)

        # Average pooling across pixels
        x = x.mean(dim=1)  # [batch_size, dmodel]

        # Optional final projection
        x = self.output_projection(x)  # [batch_size, dmodel]

        return x


def encode_raw_counts(counts, masks, encoding_dim=64):
    """
    Apply frequency encoding to raw pixel counts

    Args:
        counts: Tensor of shape [batch_size, num_pixels] (your 1323 pixels)
        masks: Binary tensor indicating valid pixels
        encoding_dim: Dimension of the encoding

    Returns:
        Encoded features of shape [batch_size, num_pixels, encoding_dim]
    """
    device = counts.device
    batch_size, num_pixels = counts.shape

    # Apply masking and clamp to ensure valid values
    masked_counts = torch.clamp(counts, min=0) * masks

    # Create a range of frequencies
    freqs_per_encoding = encoding_dim // 4  # We'll divide encoding_dim into 4 parts
    freqs = 2.0 ** torch.linspace(0, 8, freqs_per_encoding, device=device)  # 2^0 to 2^8

    # Apply log1p to counts to handle large ranges better
    log_counts = torch.log1p(masked_counts)

    # Create embeddings: [batch_size, num_pixels, freqs_per_encoding]
    sin_encoding = torch.sin(log_counts.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0))
    cos_encoding = torch.cos(log_counts.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0))

    # Create position indices tensor and expand to match batch size
    # Shape: [batch_size, num_pixels]
    pixel_pos = (
        torch.arange(num_pixels, device=device)
        .float()
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    # Normalize positions to [0, 1] range for better numerical stability
    normalized_pos = pixel_pos / num_pixels

    # Create position embeddings: [batch_size, num_pixels, freqs_per_encoding]
    sin_pos = torch.sin(normalized_pos.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0))
    cos_pos = torch.cos(normalized_pos.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0))

    # Now all tensors have shape [batch_size, num_pixels, freqs_per_encoding]
    # Combine value encoding and position encoding
    encoding = torch.cat([sin_encoding, cos_encoding, sin_pos, cos_pos], dim=-1)

    # Apply mask to ensure invalid pixels have zero encoding
    encoding = encoding * masks.unsqueeze(-1)

    return encoding  # %%


# dirichlet version
class Integrator(BaseIntegrator):
    def __init__(
        self,
        encoder,
        loss,
        qbg,
        qp,
        qI,
        mc_samples=100,
        learning_rate=1e-3,
        max_iterations=4,
        profile_threshold=0.001,
        renyi_scale=0.00,
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "mlp_encoder",
                "profile_model",
                "unet",
                "signal_preprocessor",
            ]
        )
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples

        # Model components
        self.encoder = encoder
        self.qp = qp
        self.qI = qI
        self.qbg = qbg
        self.automatic_optimization = True
        self.loss_fn = loss
        self.max_iterations = max_iterations
        # self.bg_encoder = MLPMetadataEncoder(feature_dim=60, output_dims=64)
        self.bg_encoder = MLPMetadataEncoder(feature_dim=1323)
        self.renyi_scale = renyi_scale
        B = torch.distributions.Normal(0, 1).sample((32, 10))
        self.register_buffer("B", B, persistent=True)

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks  # [B,P]
            zbg = (
                qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            )  # [B,S,1]
            # zp = qp.rsample([self.mc_samples]).permute(1, 0, 2) #
            zp = qp.mean.unsqueeze(1)  # [B,1,P]

            vi = zbg + 1e-6

            # kabsch sum
            for i in range(4):
                num = (counts.unsqueeze(1) - zbg) * zp * masks.unsqueeze(1) / vi
                denom = zp.pow(2) / vi
                I = num.sum(-1) / denom.sum(-1)  # [batch_size, mc_samples]
                vi = (I.unsqueeze(-1) * zp) + zbg
                vi = vi.mean(-1, keepdim=True)
            kabsch_sum_mean = I.mean(-1)
            kabsch_sum_var = I.var(-1)

            # profile masking
            zp = zp * masks.unsqueeze(1)  # profiles
            thresholds = torch.quantile(
                zp, 0.99, dim=-1, keepdim=True
            )  # threshold values
            profile_mask = zp > thresholds

            masked_counts = counts.unsqueeze(1) * profile_mask

            profile_masking_I = (masked_counts - zbg * profile_mask).sum(-1)

            profile_masking_mean = profile_masking_I.mean(-1)

            profile_masking_var = profile_masking_I.var(-1)

            intensities = {
                "profile_masking_mean": profile_masking_mean,
                "profile_masking_var": profile_masking_var,
                "kabsch_sum_mean": kabsch_sum_mean,
                "kabsch_sum_var": kabsch_sum_var,
            }

            return intensities

    def forward(self, counts, shoebox, metadata, masks, reference):
        # Unpack batch
        counts = torch.clamp(counts, min=0) * masks
        device = counts.device

        # num_valid_pixels = masks.sum(1)
        # total_photons = (counts).sum(1)
        # mean_photons = total_photons / num_valid_pixels
        # max_photons = counts.max(1)[0]
        # std_photons = torch.sqrt(
        # (1 / (num_valid_pixels - 1))
        # * (((counts - mean_photons.unsqueeze(1)) ** 2) * masks).sum(1)
        # )
        # q1 = torch.quantile(counts, 0.9999, dim=1)
        # q2 = torch.quantile(counts, 0.999, dim=1)
        # q3 = torch.quantile(counts, 0.9, dim=1)
        # q4 = torch.quantile(counts, 0.50, dim=1)
        # q5 = torch.quantile(counts, 0.25, dim=1)

        # vals = torch.stack(
        # [
        # torch.log1p(total_photons),
        # torch.log1p(mean_photons),
        # torch.log1p(max_photons),
        # torch.log1p(std_photons),
        # torch.log1p(q1),
        # torch.log1p(q2),
        # torch.log1p(q3),
        # torch.log1p(q4),
        # torch.log1p(q5),
        # std_photons / mean_photons,
        # ]
        # ).transpose(1, 0)

        # encoding_dim = 64
        # freqs = 2.0 ** torch.arange(
        # 0, encoding_dim // (2 * vals.shape[-1]), device=device
        # )

        # sin_encoding = torch.sin(vals.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0))
        # cos_encoding = torch.cos(vals.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0))
        # sin_encoding = sin_encoding.reshape(sin_encoding.shape[0], -1)
        # cos_encoding = cos_encoding.reshape(cos_encoding.shape[0], -1)
        # intensity_encoding = torch.concat((sin_encoding, cos_encoding), dim=1)

        # intensity_encoding = encode_raw_counts(counts, masks)

        shoebox = torch.log1p(shoebox)

        rep = self.encoder(shoebox.reshape(shoebox.shape[0], 1, 3, 21, 21), masks)
        rep2 = self.bg_encoder(shoebox)
        # bgrep = self.bg_encoder(shoebox)

        qbg = self.qbg(rep2)
        qp = self.qp(rep)
        # qI = self.qI(intensity_rep)
        qI = self.qI(rep2, metarep=rep)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qI.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        intensity_mean = qI.mean  # [batch_size]
        intensity_var = qI.variance  # [batch_size]

        rate = zI * zp + zbg

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qbg": qbg,
            "qp": qp,
            "qp_mean": qp.mean,
            "qI": qI,
            "intensity_mean": intensity_mean,
            "intensity_mean": intensity_mean,
            "intensity_var": intensity_var,
            "dials_I_sum_value": reference[:, 6],
            "dials_I_sum_var": reference[:, 7],
            "dials_I_prf_value": reference[:, 8],
            "dials_I_prf_var": reference[:, 9],
            "refl_ids": reference[:, -1],
            "profile": qp.mean,
            "zp": zp,
            "x_c": reference[:, 0],
            "y_c": reference[:, 1],
            "z_c": reference[:, 2],
            "x_c_mm": reference[:, 3],
            "y_c_mm": reference[:, 4],
            "z_c_mm": reference[:, 5],
            "dials_bg_mean": reference[:, 10],
            "dials_bg_sum_value": reference[:, 11],
            "dials_bg_sum_var": reference[:, 12],
            "d": reference[:, 13],
        }

    def training_step(self, batch, batch_idx):
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        # Calculate loss
        (loss, neg_ll, kl, kl_bg, kl_I, kl_p) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Track gradient norms here
        norms = grad_norm(self, norm_type=2)
        for name, norm in norms.items():
            self.log(f"grad_norm/{name}", norm)

        renyi_loss = (
            (-torch.log(outputs["qp"].rsample([100]).permute(1, 0, 2).pow(2).sum(-1)))
            .mean(1)
            .sum()
        ) * self.renyi_scale
        self.log("renyi_loss", renyi_loss)

        # Log metrics
        self.log("Train: -ELBO", loss.mean())
        self.log("Train: NLL", neg_ll.mean())
        self.log("Train: KL", kl.mean())
        self.log("Train: KL Bg", kl_bg.mean())
        self.log("Train: KL I", kl_I.mean())
        self.log("Train: KL Prf", kl_p.mean())
        self.log("Mean(qI.mean)", outputs["qI"].mean.mean())
        self.log("Min(qI.mean)", outputs["qI"].mean.min())
        self.log("Max(qI.mean)", outputs["qI"].mean.max())
        self.log("Mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("Min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("Max(qbg.mean)", outputs["qbg"].mean.max())
        self.log("Mean(qbg.variance)", outputs["qbg"].variance.mean())

        return loss.mean() + renyi_loss.sum()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        (
            loss,
            neg_ll,
            kl,
            kl_bg,
            kl_I,
            kl_p,
        ) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Log metrics
        self.log("Val: -ELBO", loss.mean())
        self.log("Val: NLL", neg_ll.mean())
        self.log("Val: KL", kl.mean())
        self.log("Val: KL bg", kl_bg.mean())
        self.log("Val: KL I", kl_I.mean())
        self.log("Val: KL prf", kl_p.mean())
        self.log("val_loss", neg_ll.mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        intensities = self.calculate_intensities(
            counts=outputs["counts"],
            qbg=outputs["qbg"],
            qp=outputs["qp"],
            masks=outputs["masks"],
        )

        return {
            "intensity_mean": outputs["intensity_mean"],  # qI.mean
            "intensity_var": outputs["intensity_var"],  # qI.variance
            "refl_ids": outputs["refl_ids"],
            "dials_I_sum_var": outputs["dials_I_sum_var"],
            "dials_I_prf_value": outputs["dials_I_prf_value"],
            "dials_I_prf_var": outputs["dials_I_prf_var"],
            "qbg": outputs["qbg"].mean,
            "qbg_scale": outputs["qbg"].scale,  # halfnormal param
            "profile_masking_mean": intensities["profile_masking_mean"],
            "profile_masking_var": intensities["profile_masking_var"],
            "kabsch_sum_mean": intensities["kabsch_sum_mean"],
            "kabsch_sum_var": intensities["kabsch_sum_var"],
            "x_c": outputs["x_c"],
            "y_c": outputs["y_c"],
            "z_c": outputs["z_c"],
        }

    # def on_before_optimizer_step(self, optimizer):
    # grad_norm_val = torch.nn.utils.clip_grad_norm_(
    # self.parameters(), max_norm=float("inf")
    # )

    # # Normalize gradients (scale all gradients to have unit norm)
    # if grad_norm_val > 0:
    # for param in self.parameters():
    # if param.grad is not None:
    # param.grad.data.mul_(1.0 / grad_norm_val)

    # # Log the original norm
    # self.log("grad_norm", grad_norm_val)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# %%
class IntegratorFourierFeatures(BaseIntegrator):
    def __init__(
        self,
        encoder,
        encoder2,
        loss,
        qbg,
        qp,
        qI,
        mc_samples=100,
        learning_rate=1e-3,
        max_iterations=4,
        profile_threshold=0.001,
        renyi_scale=0.00,
        ff_scale=1.0,
        num_fourier_features=10,
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "mlp_encoder",
                "profile_model",
                "unet",
                "signal_preprocessor",
            ]
        )
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples

        # Model components
        self.encoder = encoder
        self.qp = qp
        self.qI = qI
        self.qbg = qbg
        self.automatic_optimization = True
        self.loss_fn = loss
        self.max_iterations = max_iterations
        self.intensity_encoder = encoder2
        self.linear = Linear(64 * 2, 64)
        self.renyi_scale = renyi_scale
        B = torch.distributions.Normal(0, ff_scale).sample((num_fourier_features, 3))
        self.register_buffer("B", B, persistent=True)

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks  # [B,P]
            zbg = (
                qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            )  # [B,S,1]
            # zp = qp.rsample([self.mc_samples]).permute(1, 0, 2) #
            zp = qp.mean.unsqueeze(1)  # [B,1,P]

            vi = zbg + 1e-6

            # kabsch sum
            for i in range(4):
                num = (counts.unsqueeze(1) - zbg) * zp * masks.unsqueeze(1) / vi
                denom = zp.pow(2) / vi
                I = num.sum(-1) / denom.sum(-1)  # [batch_size, mc_samples]
                vi = (I.unsqueeze(-1) * zp) + zbg
                vi = vi.mean(-1, keepdim=True)
            kabsch_sum_mean = I.mean(-1)
            kabsch_sum_var = I.var(-1)

            # profile masking
            zp = zp * masks.unsqueeze(1)  # profiles
            thresholds = torch.quantile(
                zp, 0.99, dim=-1, keepdim=True
            )  # threshold values
            profile_mask = zp > thresholds

            masked_counts = counts.unsqueeze(1) * profile_mask

            profile_masking_I = (masked_counts - zbg * profile_mask).sum(-1)

            profile_masking_mean = profile_masking_I.mean(-1)

            profile_masking_var = profile_masking_I.var(-1)

            intensities = {
                "profile_masking_mean": profile_masking_mean,
                "profile_masking_var": profile_masking_var,
                "kabsch_sum_mean": kabsch_sum_mean,
                "kabsch_sum_var": kabsch_sum_var,
            }

            return intensities

    def forward(self, counts, shoebox, metadata, masks, reference):
        # Unpack batch
        counts_ = torch.clamp(counts[..., -1].clone(), min=0) * masks
        device = counts_.device

        proj = 2 * torch.pi * counts[..., :3] @ self.B.to(device).T
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        samples_ = torch.concat([features, counts[..., -1].unsqueeze(-1)], dim=-1)
        samples_ = samples_.view(samples_.shape[0], 3, 21, 21, 21)
        samples_ = samples_.permute(0, 4, 1, 2, 3)

        rep = self.encoder(samples_, masks)

        intensity_rep = self.intensity_encoder(samples_)

        qbg = self.qbg(intensity_rep)
        qp = self.qp(rep)
        qI = self.qI(intensity_rep)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qI.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        intensity_mean = qI.mean  # [batch_size]
        intensity_var = qI.variance  # [batch_size]

        rate = zI * zp + zbg

        return {
            "rates": rate,
            "counts": counts_,
            "masks": masks,
            "qbg": qbg,
            "qp": qp,
            "qp_mean": qp.mean,
            "qI": qI,
            "intensity_mean": intensity_mean,
            "intensity_mean": intensity_mean,
            "intensity_var": intensity_var,
            "dials_I_sum_value": reference[:, 6],
            "dials_I_sum_var": reference[:, 7],
            "dials_I_prf_value": reference[:, 8],
            "dials_I_prf_var": reference[:, 9],
            "refl_ids": reference[:, -1],
            "profile": qp.mean,
            "zp": zp,
            "x_c": reference[:, 0],
            "y_c": reference[:, 1],
            "z_c": reference[:, 2],
            "x_c_mm": reference[:, 3],
            "y_c_mm": reference[:, 4],
            "z_c_mm": reference[:, 5],
            "dials_bg_mean": reference[:, 10],
            "dials_bg_sum_value": reference[:, 11],
            "dials_bg_sum_var": reference[:, 12],
            "d": reference[:, 13],
        }

    def training_step(self, batch, batch_idx):
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        # Calculate loss
        (loss, neg_ll, kl, kl_bg, kl_I, kl_p) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Track gradient norms here
        norms = grad_norm(self, norm_type=2)
        for name, norm in norms.items():
            self.log(f"grad_norm/{name}", norm)

        renyi_loss = (
            (-torch.log(outputs["qp"].rsample([100]).permute(1, 0, 2).pow(2).sum(-1)))
            .mean(1)
            .sum()
        ) * self.renyi_scale
        self.log("renyi_loss", renyi_loss)

        # Log metrics
        self.log("Train: -ELBO", loss.mean())
        self.log("Train: NLL", neg_ll.mean())
        self.log("Train: KL", kl.mean())
        self.log("Train: KL Bg", kl_bg.mean())
        self.log("Train: KL I", kl_I.mean())
        self.log("Train: KL Prf", kl_p.mean())
        self.log("Mean(qI.mean)", outputs["qI"].mean.mean())
        self.log("Min(qI.mean)", outputs["qI"].mean.min())
        self.log("Max(qI.mean)", outputs["qI"].mean.max())
        self.log("Mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("Min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("Max(qbg.mean)", outputs["qbg"].mean.max())
        self.log("Mean(qbg.variance)", outputs["qbg"].variance.mean())

        return loss.mean() + renyi_loss.sum()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        (
            loss,
            neg_ll,
            kl,
            kl_bg,
            kl_I,
            kl_p,
        ) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Log metrics
        self.log("Val: -ELBO", loss.mean())
        self.log("Val: NLL", neg_ll.mean())
        self.log("Val: KL", kl.mean())
        self.log("Val: KL bg", kl_bg.mean())
        self.log("Val: KL I", kl_I.mean())
        self.log("Val: KL prf", kl_p.mean())
        self.log("val_loss", neg_ll.mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)
        intensities = self.calculate_intensities(
            counts=outputs["counts"],
            qbg=outputs["qbg"],
            qp=outputs["qp"],
            masks=outputs["masks"],
        )

        return {
            "intensity_mean": outputs["intensity_mean"],  # qI.mean
            "intensity_var": outputs["intensity_var"],  # qI.variance
            "refl_ids": outputs["refl_ids"],
            "dials_I_sum_var": outputs["dials_I_sum_var"],
            "dials_I_prf_value": outputs["dials_I_prf_value"],
            "dials_I_prf_var": outputs["dials_I_prf_var"],
            "qbg": outputs["qbg"].mean,
            "qbg_scale": outputs["qbg"].scale,  # halfnormal param
            "profile_masking_mean": intensities["profile_masking_mean"],
            "profile_masking_var": intensities["profile_masking_var"],
            "kabsch_sum_mean": intensities["kabsch_sum_mean"],
            "kabsch_sum_var": intensities["kabsch_sum_var"],
            "x_c": outputs["x_c"],
            "y_c": outputs["y_c"],
            "z_c": outputs["z_c"],
        }

    # def on_before_optimizer_step(self, optimizer):
    # grad_norm_val = torch.nn.utils.clip_grad_norm_(
    # self.parameters(), max_norm=float("inf")
    # )

    # # Normalize gradients (scale all gradients to have unit norm)
    # if grad_norm_val > 0:
    # for param in self.parameters():
    # if param.grad is not None:
    # param.grad.data.mul_(1.0 / grad_norm_val)

    # # Log the original norm
    # self.log("grad_norm", grad_norm_val)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# %%
class IntegratorLog1p(BaseIntegrator):
    def __init__(
        self,
        encoder,
        loss,
        qbg,
        qp,
        qI,
        mc_samples=100,
        learning_rate=1e-3,
        max_iterations=4,
        profile_threshold=0.001,
        renyi_scale=0.00,
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "mlp_encoder",
                "profile_model",
                "unet",
                "signal_preprocessor",
            ]
        )
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples

        # Model components
        self.encoder = encoder
        self.qp = qp
        self.qI = qI
        self.qbg = qbg
        self.automatic_optimization = True
        self.loss_fn = loss
        self.max_iterations = max_iterations
        self.intensity_encoder = MLPMetadataEncoder(feature_dim=10, output_dims=64)
        self.bg_encoder = MLPMetadataEncoder(feature_dim=10, output_dims=64)
        self.linear = Linear(64 * 2, 64)
        self.renyi_scale = renyi_scale

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks  # [B,P]
            zbg = (
                qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            )  # [B,S,1]
            # zp = qp.rsample([self.mc_samples]).permute(1, 0, 2) #
            zp = qp.mean.unsqueeze(1)  # [B,1,P]

            vi = zbg + 1e-6

            # kabsch sum
            for i in range(4):
                num = (counts.unsqueeze(1) - zbg) * zp * masks.unsqueeze(1) / vi
                denom = zp.pow(2) / vi
                I = num.sum(-1) / denom.sum(-1)  # [batch_size, mc_samples]
                vi = (I.unsqueeze(-1) * zp) + zbg
                vi = vi.mean(-1, keepdim=True)
            kabsch_sum_mean = I.mean(-1)
            kabsch_sum_var = I.var(-1)

            # profile masking
            zp = zp * masks.unsqueeze(1)  # profiles
            thresholds = torch.quantile(
                zp, 0.99, dim=-1, keepdim=True
            )  # threshold values
            profile_mask = zp > thresholds

            masked_counts = counts.unsqueeze(1) * profile_mask

            profile_masking_I = (masked_counts - zbg * profile_mask).sum(-1)

            profile_masking_mean = profile_masking_I.mean(-1)

            profile_masking_var = profile_masking_I.var(-1)

            intensities = {
                "profile_masking_mean": profile_masking_mean,
                "profile_masking_var": profile_masking_var,
                "kabsch_sum_mean": kabsch_sum_mean,
                "kabsch_sum_var": kabsch_sum_var,
            }

            return intensities

    def forward(self, counts, shoebox, metadata, masks, reference):
        # Unpack batch
        counts = torch.clamp(counts, min=0) * masks
        device = counts.device

        num_valid_pixels = masks.sum(1)
        total_photons = (counts).sum(1)
        mean_photons = total_photons / num_valid_pixels
        max_photons = counts.max(1)[0]
        std_photons = torch.sqrt(
            (1 / (num_valid_pixels - 1))
            * (((counts - mean_photons.unsqueeze(1)) ** 2) * masks).sum(1)
        )
        q1 = torch.quantile(counts, 0.9999, dim=1)
        q2 = torch.quantile(counts, 0.999, dim=1)
        q3 = torch.quantile(counts, 0.9, dim=1)
        q4 = torch.quantile(counts, 0.50, dim=1)
        q5 = torch.quantile(counts, 0.25, dim=1)

        intensity_encoding = torch.stack(
            [
                torch.log1p(total_photons),
                torch.log1p(mean_photons),
                torch.log1p(max_photons),
                torch.log1p(std_photons),
                torch.log1p(q1),
                torch.log1p(q2),
                torch.log1p(q3),
                torch.log1p(q4),
                torch.log1p(q5),
                std_photons / mean_photons,
            ]
        ).transpose(1, 0)

        intensity_rep = self.bg_encoder(intensity_encoding)
        rep = self.encoder(shoebox.reshape(shoebox.shape[0], 1, 3, 21, 21), masks)

        bgrep = self.bg_encoder(intensity_encoding)

        qbg = self.qbg(bgrep)
        qp = self.qp(rep)
        # qI = self.qI(intensity_rep)
        qI = self.qI(intensity_rep)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qI.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        intensity_mean = qI.mean  # [batch_size]
        intensity_var = qI.variance  # [batch_size]

        rate = zI * zp + zbg

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qbg": qbg,
            "qp": qp,
            "qp_mean": qp.mean,
            "qI": qI,
            "intensity_mean": intensity_mean,
            "intensity_mean": intensity_mean,
            "intensity_var": intensity_var,
            "dials_I_sum_value": reference[:, 6],
            "dials_I_sum_var": reference[:, 7],
            "dials_I_prf_value": reference[:, 8],
            "dials_I_prf_var": reference[:, 9],
            "refl_ids": reference[:, -1],
            "profile": qp.mean,
            "zp": zp,
            "x_c": reference[:, 0],
            "y_c": reference[:, 1],
            "z_c": reference[:, 2],
            "x_c_mm": reference[:, 3],
            "y_c_mm": reference[:, 4],
            "z_c_mm": reference[:, 5],
            "dials_bg_mean": reference[:, 10],
            "dials_bg_sum_value": reference[:, 11],
            "dials_bg_sum_var": reference[:, 12],
            "d": reference[:, 13],
        }

    def training_step(self, batch, batch_idx):
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        # Calculate loss
        (loss, neg_ll, kl, kl_bg, kl_I, kl_p) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Track gradient norms here
        norms = grad_norm(self, norm_type=2)
        for name, norm in norms.items():
            self.log(f"grad_norm/{name}", norm)

        renyi_loss = (
            (-torch.log(outputs["qp"].rsample([100]).permute(1, 0, 2).pow(2).sum(-1)))
            .mean(1)
            .sum()
        ) * self.renyi_scale
        self.log("renyi_loss", renyi_loss)

        # Log metrics
        self.log("Train: -ELBO", loss.mean())
        self.log("Train: NLL", neg_ll.mean())
        self.log("Train: KL", kl.mean())
        self.log("Train: KL Bg", kl_bg.mean())
        self.log("Train: KL I", kl_I.mean())
        self.log("Train: KL Prf", kl_p.mean())
        self.log("Mean(qI.mean)", outputs["qI"].mean.mean())
        self.log("Min(qI.mean)", outputs["qI"].mean.min())
        self.log("Max(qI.mean)", outputs["qI"].mean.max())
        self.log("Mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("Min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("Max(qbg.mean)", outputs["qbg"].mean.max())
        self.log("Mean(qbg.variance)", outputs["qbg"].variance.mean())

        return loss.mean() + renyi_loss.sum()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        (
            loss,
            neg_ll,
            kl,
            kl_bg,
            kl_I,
            kl_p,
        ) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Log metrics
        self.log("Val: -ELBO", loss.mean())
        self.log("Val: NLL", neg_ll.mean())
        self.log("Val: KL", kl.mean())
        self.log("Val: KL bg", kl_bg.mean())
        self.log("Val: KL I", kl_I.mean())
        self.log("Val: KL prf", kl_p.mean())
        self.log("val_loss", neg_ll.mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        intensities = self.calculate_intensities(
            counts=outputs["counts"],
            qbg=outputs["qbg"],
            qp=outputs["qp"],
            masks=outputs["masks"],
        )

        return {
            "intensity_mean": outputs["intensity_mean"],  # qI.mean
            "intensity_var": outputs["intensity_var"],  # qI.variance
            "refl_ids": outputs["refl_ids"],
            "dials_I_sum_var": outputs["dials_I_sum_var"],
            "dials_I_prf_value": outputs["dials_I_prf_value"],
            "dials_I_prf_var": outputs["dials_I_prf_var"],
            "qbg": outputs["qbg"].mean,
            "qbg_scale": outputs["qbg"].scale,  # halfnormal param
            "profile_masking_mean": intensities["profile_masking_mean"],
            "profile_masking_var": intensities["profile_masking_var"],
            "kabsch_sum_mean": intensities["kabsch_sum_mean"],
            "kabsch_sum_var": intensities["kabsch_sum_var"],
            "x_c": outputs["x_c"],
            "y_c": outputs["y_c"],
            "z_c": outputs["z_c"],
        }

    # def on_before_optimizer_step(self, optimizer):
    # grad_norm_val = torch.nn.utils.clip_grad_norm_(
    # self.parameters(), max_norm=float("inf")
    # )

    # # Normalize gradients (scale all gradients to have unit norm)
    # if grad_norm_val > 0:
    # for param in self.parameters():
    # if param.grad is not None:
    # param.grad.data.mul_(1.0 / grad_norm_val)

    # # Log the original norm
    # self.log("grad_norm", grad_norm_val)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# %%
class IntegratorLog1p2(BaseIntegrator):
    def __init__(
        self,
        encoder,
        loss,
        qbg,
        qp,
        qI,
        mc_samples=100,
        learning_rate=1e-3,
        max_iterations=4,
        profile_threshold=0.001,
        renyi_scale=0.00,
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "mlp_encoder",
                "profile_model",
                "unet",
                "signal_preprocessor",
            ]
        )
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples

        # Model components
        self.encoder = encoder
        self.qp = qp
        self.qI = qI
        self.qbg = qbg
        self.automatic_optimization = True
        self.loss_fn = loss
        self.max_iterations = max_iterations
        self.renyi_scale = renyi_scale

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks  # [B,P]
            zbg = (
                qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            )  # [B,S,1]
            # zp = qp.rsample([self.mc_samples]).permute(1, 0, 2) #integrator.py
            zp = qp.mean.unsqueeze(1)  # [B,1,P]

            vi = zbg + 1e-6

            # kabsch sum
            for i in range(4):
                num = (counts.unsqueeze(1) - zbg) * zp * masks.unsqueeze(1) / vi
                denom = zp.pow(2) / vi
                I = num.sum(-1) / denom.sum(-1)  # [batch_size, mc_samples]
                vi = (I.unsqueeze(-1) * zp) + zbg
                vi = vi.mean(-1, keepdim=True)
            kabsch_sum_mean = I.mean(-1)
            kabsch_sum_var = I.var(-1)

            # profile masking
            zp = zp * masks.unsqueeze(1)  # profiles
            thresholds = torch.quantile(
                zp, 0.99, dim=-1, keepdim=True
            )  # threshold values
            profile_mask = zp > thresholds

            masked_counts = counts.unsqueeze(1) * profile_mask

            profile_masking_I = (masked_counts - zbg * profile_mask).sum(-1)

            profile_masking_mean = profile_masking_I.mean(-1)

            profile_masking_var = profile_masking_I.var(-1)

            intensities = {
                "profile_masking_mean": profile_masking_mean,
                "profile_masking_var": profile_masking_var,
                "kabsch_sum_mean": kabsch_sum_mean,
                "kabsch_sum_var": kabsch_sum_var,
            }

            return intensities

    def forward(self, counts, shoebox, metadata, masks, reference):
        # Unpack batch
        counts = torch.clamp(counts, min=0) * masks
        device = counts.device

        rep = self.encoder(
            torch.log1p(shoebox.reshape(shoebox.shape[0], 1, 3, 21, 21)), masks
        )

        qbg = self.qbg(rep)
        qp = self.qp(rep)
        qI = self.qI(rep)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qI.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        intensity_mean = qI.mean  # [batch_size]
        intensity_var = qI.variance  # [batch_size]

        rate = zI * zp + zbg

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qbg": qbg,
            "qp": qp,
            "qp_mean": qp.mean,
            "qI": qI,
            "intensity_mean": intensity_mean,
            "intensity_mean": intensity_mean,
            "intensity_var": intensity_var,
            "dials_I_sum_value": reference[:, 6],
            "dials_I_sum_var": reference[:, 7],
            "dials_I_prf_value": reference[:, 8],
            "dials_I_prf_var": reference[:, 9],
            "refl_ids": reference[:, -1],
            "profile": qp.mean,
            "zp": zp,
            "x_c": reference[:, 0],
            "y_c": reference[:, 1],
            "z_c": reference[:, 2],
            "x_c_mm": reference[:, 3],
            "y_c_mm": reference[:, 4],
            "z_c_mm": reference[:, 5],
            "dials_bg_mean": reference[:, 10],
            "dials_bg_sum_value": reference[:, 11],
            "dials_bg_sum_var": reference[:, 12],
            "d": reference[:, 13],
        }

    def training_step(self, batch, batch_idx):
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        # Calculate loss
        (loss, neg_ll, kl, kl_bg, kl_I, kl_p) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Track gradient norms here
        norms = grad_norm(self, norm_type=2)
        for name, norm in norms.items():
            self.log(f"grad_norm/{name}", norm)

        renyi_loss = (
            (-torch.log(outputs["qp"].rsample([100]).permute(1, 0, 2).pow(2).sum(-1)))
            .mean(1)
            .sum()
        ) * self.renyi_scale
        self.log("renyi_loss", renyi_loss)

        # Log metrics
        self.log("Train: -ELBO", loss.mean())
        self.log("Train: NLL", neg_ll.mean())
        self.log("Train: KL", kl.mean())
        self.log("Train: KL Bg", kl_bg.mean())
        self.log("Train: KL I", kl_I.mean())
        self.log("Train: KL Prf", kl_p.mean())
        self.log("Mean(qI.mean)", outputs["qI"].mean.mean())
        self.log("Min(qI.mean)", outputs["qI"].mean.min())
        self.log("Max(qI.mean)", outputs["qI"].mean.max())
        self.log("Mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("Min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("Max(qbg.mean)", outputs["qbg"].mean.max())
        self.log("Mean(qbg.variance)", outputs["qbg"].variance.mean())

        return loss.mean() + renyi_loss.sum()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        (
            loss,
            neg_ll,
            kl,
            kl_bg,
            kl_I,
            kl_p,
        ) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Log metrics
        self.log("Val: -ELBO", loss.mean())
        self.log("Val: NLL", neg_ll.mean())
        self.log("Val: KL", kl.mean())
        self.log("Val: KL bg", kl_bg.mean())
        self.log("Val: KL I", kl_I.mean())
        self.log("Val: KL prf", kl_p.mean())
        self.log("val_loss", neg_ll.mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        intensities = self.calculate_intensities(
            counts=outputs["counts"],
            qbg=outputs["qbg"],
            qp=outputs["qp"],
            masks=outputs["masks"],
        )

        return {
            "intensity_mean": outputs["intensity_mean"],  # qI.mean
            "intensity_var": outputs["intensity_var"],  # qI.variance
            "refl_ids": outputs["refl_ids"],
            "dials_I_sum_var": outputs["dials_I_sum_var"],
            "dials_I_prf_value": outputs["dials_I_prf_value"],
            "dials_I_prf_var": outputs["dials_I_prf_var"],
            "qbg": outputs["qbg"].mean,
            "qbg_scale": outputs["qbg"].scale,  # halfnormal param
            "profile_masking_mean": intensities["profile_masking_mean"],
            "profile_masking_var": intensities["profile_masking_var"],
            "kabsch_sum_mean": intensities["kabsch_sum_mean"],
            "kabsch_sum_var": intensities["kabsch_sum_var"],
            "x_c": outputs["x_c"],
            "y_c": outputs["y_c"],
            "z_c": outputs["z_c"],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# %%
class IntegratorFFLog1p(BaseIntegrator):
    def __init__(
        self,
        encoder,
        loss,
        qbg,
        qp,
        qI,
        mc_samples=100,
        learning_rate=1e-3,
        max_iterations=4,
        profile_threshold=0.001,
        renyi_scale=0.00,
        ff_scale=1.0,
        num_fourier_features=10,
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "mlp_encoder",
                "profile_model",
                "unet",
                "signal_preprocessor",
            ]
        )
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples

        # Model components
        self.encoder = encoder
        self.qp = qp
        self.qI = qI
        self.qbg = qbg
        self.automatic_optimization = True
        self.loss_fn = loss
        self.max_iterations = max_iterations
        # self.intensity_encoder = MLPMetadataEncoder(feature_dim=10, output_dims=64)
        self.bg_encoder = MLPMetadataEncoder(feature_dim=10, output_dims=64)
        self.linear = Linear(64 * 2, 64)
        self.renyi_scale = renyi_scale
        B = torch.distributions.Normal(0, 2**4).sample((10, 3))
        self.register_buffer("B", B, persistent=True)

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks  # [B,P]
            zbg = (
                qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            )  # [B,S,1]
            # zp = qp.rsample([self.mc_samples]).permute(1, 0, 2) #
            zp = qp.mean.unsqueeze(1)  # [B,1,P]

            vi = zbg + 1e-6

            # kabsch sum
            for i in range(4):
                num = (counts.unsqueeze(1) - zbg) * zp * masks.unsqueeze(1) / vi
                denom = zp.pow(2) / vi
                I = num.sum(-1) / denom.sum(-1)  # [batch_size, mc_samples]
                vi = (I.unsqueeze(-1) * zp) + zbg
                vi = vi.mean(-1, keepdim=True)
            kabsch_sum_mean = I.mean(-1)
            kabsch_sum_var = I.var(-1)

            # profile masking
            zp = zp * masks.unsqueeze(1)  # profiles
            thresholds = torch.quantile(
                zp, 0.99, dim=-1, keepdim=True
            )  # threshold values
            profile_mask = zp > thresholds

            masked_counts = counts.unsqueeze(1) * profile_mask

            profile_masking_I = (masked_counts - zbg * profile_mask).sum(-1)

            profile_masking_mean = profile_masking_I.mean(-1)

            profile_masking_var = profile_masking_I.var(-1)

            intensities = {
                "profile_masking_mean": profile_masking_mean,
                "profile_masking_var": profile_masking_var,
                "kabsch_sum_mean": kabsch_sum_mean,
                "kabsch_sum_var": kabsch_sum_var,
            }

            return intensities

    def forward(self, counts, shoebox, metadata, masks, reference):
        # Unpack batch
        counts_ = torch.clamp(counts[..., -1].clone(), min=0) * masks
        device = counts_.device

        proj = 2 * torch.pi * counts[..., :3] @ self.B.to(device).T
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        samples_ = torch.concat([features, counts[..., -1].unsqueeze(-1)], dim=-1)
        samples_ = samples_.view(samples_.shape[0], 3, 21, 21, 21)
        samples_ = samples_.permute(0, 4, 1, 2, 3)

        num_valid_pixels = masks.sum(1)
        total_photons = (counts_).sum(1)
        mean_photons = total_photons / num_valid_pixels
        max_photons = counts_.max(1)[0]
        std_photons = torch.sqrt(
            (1 / (num_valid_pixels - 1))
            * (((counts_ - mean_photons.unsqueeze(1)) ** 2) * masks).sum(1)
        )
        q1 = torch.quantile(counts_, 0.9999, dim=1)
        q2 = torch.quantile(counts_, 0.999, dim=1)
        q3 = torch.quantile(counts_, 0.9, dim=1)
        q4 = torch.quantile(counts_, 0.50, dim=1)
        q5 = torch.quantile(counts_, 0.25, dim=1)

        intensity_encoding = torch.stack(
            [
                torch.log1p(total_photons),
                torch.log1p(mean_photons),
                torch.log1p(max_photons),
                torch.log1p(std_photons),
                torch.log1p(q1),
                torch.log1p(q2),
                torch.log1p(q3),
                torch.log1p(q4),
                torch.log1p(q5),
                std_photons / mean_photons,
            ]
        ).transpose(1, 0)

        profile_rep = self.encoder(samples_, masks)
        intensity_rep = self.bg_encoder(intensity_encoding)
        bgrep = self.bg_encoder(intensity_encoding)

        qp = self.qp(profile_rep)
        qbg = self.qbg(bgrep)
        qI = self.qI(intensity_rep)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qI.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        intensity_mean = qI.mean
        intensity_var = qI.variance

        rate = zI * zp + zbg

        return {
            "rates": rate,
            "counts": counts_,
            "masks": masks,
            "qbg": qbg,
            "qp": qp,
            "qp_mean": qp.mean,
            "qI": qI,
            "intensity_mean": intensity_mean,
            "intensity_mean": intensity_mean,
            "intensity_var": intensity_var,
            "dials_I_sum_value": reference[:, 6],
            "dials_I_sum_var": reference[:, 7],
            "dials_I_prf_value": reference[:, 8],
            "dials_I_prf_var": reference[:, 9],
            "refl_ids": reference[:, -1],
            "profile": qp.mean,
            "zp": zp,
            "x_c": reference[:, 0],
            "y_c": reference[:, 1],
            "z_c": reference[:, 2],
            "x_c_mm": reference[:, 3],
            "y_c_mm": reference[:, 4],
            "z_c_mm": reference[:, 5],
            "dials_bg_mean": reference[:, 10],
            "dials_bg_sum_value": reference[:, 11],
            "dials_bg_sum_var": reference[:, 12],
            "d": reference[:, 13],
        }

    def training_step(self, batch, batch_idx):
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        # Calculate loss
        (loss, neg_ll, kl, kl_bg, kl_I, kl_p) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Track gradient norms here
        norms = grad_norm(self, norm_type=2)
        for name, norm in norms.items():
            self.log(f"grad_norm/{name}", norm)

        renyi_loss = (
            (-torch.log(outputs["qp"].rsample([100]).permute(1, 0, 2).pow(2).sum(-1)))
            .mean(1)
            .sum()
        ) * self.renyi_scale
        self.log("renyi_loss", renyi_loss)

        # Log metrics
        self.log("Train: -ELBO", loss.mean())
        self.log("Train: NLL", neg_ll.mean())
        self.log("Train: KL", kl.mean())
        self.log("Train: KL Bg", kl_bg.mean())
        self.log("Train: KL I", kl_I.mean())
        self.log("Train: KL Prf", kl_p.mean())
        self.log("Mean(qI.mean)", outputs["qI"].mean.mean())
        self.log("Min(qI.mean)", outputs["qI"].mean.min())
        self.log("Max(qI.mean)", outputs["qI"].mean.max())
        self.log("Mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("Min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("Max(qbg.mean)", outputs["qbg"].mean.max())
        self.log("Mean(qbg.variance)", outputs["qbg"].variance.mean())

        return loss.mean() + renyi_loss.sum()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        (
            loss,
            neg_ll,
            kl,
            kl_bg,
            kl_I,
            kl_p,
        ) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Log metrics
        self.log("Val: -ELBO", loss.mean())
        self.log("Val: NLL", neg_ll.mean())
        self.log("Val: KL", kl.mean())
        self.log("Val: KL bg", kl_bg.mean())
        self.log("Val: KL I", kl_I.mean())
        self.log("Val: KL prf", kl_p.mean())
        self.log("val_loss", neg_ll.mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        intensities = self.calculate_intensities(
            counts=outputs["counts"],
            qbg=outputs["qbg"],
            qp=outputs["qp"],
            masks=outputs["masks"],
        )

        return {
            "intensity_mean": outputs["intensity_mean"],  # qI.mean
            "intensity_var": outputs["intensity_var"],  # qI.variance
            "refl_ids": outputs["refl_ids"],
            "dials_I_sum_var": outputs["dials_I_sum_var"],
            "dials_I_prf_value": outputs["dials_I_prf_value"],
            "dials_I_prf_var": outputs["dials_I_prf_var"],
            "qbg": outputs["qbg"].mean,
            "qbg_scale": outputs["qbg"].scale,  # halfnormal param
            "profile_masking_mean": intensities["profile_masking_mean"],
            "profile_masking_var": intensities["profile_masking_var"],
            "kabsch_sum_mean": intensities["kabsch_sum_mean"],
            "kabsch_sum_var": intensities["kabsch_sum_var"],
            "x_c": outputs["x_c"],
            "y_c": outputs["y_c"],
            "z_c": outputs["z_c"],
        }

    # def on_before_optimizer_step(self, optimizer):
    # grad_norm_val = torch.nn.utils.clip_grad_norm_(
    # self.parameters(), max_norm=float("inf")
    # )

    # # Normalize gradients (scale all gradients to have unit norm)
    # if grad_norm_val > 0:
    # for param in self.parameters():
    # if param.grad is not None:
    # param.grad.data.mul_(1.0 / grad_norm_val)

    # # Log the original norm
    # self.log("grad_norm", grad_norm_val)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# %%

torch.distributions.LogNormal(loc=0.0, scale=2.0).variance
