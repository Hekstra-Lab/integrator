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
from integrator.layers import Linear, Constraint
from torch.distributions import Dirichlet, Gamma, LogNormal
from integrator.model.encoders import MLPImageEncoder, MLPMetadataEncoder
from integrator.layers import MLP
from lightning.pytorch.utilities import grad_norm


# %%
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
        self.intensity_encoder = MLP(input_dim=60,output_dim=64)
        self.bg_encoder = MLP(input_dim=60,output_dim=64)

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks
            # zbg = qbg.rsample([self.mc_samples, 1323]).squeeze(-1).permute(2, 0, 1)
            zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            zp = qp
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
            N_used = profile_mask.sum(-1).float()  # number of pixels per mask
            masked_counts = counts.unsqueeze(1) * profile_mask
            profile_masking_I = (masked_counts - zbg * profile_mask).sum(-1)
            profile_masking_mean = profile_masking_I.mean(-1)
            centered_thresh = profile_masking_I - profile_masking_mean.unsqueeze(-1)
            profile_masking_var = (centered_thresh**2).sum(-1) / (
                N_used.mean(-1) + 1e-6
            )

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

        vals = torch.stack(
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

        encoding_dim = 64
        freqs = 2.0 ** torch.arange(
            0, encoding_dim // (2 * vals.shape[-1]), dtype=torch.float32
        )

        sin_encoding = torch.sin(vals.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0))
        cos_encoding = torch.cos(vals.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0))
        sin_encoding = sin_encoding.reshape(sin_encoding.shape[0], -1)
        cos_encoding = cos_encoding.reshape(cos_encoding.shape[0], -1)
        intensity_encoding = torch.concat((sin_encoding, cos_encoding), dim=1)

        rep = self.encoder(shoebox, masks)
        # check for nans
        if torch.isnan(rep).any():
            print("NaN detected in rep")
            raise ValueError("NaN detected in rep")
        intensity_rep = self.bg_encoder(intensity_encoding)
        bgrep = self.bg_encoder(intensity_encoding)
        # check for nans
        if torch.isnan(bgrep).any():
            print("NaN detected in bgrep")
            raise ValueError("NaN detected in bgrep")

        qbg = self.qbg(bgrep)
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
            "qp_factor": torch.tensor(0.0),
            "qp_diag": torch.tensor(0.0),
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
        # Unpack batch
        shoebox, dials, masks, metadata, counts = batch

        # Get model outputs
        outputs = self(shoebox, dials, masks, metadata, counts)

        # Calculate loss
        (loss, neg_ll, kl, kl_bg, kl_I, kl_p) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # torch.nn.utils.clip_grad_norm_(self.parameters(), norm_type=2.0, max_norm=150.0)

        # Track gradient norms here
        norms = grad_norm(self, norm_type=2)
        for name, norm in norms.items():
            self.log(f"grad_norm/{name}", norm)

        # Log metrics
        self.log("train: loss", loss.mean())
        self.log("train: nll", neg_ll.mean())
        self.log("train: kl", kl.mean())
        self.log("train: kl_bg", kl_bg.mean())
        self.log("train: kl_I", kl_I.mean())
        self.log("train: kl_p", kl_p.mean())
        self.log("qI mean mean", outputs["qI"].mean.mean())
        self.log("qI mean min", outputs["qI"].mean.min())
        self.log("qI mean max", outputs["qI"].mean.max())
        self.log("qbg mean mean", outputs["qbg"].mean.mean())
        self.log("qbg mean min", outputs["qbg"].mean.min())
        self.log("qbg mean max", outputs["qbg"].mean.max())
        self.log("qbg variance mean", outputs["qbg"].variance.mean())

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        shoebox, dials, masks, metadata, counts = batch
        # Get model outputs
        outputs = self(shoebox, dials, masks, metadata, counts)

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
        self.log("val: loss", loss.mean())
        self.log("val: nll", neg_ll.mean())
        self.log("val: kl", kl.mean())
        self.log("val: kl_bg", kl_bg.mean())
        self.log("val: kl_I", kl_I.mean())
        self.log("val: kl_p", kl_p.mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)
        intensities = self.calculate_intensities(
            counts=outputs["counts"],
            qbg=outputs["qbg"],
            qp=outputs["profile"],
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
            # "counts": outputs["counts"],
            # "profile": outputs["profile"],
            "profile_masking_mean": intensities["profile_masking_mean"],
            "profile_masking_var": intensities["profile_masking_var"],
            "kabsch_sum_mean": intensities["kabsch_sum_mean"],
            "kabsch_sum_var": intensities["kabsch_sum_var"],
            "x_c": outputs["x_c"],
            "y_c": outputs["y_c"],
        }

    def on_before_optimizer_step(self, optimizer):
        # Get current gradient norm
        grad_norm_val = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )

        # Normalize gradients (scale all gradients to have unit norm)
        if grad_norm_val > 0:
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(1.0 / grad_norm_val)

        # Log the original norm
        self.log("grad_norm", grad_norm_val)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
