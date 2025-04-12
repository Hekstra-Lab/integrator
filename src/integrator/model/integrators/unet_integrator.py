import torch
import math
import torch.nn as nn
from integrator.model.integrators import BaseIntegrator
from integrator.layers import Linear
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import torch
from integrator.model.encoders import CNNResNet2


class MLPIntegrator(BaseIntegrator):
    def __init__(
        self,
        encoder,
        loss,
        qbg,
        qp,
        decoder,
        mc_samples=100,
        learning_rate=1e-3,
        profile_threshold=0.001,
        image_encoder=None,
        max_iterations=4,
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
        self.profile_threshold = profile_threshold

        # Model components
        self.encoder = encoder
        self.qp = qp
        self.qbg = qbg
        self.decoder = decoder
        self.automatic_optimization = True
        self.loss_fn = loss
        self.max_iterations = max_iterations

    def calculate_intensities(self, counts, qbg, qp, dead_pixel_mask):
        with torch.no_grad():
            counts = counts * dead_pixel_mask
            batch_counts = counts.unsqueeze(1)  # [batch_size x 1 x pixels]
            # batch_bg_samples = (qbg.rsample([self.mc_samples]).unsqueeze(-1)).permute(
            # 1, 0, 2
            # )
            batch_bg_samples = qbg.rsample([self.mc_samples]).permute(1, 0, 2)
            batch_profile_samples = qp.rsample([self.mc_samples]).permute(
                1, 0, 2
            )  # [batch_size x mc_samples x pixels]
            batch_profile_samples = batch_profile_samples * dead_pixel_mask.unsqueeze(1)
            thresholds = torch.quantile(
                batch_profile_samples, 0.99, dim=-1, keepdim=True
            )

            weighted_sum_intensity = (
                batch_counts - batch_bg_samples
            ) * batch_profile_samples
            weighted_sum_intensity_sum = weighted_sum_intensity.sum(-1)

            summed_squared_prf = torch.norm(batch_profile_samples, p=2, dim=-1).pow(2)

            division = weighted_sum_intensity_sum / summed_squared_prf
            weighted_sum_mean = division.mean(-1)
            weighted_sum_var = division.var(-1)

            profile_masks = batch_profile_samples > thresholds
            N_used = profile_masks.sum(-1).float()  # [batch_size × mc_samples]
            masked_counts = batch_counts * profile_masks

            thresholded_intensity = (
                masked_counts - batch_bg_samples * profile_masks
            ).sum(-1)

            thresholded_mean = thresholded_intensity.mean(-1)
            centered_thresh = thresholded_intensity - thresholded_mean.unsqueeze(-1)
            thresholded_var = (centered_thresh**2).sum(-1) / (N_used.mean(-1) + 1e-6)

            intensities = {
                "thresholded_mean": thresholded_mean,
                "thresholded_var": thresholded_var,
                "weighted_sum_mean": weighted_sum_mean,
                "weighted_sum_var": weighted_sum_var,
            }
            return intensities

    def forward(self, shoebox, dials, masks, metadata, counts):
        counts = torch.clamp(counts, min=0) * masks
        shoebox = torch.cat([shoebox[:, :, -1], metadata], dim=-1)
        rep = self.encoder(shoebox, masks)
        qp = self.qp(rep)
        qbg = self.qbg(rep)

        # kabsch monte carlo

        # sample background and profiles
        zbg = qbg.sample([self.mc_samples]).permute(
            1, 0, 2
        )  # [batch_size x mc_samples x pixels]
        zp = qp.rsample([self.mc_samples]).permute(
            1, 0, 2
        )  # [batch_size x mc_samples x pixels]

        vi = zbg + 1e-6
        max_iterations = self.max_iterations

        for i in range(max_iterations):
            num = (
                torch.nn.functional.softplus(counts.unsqueeze(1) - zbg)
                * masks.unsqueeze(1)
                * zp
            ) / vi
            denom = zp.pow(2) / vi
            intensity = num.sum(-1) / denom.sum(-1)  # [batch_size, mc_samples]
            vi = (intensity.unsqueeze(-1) * zp) + zbg

        intensity_mean = intensity.mean(-1)  # [batch_size]
        intensity_var = intensity.var(-1)  # [batch_size]

        rate = intensity.unsqueeze(-1) * zp + zbg  # [batch_size, mc_samples, pixels]

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qbg": qbg,
            "qp": qp,
            "intensity_mean": intensity_mean,
            "intensity_var": intensity_var,
            "dials_I_sum_value": dials[:, 0],
            "dials_I_sum_var": dials[:, 1],
            "dials_I_prf_value": dials[:, 2],
            "dials_I_prf_var": dials[:, 3],
            "refl_ids": dials[:, 4],
        }

    def training_step(self, batch, batch_idx):
        # Unpack batch
        shoebox, dials, masks, metadata, counts = batch

        # Get model outputs
        outputs = self(shoebox, dials, masks, metadata, counts)

        # Calculate loss.
        (loss, neg_ll, kl, kl_bg, kl_p) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Clip gradients for stability
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # Log metrics
        self.log("train: loss", loss.mean())
        self.log("train: nll", neg_ll.mean())
        self.log("train: kl", kl.mean())
        self.log("train: kl_bg", kl_bg.mean())
        self.log("train: kl_p", kl_p.mean())

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
            kl_p,
        ) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Log metrics
        self.log("val: loss", loss.mean())
        self.log("val: nll", neg_ll.mean())
        self.log("val: kl", kl.mean())
        self.log("val: kl_bg", kl_bg.mean())
        self.log("val: kl_p", kl_p.mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)

        return {
            # "intensity_var": outputs["intensity_var"],
            "intensity_mean": outputs["intensity_mean"],
            "intensity_var": outputs["intensity_var"],
            "refl_ids": outputs["refl_ids"],
            "dials_I_sum_value": outputs["dials_I_sum_value"],
            "dials_I_sum_var": outputs["dials_I_sum_var"],
            "dials_I_prf_value": outputs["dials_I_prf_value"],
            "dials_I_prf_var": outputs["dials_I_prf_var"],
            "qp": outputs["qp"].mean,
            "qbg": outputs["qbg"].mean,
            "counts": outputs["counts"],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
