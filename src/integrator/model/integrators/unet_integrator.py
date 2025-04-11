import torch
import torch.nn as nn
from integrator.model.integrators import BaseIntegrator
from integrator.layers import Linear
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import torch


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
        self.image_encoder = image_encoder
        self.qp = qp
        self.qbg = qbg
        self.decoder = decoder
        self.automatic_optimization = True
        self.loss_fn = loss  # Additional layers

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
        # Preprocess input data
        counts = torch.clamp(counts, min=0) * masks
        batch_size = shoebox.shape[0]

        if self.image_encoder is not None:
            alphas = self.image_encoder(shoebox)
            qp = self.qp(alphas)

        shoebox = torch.cat([shoebox[:, :, -1], metadata], dim=-1)

        representation = self.encoder(shoebox, masks)
        qbg = self.qbg(representation)

        if self.image_encoder is None:
            qp = self.qp(representation)

        # zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zbg = qbg.rsample([self.mc_samples]).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(
            1, 0, 2
        )  # [batch_size, mc_samples, pixels]

        vi = zbg + 1e-6

        max_iterations = 3
        for i in range(max_iterations):
            # Direct subtraction instead of softplus
            numerator = (
                torch.nn.functional.softplus(counts.unsqueeze(1) - zbg)
                * masks.unsqueeze(1)
                * zp
            ) / vi
            denominator = zp.pow(2) / vi + 1e-6
            intensity = numerator.sum(-1) / denominator.sum(-1)

            vi = (intensity.unsqueeze(-1) * zp) + zbg
            vi = vi.mean(-1, keepdim=True) + 1e-6

        intensity_mean = intensity.mean(-1)
        intensity_var = intensity.var(-1)
        rate = intensity.unsqueeze(-1) * zp + zbg
        # rate = intensity_mean.unsqueeze(1)*zp.mean(1) + zbg.mean(1) # [batch_size, pixels]

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
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

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
