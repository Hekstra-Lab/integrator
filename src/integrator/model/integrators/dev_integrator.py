import torch
import torch.nn as nn
from integrator.model.integrators import BaseIntegrator
from integrator.model.loss import Loss
from integrator.model.decoders import Decoder
from integrator.layers import Linear


class DevIntegrator(BaseIntegrator):
    def __init__(
        self,
        cnn_encoder,
        metadata_encoder,
        q_bg,
        q_I,
        profile_model,
        dmodel,
        loss,
        mc_samples=100,
        learning_rate=1e-3,
        profile_threshold=0.001,
    ):
        super().__init__()
        # Save all constructor arguments except module instances
        self.save_hyperparameters(
            ignore=[
                "cnn_encoder",
                "mlp_encoder",
                "q_bg",
                "q_I",
                "profile_model",
                "loss",
            ]
        )
        self.learning_rate = learning_rate

        # Model components
        self.cnn_encoder = cnn_encoder
        self.metadata_encoder = metadata_encoder
        self.profile_model = profile_model

        # Additional layers
        self.fc_representation = Linear(dmodel * 2, dmodel)
        self.decoder = Decoder()

        # Loss function
        self.loss_fn = loss
        self.background_distribution = q_bg
        self.intensity_distribution = q_I
        self.norm = nn.LayerNorm(dmodel)
        self.mc_samples = mc_samples
        self.profile_threshold = profile_threshold
        self.automatic_optimization = True

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

            weighted_sum_mean = division.mean(-1)

            centered_w_ = weighted_sum_intensity_sum - weighted_sum_mean.unsqueeze(-1)

            weighted_sum_var = division.var(-1)

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
                "weighted_sum_mean": weighted_sum_mean,
                "weighted_sum_var": weighted_sum_var,
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
        meta_representation = self.metadata_encoder(metadata)

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
            "qI": qI,
            "qbg": qbg,
            "qp": qp,
            "dials_I_sum_value": dials[:, 0],
            "dials_I_sum_var": dials[:, 1],
            "dials_I_prf_value": dials[:, 2],
            "dials_I_prf_var": dials[:, 3],
            "refl_ids": dials[:, 4],
            "weighted_sum_mean": intensities["weighted_sum_mean"],
            "weighted_sum_var": intensities["weighted_sum_var"],
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
        (
            loss,
            neg_ll,
            kl,
        ) = self.loss_fn(
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
        (
            loss,
            neg_ll,
            kl,
        ) = self.loss_fn(
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
