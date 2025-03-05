import torch
import torch.nn as nn
from integrator.model.integrators import BaseIntegrator
from integrator.model.decoders import MVNDecoder
from integrator.layers import Linear


class SignalIndicatorNetwork(nn.Module):
    """
    Network that predicts whether a sample contains actual signal or is just noise.
    Outputs a Bernoulli distribution representing P(signal | data).
    """

    def __init__(self, dmodel):
        super().__init__()
        # A more expressive network for better feature extraction
        self.fc = nn.Sequential(
            nn.Linear(dmodel, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, representation):
        # Output logits (unbounded), which are transformed to probabilities
        # internally by the Bernoulli distribution
        logits = self.fc(representation)

        # Return a Bernoulli distribution
        # The sigmoid is applied internally when computing probabilities
        return torch.distributions.Bernoulli(logits=logits)


class MVNIntegrator(BaseIntegrator):
    def __init__(
        self,
        image_encoder,
        metadata_encoder,
        q_bg,
        q_I,
        decoder,
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
                "image_encoder",
                "mlp_encoder",
                "q_bg",
                "q_I",
                "profile_model",
                "loss",
            ]
        )
        self.learning_rate = learning_rate

        # Model components
        self.image_encoder = image_encoder
        self.metadata_encoder = metadata_encoder
        self.profile_model = profile_model

        # Additional layers
        self.fc_representation = Linear(dmodel * 2, dmodel)
        self.decoder = decoder

        # Loss function
        self.loss_fn = loss
        self.background_distribution = q_bg
        self.intensity_distribution = q_I
        self.norm = nn.LayerNorm(dmodel)
        self.mc_samples = mc_samples
        self.profile_threshold = profile_threshold
        self.automatic_optimization = True

    def calculate_intensities(self, counts, qbg, profile, dead_pixel_mask):
        with torch.no_grad():
            counts = counts * dead_pixel_mask
            batch_counts = counts.unsqueeze(1)  # [batch_size x 1 x pixels]

            # Sample background (still variational)
            batch_bg_samples = (qbg.rsample([self.mc_samples]).unsqueeze(-1)).permute(
                1, 0, 2
            )  # [batch_size x mc_samples x pixels]

            # No sampling for profile - use deterministic values directly
            # Expand profile to match MC samples dimension for background
            batch_size = profile.shape[0]
            batch_profile = profile.unsqueeze(1).expand(-1, self.mc_samples, -1)

            # Apply dead pixel mask
            batch_profile = batch_profile * dead_pixel_mask.unsqueeze(1)

            # Calculate weighted sum intensity
            weighted_sum_intensity = (batch_counts - batch_bg_samples) * batch_profile
            weighted_sum_intensity_sum = weighted_sum_intensity.sum(-1)

            # Calculate squared profile sum (for normalization)
            summed_squared_prf = torch.norm(batch_profile, p=2, dim=-1).pow(2)

            # Calculate intensity by division
            division = weighted_sum_intensity_sum / (summed_squared_prf + 1e-10)

            # Mean and variance across MC samples
            weighted_sum_intensity_mean = division.mean(-1)
            weighted_sum_intensity_var = division.var(-1)

            # Create profile masks for thresholded intensity
            profile_masks = batch_profile > self.profile_threshold

            # Count number of pixels used in thresholded calculation
            N_used = profile_masks.sum(-1).float()  # [batch_size × mc_samples]

            # Calculate masked counts
            masked_counts = batch_counts * profile_masks

            # Calculate thresholded intensity
            thresholded_intensity = (
                masked_counts - batch_bg_samples * profile_masks
            ).sum(-1)

            # Mean and variance of thresholded intensity
            thresholded_mean = thresholded_intensity.mean(-1)

            centered_thresh = thresholded_intensity - thresholded_mean.unsqueeze(-1)
            thresholded_var = (centered_thresh**2).sum(-1) / (N_used.mean(-1) + 1e-6)

            intensities = {
                "thresholded_mean": thresholded_mean,
                "thresholded_var": thresholded_var,
                "weighted_sum_mean": weighted_sum_intensity_mean,
                "weighted_sum_var": weighted_sum_intensity_var,
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
        shoebox_representation = self.image_encoder(shoebox, masks)
        meta_representation = self.metadata_encoder(metadata)

        representation = torch.cat([shoebox_representation, meta_representation], dim=1)
        representation = self.fc_representation(representation)
        representation = self.norm(representation)

        qbg = self.background_distribution(representation)
        qI = self.intensity_distribution(representation)
        profile = self.profile_model(representation)

        # Calculate intensities
        rate = self.decoder(qI, qbg, profile)
        intensities = self.calculate_intensities(counts, qbg, profile, masks)

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qI": qI,
            "qbg": qbg,
            "dials_I_sum_value": dials[:, 0],
            "dials_I_sum_var": dials[:, 1],
            "dials_I_prf_value": dials[:, 2],
            "dials_I_prf_var": dials[:, 3],
            "refl_ids": dials[:, 4],
            "profile": profile,
            "weighted_sum_mean": intensities["weighted_sum_mean"],
            "weighted_sum_var": intensities["weighted_sum_var"],
            "thresholded_mean": intensities["thresholded_mean"],
            "thresholded_var": intensities["thresholded_var"],
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
            kl_terms,
            kl_bg,
            kl_I,
            prof_reg,
        ) = self.loss_fn(
            outputs["rates"],
            outputs["counts"],
            outputs["profile"],
            outputs["qI"],
            outputs["qbg"],
            outputs["masks"],
        )

        # Log metrics
        self.log("train_loss", loss.mean())
        self.log("train_nll", neg_ll.mean())
        self.log("train_kl", kl_terms.mean())
        self.log("kl_bg", kl_bg)
        self.log("kl_I", kl_I)

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
        loss, neg_ll, kl_terms, kl_bg, kl_I, prof_reg = self.loss_fn(
            outputs["rates"],
            outputs["counts"],
            outputs["profile"],
            outputs["qI"],
            outputs["qbg"],
            outputs["masks"],
        )

        # Log metrics
        self.log("val_loss", loss.mean())
        self.log("val_nll", neg_ll.mean())
        self.log("val_kl", kl_terms.mean())
        self.log("val_kl_bg", kl_bg)
        self.log("val_kl_I", kl_I)

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
