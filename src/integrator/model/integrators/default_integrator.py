import torch
import torch.nn as nn
from integrator.model.integrators import BaseIntegrator
from integrator.model.loss import Loss
from integrator.model.decoders import Decoder, BernoulliDecoder
from integrator.layers import Linear
from integrator.model.loss import BernoulliLoss
from integrator.model.profiles import DirichletProfile
import numpy as np


def create_center_focused_dirichlet_prior(
    shape=(3, 21, 21),
    base_alpha=5.0,
    center_alpha=0.01,
    decay_factor=0.5,
    peak_percentage=0.05,
):
    """
    Create a Dirichlet prior concentration vector with lower values (higher concentration)
    near the center of the image.

    Parameters:
    -----------
    shape : tuple
        Shape of the 3D image (channels, height, width)
    base_alpha : float
        Base concentration parameter value for most elements (higher = more uniform)
    center_alpha : float
        Minimum concentration value at the center (lower = more concentrated)
    decay_factor : float
        Controls how quickly the concentration values increase with distance from center
    peak_percentage : float
        Approximate percentage of elements that should have high concentration (low alpha)

    Returns:
    --------
    alpha_vector : torch.Tensor
        Flattened concentration vector for Dirichlet prior as a PyTorch tensor
    """
    channels, height, width = shape
    total_elements = channels * height * width

    # Create a 3D array filled with the base alpha value
    alpha_3d = np.ones(shape) * base_alpha

    # Calculate center coordinates
    center_c = channels // 2
    center_h = height // 2
    center_w = width // 2

    # Calculate distance from center for each position
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                # Calculate normalized distance from center (0 to 1 scale)
                dist_c = abs(c - center_c) / (channels / 2)
                dist_h = abs(h - center_h) / (height / 2)
                dist_w = abs(w - center_w) / (width / 2)

                # Euclidean distance in normalized space
                distance = np.sqrt(dist_c**2 + dist_h**2 + dist_w**2) / np.sqrt(3)

                # Apply exponential increase based on distance
                # For elements close to center: use low alpha (high concentration)
                # For elements far from center: use high alpha (low concentration)
                if (
                    distance < peak_percentage * 5
                ):  # Adjust this multiplier to control the size of high concentration region
                    alpha_value = (
                        center_alpha
                        + (base_alpha - center_alpha)
                        * (distance / (peak_percentage * 5)) ** decay_factor
                    )
                    alpha_3d[c, h, w] = alpha_value

    # Flatten the 3D array to get the concentration vector and convert to torch tensor
    alpha_vector = torch.tensor(alpha_3d.flatten(), dtype=torch.float32)

    return alpha_vector  # %%


# NOTE: This is the stat of a mixture model
class tempBernoulliIntegrator(BaseIntegrator):
    def __init__(
        self,
        image_encoder,
        metadata_encoder,
        q_bg,
        q_I,
        profile_model,
        q_z,
        dmodel,
        decoder,
        loss,
        mc_samples=100,
        learning_rate=1e-3,
        profile_threshold=0.001,
        base_alpha=0.1,
        center_alpha=10.0,
        decay_factor=2.0,
        peak_percentage=0.01,
    ):
        """
        Integrated Integrator that works with both deterministic (MVN) and probabilistic profiles.
        """
        super().__init__()
        # Save all constructor arguments except module instances
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "metadata_encoder",
                "q_bg",
                "q_I",
                "profile_model",
                "q_z",
                "loss",
            ]
        )
        self.learning_rate = learning_rate

        # Model components
        self.image_encoder = image_encoder
        self.metadata_encoder = metadata_encoder
        self.profile_model = profile_model
        self.q_z = q_z

        # Additional layers
        self.fc_representation = Linear(dmodel * 2, dmodel)
        self.norm = nn.LayerNorm(dmodel)

        # Distributions
        self.background_distribution = q_bg
        self.intensity_distribution = q_I

        # Decoder and loss function
        self.decoder = decoder
        self.loss_fn = loss

        # Other parameters
        self.mc_samples = mc_samples
        self.profile_threshold = profile_threshold
        self.automatic_optimization = True

        # Check if profile model is deterministic (MVN) or probabilistic
        self.has_deterministic_profile = (
            self.profile_model.__class__.__name__ == "MVNProfile"
        )

    def calculate_intensities(self, counts, qbg, qp, dead_pixel_mask):
        with torch.no_grad():
            counts = counts * dead_pixel_mask
            batch_counts = counts.unsqueeze(1)  # [batch_size x 1 x pixels]

            batch_bg_samples = (qbg.rsample([self.mc_samples]).unsqueeze(-1)).permute(
                1, 0, 2
            )

            # Handle deterministic vs probabilistic profile
            if hasattr(qp, "rsample"):
                # Probabilistic profile (has distribution with rsample method)
                batch_profile_samples = qp.rsample([self.mc_samples]).permute(
                    1, 0, 2
                )  # [batch_size x mc_samples x pixels]

                batch_profile_samples = (
                    batch_profile_samples * dead_pixel_mask.unsqueeze(1)
                )

                weighted_sum_intensity = (
                    batch_counts - batch_bg_samples
                ) * batch_profile_samples

                weighted_sum_intensity_sum = weighted_sum_intensity.sum(-1)

                summed_squared_prf = torch.norm(batch_profile_samples, p=2, dim=-1).pow(
                    2
                )

                division = weighted_sum_intensity_sum / summed_squared_prf

                weighted_sum_intensity_mean = division.mean(-1)

                centered_w_ = (
                    weighted_sum_intensity_sum
                    - weighted_sum_intensity_mean.unsqueeze(-1)
                )

                weighted_sum_intensity_var = division.var(-1)

                profile_masks = batch_profile_samples > self.profile_threshold

                N_used = profile_masks.sum(-1).float()  # [batch_size × mc_samples]

                masked_counts = batch_counts * profile_masks

                thresholded_intensity = (
                    masked_counts - batch_bg_samples * profile_masks
                ).sum(-1)

                thresholded_mean = thresholded_intensity.mean(-1)

                centered_thresh = thresholded_intensity - thresholded_mean.unsqueeze(-1)

                thresholded_var = (centered_thresh**2).sum(-1) / (
                    N_used.mean(-1) + 1e-6
                )

                intensities = {
                    "thresholded_mean": thresholded_mean,
                    "thresholded_var": thresholded_var,
                    "weighted_sum_intensity_mean": weighted_sum_intensity_mean,
                    "weighted_sum_intensity_var": weighted_sum_intensity_var,
                }

                return intensities
            else:
                # Deterministic profile (direct tensor)
                # Expand profile to match MC samples dimension for background
                batch_size = qp.shape[0]
                batch_profile = qp.unsqueeze(1).expand(-1, self.mc_samples, -1)

                # Apply dead pixel mask
                batch_profile = batch_profile * dead_pixel_mask.unsqueeze(1)

                # Calculate weighted sum intensity
                weighted_sum_intensity = (
                    batch_counts - batch_bg_samples
                ) * batch_profile
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
                thresholded_var = (centered_thresh**2).sum(-1) / (
                    N_used.mean(-1) + 1e-6
                )

                intensities = {
                    "thresholded_mean": thresholded_mean,
                    "thresholded_var": thresholded_var,
                    "weighted_sum_mean": weighted_sum_intensity_mean,
                    "weighted_sum_var": weighted_sum_intensity_var,
                }

                return intensities

    def forward(self, shoebox, dials, masks, metadata, counts):
        # Original forward pass
        counts = torch.clamp(counts, min=0)
        coords = metadata[..., :3]

        batch_size, num_pixels, features = shoebox.shape

        # Get representations and distributions
        shoebox_representation = self.image_encoder(shoebox, masks)
        meta_representation = self.metadata_encoder(metadata)

        representation = torch.cat([shoebox_representation, meta_representation], dim=1)
        representation = self.fc_representation(representation)
        representation = self.norm(representation)

        qbg = self.background_distribution(representation)
        qI = self.intensity_distribution(representation)
        qz = self.q_z(representation)

        # Different handling based on profile type
        if self.profile_model.__class__.__name__ == "MVNProfile":
            # MVN Profile case (deterministic)
            # profile = self.profile_model(representation, signal_prob=qz.probs)
            profile = self.profile_model(representation)
            rate_off, rate_on, z_samples = self.decoder(qz, qI, qbg, profile)
            intensities = self.calculate_intensities(counts, qbg, profile, masks)

            return {
                "rate_on": rate_on,
                "rate_off": rate_off,
                "rates": rate_on,
                "counts": counts,
                "masks": masks,
                "z_perm": z_samples,
                "qI": qI,
                "qbg": qbg,
                "profile": profile,  # For MVNPlotter
                "qz": qz,
                "dials_I_sum_value": dials[:, 0],
                "dials_I_sum_var": dials[:, 1],
                "dials_I_prf_value": dials[:, 2],
                "dials_I_prf_var": dials[:, 3],
                "refl_ids": dials[:, 4],
                "weighted_sum_mean": intensities["weighted_sum_mean"]
                if "weighted_sum_mean" in intensities
                else intensities["weighted_sum_intensity_mean"],
                "weighted_sum_var": intensities["weighted_sum_var"]
                if "weighted_sum_var" in intensities
                else intensities["weighted_sum_intensity_var"],
                "thresholded_mean": intensities["thresholded_mean"],
                "thresholded_var": intensities["thresholded_var"],
                "intensity_mean": qI.mean,  # For UNetPlotter compatibility
                "signal_prob": torch.sigmoid(qz.loc)
                if hasattr(qz, "loc")
                else torch.sigmoid(qz.logits),
            }
        else:
            # Probabilistic profile case
            qp = self.profile_model(representation, signal_prob=qz.probs)
            rate, z_samples = self.decoder(qz, qI, qbg, qp)
            intensities = self.calculate_intensities(counts, qbg, qp, masks)

            return {
                "rate_on": rate_on,
                "rates": rate_on,
                "rate_off": rate_off,
                "z_perm": z_samples,
                "counts": counts,
                "masks": masks,
                "qI": qI,
                "qbg": qbg,
                "qp": qp,  # For UNetPlotter
                "qz": qz,
                "dials_I_sum_value": dials[:, 0],
                "dials_I_sum_var": dials[:, 1],
                "dials_I_prf_value": dials[:, 2],
                "dials_I_prf_var": dials[:, 3],
                "refl_ids": dials[:, 4],
                "weighted_sum_mean": intensities["weighted_sum_intensity_mean"]
                if "weighted_sum_intensity_mean" in intensities
                else intensities["weighted_sum_mean"],
                "weighted_sum_var": intensities["weighted_sum_intensity_var"]
                if "weighted_sum_intensity_var" in intensities
                else intensities["weighted_sum_var"],
                "thresholded_mean": intensities["thresholded_mean"],
                "thresholded_var": intensities["thresholded_var"],
                "intensity_mean": qI.mean,  # For UNetPlotter
                "signal_prob": torch.sigmoid(qz.loc)
                if hasattr(qz, "loc")
                else torch.sigmoid(qz.logits),
            }

    def training_step(self, batch, batch_idx):
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)

        # Bernoulli loss function
        (loss, neg_ll, kl, kl_z, kl_I, kl_bg, kl_p, reg_term) = self.loss_fn(
            outputs["rate_off"],
            outputs["rate_on"],
            outputs["z_perm"],
            outputs["counts"],
            outputs["qp"] if "qp" in outputs else outputs["profile"],
            outputs["qz"],
            outputs["qI"],
            outputs["qbg"],
            outputs["masks"],
        )

        # Log metrics
        self.log("train_loss", loss.mean())
        self.log("train_nll", neg_ll.mean())
        self.log("train_kl", kl.mean())
        self.log("kl_bg", kl_bg)
        self.log("kl_I", kl_I)
        self.log("kl_p", kl_p)
        self.log("kl_z", kl_z)
        self.log("reg_term", reg_term)

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)

        # Calculate validation metrics
        (loss, neg_ll, kl, kl_z, kl_I, kl_bg, kl_p, reg_term) = self.loss_fn(
            outputs["rate_off"],
            outputs["rate_on"],
            outputs["z_perm"],
            outputs["counts"],
            outputs["qp"] if "qp" in outputs else outputs["profile"],
            outputs["qz"],
            outputs["qI"],
            outputs["qbg"],
            outputs["masks"],
        )

        # Log metrics
        self.log("val_loss", loss.mean())
        self.log("val_nll", neg_ll.mean())
        self.log("val_kl", kl.mean())
        self.log("val_kl_bg", kl_bg)
        self.log("val_kl_I", kl_I)
        self.log("val_kl_p", kl_p)
        self.log("val_kl_z", kl_z)
        self.log("val_reg_term", reg_term)

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


class BernoulliIntegrator(BaseIntegrator):
    def __init__(
        self,
        image_encoder,
        metadata_encoder,
        q_bg,
        q_I,
        profile_model,
        q_z,
        dmodel,
        decoder,
        loss,
        mc_samples=100,
        learning_rate=1e-3,
        profile_threshold=0.001,
        base_alpha=0.1,
        center_alpha=10.0,
        decay_factor=2.0,
        peak_percentage=0.01,
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
        self.q_z = q_z

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

    def calculate_intensities(self, counts, qbg, qp, dead_pixel_mask):
        with torch.no_grad():
            counts = counts * dead_pixel_mask
            batch_counts = counts.unsqueeze(1)  # [batch_size x 1 x pixels]

            batch_bg_samples = (qbg.rsample([self.mc_samples]).unsqueeze(-1)).permute(
                1, 0, 2
            )
            if hasattr(qp, "rsample"):
                batch_profile_samples = qp.rsample([self.mc_samples]).permute(
                    1, 0, 2
                )  # [batch_size x mc_samples x pixels]

                batch_profile_samples = (
                    batch_profile_samples * dead_pixel_mask.unsqueeze(1)
                )

                weighted_sum_intensity = (
                    batch_counts - batch_bg_samples
                ) * batch_profile_samples

                weighted_sum_intensity_sum = weighted_sum_intensity.sum(-1)

                summed_squared_prf = torch.norm(batch_profile_samples, p=2, dim=-1).pow(
                    2
                )

                division = weighted_sum_intensity_sum / summed_squared_prf

                weighted_sum_intensity_mean = division.mean(-1)

                centered_w_ = (
                    weighted_sum_intensity_sum
                    - weighted_sum_intensity_mean.unsqueeze(-1)
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

                thresholded_var = (centered_thresh**2).sum(-1) / (
                    N_used.mean(-1) + 1e-6
                )

                intensities = {
                    "thresholded_mean": thresholded_mean,
                    "thresholded_var": thresholded_var,
                    "weighted_sum_intensity_mean": weighted_sum_intensity_mean,
                    "weighted_sum_intensity_var": weighted_sum_intensity_var,
                }

                return intensities
            else:
                # No sampling for profile - use deterministic values directly
                # Expand profile to match MC samples dimension for background
                batch_size = qp.shape[0]
                batch_profile = qp.unsqueeze(1).expand(-1, self.mc_samples, -1)

                # Apply dead pixel mask
                batch_profile = batch_profile * dead_pixel_mask.unsqueeze(1)

                # Calculate weighted sum intensity
                weighted_sum_intensity = (
                    batch_counts - batch_bg_samples
                ) * batch_profile
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
                thresholded_var = (centered_thresh**2).sum(-1) / (
                    N_used.mean(-1) + 1e-6
                )

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
        qz = self.q_z(representation)

        if self.profile_model.__class__.__name__ == "MVNProfile":
            profile = self.profile_model(representation)
            rate, z_samples = self.decoder(qz, qI, qbg, profile)
            intensities = self.calculate_intensities(counts, qbg, profile, masks)
            return {
                "rates": rate,
                "counts": counts,
                "masks": masks,
                "qI": qI,
                "qbg": qbg,
                "profile:": profile,
                "qz": qz,
                "dials_I_sum_value": dials[:, 0],
                "dials_I_sum_var": dials[:, 1],
                "dials_I_prf_value": dials[:, 2],
                "dials_I_prf_var": dials[:, 3],
                "refl_ids": dials[:, 4],
                "weighted_sum_mean": intensities["weighted_sum_mean"],
                "weighted_sum_var": intensities["weighted_sum_var"],
                "thresholded_mean": intensities["thresholded_mean"],
                "thresholded_var": intensities["thresholded_var"],
            }
        else:
            qp = self.profile_model(representation)

            # Calculate intensities
            rate, z_samples = self.decoder(qz, qI, qbg, qp)
            #        dispersion = (dxyz*qp.mean.unsqueeze(-1)).sum(1).sum(-1)
            intensities = self.calculate_intensities(counts, qbg, qp, masks)

            return {
                "rates": rate,
                "counts": counts,
                "masks": masks,
                "qI": qI,
                "qbg": qbg,
                "qp": qp,
                "qz": qz,
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
        (
            loss,
            neg_ll,
            kl,
            kl_z,
            kl_I,
            kl_bg,
            kl_p,
        ) = self.loss_fn(
            outputs["rates"],
            outputs["counts"],
            outputs["qp"],
            outputs["qz"],
            outputs["qI"],
            outputs["qbg"],
            outputs["masks"],
        )

        # Log metrics
        self.log("train_loss", loss.mean())
        self.log("train_nll", neg_ll.mean())
        self.log("train_kl", kl.mean())
        self.log("kl_bg", kl_bg)
        self.log("kl_I", kl_I)
        self.log("kl_p", kl_p)

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
            kl_z,
            kl_I,
            kl_bg,
            kl_p,
        ) = self.loss_fn(
            outputs["rates"],
            outputs["counts"],
            outputs["qp"],
            outputs["qz"],
            outputs["qI"],
            outputs["qbg"],
            outputs["masks"],
        )

        # Log metrics
        self.log("val_loss", loss.mean())
        self.log("val_nll", neg_ll.mean())
        self.log("val_kl", kl.mean())
        self.log("val_kl_bg", kl_bg)
        self.log("val_kl_I", kl_I)
        self.log("val_kl_p", kl_p)

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


class DefaultIntegrator(BaseIntegrator):
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

    # def calculate_intensities(self, counts, qbg, qp, dead_pixel_mask):
        # with torch.no_grad():
            # counts = counts * dead_pixel_mask
            # batch_counts = counts.unsqueeze(1)  # [batch_size x 1 x pixels]

            # batch_bg_samples = (qbg.rsample([self.mc_samples]).unsqueeze(-1)).permute(
                # 1, 0, 2
            # )
            # batch_profile_samples = qp.rsample([self.mc_samples]).permute(
                # 1, 0, 2
            # )  # [batch_size x mc_samples x pixels]

            # batch_profile_samples = batch_profile_samples * dead_pixel_mask.unsqueeze(1)

            # weighted_sum_intensity = (
                # batch_counts - batch_bg_samples
            # ) * batch_profile_samples

            # weighted_sum_intensity_sum = weighted_sum_intensity.sum(-1)

            # summed_squared_prf = torch.norm(batch_profile_samples, p=2, dim=-1).pow(2)

            # division = weighted_sum_intensity_sum / summed_squared_prf

            # weighted_sum_intensity_mean = division.mean(-1)

            # centered_w_ = (
                # weighted_sum_intensity_sum - weighted_sum_intensity_mean.unsqueeze(-1)
            # )

            # weighted_sum_intensity_var = division.var(-1)

            # profile_masks = batch_profile_samples > self.profile_threshold

            # N_used = profile_masks.sum(-1).float()  # [batch_size × mc_samples]

            # masked_counts = batch_counts * profile_masks

            # thresholded_intensity = (
                # masked_counts - batch_bg_samples * profile_masks
            # ).sum(-1)

            # thresholded_mean = thresholded_intensity.mean(-1)

            # # thresholded_var = thresholded_intensity.var(-1)

            # centered_thresh = thresholded_intensity - thresholded_mean.unsqueeze(-1)

            # # thresholded_var = (centered_thresh ** 2).mean(-1)

            # thresholded_var = (centered_thresh**2).sum(-1) / (N_used.mean(-1) + 1e-6)

            # intensities = {
                # "thresholded_mean": thresholded_mean,
                # "thresholded_var": thresholded_var,
                # "weighted_sum_intensity_mean": weighted_sum_intensity_mean,
                # "weighted_sum_intensity_var": weighted_sum_intensity_var,
            # }

            # return intensities

    # def forward(self, shoebox, dials, masks, metadata,counts,samples):

    def calculate_intensities(self, counts, qbg, qp, dead_pixel_mask):
        with torch.no_grad():
            # Apply mask to counts - use in-place operation
            counts = counts * dead_pixel_mask  # This creates one new tensor
            
            # Create batch_counts without an additional unsqueeze operation
            batch_size = counts.shape[0]
            batch_counts = counts.view(batch_size, 1, -1)  # Reshape instead of unsqueeze
            
            # Sample distributions once and reshape instead of permuting
            bg_samples_shape = (self.mc_samples, batch_size, 1)
            batch_bg_samples = qbg.rsample(bg_samples_shape).transpose(0, 1)
            
            profile_samples_shape = (self.mc_samples, batch_size, -1)
            batch_profile_samples = qp.rsample(profile_samples_shape).transpose(0, 1)
            
            # Apply mask in-place
            batch_profile_samples *= dead_pixel_mask.unsqueeze(1)
            
            # Calculate weighted sum intensity - try to minimize intermediate tensors
            intensity_diff = batch_counts - batch_bg_samples  # One intermediate tensor
            weighted_sum_intensity = intensity_diff * batch_profile_samples  # One more
            
            weighted_sum_intensity_sum = weighted_sum_intensity.sum(dim=-1)
            
            # Calculate norm directly without an intermediate pow operation
            summed_squared_prf = torch.sum(batch_profile_samples**2, dim=-1)  # Direct square and sum
            
            # Division operation - no change needed
            division = weighted_sum_intensity_sum / summed_squared_prf
            
            # Calculate mean
            weighted_sum_intensity_mean = division.mean(dim=-1)
            
            # Reuse tensors where possible
            centered_w_ = division - weighted_sum_intensity_mean.unsqueeze(-1)
            
            # Variance can be calculated directly from centered values
            weighted_sum_intensity_var = torch.mean(centered_w_**2, dim=-1)
            
            # Create mask directly without comparison tensor creation
            profile_masks = batch_profile_samples > self.profile_threshold
            
            # Sum directly to float without creating an additional tensor
            N_used = profile_masks.sum(dim=-1, dtype=torch.float)
            
            # Reuse profile_masks directly
            masked_counts = batch_counts * profile_masks
            masked_bg = batch_bg_samples * profile_masks
            thresholded_intensity = (masked_counts - masked_bg).sum(dim=-1)
            
            # Calculate mean directly
            thresholded_mean = thresholded_intensity.mean(dim=-1)
            
            # Reuse for variance calculation
            centered_thresh = thresholded_intensity - thresholded_mean.unsqueeze(-1)
            thresholded_var = (centered_thresh**2).sum(dim=-1) / (N_used.mean(dim=-1) + 1e-6)
            
            # Create result dictionary directly
            intensities = {
                "thresholded_mean": thresholded_mean,
                "thresholded_var": thresholded_var,
                "weighted_sum_intensity_mean": weighted_sum_intensity_mean,
                "weighted_sum_intensity_var": weighted_sum_intensity_var,
            }
            
            return intensities



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
        (
            loss,
            neg_ll,
            kl,
            kl_bg,
            kl_I,
            kl_p,
            tv_loss,
            simpson_loss,
            entropy_loss,
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
        self.log("kl_bg", kl_bg)
        self.log("kl_I", kl_I)
        self.log("kl_p", kl_p)

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
            kl_bg,
            kl_I,
            kl_p,
            tv_loss,
            profile_simpson_batch,
            entropy_loss,
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
        self.log("val_kl_bg", kl_bg)
        self.log("val_kl_I", kl_I)
        self.log("val_kl_p", kl_p)

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
