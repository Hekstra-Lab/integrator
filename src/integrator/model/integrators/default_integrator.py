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
        self.save_hyperparameters()
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

    def calculate_intensities(self, counts, qbg, qp, dead_pixel_mask):
        with torch.no_grad():
            counts = counts * dead_pixel_mask
            batch_counts = counts.unsqueeze(1)

            batch_bg_samples = qbg.rsample([self.mc_samples]).unsqueeze(-1)
            batch_bg_samples = batch_bg_samples.transpose(0, 1)

            batch_profile_samples = qp.rsample([self.mc_samples])
            batch_profile_samples = batch_profile_samples.transpose(0, 1)
            batch_profile_samples = batch_profile_samples * dead_pixel_mask.unsqueeze(1)

            weighted_sum_intensity = (
                batch_counts - batch_bg_samples
            ) * batch_profile_samples
            weighted_sum_intensity_sum = weighted_sum_intensity.sum(-1)

            summed_squared_prf = torch.sum(batch_profile_samples**2, dim=-1)

            division = weighted_sum_intensity_sum / summed_squared_prf
            weighted_sum_mean = division.mean(-1)
            # Variance calculation
            weighted_sum_var = division.var(-1)
            profile_masks = batch_profile_samples > self.profile_threshold
            N_used = profile_masks.sum(-1).float()
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
        counts = torch.clamp(counts, min=0)

        # Get representations and distributions
        shoebox_representation = self.image_encoder(shoebox, masks)
        meta_representation = self.metadata_encoder(metadata)

        representation = torch.cat([shoebox_representation, meta_representation], dim=1)
        representation = self.fc_representation(representation)
        representation = self.norm(representation)

        qbg = self.background_distribution(representation)
        qI = self.intensity_distribution(representation)
        qp = self.profile_model(representation)

        rate = self.decoder(qI, qbg, qp)

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
        }

    def training_step(self, batch, batch_idx):
        # shoebox, dials, masks, metadata,counts,samples = batch
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)

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

    def validation_step(self, batch, batch_idx):
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)

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

        return outputs

    def predict_step(self, batch, batch_idx):
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)
        intensities = self.calculate_intensities(
            outputs["counts"], outputs["qbg"], outputs["qp"], outputs["masks"]
        )
        return {
            "qI_mean": outputs["qI"].mean,
            "qI_variance": outputs["qI"].variance,
            "weighted_sum_mean": intensities["weighted_sum_mean"],
            "weighted_sum_var": intensities["weighted_sum_var"],
            "thresholded_mean": intensities["thresholded_mean"],
            "thresholded_var": intensities["thresholded_var"],
            "refl_ids": outputs["refl_ids"],
            "qp_mean": outputs["qp"].mean,
            "qp_variance": outputs["qp"].variance,
            "dials_I_sum_value": outputs["dials_I_sum_value"],
            "dials_I_sum_var": outputs["dials_I_sum_var"],
            "dials_I_prf_value": outputs["dials_I_prf_value"],
            "dials_I_prf_var": outputs["dials_I_prf_var"],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
