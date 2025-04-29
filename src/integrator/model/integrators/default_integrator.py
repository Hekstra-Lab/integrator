import torch
import torch.nn as nn
from integrator.model.integrators import BaseIntegrator
from integrator.model.loss import Loss
from integrator.model.decoders import Decoder
from integrator.layers import Linear
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
        use_metarep=True,
        use_metaonly=False,
    ):
        super().__init__()
        # Save all constructor arguments except module instances
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.use_metarep = use_metarep
        self.use_metaonly = use_metaonly

        # Handle model components based on flags
        if self.use_metaonly:
            self.metadata_encoder = metadata_encoder
            self.encoder = None
            print("Using metadata encoder only")
        else:
            self.image_encoder = image_encoder
            if self.use_metarep:
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

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks
            batch_counts = counts.unsqueeze(1)

            zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            zp = qp.rsample([self.mc_samples])
            zp = zp.transpose(0, 1)
            zp = zp * masks.unsqueeze(1)
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

            profile_masks = zp > self.profile_threshold

            N_used = profile_masks.sum(-1).float()
            masked_counts = batch_counts * profile_masks

            # %%
            profile_masking_intensity = (masked_counts - zbg * profile_masks).sum(-1)
            profile_masking_mean = profile_masking_intensity.mean(-1)
            centered_thresh = (
                profile_masking_intensity - profile_masking_mean.unsqueeze(-1)
            )
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
        counts = torch.clamp(counts, min=0)

        # Get representations and distributions

        if self.use_metaonly:
            representation = self.metadata_encoder(metadata)
        else:
            shoebox_representation = self.image_encoder(shoebox, masks)
            if self.use_metarep:
                meta_representation = self.metadata_encoder(metadata)
                representation = torch.cat(
                    [shoebox_representation, meta_representation], dim=1
                )
                representation = self.fc_representation(representation)
            else:
                representation = shoebox_representation

        # representation = self.norm(representation)

        qbg = self.background_distribution(representation)
        qI = self.intensity_distribution(representation)
        qp = self.profile_model(representation)

        rate = self.decoder(qI, qbg, qp)

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qI": qI,
            "intensity_mean": qI.mean,
            "intensity_var": qI.variance,
            "qbg": qbg,
            "qbg_mean": qbg.mean,
            "qp": qp,
            "dials_I_sum_value": reference[:, 6],
            "dials_I_sum_var": reference[:, 7],
            "dials_I_prf_value": reference[:, 8],
            "dials_I_prf_var": reference[:, 9],
            "refl_ids": reference[:, -1],
            "metadata": metadata,
            "x_c": reference[:, 0],
            "y_c": reference[:, 1],
            "z_c": reference[:, 2],
            "x_c_mm": reference[:, 3],
            "y_c_mm": reference[:, 4],
            "z_c_mm": reference[:, 5],
            "dials_bg_mean": reference[:, 10],
            "dials_bg_sum_value": reference[:, 11],
            "dials_bg_sum_var": reference[:, 12],
            "profile": qp.mean,
            "d": reference[:, 13],
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
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)
        intensities = self.calculate_intensities(
            outputs["counts"], outputs["qbg"], outputs["qp"], outputs["masks"]
        )
        return {
            "intensity_mean": outputs["qI"].mean,
            "intensity_var": outputs["qI"].variance,
            "kabsch_sum_mean": intensities["kabsch_sum_mean"],
            "kabsch_sum_var": intensities["kabsch_sum_var"],
            # "profile": outputs["qp"].mean,
            "profile_masking_mean": intensities["profile_masking_mean"],
            "profile_masking_var": intensities["profile_masking_var"],
            "refl_ids": outputs["refl_ids"],
            "dials_I_prf_value": outputs["dials_I_prf_value"],
            "dials_I_prf_var": outputs["dials_I_prf_var"],
            "qbg": outputs["qbg"].mean,
            "qbg_variance": outputs["qbg"].variance,
            "qp_variance": outputs["qp"].variance,
            # "qp_mean": outputs["qp"].mean,
            # "counts": outputs["counts"],
            # "masks": outputs["masks"],
            "x_c": outputs["x_c"],
            "y_c": outputs["y_c"],
            "z_c": outputs["z_c"],
            "d": outputs["d"],
            "dials_bg_mean": outputs["dials_bg_mean"],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
