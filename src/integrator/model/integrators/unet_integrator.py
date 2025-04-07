import torch
import torch.nn as nn
from integrator.model.integrators import BaseIntegrator
from integrator.layers import Linear
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import torch


class DirichletProfile(torch.nn.Module):
    """
    Dirichlet profile model
    """

    def __init__(self, dmodel, rank=None, mc_samples=100, num_components=3 * 21 * 21):
        super().__init__()
        self.dmodel = dmodel
        self.mc_samples = mc_samples
        self.num_components = num_components
        self.alpha_layer = Linear(self.dmodel, self.num_components)
        self.rank = rank
        self.eps = 1e-6

    # def forward(self, representation):
    def forward(self, alphas):
        # alphas = self.alpha_layer(representation)
        alphas = F.softplus(alphas) + self.eps
        q_p = torch.distributions.Dirichlet(alphas)

        return q_p


# %%
class UNetIntegrator(BaseIntegrator):
    def __init__(
        self,
        image_encoder,
        metadata_encoder,
        loss,
        q_bg,
        profile_model,
        decoder,
        dmodel=64,
        output_dims=1323,
        mc_samples=100,
        learning_rate=1e-3,
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
        self.profile_threshold = profile_threshold

        # Model components
        self.image_encoder = image_encoder
        self.metadata_encoder = metadata_encoder
        self.profile_model = profile_model

        self.background_distribution = q_bg
        self.decoder = decoder

        self.loss_fn = loss  # Additional layers
        self.fc_representation = Linear(1323, dmodel)
        # self.norm = nn.LayerNorm(dmodel)

        # Enable automatic optimization
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
            weighted_sum_var = division.var(-1)
            profile_masks = batch_profile_samples > self.profile_threshold
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

        # Extract representations from the shoebox and metadata

        shoebox_representation = self.image_encoder(shoebox, masks)

        meta_representation = self.metadata_encoder(metadata)

        # Combine representations and normalize
        if shoebox_representation.shape[1] != meta_representation.shape[1]:
            representation = (
                self.fc_representation(shoebox_representation) + meta_representation
            )
            qp = self.profile_model(shoebox_representation)
        else:
            representation = shoebox_representation + meta_representation
            qp = self.profile_model(representation)

        # Get distributions from confidence model
        q_bg = self.background_distribution(representation)

        # Calculate rate using the decoder
        rate, intensity_mean, intensity_var = self.decoder(q_bg, qp, counts, masks)

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qbg": q_bg,
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
        # Updated call: note we no longer pass a separate q_I_nosignal.
        (
            loss,
            neg_ll,
            kl,
            kl_bg,
            kl_p,
        ) = self.loss_fn(
            outputs["rates"],
            outputs["counts"],
            outputs["qp"],
            outputs["qbg"],
            outputs["masks"],
        )

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # Log metrics
        self.log("train_loss", loss.mean())
        self.log("train_nll", neg_ll.mean())
        self.log("train_kl", kl.mean())
        self.log("train_kl_bg", kl_bg.mean())
        self.log("train_kl_p", kl_p.mean())

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
            outputs["rates"],
            outputs["counts"],
            outputs["qp"],
            outputs["qbg"],
            outputs["masks"],
        )

        # Log metrics
        self.log("val_loss", loss.mean())
        self.log("val_nll", neg_ll.mean())
        self.log("val_kl", kl.mean())
        self.log("val_kl_bg", kl_bg.mean())
        self.log("val_kl_p", kl_p.mean())

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


if __name__ == "__main__":
    from integrator.model.encoders import MLPMetadataEncoder
    from integrator.model.encoders import UNetDirichletConcentration, CNNResNet2
    from integrator.model.distribution import GammaDistribution
    from integrator.model.profiles import UnetDirichletProfile, DirichletProfile

    Z, H, W = 3, 21, 21
    dmodel = 64
    batch_size = 4

    # create networks
    shoebox_encoder = CNNResNet2(conv1_out_channel=dmodel)
    metadata_encoder = MLPMetadataEncoder(depth=10, dmodel=dmodel, feature_dim=7)
    background_distribution = GammaDistribution(dmodel)
    profile_model = DirichletProfile(dmodel)

    x = torch.randn(batch_size, Z * H * W, 7)
    masks = torch.ones(batch_size, Z * H * W)
    metadata = torch.randn(batch_size, 7)

    shoebox_representation = shoebox_encoder(x, masks)  # [batch_size, dmodel]
    meta_representation = metadata_encoder(metadata)  # [batch_size, dmodel]

    representation = (
        shoebox_representation + meta_representation
    )  # [batch_size, dmodel]

    qbg = background_distribution(representation)
    qp = profile_model(representation)  # [batch_size, num_components]
