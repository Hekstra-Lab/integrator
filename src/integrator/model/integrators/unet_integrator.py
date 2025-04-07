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


class BasicResBlock3D(nn.Module):
    """
    A simple 3D residual block.
    By default, it keeps the same spatial resolution unless `stride>1`.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Convert 'stride' to a tuple if it's just an integer
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,  # "same" padding for 3D
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # If in/out channels or stride differ, we project the identity
        # to match shape for the residual addition.
        self.downsample = None
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class ResNet3D_EncoderDecoder(nn.Module):
    """
    A 3D ResNet-style encoder-decoder that:
      - Keeps the depth dimension (Z=3) unchanged (stride=(1,2,2) in encoder).
      - Uses interpolation to ensure exact output size matching.
      - Ensures final shape == input shape (except for channel count).
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
        super().__init__()
        self.base_channels = base_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # -----------------------
        # Encoder
        # -----------------------
        # Stage 1: stride=1
        self.enc1 = BasicResBlock3D(in_channels, base_channels, stride=(1, 1, 1))
        # Downsample 1 in H,W only (stride=(1,2,2)):
        self.enc2 = BasicResBlock3D(base_channels, base_channels * 2, stride=(1, 2, 2))

        # Another block at this scale (stride=1)
        self.enc3 = BasicResBlock3D(
            base_channels * 2, base_channels * 2, stride=(1, 1, 1)
        )
        # Downsample 2 in H,W only:
        self.enc4 = BasicResBlock3D(
            base_channels * 2, base_channels * 4, stride=(1, 2, 2)
        )

        # Deeper features (stride=1)
        self.enc5 = BasicResBlock3D(
            base_channels * 4, base_channels * 4, stride=(1, 1, 1)
        )

        # -----------------------
        # Decoder - Using interpolation instead of transposed convolutions
        # -----------------------

        # Upsampling block 1
        self.up_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            nn.Conv3d(
                base_channels * 4,
                base_channels * 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.dec1 = BasicResBlock3D(
            base_channels * 2, base_channels * 2, stride=(1, 1, 1)
        )

        # Upsampling block 2
        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            nn.Conv3d(
                base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.dec2 = BasicResBlock3D(base_channels, base_channels, stride=(1, 1, 1))

        # Final projection to out_channels
        self.final_conv = nn.Conv3d(
            in_channels=base_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        """
        Input: [B, in_channels, Z=3, H=21, W=21]
        Output: [B, out_channels, 3, 21, 21]
        """
        # Store original input size for final resize
        input_shape = x.shape

        # 1) Encoder
        x1 = self.enc1(x)  # [B, base_channels, 3, 21, 21]
        x2 = self.enc2(x1)  # [B, base_channels*2, 3, 10/11, 10/11]
        x3 = self.enc3(x2)  # same shape
        x4 = self.enc4(x3)  # [B, base_channels*4, 3, 5/6, 5/6]
        x5 = self.enc5(x4)  # same shape

        # 2) Decoder with interpolation-based upsampling
        # First upsampling
        x = self.up_conv1(x5)
        x = self.dec1(x)

        # Second upsampling
        x = self.up_conv2(x)
        x = self.dec2(x)

        # Final projection
        x = self.final_conv(x)

        # Final resize to guarantee exact match with input dimensions
        if x.shape[2:] != input_shape[2:]:
            x = F.interpolate(
                x, size=input_shape[2:], mode="trilinear", align_corners=False
            )

        return x.view(-1, 3 * 21 * 21)


# Test the fixed model
def test_model():
    model = ResNet3D_EncoderDecoder(in_channels=1, out_channels=1, base_channels=8)
    x = torch.randn(2, 1, 3, 21, 21)
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)  # Should be [2, 1, 3, 21, 21]
    return out.shape == x.shape


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
        # self.fc_representation = Linear(dmodel, dmodel)
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
        representation = shoebox_representation + meta_representation

        # Get distributions from confidence model
        q_bg = self.background_distribution(representation)
        qp = self.profile_model(representation)

        # Calculate rate using the decoder
        rate, intensity_mean, intensity_var = self.decoder(q_bg, qp, counts)

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
            tv_loss,
            simpson_loss,
            entropy_loss,
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
            tv_loss,
            simpson_loss,
            entropy_loss,
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
        self.log("val_tv_loss", tv_loss.mean())
        self.log("val_simpson_loss", simpson_loss.mean())
        self.log("val_entropy_loss", entropy_loss.mean())

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
