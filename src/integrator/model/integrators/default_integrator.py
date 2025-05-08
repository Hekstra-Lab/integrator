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
        # self.loss_fn = loss
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
        self.log("Train: -ELBO", loss.mean())
        self.log("Train: NLL", neg_ll.mean())
        self.log("Train: KL", kl.mean())
        self.log("Train: KL Bg", kl_bg.mean())
        self.log("Train: KL I", kl_I.mean())
        self.log("Train: KL Prf", kl_p.mean())

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
        self.log("Val: -ELBO", loss.mean())
        self.log("Val: NLL", neg_ll.mean())
        self.log("Val: KL", kl.mean())
        self.log("Val: KL bg", kl_bg.mean())
        self.log("Val: KL I", kl_I.mean())
        self.log("Val: KL prf", kl_p.mean())

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


# %%


# %%
import math


def weight_initializer(weight):
    fan_avg = 0.5 * (weight.shape[-1] + weight.shape[-2])
    std = math.sqrt(1.0 / fan_avg / 10.0)
    a = -2.0 * std
    b = 2.0 * std
    torch.nn.init.trunc_normal_(weight, 0.0, std, a, b)
    return weight


class Linear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)  # Set bias=False

    def reset_parameters(self) -> None:
        self.weight = weight_initializer(self.weight)


class ResidualLayer(nn.Module):
    def __init__(self, width, dropout=None):
        super().__init__()

        self.fc1 = Linear(width, width)
        self.fc2 = Linear(width, width)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        residual = x
        out = x
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)

        if self.dropout:
            out = self.dropout(out)
        out = self.fc2(out)
        out += residual

        return out


class MLP(torch.nn.Module):
    def __init__(self, width, depth, dropout=None, output_dims=None):
        super().__init__()
        layers = [ResidualLayer(width, dropout=dropout) for _ in range(depth)]
        if output_dims is not None:
            layers.append(Linear(width, output_dims))
        self.main = torch.nn.Sequential(*layers)

    def forward(self, data):
        # Check if the input has 2 or 3 dimensions
        if len(data.shape) == 3:
            batch_size, num_pixels, features = data.shape
            data = data.view(
                -1, features
            )  # Flatten to [batch_size * num_pixels, features]
        elif len(data.shape) == 2:
            batch_size, features = data.shape
            num_pixels = None  # No pixels in this case

        # data = data.view(-1, features)
        out = self.main(data)

        # If there were pixels, reshape back to [batch_size, num_pixels, output_dims]
        if num_pixels is not None:
            out = out.view(batch_size, num_pixels, -1)  # Reshape back if needed

        return out


class MeanPool(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.register_buffer(
            "dim",
            torch.tensor(dim),
        )

    def forward(self, data, mask=None):
        data = data * mask
        out = torch.sum(data, dim=1, keepdim=True)
        if mask is None:
            denom = data.shape[-1]
        else:
            denom = torch.sum(mask, dim=-2, keepdim=True)
        out = out / denom

        return out.squeeze(1)


class MLPResNet(torch.nn.Module):
    def __init__(self, depth=10, dmodel=64, feature_dim=7, dropout=None):
        super().__init__()
        self.linear = Linear(feature_dim, dmodel)
        self.relu = torch.nn.ReLU()
        self.mlp_1 = MLP(dmodel, depth, dropout=dropout, output_dims=dmodel)
        self.mean_pool = MeanPool()

    def forward(self, shoebox_data, mask):
        batch_size, num_pixels, _ = shoebox_data.shape

        # Initial transformations
        out = self.linear(shoebox_data)
        out = self.relu(out)

        # Reshape for BatchNorm1d, apply it, then reshape back
        out = out.view(batch_size * num_pixels, -1)
        out = out.view(batch_size, num_pixels, -1)

        # Pass through residual blocks
        out = self.mlp_1(out)
        pooled_out = self.mean_pool(out, mask.unsqueeze(-1))

        return pooled_out


class IntegratorMLP(BaseIntegrator):
    def __init__(
        self,
        qbg,
        qp,
        qI,
        mc_samples=100,
        learning_rate=1e-3,
        max_iterations=4,
        prior_tensor=None,
        pI_params={"loc": 1.5, "scale": 0.5},
        pbg_params={"loc": 0.0, "scale": 1.0},
        pI_scale=0.0001,
        pbg_scale=0.0001,
        pp_scale=0.0001,
        eps=1e-6,
        count_stats="stats.pt",
        coord_stats="coords.pt",
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
        self.qp = qp
        self.qI = qI
        self.qbg = qbg
        self.automatic_optimization = True
        self.max_iterations = max_iterations
        self.encoder = MLPResNet(feature_dim=7, dmodel=64)
        if prior_tensor is not None:
            self.concentration = torch.load(prior_tensor, weights_only=False)
            self.concentration[self.concentration > 2] *= 40
            self.concentration /= self.concentration.sum()
        else:
            self.concentration = torch.ones(1323) * 0.0001
        self.pI_scale = pI_scale
        self.pbg_scale = pbg_scale
        self.pp_scale = pp_scale
        self.pI_params = pI_params
        self.pbg_params = pbg_params
        self.eps = eps
        self.coord_stats = torch.load(coord_stats, weights_only=False)
        self.count_stats = torch.load(count_stats, weights_only=False)
        self.coord_mean = self.coord_stats["mean_coords"]
        self.coord_std = self.coord_stats["std_coords"]
        self.count_mean = self.count_stats[0]
        self.count_std = self.count_stats[1].sqrt()

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks  # [B,P]
            zbg = (
                qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            )  # [B,S,1]
            # zp = qp.rsample([self.mc_samples]).permute(1, 0, 2) #
            zp = qp.mean.unsqueeze(1)  # [B,1,P]

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

            masked_counts = counts.unsqueeze(1) * profile_mask

            profile_masking_I = (masked_counts - zbg * profile_mask).sum(-1)

            profile_masking_mean = profile_masking_I.mean(-1)

            profile_masking_var = profile_masking_I.var(-1)

            intensities = {
                "profile_masking_mean": profile_masking_mean,
                "profile_masking_var": profile_masking_var,
                "kabsch_sum_mean": kabsch_sum_mean,
                "kabsch_sum_var": kabsch_sum_var,
            }

            return intensities

    def loss_fn(self, rate, counts, qp, qI, qbg, masks):
        device = rate.device
        batch_size = rate.shape[0]

        pp = torch.distributions.Dirichlet(self.concentration.to(device))
        pbg = torch.distributions.LogNormal(
            loc=torch.tensor(self.pbg_params["loc"], device=device),
            scale=torch.tensor(self.pbg_params["scale"], device=device),
        )
        pI = torch.distributions.LogNormal(
            loc=torch.tensor(self.pI_params["loc"], device=device),
            scale=torch.tensor(self.pI_params["scale"], device=device),
        )

        # calculate kl div
        kl_terms = torch.zeros(batch_size, device=device)

        kl_p = torch.distributions.kl.kl_divergence(qp, pp)
        # kl_terms += kl_p * self.pp_scale

        kl_bg = torch.distributions.kl.kl_divergence(qbg, pbg)
        # kl_terms += kl_bg * self.pbg_scale

        kl_I = torch.distributions.kl.kl_divergence(qI, pI)
        # kl_terms += kl_I * self.pI_scale

        kl_terms = kl_I * self.pI_scale + kl_bg * self.pbg_scale + kl_p * self.pp_scale

        # calculate expected log likelihood
        ll = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))

        ll_mean = torch.mean(ll, dim=1) * masks.squeeze(-1)

        neg_ll_batch = -ll_mean.sum(dim=1)

        batch_loss = neg_ll_batch + kl_terms

        total_loss = torch.mean(batch_loss)

        return (
            total_loss,
            neg_ll_batch.mean(),
            kl_terms.mean(),
            kl_bg * self.pbg_scale,
            kl_I * self.pI_scale,
            kl_p * self.pp_scale,
        )

    def forward(self, counts, masks, reference):
        # Unpack batch
        coords = counts[:, :, :6].clone()
        counts = torch.clamp(counts[..., -1], min=0) * masks

        device = counts.device

        standardized_counts = (counts - self.count_mean.to(device)) / self.count_std.to(
            device
        )
        standardized_coords = (coords - self.coord_mean.to(device)) / self.coord_std.to(
            device
        )

        shoebox = torch.cat(
            [
                standardized_coords,
                standardized_counts.unsqueeze(-1),
            ],
            dim=-1,
        )

        rep = self.encoder(shoebox, masks)

        qp = self.qp(rep)
        qbg = self.qbg(rep)
        qI = self.qI(rep)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qI.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        intensity_mean = qI.mean
        intensity_var = qI.variance

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
        counts, masks, reference = batch
        outputs = self(counts, masks, reference)

        # Calculate loss
        (loss, neg_ll, kl, kl_bg, kl_I, kl_p) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            qp=outputs["qp"],
            qI=outputs["qI"],
            qbg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Log metrics
        self.log("Train: -ELBO", loss.mean())
        self.log("Train: NLL", neg_ll.mean())
        self.log("Train: KL", kl.mean())
        self.log("Train: KL Bg", kl_bg.mean())
        self.log("Train: KL I", kl_I.mean())
        self.log("Train: KL Prf", kl_p.mean())
        self.log("Mean(qI.mean)", outputs["qI"].mean.mean())
        self.log("Min(qI.mean)", outputs["qI"].mean.min())
        self.log("Max(qI.mean)", outputs["qI"].mean.max())
        self.log("Mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("Min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("Max(qbg.mean)", outputs["qbg"].mean.max())
        self.log("Mean(qbg.variance)", outputs["qbg"].variance.mean())

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        counts, masks, reference = batch
        outputs = self(counts, masks, reference)

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
            qp=outputs["qp"],
            qI=outputs["qI"],
            qbg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Log metrics
        self.log("Val: -ELBO", loss.mean())
        self.log("Val: NLL", neg_ll.mean())
        self.log("Val: KL", kl.mean())
        self.log("Val: KL bg", kl_bg.mean())
        self.log("Val: KL I", kl_I.mean())
        self.log("Val: KL prf", kl_p.mean())
        self.log("val_loss", neg_ll.mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        counts, masks, reference = batch
        outputs = self(counts, masks, reference)

        intensities = self.calculate_intensities(
            counts=outputs["counts"],
            qbg=outputs["qbg"],
            qp=outputs["qp"],
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
            "profile_masking_mean": intensities["profile_masking_mean"],
            "profile_masking_var": intensities["profile_masking_var"],
            "kabsch_sum_mean": intensities["kabsch_sum_mean"],
            "kabsch_sum_var": intensities["kabsch_sum_var"],
            "x_c": outputs["x_c"],
            "y_c": outputs["y_c"],
            "z_c": outputs["z_c"],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
