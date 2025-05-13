import torch
import torch.nn as nn
from integrator.model.integrators import BaseIntegrator
from integrator.model.loss import Loss
from integrator.model.decoders import Decoder
from integrator.layers import Linear
from integrator.model.profiles import DirichletProfile, MVNProfile
import torch.nn.functional as F
import numpy as np
from integrator.model.encoders import ShoeboxEncoder


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
        weight_initializer(self.weight)


class ResidualLayer(torch.nn.Module):
    def __init__(self, dims, dropout=None):
        super().__init__()
        self.linear_1 = Linear(dims, 2 * dims)
        self.linear_2 = Linear(2 * dims, dims)
        self.dropout = (
            torch.nn.Dropout(dropout) if dropout is not None else torch.nn.Identity()
        )

    def activation(self, data):
        return torch.relu(data)

    def forward(self, data):
        out = self.activation(data)
        out = self.linear_1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        return out + data


class MLP(torch.nn.Module):
    """
    If d_in \neq width, you must specify it .
    """

    # If d_in \neq width, you must specify it
    def __init__(self, width, depth, dropout=None, d_in=None, output_dims=None):
        """
        Multi-layer perceptron (MLP) module
        """
        super().__init__()
        layers = []
        if d_in is not None:
            layers.append(Linear(d_in, width))
        layers.extend([ResidualLayer(width, dropout=dropout) for i in range(depth)])
        if output_dims is not None:
            layers.append(Linear(width, output_dims))
        self.main = torch.nn.Sequential(*layers)

    def forward(self, data, **kwargs):
        out = self.main(data)
        return out


class MeanPool(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.register_buffer(
            "dim",
            torch.tensor(dim),
        )

    def forward(self, data, mask=None):
        if mask is not None:
            data = data * mask.unsqueeze(-1)

        out = torch.sum(data, dim=1)

        if mask is None:
            denom = data.shape[-1]
        else:
            denom = torch.sum(mask, dim=-1, keepdim=True)
        out = out / (denom + 1e-6)

        return out.squeeze(1)


class Encoder(torch.nn.Module):
    def __init__(self, depth=10, dmodel=64, feature_dim=7, dropout=None):
        super().__init__()
        self.dropout = None
        self.dmodel = dmodel
        self.mlp_1 = MLP(
            dmodel, depth, d_in=feature_dim, dropout=self.dropout, output_dims=dmodel
        )
        self.mean_pool = MeanPool()

    def forward(self, shoebox_data, mask=None):
        batch_size = shoebox_data.shape[0]
        out = self.mlp_1(shoebox_data)
        # pooled_out = self.mean_pool(out, mask)
        # outputs = self.linear(pooled_out)
        # return pooled_out.view(batch_size,self.dmodel)
        return out.view(batch_size, self.dmodel)


class DirichletProfile(torch.nn.Module):
    """
    Dirichlet profile model
    """

    def __init__(self, dmodel=None, num_components=3 * 21 * 21):
        super().__init__()
        if dmodel is not None:
            self.alpha_layer = Linear(dmodel, num_components)
        self.dmodel = dmodel
        self.eps = 1e-6

    def forward(self, alphas):
        if self.dmodel is not None:
            alphas = self.alpha_layer(alphas)

        alphas = F.softplus(alphas) + self.eps
        alphas = torch.clamp(alphas, max=1e3)
        print("max alpha", alphas.max())
        q_p = torch.distributions.Dirichlet(alphas + 1e-4)

        return q_p


class TinyNet(nn.Module):
    """
    Minimal MLP for 1323-D inputs.
    1323 → 512 → 128 → d_out
    """

    def __init__(self, d_out: int, dropout_rate: float, d_in= 1323):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 512)  # layer 1
        self.fc2 = nn.Linear(512, 128)  # layer 2
        self.fc3 = nn.Linear(128, d_out)  # output layer
        self.dropout = nn.Dropout(dropout_rate)  # optional; set p=0.0 to disable

    def forward(self, x, mask=None):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # has no effect if p=0
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # leave last activation to the caller
        return x


WIDTH = 2048  # 1–4 × 1323 is typical

class TinyCNN(nn.Module):
    def __init__(self, d_out: int):
        super().__init__()
        # Simple CNN layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3×21×21 -> 32×21×21
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32×21×21 -> 64×21×21
        self.pool = nn.MaxPool2d(2)                             # 64×21×21 -> 64×10×10

        # Calculate flattened size
        flattened_size = 64 * 10 * 10  # = 6400

        # Dense layers
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, d_out)
        self.act = nn.GELU()  # Same as your WideTinyNet
        self.drop = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # x shape: [batch, 3, 21, 21]
        #x = self.act(self.conv1(x.view(x.shape[0],3,21,21)))
        if x.ndim != 4:
            x = x.view(x.shape[0], 3, 21, 21)
        x = self.act(self.conv2(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


class TinyConv3D(nn.Module):
    def __init__(
        self,
        d_out=64,
        in_channels=1,
        input_shape=(21, 21, 3),  # (H, W, D)
    ):
        super().__init__()
        
        # First define all activation functions
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)
        
        # Conv layers
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 2, 2), 
            stride=(1, 2, 2), 
            ceil_mode=True
        )
        
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding=(0, 0, 0),
        )
        
        # Calculate flattened size
        self.flattened_size = self._infer_flattened_size(
            input_shape=input_shape, 
            in_channels=in_channels
        )
        
        # Final linear layer
        self.fc = nn.Linear(self.flattened_size, d_out)
        
    def _infer_flattened_size(self, input_shape, in_channels):
        # input_shape: (H, W, D)
        with torch.no_grad():
            # (B, C, D, H, W)
            dummy = torch.zeros(
                1, in_channels, input_shape[2], input_shape[0], input_shape[1]
            )
            x = self.pool(self.act(self.conv1(dummy)))
            x = self.act(self.conv2(x))
            return x.numel()
        
    def forward(self, x, mask=None):
        # Make sure input is correctly shaped: [B, C, D, H, W]
        # If not already shaped correctly, reshape it
        if x.ndim != 5:
            x = x.view(x.shape[0], 1, 3, 21, 21)
            
        # Apply first conv and pooling
        x = self.act(self.conv1(x))
        x = self.pool(x)
        
        # Apply second conv
        x = self.act(self.conv2(x))
        
        # Flatten and apply fully connected layer
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        
        return x



class WideTinyNet(nn.Module):
    def __init__(self, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(1323, WIDTH)
        self.fc2 = nn.Linear(WIDTH, d_out)
        self.act = nn.GELU()  # smoother than ReLU, still norm-free
        self.drop = nn.Dropout(0.1)  # mild regularisation

    def forward(self, x, mask=None):
        x = self.act(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

def kl_beta(epoch, total_warmup_epochs=20):
    """
    Linear warm-up:
    β = 0 at epoch 0,  β = 1 after `total_warmup_epochs`.
    """
    return min(1.0, epoch / total_warmup_epochs)



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
        pp_scale=10.0,
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
        self.kl_warmup_epochs = 40

        # Model components
        self.qp = DirichletProfile(dmodel=64)
        self.qI = qI
        self.qbg = qbg
        self.automatic_optimization = True
        self.max_iterations = max_iterations
        #self.encoder = TinyNet(d_out = 64,dropout_rate=0.0)
        self.encoder2 = TinyNet(d_out = 64,dropout_rate=0.0,d_in=1323+10)
        #self.encoder = WideTinyNet(d_out=64)
        #self.encoder2 = WideTinyNet(d_out=64)
        #self.encoder = TinyCNN(d_out=64)
        #self.encoder2 = TinyCNN(d_out=64)
        self.encoder = TinyConv3D(d_out=64)
        #self.encoder2 = TinyConv3D(d_out=64)



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
        self.max = 1048576

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks  # [B,P]
            zbg = (
                qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            )  # [B,S,1]

            zp = qp.mean.unsqueeze(1)  # [B,1,P]
            # zp = qp.unsqueeze(1)

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

    def loss_fn(self, rate, counts, qp, qI, qbg, masks,beta):
        device = rate.device
        batch_size = rate.shape[0]

        pp = torch.distributions.Dirichlet(self.concentration.to(device) + 1e-4)
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

        kl_bg = torch.distributions.kl.kl_divergence(qbg, pbg)

        kl_I = torch.distributions.kl.kl_divergence(qI, pI)

        kl_terms = beta* (kl_I * self.pI_scale + kl_bg * self.pbg_scale + kl_p * self.pp_scale)
        # kl_terms = kl_I * self.pI_scale + kl_bg * self.pbg_scale

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
            # torch.tensor([0.0])
        )

    def forward(self, counts, masks, reference):
        # Unpack batch
        # coords = counts[:, :, :6].clone()
        counts = torch.clamp(counts, min=0) * masks
        #print("counts max", counts.sum(-1).max())

        device = counts.device

        num_valid_pixels = masks.sum(1)
        total_photons = (counts).sum(1)
        mean_photons = total_photons / num_valid_pixels
        max_photons = counts.max(1)[0]
        std_photons = torch.sqrt(
            (1 / (num_valid_pixels - 1))
            * (((counts - mean_photons.unsqueeze(1)) ** 2) * masks).sum(1)
        )
        q1 = torch.quantile(counts, 0.9999, dim=1)
        q2 = torch.quantile(counts, 0.999, dim=1)
        q3 = torch.quantile(counts, 0.9, dim=1)
        q4 = torch.quantile(counts, 0.50, dim=1)
        q5 = torch.quantile(counts, 0.25, dim=1)

        vals = torch.stack(
            [
                torch.log1p(total_photons),
                torch.log1p(mean_photons),
                torch.log1p(max_photons),
                torch.log1p(std_photons),
                torch.log1p(q1),
                torch.log1p(q2),
                torch.log1p(q3),
                torch.log1p(q4),
                torch.log1p(q5),
                std_photons / mean_photons,
            ]
        ).transpose(1, 0)
        



        #standardized_counts = (counts - self.count_mean.to(device)) / self.count_std.to(
        #   device
        #)
        # standardized_coords = (coords - self.coord_mean.to(device)) / self.coord_std.to(
        #    device
        # )
        #normed_counts = (counts/counts.max(-1)[0].unsqueeze(-1)).unsqueeze(-1)

        # print('count mean',standardized_counts.mean())
        # print('count var',standardized_counts.var())

        # shoebox = torch.cat(
        #    [
        #        standardized_coords,
        #        standardized_counts.unsqueeze(-1),
        #    ],
        #    dim=-1,
        # )

        logged_counts = torch.log1p(counts.float())

        samples = torch.concat([logged_counts,vals],dim=-1)


        #logged_counts = counts/self.max
        #logged_counts = 2*torch.sqrt(counts.float() + (3/8))

        rep = self.encoder(logged_counts, masks)
        #rep = self.encoder(standardized_counts.reshape(standardized_counts.shape[0], 1, 3, 21, 21), masks)
        #rep = self.encoder(logged_counts.reshape(logged_counts.shape[0], 1, 3, 21, 21), masks)
        # rep2 = self.encoder2(shoebox,masks)
        rep2 = self.encoder2(samples, masks)
        #rep2 = self.encoder2(standardized_counts, masks)

        qp = self.qp(rep)
        qbg = self.qbg(rep2)
        qI = self.qI(rep2)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qI.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        intensity_mean = qI.mean
        intensity_var = qI.variance

        rate = zI * zp + zbg
        # rate = zI * qp.unsqueeze(1) + zbg

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qbg": qbg,
            "qp": qp,
            "qp_mean": qp.mean,
            # "qp_mean": qp,
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
            # "profile":qp,
            "zp": qp,
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

        #beta = kl_beta(self.current_epoch,self.kl_warmup_epochs)
        beta = 1.0

        counts, masks, reference = batch
        outputs = self(counts, masks, reference)

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # Calculate loss
        (loss, neg_ll, kl, kl_bg, kl_I, kl_p) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            qp=outputs["qp"],
            qI=outputs["qI"],
            qbg=outputs["qbg"],
            masks=outputs["masks"],
            beta=beta,
        )

        self.log("Train/β", beta)

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
        self.log("Max(qI.var)", outputs["qI"].variance.max())
        self.log("Min(qI.var)", outputs["qI"].variance.min())
        self.log("Mean(qI.var)", outputs["qI"].variance.mean())
        self.log("Mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("Min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("Max(qbg.mean)", outputs["qbg"].mean.max())
        self.log("Mean(qbg.variance)", outputs["qbg"].variance.mean())

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        beta = 1.0

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
            beta=beta
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
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate,weight_decay=1e-6)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
