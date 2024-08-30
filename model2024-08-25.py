import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning
from integrator.layers import Linear, Constraint
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from integrator.layers import Linear
from rs_distributions.transforms import FillScaleTriL
from torch.distributions.transforms import SoftplusTransform
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import rs_distributions.distributions as rsd
from integrator.io import ShoeboxDataModule


class Residual(nn.Module):
    """The Residual block of ResNet models."""

    def __init__(self, in_channels, out_channels, strides=1, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels or strides != 1:
            self.conv3 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=strides
            )
        else:
            self.conv3 = None

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    #            if self.conv3:
    #                self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = self.conv1(X)
        if self.use_bn:
            Y = self.bn1(Y)
        Y = F.relu(Y)
        Y = self.conv2(Y)
        if self.use_bn:
            Y = self.bn2(Y)

        if self.conv3:
            X = self.conv3(X)
            # if self.use_bn:
            #    X = self.bn3(X)
        Y += X
        return F.relu(Y)


class Encoder(nn.Module):
    def __init__(
        self,
        use_bn=True,
        conv1_in_channel=3,
        conv1_out_channel=64,
        conv1_kernel_size=7,
        conv1_stride=2,
        conv1_padding=3,
        layer1_num_blocks=3,
        conv2_out_channel=128,
        layer2_stride=1,
        layer2_num_blocks=3,
        maxpool_in_channel=64,
        maxpool_out_channel=64,
        maxpool_kernel_size=3,
        maxpool_stride=2,
        maxpool_padding=1,
    ):
        super(Encoder, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(
            conv1_in_channel,
            conv1_out_channel,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=conv1_padding,
            bias=False,
        )
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(conv1_out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        #        self.maxpool = nn.Conv2d(
        #            in_channels=maxpool_in_channel,
        #            out_channels=maxpool_out_channel,
        #            kernel_size=maxpool_kernel_size,
        #            stride=maxpool_stride,
        #            padding=maxpool_padding,
        #            bias=False,
        #        )

        # Define the layers in the sequence they are applied
        self.layer1 = self._make_layer(
            conv1_out_channel, conv1_out_channel, layer1_num_blocks, use_bn=self.use_bn
        )
        # self.layer2 = self._make_layer(
        #    conv1_out_channel,
        #    conv2_out_channel,
        #    layer2_num_blocks,
        #    stride=layer2_stride,
        #    use_bn=self.use_bn,
        # )
        # self.layer3 = self._make_layer(128, 256, 3, stride=2, use_bn=self.use_bn)
        # self.layer4 = self._make_layer(256, 512, 2, stride=2, use_bn=self.use_bn)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1, use_bn=True):
        layers = []
        layers.append(
            Residual(in_channels, out_channels, strides=stride, use_bn=use_bn)
        )
        for _ in range(1, num_blocks):
            layers.append(Residual(out_channels, out_channels, use_bn=use_bn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x


class Standardize(torch.nn.Module):
    def __init__(
        self, center=True, feature_dim=7, max_counts=float("inf"), epsilon=1e-6
    ):
        super().__init__()
        self.epsilon = epsilon
        self.center = center
        self.max_counts = max_counts
        self.register_buffer("mean", torch.zeros((1, 1, feature_dim)))
        self.register_buffer("m2", torch.zeros((1, 1, feature_dim)))
        self.register_buffer("pixel_count", torch.tensor(0.0))  # Counter for pixels
        self.register_buffer("image_count", torch.tensor(0.0))  # Counter for images

        # Mask to exclude certain features from mean subtraction (0 for exclusion, 1 for inclusion)
        # self.mean_mask = torch.ones((1, 1, feature_dim)).to(device)
        # self.mean_mask[
        #    ..., 3:6
        # ] = 0  # Exclude 4th, 5th, and 6th features (0-based index)

    @property
    def var(self):
        m2 = torch.clamp(self.m2, min=self.epsilon)
        return m2 / self.pixel_count.clamp(min=1)

    @property
    def std(self):
        return torch.sqrt(self.var)

    def update(self, im, mask=None):
        if mask is None:
            k = len(
                im
            )  # Assuming 'k' should be number of elements in 'im' if no mask is provided
        else:
            k = mask.sum()  # count num of pixels in batch
        self.pixel_count += k
        self.image_count += len(im)

        if mask is None:
            diff = im - self.mean
        else:
            diff = (im - self.mean) * mask.unsqueeze(-1)

        new_mean = self.mean + torch.sum(diff, dim=(0, 1)) / self.pixel_count
        if mask is None:
            self.m2 += torch.sum((im - new_mean) * diff, dim=(0, 1))
        else:
            self.m2 += torch.sum(
                (im - new_mean) * mask.unsqueeze(-1) * diff, dim=(0, 1)
            )
        self.mean = new_mean

    def standardize(self, im, mask=None):
        if self.center:
            if mask is None:
                return (im - self.mean) / self.std
            else:
                return ((im - self.mean) * mask.unsqueeze(-1)) / self.std
        return im / self.std

    def forward(self, im, mask=None, training=True):
        if self.image_count >= self.max_counts:
            training = False

        if training:
            self.update(im, mask)

        return self.standardize(im, mask)


class Profile(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        num_components=5,  # number of components in the mixture model
    ):
        super().__init__()
        self.dmodel = dmodel
        self.L_transform = FillScaleTriL(diag_transform=SoftplusTransform())
        self.num_components = num_components

        if self.num_components == 1:
            self.scale_layer = Linear(self.dmodel, 6)
        else:
            self.mixture_weight_layer = Linear(dmodel, num_components)
            self.mean_layer = Linear(dmodel, (num_components - 1) * 3)
            self.scale_layer = Linear(self.dmodel, num_components * 6)

    def forward(self, representation, dxyz, num_planes=3):
        num_components = self.num_components

        batch_size = representation.size(0)

        if self.num_components == 1:
            means = torch.zeros(
                (batch_size, 1, 3), device=representation.device, requires_grad=False
            ).to(torch.float32)

            scales = self.scale_layer(representation).view(batch_size, 1, 6)

            L = self.L_transform(scales).to(torch.float32)

            mvn = MultivariateNormal(means, scale_tril=L)

            log_probs = mvn.log_prob(dxyz)

            profile = torch.exp(log_probs)

            return profile, L

        else:
            mixture_weights = self.mixture_weight_layer(representation).view(
                batch_size, num_components
            )

            mixture_weights = F.softmax(mixture_weights, dim=-1)

            means = self.mean_layer(representation).view(
                batch_size, num_components - 1, 3
            )

            zero_means = torch.zeros((batch_size, 1, 3), device=representation.device)

            means = torch.cat([zero_means, means], dim=1).to(torch.float32)

            scales = self.scale_layer(representation).view(
                batch_size, num_components, 6
            )

            L = self.L_tranform(scales).to(torch.float32)

            mvn = MultivariateNormal(means, scale_tril=L)

            mix = Categorical(mixture_weights)

            gmm = MixtureSameFamily(
                mixture_distribution=mix, component_distribution=mvn
            )

            log_probs = gmm.log_prob(dxyz.view(441 * 3, batch_size, 3))

            profile = torch.exp(log_probs).view(batch_size, num_planes * 441)

            return profile, L


class IntensityDistribution(torch.nn.Module):
    def __init__(self, dmodel, intensity_dist):
        super().__init__()

        self.linear_intensity_params = Linear(dmodel, 2)

        self.intensity_dist = intensity_dist

        self.constraint = Constraint()

    def intensity_distribution(self, intensity_params):
        loc = self.constraint(intensity_params[..., 0])

        # loc = intensity_params[..., 0]

        scale = self.constraint(intensity_params[..., 1])

        return self.intensity_dist(loc, scale)

    def forward(self, representation):
        intensity_params = self.linear_intensity_params(representation)

        q_I = self.intensity_distribution(intensity_params)

        return q_I


class DistributionBuilder(torch.nn.Module):
    def __init__(
        self,
        intensity_distribution,
        spot_profile_model,
        bg_indicator=None,
        eps=1e-12,
        beta=1.0,
    ):
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        self.intensity_distribution = intensity_distribution
        self.spot_profile_model = spot_profile_model
        self.bg_indicator = bg_indicator if bg_indicator is not None else None

    def forward(
        self,
        representation,
        dxyz,
    ):
        q_I = self.intensity_distribution(representation)

        return q_I


class Decoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        q_I,
        q_bg,
        profile,
        bg,
        mc_samples=100,
    ):
        # Sample from variational distributions
        z = q_I.rsample([mc_samples]).unsqueeze(-1)
        bg = q_bg.rsample([mc_samples]).unsqueeze(-1)

        # rate = z.permute(1, 0, 2) * profile.unsqueeze(1)
        # rate = z.permute(1, 0, 2) * profile.unsqueeze(1)

        # rate = z.permute(1, 0, 2) * profile.unsqueeze(1) + bg.unsqueeze(1)
        rate = z.permute(1, 0, 2) * profile.unsqueeze(1) + bg.permute(1, 0, 2)

        return rate, z


# %%
# %%
class Loss(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        prior_I=torch.distributions.exponential.Exponential(1.0),
        prior_bg=torch.distributions.exponential.Exponential(1.0),
        # prior_I=torch.distributions.log_normal.LogNormal(0.0, 1.0),
        p_I_scale=0.0001,
        p_bg_scale=0.0001,
        mc_samples=100,
    ):
        super().__init__()
        self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
        self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)
        self.prior_I = prior_I
        self.prior_bg = prior_bg
        self.p_I_scale = torch.nn.Parameter(
            data=torch.tensor(p_I_scale), requires_grad=False
        )
        self.p_bg_scale = torch.nn.Parameter(
            data=torch.tensor(p_bg_scale), requires_grad=False
        )
        self.mc_samples = mc_samples

    def forward(
        self,
        rate,
        z,
        counts,
        q_I,
        q_bg,
        dead_pixel_mask,
        eps=1e-5,
    ):
        ll = torch.distributions.Poisson(rate + eps).log_prob(counts.unsqueeze(1))

        kl_term = 0

        # Calculate KL-divergence only if the corresponding priors and distributions are available

        if q_I is not None and self.prior_I is not None:
            # kl_I = q_I.log_prob(z) - self.prior_I.log_prob(z)

            kl_I = torch.distributions.kl.kl_divergence(q_I, self.prior_I)
            kl_term += kl_I.mean() * self.p_I_scale

        if q_bg is not None and self.prior_bg is not None:
            # kl_bg = q_bg.log_prob(z) - self.prior_bg.log_prob(z)

            kl_bg = torch.distributions.kl.kl_divergence(q_bg, self.prior_bg)
            kl_term += kl_bg.mean() * self.p_bg_scale

        # if q_I is not None and self.prior_I is not None:
        # # Monte Carlo sampling from the variational distribution
        # z_samples = q_I.rsample([self.mc_samples])

        # # Compute log probabilities for the samples
        # log_qz = q_I.log_prob(z_samples)
        # log_pz = self.prior_I.log_prob(z_samples)

        # # Monte Carlo estimate of KL divergence
        # kl_I = torch.mean(log_qz - log_pz, dim=0)  # Averaging over samples
        # kl_term += kl_I.mean() * self.p_I_scale

        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)

        nll = -(torch.sum(ll_mean) / torch.sum(dead_pixel_mask))

        return nll, kl_term


# %%
class ProfileDecomposition(nn.Module):
    def __init__(self, input_dim, rank, channels=3, height=21, width=21):
        super(ProfileDecomposition, self).__init__()
        self.rank = rank
        self.channels = channels
        self.height = height
        self.width = width

        # Linear layer to produce the decomposition parameters
        self.fc = nn.Linear(input_dim, rank * (channels + height + width))

    def forward(self, encoding):
        # Input encoding shape: (batch_size, 1, input_dim)
        batch_size = encoding.size(0)
        encoding = encoding.view(batch_size, -1)  # Flatten to (batch_size, input_dim)

        # Linear transformation to produce decomposition parameters
        decomposition_params = self.fc(encoding)  # (batch_size, rank * (3 + 21 + 21))

        # Split the parameters into separate factors
        A = decomposition_params[:, : self.rank * self.channels].view(
            batch_size, self.rank, self.channels
        )  # (batch_size, rank, 3)
        B = decomposition_params[
            :, self.rank * self.channels : self.rank * (self.channels + self.height)
        ].view(
            batch_size, self.rank, self.height
        )  # (batch_size, rank, 21)
        C = decomposition_params[:, self.rank * (self.channels + self.height) :].view(
            batch_size, self.rank, self.width
        )  # (batch_size, rank, 21)
        # A = torch.nn.functional.softmax(A)
        # B = torch.nn.functional.softmax(B)
        # C = torch.nn.functional.softmax(C)
        # A = torch.nn.functional.softplus(A)
        # B = torch.nn.functional.softplus(B)
        # C = torch.nn.functional.softplus(C)

        # Reconstruct the background tensor using CP decomposition
        background = torch.einsum(
            "brc,brh,brw->bchw", A, B, C
        )  # (batch_size, 3, 21, 21)

        background = torch.softmax(
            background.view(batch_size, self.channels * self.height * self.width),
            dim=-1,
        )

        return background


# %%
class BackgroundDecomposition(torch.nn.Module):
    def __init__(self, dmodel, background_distribution, constraint=Constraint()):
        super().__init__()
        self.linear_bg_params = Linear(dmodel, 2)
        self.background_distribution = background_distribution
        self.constraint = constraint

    def background(self, bgparams):
        mu = self.constraint(bgparams[..., 0])
        sigma = self.constraint(bgparams[..., 1])
        return self.background_distribution(mu, sigma)

    def forward(self, representation):
        bgparams = self.linear_bg_params(representation)
        q_bg = self.background(bgparams)
        return q_bg


# %%
class Integrator(pytorch_lightning.LightningModule):
    def __init__(
        self,
        encoder,
        profile_decomposition,
        builder,
        decoder,
        loss_model,
        standardize,
        dmodel,
        rank,
        C,
        Z,
        H,
        W,
        batch_size,
        intensity_distribution,
        bg_distribution,
        spot_profile,
        total_steps,
        bg_decomp,
        lr=0.001,
        lambda_l2=0.001,
        lambda_tv=0.001,
        lambda_l1=0.001,
    ):
        super().__init__()

        self.encoder = encoder
        self.lambda_tv = lambda_tv
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.profile_decomposition = profile_decomposition

        self.builder = builder
        self.decoder = decoder
        self.loss_model = loss_model
        self.standardize = standardize
        self.dmodel = dmodel
        self.rank = rank
        self.C = C
        self.Z = Z
        self.H = H
        self.W = W
        self.batch_size = batch_size
        self.intensity_distribution = intensity_distribution
        self.bg_distribution = bg_distribution
        self.spot_profile = spot_profile
        self.current_step = 0
        self.lr = lr
        self.training_preds = {
            "q_I_mean": [],
            "q_I_stddev": [],
            "q_bg_mean": [],
            "q_bg_stddev": [],
            "DIALS_I_prf_val": [],
            "DIALS_I_prf_var": [],
            "DIALS_I_sum_val": [],
            "DIALS_I_sum_var": [],
            "shape": [],
            "refl_id": [],
            "tbl_id": [],
            "counts": [],
            "profile": [],
        }

        self.l2_lambda = lambda_l2
        self.validation_preds = {
            "q_I_mean": [],
            "q_I_stddev": [],
            "q_bg_mean": [],
            "q_bg_stddev": [],
            "DIALS_I_prf_val": [],
            "DIALS_I_prf_var": [],
            "DIALS_I_sum_val": [],
            "DIALS_I_sum_var": [],
            "shape": [],
            "refl_id": [],
            "tbl_id": [],
            "counts": [],
            "profile": [],
        }
        self.total_steps = total_steps + 1
        self.background_decomposition = bg_decomp

        self.current_step = 0

    def forward(
        self,
        samples,
        dead_pixel_mask,
    ):
        counts = torch.clamp(samples[..., -1], min=0)
        shoebox_ = self.standardize(samples, dead_pixel_mask.squeeze(-1))

        dxyz = samples[..., 3:6]

        counts_ = shoebox_[..., -1].reshape(samples.size(0), Z, H, W)

        batch_size = counts.size(0)

        counts_ = counts_ * dead_pixel_mask.reshape(batch_size, Z, H, W)

        batch_ = self.standardize(samples, dead_pixel_mask)

        X_coords = batch_[..., 0][: H * W][:, : H * W].view(batch_size, 1, H, W)

        Y_coords = batch_[..., 1][: H * W][:, : H * W].view(batch_size, 1, H, W)

        Z_coords = batch_[..., 2][: H * W][:, : H * W].view(batch_size, 1, H, W)

        shoebox = torch.cat([X_coords, Y_coords, Z_coords, counts_], dim=1)

        representation = self.encoder(shoebox)

        profile = self.profile_decomposition(representation).view(batch_size, Z * H * W)

        # background_decomposition = self.background_decomposition(representation).view(
        # batch_size, Z * H * W
        # )
        background_decomposition = self.background_decomposition(representation)

        q_I = self.builder(representation, dxyz)

        rate, z = self.decoder(
            q_I, background_decomposition, profile, background_decomposition
        )

        nll, kl_term = self.loss_model(
            rate,
            z,
            counts=counts,
            q_I=q_I,
            q_bg=background_decomposition,
            dead_pixel_mask=dead_pixel_mask,
        )

        return nll, kl_term, rate, q_I, profile, background_decomposition, counts

    def training_step(self, batch):
        device = self.device
        samples, metadata, dead_pixel_mask = batch
        samples = samples.to(device)
        dead_pixel_mask = dead_pixel_mask.to(device)
        nll, kl_term, rate, q_I, profile, bg, counts = self(samples, dead_pixel_mask)
        self.current_step += 1

        loss = nll + kl_term

        # l1_penalty = self.background_decomposition.compute_l1_penalty(bg)

        # tv_penalty = self.background_decomposition.compute_tv_penalty(bg)

        # loss += self.l2_lambda * l2_reg  # Add L2 penalty to the loss

        # smoothness_penalty = self.background_decomposition.compute_smoothness_penalty(
        # bg
        # )
        # smoothness_lambda = 0.1
        # loss += self.lambda_tv * tv_penalty + l1_penalty * self.lambda_l1

        self.log("train_loss", loss, prog_bar=True)
        self.log("nll", nll, prog_bar=False)
        self.log("kl_term", kl_term, prog_bar=True)
        # self.log("mean_bg", bg.mean(), prog_bar=True)
        self.log("mean_bg", bg.mean.mean(), prog_bar=True)
        # self.log("std_bg", bg.std(), prog_bar=False)
        # self.log("l1", l1_penalty, prog_bar=True)
        # self.log("f'", smoothness_penalty, prog_bar=True)
        # self.log("l2_reg", l2_reg * self.l2_lambda, prog_bar=True)

        if self.current_epoch == self.trainer.max_epochs - 1:
            self.training_preds["q_I_mean"].extend(
                q_I.mean.detach().cpu().ravel().tolist()
            )
            self.training_preds["q_I_stddev"].extend(
                q_I.stddev.detach().cpu().ravel().tolist()
            )

            self.training_preds["q_bg_mean"].extend(bg.mean.detach().cpu().ravel())
            self.training_preds["DIALS_I_prf_val"].extend(metadata[:, 2].detach().cpu())

            self.training_preds["DIALS_I_prf_var"].extend(metadata[:, 3].detach().cpu())

            self.training_preds["DIALS_I_sum_val"].extend(metadata[:, 0].detach().cpu())

            self.training_preds["DIALS_I_sum_var"].extend(metadata[:, 1].detach().cpu())

            self.training_preds["refl_id"].extend(metadata[:, 4].detach().cpu().numpy())
            self.training_preds["counts"].extend(counts.detach().cpu().numpy())
            self.training_preds["profile"].extend(profile.detach().cpu().numpy())

        return loss

    def validation_step(self, batch):
        device = self.device
        samples, metadata, dead_pixel_mask = batch
        samples = samples.to(device)
        nll, kl_term, rate, q_I, profile, bg, counts = self(samples, dead_pixel_mask)
        self.current_step += 1
        # l1_lambda = 1e-3
        # l1_penalty = torch.sum(torch.abs(bg))
        loss = nll + kl_term

        # smoothness_penalty = self.background_decomposition.compute_smoothness_penalty(
        # bg
        # )
        # smoothness_lambda = 0.1
        # l1_penalty = self.background_decomposition.compute_l1_penalty(bg)
        # tv_penalty = self.background_decomposition.compute_tv_penalty(bg)
        # loss += smoothness_lambda * smoothness_penalty + l1_penalty * self.l2_lambda
        # loss += tv_penalty

        # l2_reg = 0
        # for param in self.background_decomposition.parameters():
        # l2_reg += torch.norm(param, 2)
        # # l2_reg += torch.sum(torch.abs(params)

        # loss += self.l2_lambda * l2_reg  # Add L2 penalty to the loss
        # loss = loss + l1_lambda * l1_penalty
        self.log("val_loss", loss, prog_bar=True)
        # self.log("l1_penalty", l1_penalty, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# %%

subset_size = 10
batch_size = 8

data_Module = ShoeboxDataModule(
    shoebox_data="./samples.pt",
    metadata="./metadata.pt",
    dead_pixel_mask="./masks.pt",
    batch_size=batch_size,
    val_split=0.2,
    test_split=0.1,
    num_workers=1,
    include_test=False,
    subset_size=subset_size,
    single_sample_index=None,
)

data_Module.setup()

# %%
epochs = 100
p_I_scale = 0.0001
p_bg_scale = 0.001
dmodel = 64
C = 6
Z = 3
H = 21
W = 21
rank = 5


# Profile layers
spot_profile = Profile(dmodel, num_components=1)
intensity_distribution = IntensityDistribution(
    dmodel, intensity_dist=torch.distributions.gamma.Gamma
)
bg_distribution = torch.distributions.gamma.Gamma

# intensity_distribution = IntensityDistribution(dmodel, intensity_dist=rsd.FoldedNormal)

# intensity_distribution = IntensityDistribution(
# dmodel,

# intensity_dist=torch.distributions.log_normal.LogNormal,
# )


train_loader_len = len(data_Module.train_dataloader())


# Layers
standardize = Standardize()
encoder = Encoder(
    use_bn=True,
    conv1_in_channel=C,
    conv1_out_channel=64,
    conv1_kernel_size=7,
    conv1_stride=2,
    conv1_padding=3,
    layer1_num_blocks=3,
)
profile_decomposition = ProfileDecomposition(dmodel, rank)

builder = DistributionBuilder(intensity_distribution, spot_profile)

decoder = Decoder()

loss_model = Loss(
    p_I_scale=p_I_scale,
    p_bg_scale=p_bg_scale,
    prior_I=torch.distributions.exponential.Exponential(0.1),
    # prior_bg=torch.distributions.exponential.Exponential(10),
    prior_bg=torch.distributions.normal.Normal(0.0, 0.5),
)

steps = 1000 * train_loader_len

logger = TensorBoardLogger(save_dir="./integrator_logs", name="integrator_model")

checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",
    dirpath="./integrator_logs/checkpoints/",
    filename="integrator-{epoch:02d}-{train_loss:.2f}",
    save_top_k=3,
    mode="min",
)

progress_bar = TQDMProgressBar(refresh_rate=1)

# bg_decomp = BackgroundDecomposition(dmodel)
bg_decomp = BackgroundDecomposition(dmodel, background_distribution=bg_distribution)

integrator_model = Integrator(
    encoder,
    profile_decomposition,
    builder,
    decoder,
    loss_model,
    standardize,
    bg_decomp=bg_decomp,
    dmodel=64,
    rank=rank,
    C=6,
    Z=3,
    H=21,
    W=21,
    batch_size=batch_size,
    # intensity_distribution=torch.distributions.gamma.Gamma,
    intensity_distribution=rsd.FoldedNormal,
    bg_distribution=torch.distributions.gamma.Gamma,
    spot_profile=spot_profile,
    total_steps=steps,
    lambda_l1=0.08,
    lambda_tv=0.001,
)

trainer = Trainer(
    max_epochs=epochs,
    accelerator="cpu",  # Use "cpu" for CPU training
    devices="auto",
    num_nodes=1,
    precision=32,  # Use 32-bit precision for CPU
    accumulate_grad_batches=1,
    check_val_every_n_epoch=1,
    callbacks=[checkpoint_callback, progress_bar],
    logger=logger,
    log_every_n_steps=10,
)

trainer.fit(integrator_model, data_Module)

# %%
idx = 3

line = np.linspace(1, 1e6, 10000)

plt.plot(line, line, "k--", alpha=0.3)
plt.scatter(
    integrator_model.training_preds["q_I_mean"],
    integrator_model.training_preds["DIALS_I_prf_val"],
)
plt.yscale("log")
plt.grid()
plt.xscale("log")
plt.gca().set_aspect("equal", adjustable="box")
plt.xlim(1, 1e6)
plt.ylim(1, 1e6)
plt.show()

rates = torch.tensor(integrator_model.training_preds["q_I_mean"]).unsqueeze(
    -1
) * torch.tensor(integrator_model.training_preds["profile"])
samples = torch.distributions.Poisson(rates).sample().view(rates.size(0), Z, H, W)

figure, ax = plt.subplots(4, 3, figsize=(15, 5))

integrator_model.training_preds["q_bg_mean"][idx]

for j in range(3):
    vmin = integrator_model.training_preds["q_bg_mean"][idx].min()
    vmax = integrator_model.training_preds["q_bg_mean"][idx].max()
    ax[0, j].imshow(
        integrator_model.training_preds["q_bg_mean"][idx].reshape(Z, H, W)[j],
        vmin=vmin,
        vmax=vmax,
    )

for j in range(3):
    vmin = (
        integrator_model.training_preds["profile"][idx].min()
        * integrator_model.training_preds["q_I_mean"][idx]
    )
    vmax = (
        integrator_model.training_preds["profile"][idx].max()
        * integrator_model.training_preds["q_I_mean"][idx]
    )
    ax[1, j].imshow(
        integrator_model.training_preds["profile"][idx].reshape(Z, H, W)[j]
        * integrator_model.training_preds["q_I_mean"][idx],
        vmin=vmin,
        vmax=vmax,
    )

for j in range(3):
    vmin = integrator_model.training_preds["counts"][idx].min()
    vmax = integrator_model.training_preds["counts"][idx].max()
    ax[2, j].imshow(
        integrator_model.training_preds["counts"][idx].reshape(Z, H, W)[j],
        vmin=vmin,
        vmax=vmax,
    )

for j in range(3):
    vmin = samples[idx].min()
    vmax = samples[idx].max()

    ax[3, j].imshow(
        samples[idx][j],
        vmin=vmin,
        vmax=vmax,
    )

plt.show()

torch.stack(integrator_model.training_preds["q_bg_mean"]).min()
torch.stack(integrator_model.training_preds["q_bg_mean"]).max()


# %%
integrator_model.training_preds["q_bg_mean"]

# %%

integrator_model.training_preds["q_I_mean"][idx]

integrator_model.training_preds["DIALS_I_prf_val"][idx]
integrator_model.training_preds["DIALS_I_sum_val"][idx]

# %%
bg = torch.stack(integrator_model.training_preds["q_bg_mean"])
mean_bg = torch.stack(integrator_model.training_preds["q_bg_mean"]).mean

plt.plot(bg.sum(1), integrator_model.training_preds["DIALS_I_prf_val"], "o")
plt.yscale("log")
plt.xscale("log")
plt.show()

bg.sum(1).mean()
bg.sum(1).max()
bg.sum(1).min()

torch.sigmoid(integrator_model.training_preds["q_bg_mean"][10].view(3, 21, 21))

batch = torch.stack(integrator_model.training_preds["q_bg_mean"]).reshape(30, 3, 21, 21)


# %%
# SCRATCH WORK
batch_size = 10
dmodel = 64
rank = 10
C = 6
Z = 3
H = 21
W = 21

# Profile layers
spot_profile = Profile(dmodel, num_components=1)
# intensity_distribution = IntensityDistribution(
# dmodel, intensity_dist=torch.distributions.gamma.Gamma
# )
intensity_distribution = IntensityDistribution(
    dmodel,
    intensity_dist=torch.distributions.log_normal.LogNormal,
)

# Layers
standardize = Standardize()
encoder = Encoder(
    use_bn=True,
    conv1_in_channel=C,
    conv1_out_channel=64,
    conv1_kernel_size=7,
    conv1_stride=2,
    conv1_padding=3,
    layer1_num_blocks=3,
)
profile_decomposition = ProfileDecomposition(dmodel, 5)
builder = DistributionBuilder(intensity_distribution, spot_profile)
decoder = Decoder()
loss_model = Loss(p_I_scale=0.0001)

# Load samples and their masks
# samples = torch.load("samples.pt")
masks = torch.load("masks.pt")
batch = samples[:batch_size]
counts_ = torch.clamp(batch[..., -1], min=0)

mask_batch = masks[:batch_size]

dxyz = batch[..., 3:6]

batch_ = standardize(batch, mask_batch)

# Get the intensities and reshape into a Z x H x W image
counts = batch_[..., -1].reshape(batch_size, Z, H, W)

# Create feature channels of the X, Y, and Z coordinates
X_coords = batch_[..., 0][: H * W][:, : H * W].view(batch_size, 1, H, W)

Y_coords = batch_[..., 1][: H * W][:, : H * W].view(batch_size, 1, H, W)

Z_coords = batch_[..., 2][: H * W][:, : H * W].view(batch_size, 1, H, W)

# Concatenate the feature channels
shoebox = torch.cat([X_coords, Y_coords, Z_coords, counts], dim=1)

representation = encoder(shoebox)

background = profile_decomposition(representation).view(batch_size, Z * H * W)

q_I, profile, L = builder(representation, dxyz)

rate, z = decoder(q_I, profile, background)

nll, kl_term = loss_model(rate, z, counts=counts_, q_I=q_I, dead_pixel_mask=masks)
