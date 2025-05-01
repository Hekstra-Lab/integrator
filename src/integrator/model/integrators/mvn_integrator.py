import torch
import torch.nn.functional as F
import torch.nn as nn
from integrator.model.integrators import BaseIntegrator
from integrator.model.decoders import MVNDecoder
from integrator.layers import Linear


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
            division = weighted_sum_intensity_sum / (summed_squared_prf + 1e-10)

            # Mean and variance across MC samples
            weighted_sum_mean = division.mean(-1)
            weighted_sum_var = division.var(-1)

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
                "weighted_sum_mean": weighted_sum_mean,
                "weighted_sum_var": weighted_sum_var,
            }

            return intensities

        # def forward(self, shoebox, dials, masks, metadata,counts,samples):

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
        profile = self.profile_model(representation)

        if self.intensity_distribution is not None:
            qI = self.intensity_distribution(representation)
            rate = self.decoder(qI, qbg, profile)
        else:
            qI = None
            rate, intensity_mean, intensity_variance = self.decoder(
                qbg, profile, counts, masks
            )

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qbg": qbg,
            # "qI": qI,
            "intensity_mean": intensity_mean if qI is None else None,
            "intensity_var": intensity_variance if qI is None else None,
            "dials_I_sum_value": dials[:, 0],
            "dials_I_sum_var": dials[:, 1],
            "dials_I_prf_value": dials[:, 2],
            "dials_I_prf_var": dials[:, 3],
            "refl_ids": dials[:, 4],
            "profile": profile,
        }

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
            None,
            # outputs["qI"],
            outputs["qbg"],
            outputs["masks"],
        )

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # Log metrics
        self.log("train_loss", loss.mean())
        self.log("train_nll", neg_ll.mean())
        self.log("train_kl", kl_terms.mean())
        self.log("kl_bg", kl_bg)
        self.log("kl_I", kl_I)

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)

        # Calculate validation metrics
        loss, neg_ll, kl_terms, kl_bg, kl_I, prof_reg = self.loss_fn(
            outputs["rates"],
            outputs["counts"],
            outputs["profile"],
            # outputs["qI"],
            None,
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
        intensities = self.calculate_intensities(
            outputs["counts"], outputs["qbg"], outputs["profile"], outputs["masks"]
        )
        return {
            "qI_mean": outputs["qI"].mean,
            "qI_variance": outputs["qI"].variance,
            "weighted_sum_mean": intensities["weighted_sum_mean"],
            "weighted_sum_var": intensities["weighted_sum_var"],
            "thresholded_mean": intensities["thresholded_mean"],
            "thresholded_var": intensities["thresholded_var"],
            "refl_ids": outputs["refl_ids"],
            "intensity_mean": outputs["intensity_mean"],
            "intensity_var": outputs["intensity_var"],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LRMVNIntegrator(BaseIntegrator):
    def __init__(
        self,
        encoder,
        metadata_encoder,
        loss,
        qbg,
        qp,
        qI,
        mc_samples=100,
        learning_rate=1e-3,
        max_iterations=4,
        profile_threshold=0.001,
        dmodel=64,
        use_metarep=True,
        use_metaonly=False,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples
        self.qI = qI
        self.qbg = qbg
        self.automatic_optimization = True
        self.loss_fn = loss
        self.max_iterations = max_iterations
        self.d = 3
        self.h = 21
        self.w = 21

        self.use_metarep = use_metarep
        self.use_metaonly = use_metaonly

        # Handle model components based on flags
        if self.use_metaonly:
            self.metadata_encoder = metadata_encoder
            self.encoder = None
            print("Using metadata encoder only")
        else:
            self.encoder = encoder
            if self.use_metarep:
                self.metadata_encoder = metadata_encoder

        # Create centered coordinate grid
        z_coords = torch.arange(self.d).float() - (self.d - 1) / 2
        y_coords = torch.arange(self.h).float() - (self.h - 1) / 2
        x_coords = torch.arange(self.w).float() - (self.w - 1) / 2

        z_coords = z_coords.view(self.d, 1, 1).expand(self.d, self.h, self.w)
        y_coords = y_coords.view(1, self.h, 1).expand(self.d, self.h, self.w)
        x_coords = x_coords.view(1, 1, self.w).expand(self.d, self.h, self.w)

        # Stack coordinates
        pixel_positions = torch.stack([x_coords, y_coords, z_coords], dim=-1)
        pixel_positions = pixel_positions.view(-1, 3)

        # Register buffer
        self.register_buffer("pixel_positions", pixel_positions)

        # independent normals for mean
        self.mean_layer = Linear(
            in_features=self.dmodel,
            out_features=3,
        )
        self.std_layer = Linear(
            in_features=self.dmodel,
            out_features=3,
        )

        # indepdent normals for factor
        self.mean2_layer = Linear(
            in_features=self.dmodel,
            out_features=3,
        )
        self.std2_layer = Linear(
            in_features=self.dmodel,
            out_features=3,
        )

        # halfnormals for diag
        self.scale_layer = Linear(
            in_features=self.dmodel,
            out_features=3,
        )

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks
            zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            zp = qp
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

    def forward(self, counts, shoebox, metadata, masks, reference):
        # Unpack batch
        counts = torch.clamp(counts, min=0) * masks
        batch_size = shoebox.shape[0]

        if self.use_metarep:
            meta_rep = self.metadata_encoder(metadata)

        if self.use_metaonly:
            rep = self.metadata_encoder(metadata)
        else:
            rep = self.encoder(shoebox, masks)

        mean = self.mean_layer(rep).unsqueeze(-1)
        std = F.softplus(self.std_layer(rep)).unsqueeze(-1)

        if self.use_metarep:
            mean_factor = self.mean2_layer(rep + meta_rep).unsqueeze(-1)
            std_factor = F.softplus(self.std2_layer(rep + meta_rep)).unsqueeze(-1)
        else:
            mean_factor = self.mean2_layer(rep).unsqueeze(-1)
            std_factor = F.softplus(self.std2_layer(rep)).unsqueeze(-1)

        # for halfnormal
        scale = F.softplus(self.scale_layer(rep)).unsqueeze(-1)

        qp_mean = torch.distributions.Normal(
            loc=mean,
            scale=std,
        )

        qp_factor = torch.distributions.Normal(
            loc=mean_factor,
            scale=std_factor,
        )

        qp_diag = torch.distributions.half_normal.HalfNormal(
            scale=scale,
        )

        mean_samples = qp_mean.rsample([self.mc_samples]).squeeze(-1).permute(1, 0, 2)
        factor_samples = qp_factor.rsample([self.mc_samples]).permute(1, 0, 2, 3)
        diag_samples = (
            qp_diag.rsample([self.mc_samples]).squeeze(-1).permute(1, 0, 2) + 1e-6
        )

        prof_dist = (
            torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
                loc=mean_samples.view(batch_size, self.mc_samples, 1, 3),
                cov_factor=factor_samples.view(batch_size, self.mc_samples, 1, 3, 1),
                cov_diag=diag_samples.view(batch_size, self.mc_samples, 1, 3),
            )
        )

        log_probs = prof_dist.log_prob(
            self.pixel_positions.expand(batch_size, 1, 1323, 3)
        )

        log_probs_stable = log_probs - log_probs.max(dim=-1, keepdim=True)[0]

        profile = torch.exp(log_probs_stable)

        avg_profile = profile.mean(dim=1)
        avg_profile = avg_profile / (avg_profile.sum(dim=-1, keepdim=True) + 1e-10)

        zp = profile / profile.sum(dim=-1, keepdim=True) + 1e-10

        qbg = self.qbg(rep)

        if self.use_metarep:
            qI = self.qI(rep, meta_rep)
        else:
            qbg = self.qbg(rep)
            qI = self.qI(rep)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        zI = qI.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        intensity_mean = qI.mean  # [batch_size]
        intensity_var = qI.variance  # [batch_size]

        rate = zI * zp + zbg  # [batch_size, mc_samples, pixels]

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qbg": qbg,
            "qp": avg_profile,
            "qp_mean": qp_mean,
            "qI": qI,
            "intensity_mean": intensity_mean,
            "intensity_var": intensity_var,
            "dials_I_sum_value": reference[:, 6],
            "dials_I_sum_var": reference[:, 7],
            "dials_I_prf_value": reference[:, 8],
            "dials_I_prf_var": reference[:, 9],
            "refl_ids": reference[:, -1],
            "profile": zp,
            "qp_factor": qp_factor,
            "qp_diag": qp_diag,
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
            "d": reference[:, 13],
        }

    def training_step(self, batch, batch_idx):
        # Unpack batch
        shoebox, dials, masks, metadata, counts = batch

        # Get model outputs
        outputs = self(shoebox, dials, masks, metadata, counts)

        # Calculate loss
        (
            loss,
            neg_ll,
            kl,
            kl_bg,
            kl_I,
            kl_p_mean,
            kl_p_diag,
            kl_p_factor,
        ) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            # q_p=outputs["qp"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
            q_I=outputs["qI"],
            q_p_mean=outputs["qp_mean"],
            q_p_diag=outputs["qp_diag"],
            q_p_factor=outputs["qp_factor"],
        )

        # Clip gradients for stability
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # Log metrics
        self.log("train: loss", loss.mean())
        self.log("train: nll", neg_ll.mean())
        self.log("train: kl", kl.mean())
        self.log("train: kl_bg", kl_bg.mean())
        self.log("train: kl_I", kl_I.mean())
        self.log("train: kl_p_mean", kl_p_mean.mean())
        self.log("train: kl_p_factor", kl_p_factor.mean())
        self.log("train: kl_p_diag", kl_p_diag.mean())
        # self.log("mean(qI.mean)", outputs["qI"].mean.mean())
        # self.log("min(qI.mean)", outputs["qI"].mean.min())
        # self.log("max(qI.mean)", outputs["qI"].mean.max())
        # self.log("mean(qp.mean)", outputs["qp_mean"].mean.mean())
        # self.log("min(qp.mean)", outputs["qp_mean"].mean.min())
        # self.log("max(qp.mean)", outputs["qp_mean"].mean.max())
        # self.log("mean(qp.variance)", outputs["qp_mean"].variance.mean())
        # self.log("mean(qp_factor.mean)", outputs["qp_factor"].mean.mean())
        # self.log("min(qp_factor.mean)", outputs["qp_factor"].mean.min())
        # self.log("max(qp_factor.mean)", outputs["qp_factor"].mean.max())
        # self.log("mean(qp_factor.variance)", outputs["qp_factor"].variance.mean())
        # self.log("mean(qp_diag.mean)", outputs["qp_diag"].mean.mean())
        # self.log("min(qp_diag.mean)", outputs["qp_diag"].mean.min())
        # self.log("max(qp_diag.mean)", outputs["qp_diag"].mean.max())
        # self.log("mean(qp_diag.variance)", outputs["qp_diag"].variance.mean())
        # self.log("mean(qbg.mean)", outputs["qbg"].mean.mean())
        # self.log("min(qbg.mean)", outputs["qbg"].mean.min())
        # self.log("max(qbg.mean)", outputs["qbg"].mean.max())
        # self.log("mean(qbg.variance)", outputs["qbg"].variance.mean())
        # self.log("mean(qI.variance)", outputs["qI"].variance.mean())
        # self.log("mean(qI.mean)", outputs["qI"].mean.mean())
        # self.log("max(qI.variance)", outputs["qI"].variance.max())
        # self.log("min(qI.variance)", outputs["qI"].variance.min())

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
            kl_I,
            kl_p_mean,
            kl_p_diag,
            kl_p_factor,
        ) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
            q_I=outputs["qI"],
            q_p_mean=outputs["qp_mean"],
            q_p_diag=outputs["qp_diag"],
            q_p_factor=outputs["qp_factor"],
        )

        # Log metrics
        self.log("val: -ELBO", loss.mean())
        self.log("val: NLL", neg_ll.mean())
        self.log("val: KL", kl.mean())
        self.log("val: KL bg", kl_bg.mean())
        self.log("val: KL I", kl_I.mean())
        self.log("val: KL p_mean", kl_p_mean.mean())
        self.log("val: KL p_factor", kl_p_factor.mean())
        self.log("val: KL p_diag", kl_p_diag.mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)
        intensities = self.calculate_intensities(
            counts=outputs["counts"],
            qbg=outputs["qbg"],
            qp=outputs["profile"],
            masks=outputs["masks"],
        )

        return {
            "intensity_mean": outputs["intensity_mean"],
            "intensity_var": outputs["intensity_var"],
            "refl_ids": outputs["refl_ids"],
            "dials_I_sum_var": outputs["dials_I_sum_var"],
            "dials_I_prf_value": outputs["dials_I_prf_value"],
            "dials_I_prf_var": outputs["dials_I_prf_var"],
            "qbg": outputs["qbg"].mean,
            "qbg_scale": outputs["qbg"].scale,
            "profile_masking_mean": intensities["profile_masking_mean"],
            "profile_masking_var": intensities["profile_masking_var"],
            "kabsch_sum_mean": intensities["kabsch_sum_mean"],
            "kabsch_sum_var": intensities["kabsch_sum_var"],
            "x_c": outputs["x_c"],
            "y_c": outputs["y_c"],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
