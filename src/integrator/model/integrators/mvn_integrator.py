import torch
import torch.nn.functional as F

from integrator.layers import Linear
from integrator.model.integrators import BaseIntegrator


class LRMVNIntegrator(BaseIntegrator):
    def __init__(
        self,
        encoder,
        metadata_encoder,
        loss,
        qbg,
        qi,
        mc_samples=100,
        learning_rate=1e-3,
        max_iterations=4,
        dmodel=64,
        use_metarep=True,
        use_metaonly=False,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples
        self.qi = qi
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
            thresholds = torch.quantile(zp, 0.99, dim=-1, keepdim=True)  # threshold values

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
            rep = self.encoder(shoebox.reshape(shoebox.shape[0], 1, 3, 21, 21), masks)

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
        diag_samples = qp_diag.rsample([self.mc_samples]).squeeze(-1).permute(1, 0, 2) + 1e-6

        prof_dist = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
            loc=mean_samples.view(batch_size, self.mc_samples, 1, 3),
            cov_factor=factor_samples.view(batch_size, self.mc_samples, 1, 3, 1),
            cov_diag=diag_samples.view(batch_size, self.mc_samples, 1, 3),
        )

        log_probs = prof_dist.log_prob(self.pixel_positions.expand(batch_size, 1, 1323, 3))

        log_probs_stable = log_probs - log_probs.max(dim=-1, keepdim=True)[0]

        profile = torch.exp(log_probs_stable)

        avg_profile = profile.mean(dim=1)
        avg_profile = avg_profile / (avg_profile.sum(dim=-1, keepdim=True) + 1e-10)

        zp = profile / profile.sum(dim=-1, keepdim=True) + 1e-10

        qbg = self.qbg(rep)

        if self.use_metarep:
            qi = self.qI(rep, meta_rep)
        else:
            qbg = self.qbg(rep)
            qi = self.qI(rep)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        intensity_mean = qi.mean  # [batch_size]
        intensity_var = qi.variance  # [batch_size]

        rate = zI * zp + zbg  # [batch_size, mc_samples, pixels]

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qbg": qbg,
            "qp": avg_profile,
            "qp_mean": qp_mean,
            "qi": qi,
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
            kl_i,
            kl_p_mean,
            kl_p_diag,
            kl_p_factor,
        ) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            # q_p=outputs["qp"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
            q_i=outputs["qi"],
            q_p_mean=outputs["qp_mean"],
            q_p_diag=outputs["qp_diag"],
            q_p_factor=outputs["qp_factor"],
        )

        # Clip gradients for stability
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # Log metrics
        self.log("Train: loss", loss.mean())
        self.log("Train: nll", neg_ll.mean())
        self.log("Train: kl", kl.mean())
        self.log("Train: kl_bg", kl_bg.mean())
        self.log("Train: kl_i", kl_i.mean())
        self.log("Train: kl_p_mean", kl_p_mean.mean())
        self.log("Train: kl_p_factor", kl_p_factor.mean())
        self.log("Train: kl_p_diag", kl_p_diag.mean())
        # self.log("Mean(qi.mean)", outputs["qi"].mean.mean())
        # self.log("Min(qi.mean)", outputs["qi"].mean.min())
        # self.log("Max(qi.mean)", outputs["qi"].mean.max())
        # self.log("Mean(qp.mean)", outputs["qp_mean"].mean.mean())
        # self.log("Min(qp.mean)", outputs["qp_mean"].mean.min())
        # self.log("Max(qp.mean)", outputs["qp_mean"].mean.max())
        # self.log("Mean(qp.variance)", outputs["qp_mean"].variance.mean())
        # self.log("Mean(qp_factor.mean)", outputs["qp_factor"].mean.mean())
        # self.log("Min(qp_factor.mean)", outputs["qp_factor"].mean.min())
        # self.log("Max(qp_factor.mean)", outputs["qp_factor"].mean.max())
        # self.log("Mean(qp_factor.variance)", outputs["qp_factor"].variance.mean())
        # self.log("Mean(qp_diag.mean)", outputs["qp_diag"].mean.mean())
        # self.log("Min(qp_diag.mean)", outputs["qp_diag"].mean.min())
        # self.log("Max(qp_diag.mean)", outputs["qp_diag"].mean.max())
        # self.log("Mean(qp_diag.variance)", outputs["qp_diag"].variance.mean())
        # self.log("Mean(qbg.mean)", outputs["qbg"].mean.mean())
        # self.log("Min(qbg.mean)", outputs["qbg"].mean.min())
        # self.log("Max(qbg.mean)", outputs["qbg"].mean.max())
        # self.log("Mean(qbg.variance)", outputs["qbg"].variance.mean())
        # self.log("Mean(qi.variance)", outputs["qi"].variance.mean())
        # self.log("Mean(qi.mean)", outputs["qi"].mean.mean())
        # self.log("Max(qi.variance)", outputs["qi"].variance.max())
        # self.log("Min(qi.variance)", outputs["qi"].variance.min())

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
            kl_i,
            kl_p_mean,
            kl_p_diag,
            kl_p_factor,
        ) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
            q_i=outputs["qi"],
            q_p_mean=outputs["qp_mean"],
            q_p_diag=outputs["qp_diag"],
            q_p_factor=outputs["qp_factor"],
        )

        # Log metrics
        self.log("Val: -ELBO", loss.mean())
        self.log("Val: NLL", neg_ll.mean())
        self.log("Val: KL", kl.mean())
        self.log("Val: KL bg", kl_bg.mean())
        self.log("Val: KL I", kl_i.mean())
        self.log("Val: KL p_mean", kl_p_mean.mean())
        self.log("Val: KL p_factor", kl_p_factor.mean())
        self.log("Val: KL p_diag", kl_p_diag.mean())

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
