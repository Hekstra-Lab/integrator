import torch
import os
import matplotlib.pyplot as plt
import pytorch_lightning
import numpy as np


class Integrator(pytorch_lightning.LightningModule):
    def __init__(
        self,
        encoder,
        profile,
        q_bg,
        q_I,
        decoder,
        loss,
        standardize,
        encoder_type,  # Add encoder_type parameter
        profile_type,  # Add profile_type parameter
        batch_size,  # Add batch_size parameter
        total_steps=10,
        max_epochs=10,
        dmodel=64,
        rank=None,
        C=6,
        Z=3,
        H=21,
        W=21,
        lr=0.001,
        images_dir=None,  # Add the images_dir parameter
        dirichlet=False,
    ):
        super().__init__()

        self.dirichlet = dirichlet
        self.encoder_type = encoder_type  # Store encoder_type
        self.profile_type = profile_type  # Store profile_type
        self.batch_size = batch_size  # Store batch_size
        self.max_epochs = max_epochs
        self.encoder = encoder
        self.profile = profile
        self.decoder = decoder
        self.loss = loss
        self.standardize = standardize
        self.dmodel = dmodel
        self.rank = rank
        self.C = C
        self.Z = Z
        self.H = H
        self.W = W

        self.num_pixels = H * W * Z

        self.lr = lr
        self.images_dir = images_dir  # Save the images_dir for later use
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

        self.q_bg = q_bg
        self.q_I = q_I
        self.current_step = 0

    def forward(
        self,
        samples,
        dead_pixel_mask,
    ):
        counts = torch.clamp(samples[..., -1], min=0)
        dxyz = samples[..., 3:6]
        # deal_pixel_mask = torch.ones_like(dead_pixel_mask)
        shoebox_ = self.standardize(samples, dead_pixel_mask.squeeze(-1))

        representation = self.encoder(shoebox_, dead_pixel_mask)
        q_bg = self.q_bg(representation)
        q_I = self.q_I(representation)

        if self.dirichlet is True:
            profile, qp = self.profile(representation)

            rate = self.decoder(q_I, q_bg, profile, mc_samples=100)

            nll, kl_term = self.loss(
                rate,
                counts,
                q_I,
                q_bg,
                qp,
                dead_pixel_mask,
                eps=1e-5,
            )

            return nll, kl_term, rate, q_I, profile, qp, q_bg, counts

        else:
            profile = self.profile(representation, dxyz).view(
                representation.size(0), self.num_pixels
            )

            rate = self.decoder(q_I, q_bg, profile, mc_samples=100)

            nll, kl_term = self.loss(
                rate,
                counts,
                q_I,
                q_bg,
                dead_pixel_mask,
                eps=1e-5,
            )

            return nll, kl_term, rate, q_I, profile, q_bg, counts

    def training_step(self, batch, batch_idx):
        device = self.device
        num_samples_to_save = (
            5  # Default value; you can also make this a hyperparameter
        )

        samples, metadata, dead_pixel_mask = batch

        samples = samples.to(device)
        dead_pixel_mask = dead_pixel_mask.to(device)

        if self.dirichlet:
            nll, kl_term, rate, q_I, profile, qp, bg, counts = self(
                samples, dead_pixel_mask
            )

        else:
            nll, kl_term, rate, q_I, profile, bg, counts = self(
                samples, dead_pixel_mask
            )

        self.current_step += 1

        loss = nll + kl_term

        self.log("train_loss", loss, prog_bar=True)
        self.log("nll", nll, prog_bar=False)
        self.log("kl_term", kl_term, prog_bar=True)
        self.log("mean_bg", bg.mean.mean(), prog_bar=True)

        return loss

    def validation_step(self, batch):
        device = self.device
        num_samples_to_save = (
            5  # Default value; you can also make this a hyperparameter
        )

        samples, metadata, dead_pixel_mask = batch
        samples = samples.to(device)
        dead_pixel_mask = dead_pixel_mask.to(device)

        if self.dirichlet:
            nll, kl_term, rate, q_I, profile, qp, bg, counts = self(
                samples, dead_pixel_mask
            )

        else:
            nll, kl_term, rate, q_I, profile, bg, counts = self(
                samples, dead_pixel_mask
            )

        loss = nll + kl_term

        self.log("val_loss", loss, prog_bar=True)
        self.log("nll", nll, prog_bar=False)
        self.log("kl_term", kl_term, prog_bar=True)
        self.log("mean_bg", bg.mean.mean(), prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
            samples, metadata, dead_pixel_mask = batch
            samples = samples.to(self.device)
            dead_pixel_mask = dead_pixel_mask.to(self.device)

            if self.dirichlet:
                nll,kl_term, rate, q_I, profile, qp, bg, counts = self(samples, dead_pixel_mask)
                prof_intensity = torch.sum((counts - bg.mean.unsqueeze(-1)) * qp.mean, dim=-1)
                
                # Compute weighted_sum
                bg_samples = bg.sample([100])
                bg_expanded = bg_samples.unsqueeze(-1).expand(-1, -1, profile.size(-1))
                result_tensor = counts.unsqueeze(0) - bg_expanded
                weights = qp.sample([100])
                weighted_sum = (result_tensor * weights).sum(-1).mean(0)
                
                # Compute masked_sum
                prof_mask = qp.mean > self.hparams.get("threshold", 0.01)
                masked_sum = torch.sum((counts - bg.mean.unsqueeze(-1)) * prof_mask, dim=-1)
                
                return {
                    "q_I_mean": q_I.mean,
                    "q_I_stddev": q_I.stddev,
                    "q_bg_mean": bg.mean,
                    "q_bg_stddev": bg.stddev,
                    "counts": counts,
                    "profile": qp.mean,
                    "refl_id": metadata[:, 4],
                    "DIALS_I_sum_val": metadata[:, 0],
                    "DIALS_I_sum_var": metadata[:, 1],
                    "DIALS_I_prf_val": metadata[:, 2],
                    "DIALS_I_prf_var": metadata[:, 3],
                    "profile_intensity": prof_intensity,
                    "weighted_sum": weighted_sum,
                    "masked_sum": masked_sum,
                    "alphas": qp.concentration
                }
            else:
                rate, q_I, profile, bg, counts = self(samples, dead_pixel_mask)
                prof_mask = profile > self.hparams.get("threshold", 0.01)
                prof_intensity = torch.sum((counts - bg.mean.unsqueeze(-1)) * prof_mask, dim=-1)
                
                return {
                    "q_I_mean": q_I.mean,
                    "q_I_stddev": q_I.stddev,
                    "q_bg_mean": bg.mean,
                    "q_bg_stddev": bg.stddev,
                    "counts": counts,
                    "profile": profile,
                    "refl_id": metadata[:, 4],
                    "DIALS_I_sum_val": metadata[:, 0],
                    "DIALS_I_sum_var": metadata[:, 1],
                    "DIALS_I_prf_val": metadata[:, 2],
                    "DIALS_I_prf_var": metadata[:, 3],
                    "profile_intensity": prof_intensity
                }


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


