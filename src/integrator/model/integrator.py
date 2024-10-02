import torch
import os
import matplotlib.pyplot as plt
import pytorch_lightning


# in integrator.py


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
    ):
        super().__init__()

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
        deal_pixel_mask = torch.ones_like(dead_pixel_mask)
        shoebox_ = self.standardize(samples, dead_pixel_mask.squeeze(-1))

        representation = self.encoder(shoebox_, dead_pixel_mask)
        q_bg = self.q_bg(representation)
        q_I = self.q_I(representation)
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

        nll, kl_term, rate, q_I, profile, bg, counts = self(samples, dead_pixel_mask)

        self.current_step += 1

        loss = nll + kl_term

        self.log("train_loss", loss, prog_bar=True)
        self.log("nll", nll, prog_bar=False)
        self.log("kl_term", kl_term, prog_bar=True)
        self.log("mean_bg", bg.mean.mean(), prog_bar=True)


        if self.current_epoch == self.trainer.max_epochs - 1:
            # Save predictions
            self.training_preds["q_I_mean"].extend(
                q_I.mean.detach().cpu().ravel().tolist()
            )
            self.training_preds["q_I_stddev"].extend(
                q_I.stddev.detach().cpu().ravel().tolist()
            )
            self.training_preds["q_bg_mean"].extend(bg.mean.detach().cpu().numpy())
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
        num_samples_to_save = (
            5  # Default value; you can also make this a hyperparameter
        )

        samples, metadata, dead_pixel_mask = batch
        samples = samples.to(device)
        dead_pixel_mask = dead_pixel_mask.to(device)

        nll, kl_term, rate, q_I, profile, bg, counts = self(samples, dead_pixel_mask)
        loss = nll + kl_term

        self.log("val_loss", loss, prog_bar=True)
        self.log("nll", nll, prog_bar=False)
        self.log("kl_term", kl_term, prog_bar=True)
        self.log("mean_bg", bg.mean.mean(), prog_bar=True)



        if self.current_epoch == self.trainer.max_epochs - 1:
            self.validation_preds["q_I_mean"].extend(
                q_I.mean.detach().cpu().ravel().tolist()
            )
            self.validation_preds["q_I_stddev"].extend(
                q_I.stddev.detach().cpu().ravel().tolist()
            )
            self.validation_preds["q_bg_mean"].extend(bg.mean.detach().cpu().numpy())
            self.validation_preds["DIALS_I_prf_val"].extend(
                metadata[:, 2].detach().cpu()
            )
            self.validation_preds["DIALS_I_prf_var"].extend(
                metadata[:, 3].detach().cpu()
            )
            self.validation_preds["DIALS_I_sum_val"].extend(
                metadata[:, 0].detach().cpu()
            )
            self.validation_preds["DIALS_I_sum_var"].extend(
                metadata[:, 1].detach().cpu()
            )
            self.validation_preds["refl_id"].extend(
                metadata[:, 4].detach().cpu().numpy()
            )
            self.validation_preds["counts"].extend(counts.detach().cpu().numpy())
            self.validation_preds["profile"].extend(profile.detach().cpu().numpy())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
