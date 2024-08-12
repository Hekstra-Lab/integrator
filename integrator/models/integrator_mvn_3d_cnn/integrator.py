import torch
import pytorch_lightning


class Integrator(pytorch_lightning.LightningModule):
    def __init__(
        self,
        encoder,
        distribution_builder,
        standardize,
        decoder,
        loss_model,
        total_steps,
        n_cycle=4,
        ratio=0.5,
        lr=0.001,
        max_epochs=10,
        penalty_scale=0.01,
        use_bg_profile=True,
    ):
        super().__init__()
        self.lr = lr
        self.standardize = standardize
        self.encoder = encoder
        self.distribution_builder = distribution_builder
        self.decoder = decoder
        self.train_avg_loss = []
        self.validation_avg_loss = []
        self.max_epochs = max_epochs
        self.penalty_scale = self.register_buffer(
            "penalty_scale", torch.tensor(penalty_scale)
        )
        self.training_step_loss = []
        self.validation_step_loss = []
        self.loss_model = loss_model
        self.training_preds = {
            "q_I_mean": [],
            "q_I_stddev": [],
            "q_bg_mean": [],
            "q_bg_stddev": [],
            "L_pred": [],
            "DIALS_I_prf_val": [],
            "DIALS_I_prf_var": [],
            "DIALS_I_sum_val": [],
            "DIALS_I_sum_var": [],
            "shape": [],
            "refl_id": [],
            "tbl_id": [],
        }

        self.validation_preds = {
            "q_I_mean": [],
            "q_I_stddev": [],
            "q_bg_mean": [],
            "q_bg_stddev": [],
            "L_pred": [],
            "DIALS_I_prf_val": [],
            "DIALS_I_prf_var": [],
            "DIALS_I_sum_val": [],
            "DIALS_I_sum_var": [],
            "shape": [],
            "refl_id": [],
            "tbl_id": [],
        }
        self.total_steps = total_steps + 1

        self.current_step = 0
        self.use_bg_profile = use_bg_profile

    def forward(
        self,
        shoebox,
        dead_pixel_mask,
    ):
        counts = torch.clamp(shoebox[..., -1], min=0)

        shoebox_ = self.standardize(shoebox, dead_pixel_mask.squeeze(-1))

        dxyz = shoebox[..., 3:6]

        counts_ = shoebox_[..., -1].reshape(shoebox.size(0), 3, 21, 21)
        batch_size = counts_.size(0)

        #counts_ = counts_ * dead_pixel_mask.reshape(batch_size, 3, 21, 21)
        counts_ = counts.reshape(batch_size,3,21,21) * dead_pixel_mask.reshape(batch_size, 3, 21, 21)

        #x = shoebox_[..., 0].reshape(shoebox.size(0), 3, 21, 21)[:, 0, :, :]
        #y = shoebox_[..., 1].reshape(shoebox.size(0), 3, 21, 21)[:, 0, :, :]
        #z = shoebox_[..., 2].reshape(shoebox.size(0), 3, 21, 21)[:, 0, :, :]

        # shoebox_ = torch.cat(
        # [x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1), counts_], dim=1
        # )

        representation = self.encoder(counts_).unsqueeze(1)

        q_bg, q_I, profile, L, bg_profile = self.distribution_builder(
            representation, dxyz
        )

        rate, z, bg = self.decoder(
            q_bg, q_I, profile, bg_profile, use_bg_profile=self.use_bg_profile
        )

        nll, kl_term = self.loss_model(
            rate,
            z,
            bg,
            counts,
            q_bg,
            q_I,
            dead_pixel_mask,
            eps=1e-5,
        )

        return nll, kl_term, rate, q_I, profile, q_bg, counts, L

    def training_step(self, batch, batch_idx):
        try:
            device = self.device
            (sbox, metadata, mask) = batch
            (
                sbox,
                mask,
            ) = sbox.to(device), mask.to(device)

            nll, kl_term, rate, q_I, profile, q_bg, counts, L = self(sbox, mask)

            self.current_step += 1

            loss = nll +  kl_term
            self.training_step_loss.append(loss)
            self.log("train_loss", loss, prog_bar=True)

            if self.current_epoch == self.trainer.max_epochs - 1:
                self.training_preds["q_I_mean"].extend(
                    q_I.mean.detach().cpu().ravel().tolist()
                )

                self.training_preds["q_I_stddev"].extend(
                    q_I.stddev.detach().cpu().ravel().tolist()
                )

                self.training_preds["q_bg_mean"].extend(
                    q_bg.mean.detach().cpu().ravel().tolist()
                )

                self.training_preds["q_bg_stddev"].extend(
                    q_bg.stddev.detach().cpu().ravel().tolist()
                )

                self.training_preds["L_pred"].extend(L.detach().cpu())

                self.training_preds["DIALS_I_prf_val"].extend(
                    metadata[:, 2].detach().cpu()
                )

                self.training_preds["DIALS_I_prf_var"].extend(
                    metadata[:, 3].detach().cpu()
                )

                self.training_preds["DIALS_I_sum_val"].extend(
                    metadata[:, 0].detach().cpu()
                )

                self.training_preds["DIALS_I_sum_var"].extend(
                    metadata[:, 1].detach().cpu()
                )

                # self.training_preds["shape"].extend(metadata[:, 6:].detach().cpu())

                self.training_preds["refl_id"].extend(
                    metadata[:, 4].detach().cpu().numpy()
                )

                # self.training_preds["tbl_id"].extend(metadata[:, 4].detach().cpu().numpy())

            return loss

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            print(f"Batch data: {batch}")
            # Optionally, save the batch to a file for further inspection
            torch.save(batch, f"error_batch_{batch_idx}.pt")
            raise e  # Re-raise the exception to stop training

    def validation_step(self, batch):
        device = self.device

        (sbox, metadata, mask) = batch

        sbox, mask = sbox.to(device), mask.to(device)

        nll, kl_term, rate, q_I, profile, q_bg, counts, L = self(sbox, mask)

        loss = nll +  kl_term

        self.validation_step_loss.append(loss)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        if self.current_epoch == self.trainer.max_epochs - 1:
            self.validation_preds["q_I_mean"].extend(
                q_I.mean.detach().cpu().ravel().tolist()
            )
            self.validation_preds["q_I_stddev"].extend(
                q_I.stddev.detach().cpu().ravel().tolist()
            )
            self.validation_preds["q_bg_mean"].extend(
                q_bg.mean.detach().cpu().ravel().tolist()
            )
            self.validation_preds["q_bg_stddev"].extend(
                q_bg.stddev.detach().cpu().ravel().tolist()
            )
            self.validation_preds["L_pred"].extend(L.detach().cpu())
            self.validation_preds["DIALS_I_prf_val"].extend(
                metadata[:, 2].detach().cpu()
            )
            self.validation_preds["DIALS_I_sum_val"].extend(
                metadata[:, 0].detach().cpu()
            )
            self.validation_preds["DIALS_I_prf_var"].extend(
                metadata[:, 3].detach().cpu()
            )

            self.validation_preds["DIALS_I_sum_var"].extend(
                metadata[:, 1].detach().cpu()
            )
            self.validation_preds["shape"].extend(metadata[:, 6:].detach().cpu())
            self.validation_preds["refl_id"].extend(
                metadata[:, 4].detach().cpu().numpy()
            )
            # self.validation_preds["tbl_id"].extend(
            # metadata[:, 4].detach().cpu().numpy()
            # )

        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.mean(torch.tensor(self.training_step_loss, device=self.device))
        self.train_avg_loss.append(avg_loss.detach().cpu())
        self.training_step_loss.clear()

    def on_validation_epoch_end(self):
        avg_loss = torch.mean(
            torch.tensor(self.validation_step_loss, device=self.device)
        )
        self.validation_avg_loss.append(avg_loss.detach().cpu())
        self.validation_step_loss.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
