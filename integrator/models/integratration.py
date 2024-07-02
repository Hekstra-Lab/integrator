from pylab import *
import pytorch_lightning
import torch


class Integrator(torch.nn.Module):
    """
    Integration module

    Attributes:
        encoder (torch.nn.Module): Encodes shoeboxes.
        distribution_builder (torch.nn.Module): Builds variational distributions and profile.
        likelihood (torch.nn.Module): MLE cost function.
        counts_std (torch.nn.Parameter): Standard deviation of counts. Not trainable.
    """

    def __init__(
        self,
        standardize,
        encoder,
        distribution_builder,
        likelihood,
    ):
        super().__init__()
        self.standardize = standardize
        self.encoder = encoder
        self.distribution_builder = distribution_builder
        self.likelihood = likelihood

    def check_gradients(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"{name} gradient norm: {param.grad.norm().item()}")
            else:
                print(f"{name} has no gradients")

    def forward(
        self,
        shoebox,
        dead_pixel_mask,
        is_flat,
        mc_samples=100,
        min_voxel= None,
    ):
        """
        Forward pass of the integrator.

        Args:
            shoebox (torch.Tensor): Shoebox tensor
            padding_mask (torch.Tensor): Mask of padded entries
            dead_pixel_mask (torch.Tensor): Mask of dead pixels and padded entries
            mc_samples (int): Number of Monte Carlo samples. Defaults to 100.

        Returns:
            torch.Tensor: Negative log-likelihood loss
        """

        # get counts
        counts = torch.clamp(shoebox[..., -1], min=0)

        # standardize data
        shoebox_ = self.standardize(shoebox, dead_pixel_mask.squeeze(-1))

        # get centroid offsets
        dxyz = shoebox[..., 3:6]

        # encode shoebox
        representation = self.encoder(shoebox_, dead_pixel_mask.unsqueeze(-1))

        # build q_I, q_bg, and profile
        q_bg, q_I, profile, L = self.distribution_builder(
            representation, dxyz, dead_pixel_mask, is_flat
        )
        # q_I, profile = self.distribution_builder(representation, dxyz, dead_pixel_mask)

        # calculate ll and kl
        ll, kl_term, rate_ = self.likelihood(
            counts,
            q_bg,
            q_I,
            profile,
            L,
            mc_samples=mc_samples,
            mask=dead_pixel_mask.squeeze(-1),
        )
        num_vox = dead_pixel_mask.sum(1)
        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(
            -1
        )  # mean across mc_samples
        weights = np.log(torch.tensor(min_voxel))/torch.log(num_vox)
        ll_mean = ll_mean.sum(-1)*weights
        nll = -(torch.sum(ll_mean) / torch.sum(dead_pixel_mask))

        return (nll + kl_term, rate_, q_I, profile, q_bg, counts, L)

class IntegratorModel(pytorch_lightning.LightningModule):
    def __init__(self, encoder, distribution_builder, likelihood, standardize):
        super().__init__()
        self.standardize = standardize
        self.encoder = encoder
        self.distribution_builder = distribution_builder
        self.likelihood = likelihood

        self.train_avg_loss = []
        self.validation_avg_loss = []

        self.training_step_loss = []
        self.validation_step_loss = []

        self.training_preds = {
            "q_I_mean": [],
            "q_I_stddev": [],
            "q_bg_mean": [],
            "q_bg_stddev": [],
            "L_pred": [],
            "DIALS_I": [],
            "DIALS_bg": [],
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
            "DIALS_I": [],
            "DIALS_bg": [],
            "shape": [],
            "refl_id": [],
            "tbl_id": [],
        }

    def forward(self, shoebox, dead_pixel_mask, is_flat):
        device = shoebox.device
        counts = torch.clamp(shoebox[..., -1], min=0)
        shoebox_ = self.standardize(shoebox, dead_pixel_mask.squeeze(-1))
        dxyz = shoebox[..., 3:6]
        representation = self.encoder(shoebox_, dead_pixel_mask.unsqueeze(-1))
        q_bg, q_I, profile, L = self.distribution_builder(
            representation, dxyz, dead_pixel_mask, is_flat
        )
        ll, kl_term, rate = self.likelihood(
            counts, q_bg, q_I, profile, L, mask=dead_pixel_mask.squeeze(-1)
        )
        num_vox = dead_pixel_mask.sum(1)
        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)
        nll = -(torch.sum(ll_mean) / torch.sum(dead_pixel_mask))
        return nll + kl_term, rate, q_I, profile, q_bg, counts, L

    def training_step(self, batch, batch_idx):
        device = self.device
        sbox, mask, DIALS_I, DIALS_bg, idx, pad_mask, is_flat, id, tbl_id, shape = batch
        sbox, mask, is_flat = sbox.to(device), mask.to(device), is_flat.to(device)
        loss, rate, q_I, profile, q_bg, counts, L = self(sbox, mask, is_flat)
        self.training_step_loss.append(loss)
        self.log("train_loss", loss)

        if self.current_epoch == self.trainer.max_epochs - 1:
            self.training_preds["q_I_mean"].extend(q_I.mean.detach().ravel().tolist())
            self.training_preds["q_I_stddev"].extend(
                q_I.stddev.detach().ravel().tolist()
            )
            self.training_preds["q_bg_mean"].extend(q_bg.mean.detach().ravel().tolist())
            self.training_preds["q_bg_stddev"].extend(
                q_bg.stddev.detach().ravel().tolist()
            )
            self.training_preds["L_pred"].extend(L.detach().cpu())
            self.training_preds["DIALS_I"].extend(DIALS_I)
            self.training_preds["DIALS_bg"].extend(DIALS_bg)
            self.training_preds["shape"].extend(shape)
            self.training_preds["refl_id"].extend(id)
            self.training_preds["tbl_id"].extend(tbl_id)

        return loss

    def validation_step(self, batch, batch_idx):
        device = self.device
        sbox, mask, DIALS_I, DIALS_bg, idx, pad_mask, is_flat, id, tbl_id, shape = batch
        sbox, mask, is_flat = sbox.to(device), mask.to(device), is_flat.to(device)
        loss, rate, q_I, profile, q_bg, counts, L = self(sbox, mask, is_flat)
        self.validation_step_loss.append(loss)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        if self.current_epoch == self.trainer.max_epochs - 1:
            self.validation_preds["q_I_mean"].extend(q_I.mean.detach().ravel().tolist())
            self.validation_preds["q_I_stddev"].extend(
                q_I.stddev.detach().ravel().tolist()
            )
            self.validation_preds["q_bg_mean"].extend(
                q_bg.mean.detach().ravel().tolist()
            )
            self.validation_preds["q_bg_stddev"].extend(
                q_bg.stddev.detach().ravel().tolist()
            )
            self.validation_preds["L_pred"].extend(L.detach().cpu())
            self.validation_preds["DIALS_I"].extend(DIALS_I)
            self.validation_preds["DIALS_bg"].extend(DIALS_bg)
            self.validation_preds["shape"].extend(shape)
            self.validation_preds["refl_id"].extend(id)
            self.validation_preds["tbl_id"].extend(tbl_id)

        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.mean(torch.tensor(self.training_step_loss, device=self.device))
        self.train_avg_loss.append(avg_loss)
        self.training_step_loss.clear()

    def on_validation_epoch_end(self):
        avg_loss = torch.mean(
            torch.tensor(self.validation_step_loss, device=self.device)
        )
        self.validation_avg_loss.append(avg_loss)
        self.validation_step_loss.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
