from pylab import *
import numpy as np
import math
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

        # get pixel intensity values
        counts = torch.clamp(shoebox[..., -1], min=0)

        # standardize data
        shoebox_ = self.standardize(shoebox, dead_pixel_mask.squeeze(-1))

        # get centroid offsets
        dxyz = shoebox[..., 3:6]

        # encode shoeboxes
        representation = self.encoder(shoebox_, dead_pixel_mask.unsqueeze(-1))

        # build q_I, q_bg, and profile
        q_bg, q_I, profile, L = self.distribution_builder(
            representation, dxyz, dead_pixel_mask, is_flat
        )

        # q_I, profile = self.distribution_builder(representation, dxyz, dead_pixel_mask)

        # calculate likelihood and kl-divergence
        ll, kl_term, rate_ = self.likelihood(
            counts,
            q_bg,
            q_I,
            profile,
            L,
            mc_samples=mc_samples,
            mask=dead_pixel_mask.squeeze(-1),
        )

        # mean across monte carlo samples
        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)

        # negative log-likelihood
        nll = -(torch.sum(ll_mean) / torch.sum(dead_pixel_mask))

        return (nll + kl_term, rate_, q_I, profile, q_bg, counts, L)


def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.8):
    """
    A cosine function that cycles over n_epoch with n_cycle periods.
    The cosine function is scaled by ratio and shifted by start and stop.

    Args:
        start (float): start value of the cosine function
        stop (float): stop value of the cosine function
        n_epoch (int): number of epochs
        n_cycle (int): number of cycles
        ratio (float): scaling factor of the cosine function
    """
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    # transform into [0, pi] for plots:

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop:
            L[int(i + c * period)] = 0.5 - 0.5 * math.cos(v * math.pi)
            v += step
            i += 1
    return L


# class IntegratorModel(pytorch_lightning.LightningModule):
# def __init__(
# self,
# encoder,
# distribution_builder,
# likelihood,
# standardize,
# total_steps,
# n_cycle=4,
# ratio=0.5,
# anneal=True,
# lr=0.001,
# max_epochs=10,
# ):
# super().__init__()
# self.lr = lr
# self.standardize = standardize
# self.encoder = encoder
# self.distribution_builder = distribution_builder
# self.likelihood = likelihood

# self.train_avg_loss = []
# self.validation_avg_loss = []
# self.max_epochs = max_epochs

# self.training_step_loss = []
# self.validation_step_loss = []

# self.training_preds = {
# "q_I_mean": [],
# "q_I_stddev": [],
# "q_bg_mean": [],
# "q_bg_stddev": [],
# "L_pred": [],
# "DIALS_I_prf_val": [],
# "DIALS_I_prf_var": [],
# "DIALS_I_sum_val": [],
# "DIALS_I_sum_var": [],
# "shape": [],
# "refl_id": [],
# "tbl_id": [],
# }

# self.validation_preds = {
# "q_I_mean": [],
# "q_I_stddev": [],
# "q_bg_mean": [],
# "q_bg_stddev": [],
# "L_pred": [],
# "DIALS_I_prf_val": [],
# "DIALS_I_prf_var": [],
# "DIALS_I_sum_val": [],
# "DIALS_I_sum_var": [],
# "shape": [],
# "refl_id": [],
# "tbl_id": [],
# }
# self.total_steps = total_steps + 1
# self.anneal = anneal
# if self.anneal:
# self.anneal_schedule = frange_cycle_cosine(
# 0.0, 1.0, self.total_steps, n_cycle=n_cycle, ratio=ratio
# )
# self.current_step = 0

# def forward(self, shoebox, dead_pixel_mask, is_flat):
# counts = torch.clamp(shoebox[..., -1], min=0)
# shoebox_ = self.standardize(shoebox, dead_pixel_mask.squeeze(-1))
# dxyz = shoebox[..., 3:6]
# representation = self.encoder(shoebox_, dead_pixel_mask.unsqueeze(-1))
# q_bg, q_I, profile, L = self.distribution_builder(
# representation, dxyz, dead_pixel_mask, is_flat
# )

# ll, kl_term, rate = self.likelihood(
# counts,
# q_bg,
# q_I,
# profile,
# # mcsamples=100,
# )

# ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)

# nll = -(torch.sum(ll_mean) / torch.sum(dead_pixel_mask))

# return nll, kl_term, rate, q_I, profile, q_bg, counts, L

# def training_step(self, batch):
# device = self.device
# (
# sbox,
# mask,
# DIALS_I_prf_val,
# DIALS_I_prf_var,
# DIALS_I_sum_val,
# DIALS_I_sum_var,
# # idx,
# # pad_mask,
# is_flat,
# id,
# tbl_id,
# shape,
# ) = batch
# sbox, mask, is_flat = sbox.to(device), mask.to(device), is_flat.to(device)
# nll, kl_term, rate, q_I, profile, q_bg, counts, L = self(sbox, mask, is_flat)

# if self.anneal:
# anneal_rate = self.anneal_schedule[self.current_step]
# else:
# anneal_rate = 1.0
# self.current_step += 1

# loss = nll + anneal_rate * kl_term

# self.training_step_loss.append(loss)
# self.log("train_loss", loss, prog_bar=True)

# if self.current_epoch == self.trainer.max_epochs - 1:
# self.training_preds["q_I_mean"].extend(
# q_I.mean.detach().cpu().ravel().tolist()
# )
# self.training_preds["q_I_stddev"].extend(
# q_I.stddev.detach().cpu().ravel().tolist()
# )
# self.training_preds["q_bg_mean"].extend(
# q_bg.mean.detach().cpu().ravel().tolist()
# )
# self.training_preds["q_bg_stddev"].extend(
# q_bg.stddev.detach().cpu().ravel().tolist()
# )
# self.training_preds["L_pred"].extend(L.detach().cpu())
# self.training_preds["DIALS_I_prf_val"].extend(
# DIALS_I_prf_val.detach().cpu()
# )
# self.training_preds["DIALS_I_prf_var"].extend(
# DIALS_I_prf_var.detach().cpu()
# )
# self.training_preds["DIALS_I_sum_val"].extend(
# DIALS_I_sum_val.detach().cpu()
# )
# self.training_preds["DIALS_I_sum_var"].extend(
# DIALS_I_sum_var.detach().cpu()
# )
# self.training_preds["shape"].extend(shape.detach().cpu())
# self.training_preds["refl_id"].extend(id.detach().cpu().numpy())
# self.training_preds["tbl_id"].extend(tbl_id.detach().cpu().numpy())

# return loss

# def validation_step(self, batch):
# device = self.device

# (
# sbox,
# mask,
# DIALS_I_prf_val,
# DIALS_I_prf_var,
# DIALS_I_sum_val,
# DIALS_I_sum_var,
# # idx,
# # pad_mask,
# is_flat,
# id,
# tbl_id,
# shape,
# ) = batch
# sbox, mask, is_flat = sbox.to(device), mask.to(device), is_flat.to(device)
# nll, kl_term, rate, q_I, profile, q_bg, counts, L = self(sbox, mask, is_flat)

# if self.anneal:
# anneal_rate = self.anneal_schedule[self.current_step]
# else:
# anneal_rate = 1.0

# loss = nll + anneal_rate * kl_term

# self.validation_step_loss.append(loss)
# self.log("val_loss", loss, prog_bar=True, sync_dist=True)

# if self.current_epoch == self.trainer.max_epochs - 1:
# self.validation_preds["q_I_mean"].extend(
# q_I.mean.detach().cpu().ravel().tolist()
# )
# self.validation_preds["q_I_stddev"].extend(
# q_I.stddev.detach().cpu().ravel().tolist()
# )
# self.validation_preds["q_bg_mean"].extend(
# q_bg.mean.detach().cpu().ravel().tolist()
# )
# self.validation_preds["q_bg_stddev"].extend(
# q_bg.stddev.detach().cpu().ravel().tolist()
# )
# self.validation_preds["L_pred"].extend(L.detach().cpu())
# self.validation_preds["DIALS_I_prf_val"].extend(
# DIALS_I_prf_val.detach().cpu()
# )
# self.validation_preds["DIALS_I_prf_var"].extend(
# DIALS_I_prf_var.detach().cpu()
# )
# self.validation_preds["DIALS_I_sum_val"].extend(
# DIALS_I_sum_val.detach().cpu()
# )
# self.validation_preds["DIALS_I_sum_var"].extend(
# DIALS_I_sum_var.detach().cpu()
# )
# self.validation_preds["shape"].extend(shape.detach().cpu())
# self.validation_preds["refl_id"].extend(id.detach().cpu().numpy())
# self.validation_preds["tbl_id"].extend(tbl_id.detach().cpu().numpy())

# return loss

# def on_train_epoch_end(self):
# avg_loss = torch.mean(torch.tensor(self.training_step_loss, device=self.device))
# self.train_avg_loss.append(avg_loss.detach().cpu())
# self.training_step_loss.clear()

# def on_validation_epoch_end(self):
# avg_loss = torch.mean(
# torch.tensor(self.validation_step_loss, device=self.device)
# )
# self.validation_avg_loss.append(avg_loss.detach().cpu())
# self.validation_step_loss.clear()

# def configure_optimizers(self):
# optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
# return optimizer


class IntegratorModelSim(pytorch_lightning.LightningModule):
    """

    Attributes:
        standardize: standardization module
        lr: learning rate
        encoder: encoder module
        distribution_builder: distribution builder module
        likelihood: likelihood module
        train_avg_loss: list of average training losses
        validation_avg_loss: list of average validation losses
        training_step_loss: list of training step losses
        validation_step_loss: list of validation step losses
        training_preds: dictionary of training predictions
        validation_preds: dictionary of validation predictions
        total_steps: total number of iterations
        anneal: boolean flag to anneal the KL term
        current_step: current iteration
    """

    def __init__(
        self,
        encoder,
        distribution_builder,
        likelihood,
        standardize,
        total_steps,
        n_cycle=4,
        ratio=0.5,
        anneal=True,
        lr=0.001,
    ):
        super().__init__()
        self.standardize = standardize
        self.lr = lr
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
            "true_I": [],
            "true_L": [],
            "true_bg": [],
            "shape": [],
        }

        self.validation_preds = {
            "q_I_mean": [],
            "q_I_stddev": [],
            "q_bg_mean": [],
            "q_bg_stddev": [],
            "L_pred": [],
            "true_I": [],
            "true_L": [],
            "true_bg": [],
            "shape": [],
        }
        self.total_steps = total_steps + 1
        self.anneal = anneal
        if self.anneal:
            self.anneal_schedule = frange_cycle_cosine(
                0.0, 1.0, self.total_steps, n_cycle=n_cycle, ratio=ratio
            )
        self.current_step = 0

    def forward(self, shoebox, dead_pixel_mask, is_flat):
        counts = torch.clamp(shoebox[..., -1], min=0)
        shoebox_ = self.standardize(shoebox, dead_pixel_mask.squeeze(-1))
        dxyz = shoebox[..., 3:6]
        representation = self.encoder(shoebox_, dead_pixel_mask.unsqueeze(-1))
        q_bg, q_I, profile, L = self.distribution_builder(
            representation, dxyz, dead_pixel_mask, is_flat
        )
        ll, kl_term, rate = self.likelihood(counts, q_bg, q_I, profile)
        num_vox = dead_pixel_mask.sum(1)
        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)
        nll = -(torch.sum(ll_mean) / torch.sum(dead_pixel_mask))
        return nll, kl_term, rate, q_I, profile, q_bg, counts, L

    def training_step(self, batch):
        (
            sbox,
            mask,
            true_I,
            true_L,
            true_bg,
            is_flat,
            shape,
        ) = batch
        is_flat = is_flat.to(self.device)
        mask = mask.to(self.device)
        sbox = sbox.to(self.device)
        nll, kl_term, rate, q_I, profile, q_bg, counts, L = self(sbox, mask, is_flat)

        if self.anneal:
            anneal_rate = self.anneal_schedule[self.current_step]
        else:
            anneal_rate = 1.0

        self.current_step += 1

        loss = nll + anneal_rate * kl_term

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
            self.training_preds["true_I"].extend(true_I.detach().ravel().tolist())
            self.training_preds["true_L"].extend(true_L.detach().cpu())
            self.training_preds["true_bg"].extend(true_bg.detach().cpu())
            self.training_preds["shape"].extend(shape.detach().cpu())

        return loss

    def validation_step(self, batch):
        (
            sbox,
            mask,
            true_I,
            true_L,
            true_bg,
            is_flat,
            shape,
        ) = batch
        is_flat = is_flat.to(self.device)
        mask = mask.to(self.device)
        sbox = sbox.to(self.device)

        nll, kl_term, rate, q_I, profile, q_bg, counts, L = self(sbox, mask, is_flat)

        if self.anneal:
            anneal_rate = self.anneal_schedule[self.current_step]
        else:
            anneal_rate = 1.0

        loss = nll + anneal_rate * kl_term

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
            self.validation_preds["true_I"].extend(true_I.detach().ravel().tolist())
            self.validation_preds["true_L"].extend(true_L.detach().cpu())
            self.validation_preds["true_bg"].extend(true_bg.detach().cpu())
            self.validation_preds["shape"].extend(shape.detach().cpu())

        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.mean(torch.tensor(self.training_step_loss))
        self.train_avg_loss.append(avg_loss)
        self.training_step_loss.clear()

    def on_validation_epoch_end(self):
        avg_loss = torch.mean(torch.tensor(self.validation_step_loss))
        self.validation_avg_loss.append(avg_loss)
        self.validation_step_loss.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# %%
class IntegratorModel(pytorch_lightning.LightningModule):
    def __init__(
        self,
        encoder,
        distribution_builder,
        likelihood,
        standardize,
        total_steps,
        n_cycle=4,
        ratio=0.5,
        anneal=True,
        lr=0.001,
        max_epochs=10,
    ):
        super().__init__()
        self.lr = lr
        self.standardize = standardize
        self.encoder = encoder
        self.distribution_builder = distribution_builder
        self.likelihood = likelihood

        self.train_avg_loss = []
        self.validation_avg_loss = []
        self.max_epochs = max_epochs

        self.training_step_loss = []
        self.validation_step_loss = []

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
        self.anneal = anneal
        if self.anneal:
            self.anneal_schedule = frange_cycle_cosine(
                0.0, 1.0, self.total_steps, n_cycle=n_cycle, ratio=ratio
            )
        self.current_step = 0

    def forward(self, shoebox, dead_pixel_mask, is_flat):
        counts = torch.clamp(shoebox[..., -1], min=0)
        shoebox_ = self.standardize(shoebox, dead_pixel_mask.squeeze(-1))
        dxyz = shoebox[..., 3:6]
        representation = self.encoder(shoebox_, dead_pixel_mask.unsqueeze(-1))
        q_bg, q_I, profile, L, image_weights = self.distribution_builder(
            representation, dxyz, dead_pixel_mask, is_flat.squeeze(-1)
        )

        ll, kl_term, rate = self.likelihood(
            counts,
            q_bg,
            q_I,
            profile,
            image_weights,
            # mcsamples=100,
        )

        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)

        nll = -(torch.sum(ll_mean) / torch.sum(dead_pixel_mask))

        return nll, kl_term, rate, q_I, profile, q_bg, counts, L

    def training_step(self, batch):
        device = self.device
        (sbox, metadata, is_flat, mask) = batch
        sbox, mask, is_flat = sbox.to(device), mask.to(device), is_flat.to(device)

        nll, kl_term, rate, q_I, profile, q_bg, counts, L = self(sbox, mask, is_flat)

        if self.anneal:
            anneal_rate = self.anneal_schedule[self.current_step]
        else:
            anneal_rate = 1.0
        self.current_step += 1

        loss = nll + anneal_rate * kl_term

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

            self.training_preds["DIALS_I_prf_val"].extend(metadata[:, 2].detach().cpu())
            self.training_preds["DIALS_I_prf_var"].extend(metadata[:, 3].detach().cpu())

            self.training_preds["DIALS_I_sum_val"].extend(metadata[:, 0].detach().cpu())

            self.training_preds["DIALS_I_sum_var"].extend(metadata[:, 1].detach().cpu())

            # self.training_preds["shape"].extend(metadata[:, 6:].detach().cpu())

            self.training_preds["refl_id"].extend(metadata[:, 4].detach().cpu().numpy())

            # self.training_preds["tbl_id"].extend(metadata[:, 4].detach().cpu().numpy())

        return loss

    def validation_step(self, batch):
        device = self.device

        (sbox, metadata, is_flat, mask) = batch
        sbox, mask, is_flat = sbox.to(device), mask.to(device), is_flat.to(device)

        nll, kl_term, rate, q_I, profile, q_bg, counts, L = self(sbox, mask, is_flat)

        if self.anneal:
            anneal_rate = self.anneal_schedule[self.current_step]
        else:
            anneal_rate = 1.0

        loss = nll + anneal_rate * kl_term

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
