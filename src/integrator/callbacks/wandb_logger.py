import wandb
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np

# Import needed for colorbar positioning


# class UNetPlotter(Callback):
# def __init__(self, num_profiles=5, plot_every_n_epochs=5, d_vectors=None):
# super().__init__()
# self.train_predictions = {}
# self.val_predictions = {}
# self.num_profiles = num_profiles
# self.tracked_refl_ids = None
# self.all_seen_ids = set()
# self.tracked_predictions = {
# "qp": {},  # Profile prediction image
# "counts": {},  # Input counts image
# "qbg": {},  # Background (mean)
# "rates": {},  # Predicted rate image (I*p + bg)
# "intensity_mean": {},  # Overall intensity (q_I.mean × signal_prob)
# "dials_I_prf_value": {},  # DIALS profile intensity value
# }
# self.epoch_predictions = None
# self.plot_every_n_epochs = plot_every_n_epochs
# self.current_epoch = 0
# self.d_vectors = d_vectors

# def on_train_epoch_start(self, trainer, pl_module):
# self.epoch_predictions = {
# "qp": [],
# "counts": [],
# "refl_ids": [],
# "intensity_mean": [],
# "dials_I_prf_value": [],
# "qbg": [],
# "rates": [],
# }
# # Clear tracked predictions at start of epoch
# self.tracked_predictions = {
# "qp": {},
# "counts": {},
# "qbg": {},
# "rates": {},
# "intensity_mean": {},
# "dials_I_prf_value": {},
# }

# def update_tracked_predictions(
# self,
# qp_preds,
# qbg_preds,
# rates,
# count_preds,
# refl_ids,
# dials_I,
# intensity_mean,
# ):
# """
# Update tracked predictions for a set of reflection IDs.
# Expects:
# - qp_preds: a distribution with a `.mean` that can be reshaped to [batch, 3, 21, 21]
# - qbg_preds: a distribution with a `.mean`
# - rates: tensor of shape [batch, mc_samples, num_components]
# - count_preds: tensor of shape [batch, num_components]
# - refl_ids: tensor of reflection IDs (shape [batch])
# - dials_I: tensor (e.g., dials_I_prf_value) of shape [batch]
# - intensity_mean: tensor of shape [batch, 1] (overall intensity)
# """
# current_refl_ids = refl_ids.cpu().numpy()
# self.all_seen_ids.update(current_refl_ids)
# if self.tracked_refl_ids is None:
# self.tracked_refl_ids = sorted(list(self.all_seen_ids))[: self.num_profiles]
# print(
# f"Selected {self.num_profiles} reflection IDs to track: {self.tracked_refl_ids}"
# )

# # Reshape predictions to images (assumes 3 channels of 21x21 and we select the middle channel)
# qp_images = qp_preds.mean.reshape(-1, 3, 21, 21)[..., 1, :, :]
# count_images = count_preds.reshape(-1, 3, 21, 21)[..., 1, :, :]
# rate_images = rates.mean(1).reshape(-1, 3, 21, 21)[..., 1, :, :]
# # intensity_mean and signal_prob are assumed to be [batch, 1]; squeeze them to [batch]
# intensity_mean_val = (
# intensity_mean.squeeze(1) if intensity_mean.dim() > 1 else intensity_mean
# )

# for ref_id in self.tracked_refl_ids:
# matches = np.where(current_refl_ids == ref_id)[0]
# if len(matches) > 0:
# idx = matches[0]
# self.tracked_predictions["qp"][ref_id] = qp_images[idx].cpu()
# self.tracked_predictions["counts"][ref_id] = count_images[idx].cpu()
# self.tracked_predictions["rates"][ref_id] = rate_images[idx].cpu()
# self.tracked_predictions["qbg"][ref_id] = qbg_preds.mean[idx].cpu()
# self.tracked_predictions["intensity_mean"][ref_id] = intensity_mean_val[
# idx
# ].cpu()
# self.tracked_predictions["dials_I_prf_value"][ref_id] = dials_I[
# idx
# ].cpu()

# def create_comparison_grid(self, cmap="cividis"):
# if not self.tracked_refl_ids:
# return None

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import numpy as np
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(
# 3, self.num_profiles, figsize=(5 * self.num_profiles, 8)
# )
# if self.num_profiles == 1:
# axes = axes.reshape(-1, 1)

# for i, ref_id in enumerate(self.tracked_refl_ids):
# try:
# counts_data = self.tracked_predictions["counts"][ref_id]
# profile_data = self.tracked_predictions["qp"][ref_id]
# rates_data = self.tracked_predictions["rates"][ref_id]
# bg_value = float(self.tracked_predictions["qbg"][ref_id])
# intensity_val = float(
# self.tracked_predictions["intensity_mean"][ref_id]
# )
# dials_val = float(self.tracked_predictions["dials_I_prf_value"][ref_id])

# # Shared min/max for counts and rates
# vmin_13 = min(counts_data.min().item(), rates_data.min().item())
# vmax_13 = max(counts_data.max().item(), rates_data.max().item())

# # Row 1: Raw counts image
# im0 = axes[0, i].imshow(
# counts_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13
# )
# axes[0, i].set_title(f"ID: {ref_id}\nDIALS: {dials_val:.2f}")
# axes[0, i].set_ylabel("raw image", labelpad=5)
# axes[0, i].tick_params(
# left=False, bottom=False, labelleft=False, labelbottom=False
# )

# # Row 2: Profile prediction image
# im1 = axes[1, i].imshow(profile_data, cmap=cmap)
# axes[1, i].set_ylabel("profile", labelpad=5)
# axes[1, i].tick_params(
# left=False, bottom=False, labelleft=False, labelbottom=False
# )

# # Row 3: Rates image
# im2 = axes[2, i].imshow(
# rates_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13
# )
# axes[2, i].set_title(
# f"Bg: {bg_value:.2f} | I: {intensity_val:.2f}\nP(sig): "
# )
# axes[2, i].set_ylabel("rate = I*p + Bg", labelpad=5)
# axes[2, i].tick_params(
# left=False, bottom=False, labelleft=False, labelbottom=False
# )

# # Add colorbars
# for ax, im in zip(axes[:, i], [im0, im1, im2]):
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)
# except Exception as e:
# print(f"Error plotting reflection {ref_id}: {e}")
# for row in range(3):
# axes[row, i].text(
# 0.5,
# 0.5,
# f"Error plotting\nreflection {ref_id}",
# ha="center",
# va="center",
# transform=axes[row, i].transAxes,
# )
# plt.tight_layout()
# return fig

# def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
# import numpy as np

# with torch.no_grad():
# shoebox, dials, masks, metadata, counts = batch
# predictions = pl_module(shoebox, dials, masks, metadata, counts)
# # Update tracked predictions on epochs when plotting is scheduled
# if self.current_epoch % self.plot_every_n_epochs == 0:
# self.update_tracked_predictions(
# predictions["qp"],
# predictions["qbg"],
# predictions["rates"],
# predictions["counts"],
# predictions["refl_ids"],
# predictions["dials_I_prf_value"],
# predictions["intensity_mean"],
# )
# for key in self.epoch_predictions.keys():
# if key in predictions:
# self.epoch_predictions[key].append(predictions[key])
# self.train_predictions = predictions

# def on_train_epoch_end(self, trainer, pl_module):
# if self.train_predictions:
# import wandb

# data = []
# # Use intensity_mean, dials_I_prf_value, and signal_prob for table data.
# intensity_data = self.train_predictions["intensity_mean"]
# dials_data = self.train_predictions["dials_I_prf_value"]
# intensity_flat = intensity_data.flatten()
# dials_flat = dials_data.flatten()

# for i in range(len(intensity_flat)):
# try:
# data.append(
# [
# float(torch.log(intensity_flat[i] + 1e-8)),
# float(torch.log(dials_flat[i] + 1e-8)),
# ]
# )
# except Exception as e:
# print(f"Error creating data point {i}: {e}")
# table = wandb.Table(
# data=data,
# columns=["log_intensity", "log_dials_I_prf"],
# )
# try:
# stacked = torch.vstack([intensity_flat, dials_flat])
# corr = torch.corrcoef(stacked)[0, 1].item()
# except Exception as e:
# print(f"Error calculating correlation: {e}")
# corr = float("nan")
# log_dict = {
# "train_intensity_vs_prf": wandb.plot.scatter(
# table,
# "log_intensity",
# "log_dials_I_prf",
# ),
# "corrcoef_intensity": corr,
# }
# try:
# log_dict["mean_intensity"] = torch.mean(intensity_flat).item()
# except Exception as e:
# print(f"Error calculating mean_intensity: {e}")
# try:
# if "qbg" in self.train_predictions and hasattr(
# self.train_predictions["qbg"], "mean"
# ):
# bg_mean = self.train_predictions["qbg"].mean
# if callable(bg_mean):
# bg_mean = bg_mean()
# log_dict["mean_bg"] = torch.mean(bg_mean.detach().cpu()).item()
# except Exception as e:
# print(f"Error calculating mean_bg: {e}")
# if self.current_epoch % self.plot_every_n_epochs == 0:
# comparison_fig = self.create_comparison_grid()
# if comparison_fig is not None:
# log_dict["profile_comparisons"] = wandb.Image(comparison_fig)
# import matplotlib.pyplot as plt

# plt.close(comparison_fig)
# wandb.log(log_dict)
# self.current_epoch += 1

# def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
# with torch.no_grad():
# shoebox, dials, masks, metadata, counts = batch
# self.val_predictions = pl_module(shoebox, dials, masks, metadata, counts)


class UNetPlotter2(Callback):
    def __init__(self, num_profiles=5, plot_every_n_epochs=5, d_vectors=None):
        super().__init__()
        self.train_predictions = {}
        self.val_predictions = {}
        self.num_profiles = num_profiles
        self.tracked_refl_ids = None
        self.all_seen_ids = set()
        self.tracked_predictions = {
            "qp": {},  # Profile prediction image
            "counts": {},  # Input counts image
            "qbg": {},  # Background (mean)
            "rates": {},  # Predicted rate image (I*p + bg)
            "intensity_mean": {},  # Overall intensity (q_I.mean × signal_prob)
            "dials_I_prf_value": {},  # DIALS profile intensity value
            "signal_prob": {},  # Signal existence probability (from q_z)
        }
        self.epoch_predictions = None
        self.plot_every_n_epochs = plot_every_n_epochs
        self.current_epoch = 0
        self.d_vectors = d_vectors

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_predictions = {
            "qp": [],
            "counts": [],
            "refl_ids": [],
            "intensity_mean": [],
            "dials_I_prf_value": [],
            "qbg": [],
            "rates": [],
            "signal_prob": [],
        }
        # Clear tracked predictions at start of epoch
        self.tracked_predictions = {
            "qp": {},
            "counts": {},
            "qbg": {},
            "rates": {},
            "intensity_mean": {},
            "dials_I_prf_value": {},
            "signal_prob": {},
        }

    def update_tracked_predictions(
        self,
        qp_preds,
        qbg_preds,
        rates,
        count_preds,
        refl_ids,
        dials_I,
        intensity_mean,
        signal_prob,
    ):
        """
        Update tracked predictions for a set of reflection IDs.
        Expects:
          - qp_preds: a distribution with a `.mean` that can be reshaped to [batch, 3, 21, 21]
          - qbg_preds: a distribution with a `.mean`
          - rates: tensor of shape [batch, mc_samples, num_components]
          - count_preds: tensor of shape [batch, num_components]
          - refl_ids: tensor of reflection IDs (shape [batch])
          - dials_I: tensor (e.g., dials_I_prf_value) of shape [batch]
          - intensity_mean: tensor of shape [batch, 1] (overall intensity)
          - signal_prob: tensor of shape [batch, 1]
        """
        current_refl_ids = refl_ids.cpu().numpy()
        self.all_seen_ids.update(current_refl_ids)
        if self.tracked_refl_ids is None:
            self.tracked_refl_ids = sorted(list(self.all_seen_ids))[: self.num_profiles]
            print(
                f"Selected {self.num_profiles} reflection IDs to track: {self.tracked_refl_ids}"
            )

        # Reshape predictions to images (assumes 3 channels of 21x21 and we select the middle channel)
        qp_images = qp_preds.mean.reshape(-1, 3, 21, 21)[..., 1, :, :]
        count_images = count_preds.reshape(-1, 3, 21, 21)[..., 1, :, :]
        rate_images = rates.mean(1).reshape(-1, 3, 21, 21)[..., 1, :, :]
        # intensity_mean and signal_prob are assumed to be [batch, 1]; squeeze them to [batch]
        intensity_mean_val = (
            intensity_mean.squeeze(1) if intensity_mean.dim() > 1 else intensity_mean
        )
        signal_prob_val = (
            signal_prob.squeeze(1) if signal_prob.dim() > 1 else signal_prob
        )

        for ref_id in self.tracked_refl_ids:
            matches = np.where(current_refl_ids == ref_id)[0]
            if len(matches) > 0:
                idx = matches[0]
                self.tracked_predictions["qp"][ref_id] = qp_images[idx].cpu()
                self.tracked_predictions["counts"][ref_id] = count_images[idx].cpu()
                self.tracked_predictions["rates"][ref_id] = rate_images[idx].cpu()
                self.tracked_predictions["qbg"][ref_id] = qbg_preds.mean[idx].cpu()
                self.tracked_predictions["intensity_mean"][ref_id] = intensity_mean_val[
                    idx
                ].cpu()
                self.tracked_predictions["dials_I_prf_value"][ref_id] = dials_I[
                    idx
                ].cpu()
                self.tracked_predictions["signal_prob"][ref_id] = signal_prob_val[
                    idx
                ].cpu()

    def create_comparison_grid(self, cmap="cividis"):
        if not self.tracked_refl_ids:
            return None

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import numpy as np
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            3, self.num_profiles, figsize=(5 * self.num_profiles, 8)
        )
        if self.num_profiles == 1:
            axes = axes.reshape(-1, 1)

        for i, ref_id in enumerate(self.tracked_refl_ids):
            try:
                counts_data = self.tracked_predictions["counts"][ref_id]
                profile_data = self.tracked_predictions["qp"][ref_id]
                rates_data = self.tracked_predictions["rates"][ref_id]
                bg_value = float(self.tracked_predictions["qbg"][ref_id])
                intensity_val = float(
                    self.tracked_predictions["intensity_mean"][ref_id]
                )
                sig_prob = float(self.tracked_predictions["signal_prob"][ref_id])
                dials_val = float(self.tracked_predictions["dials_I_prf_value"][ref_id])

                # Shared min/max for counts and rates
                vmin_13 = min(counts_data.min().item(), rates_data.min().item())
                vmax_13 = max(counts_data.max().item(), rates_data.max().item())

                # Row 1: Raw counts image
                im0 = axes[0, i].imshow(
                    counts_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13
                )
                axes[0, i].set_title(f"ID: {ref_id}\nDIALS: {dials_val:.2f}")
                axes[0, i].set_ylabel("raw image", labelpad=5)
                axes[0, i].tick_params(
                    left=False, bottom=False, labelleft=False, labelbottom=False
                )

                # Row 2: Profile prediction image
                im1 = axes[1, i].imshow(profile_data, cmap=cmap)
                axes[1, i].set_ylabel("profile", labelpad=5)
                axes[1, i].tick_params(
                    left=False, bottom=False, labelleft=False, labelbottom=False
                )

                # Row 3: Rates image
                im2 = axes[2, i].imshow(
                    rates_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13
                )
                axes[2, i].set_title(
                    f"Bg: {bg_value:.2f} | I: {intensity_val:.2f}\nP(sig): {sig_prob:.2f}"
                )
                axes[2, i].set_ylabel("rate = I*p + Bg", labelpad=5)
                axes[2, i].tick_params(
                    left=False, bottom=False, labelleft=False, labelbottom=False
                )

                # Add colorbars
                for ax, im in zip(axes[:, i], [im0, im1, im2]):
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)
            except Exception as e:
                print(f"Error plotting reflection {ref_id}: {e}")
                for row in range(3):
                    axes[row, i].text(
                        0.5,
                        0.5,
                        f"Error plotting\nreflection {ref_id}",
                        ha="center",
                        va="center",
                        transform=axes[row, i].transAxes,
                    )
        plt.tight_layout()
        return fig

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        import numpy as np

        with torch.no_grad():
            shoebox, dials, masks, metadata, counts = batch
            predictions = pl_module(shoebox, dials, masks, metadata, counts)
            # Update tracked predictions on epochs when plotting is scheduled
            if self.current_epoch % self.plot_every_n_epochs == 0:
                self.update_tracked_predictions(
                    predictions["qp"],
                    predictions["qbg"],
                    predictions["rates"],
                    predictions["counts"],
                    predictions["refl_ids"],
                    predictions["dials_I_prf_value"],
                    predictions["intensity_mean"],
                    predictions["signal_prob"],
                )
            for key in self.epoch_predictions.keys():
                if key in predictions:
                    self.epoch_predictions[key].append(predictions[key])
            self.train_predictions = predictions

    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_predictions:
            import wandb

            data = []
            # Use intensity_mean, dials_I_prf_value, and signal_prob for table data.
            intensity_data = self.train_predictions["intensity_mean"]
            dials_data = self.train_predictions["dials_I_prf_value"]
            signal_prob_data = self.train_predictions["signal_prob"]
            intensity_flat = intensity_data.flatten()
            dials_flat = dials_data.flatten()
            signal_flat = signal_prob_data.flatten()
            for i in range(len(intensity_flat)):
                try:
                    data.append(
                        [
                            float(torch.log(intensity_flat[i] + 1e-8)),
                            float(torch.log(dials_flat[i] + 1e-8)),
                            float(signal_flat[i]),
                        ]
                    )
                except Exception as e:
                    print(f"Error creating data point {i}: {e}")
            table = wandb.Table(
                data=data,
                columns=["log_intensity", "log_dials_I_prf", "signal_probability"],
            )
            try:
                stacked = torch.vstack([intensity_flat, dials_flat])
                corr = torch.corrcoef(stacked)[0, 1].item()
            except Exception as e:
                print(f"Error calculating correlation: {e}")
                corr = float("nan")
            log_dict = {
                "train_intensity_vs_prf": wandb.plot.scatter(
                    table, "log_intensity", "log_dials_I_prf", "signal_probability"
                ),
                "corrcoef_intensity": corr,
            }
            try:
                log_dict["mean_signal_prob"] = torch.mean(signal_flat).item()
            except Exception as e:
                print(f"Error calculating mean_signal_prob: {e}")
            try:
                log_dict["mean_intensity"] = torch.mean(intensity_flat).item()
            except Exception as e:
                print(f"Error calculating mean_intensity: {e}")
            try:
                if "qbg" in self.train_predictions and hasattr(
                    self.train_predictions["qbg"], "mean"
                ):
                    bg_mean = self.train_predictions["qbg"].mean
                    if callable(bg_mean):
                        bg_mean = bg_mean()
                    log_dict["mean_bg"] = torch.mean(bg_mean.detach().cpu()).item()
            except Exception as e:
                print(f"Error calculating mean_bg: {e}")
            if self.current_epoch % self.plot_every_n_epochs == 0:
                comparison_fig = self.create_comparison_grid()
                if comparison_fig is not None:
                    log_dict["profile_comparisons"] = wandb.Image(comparison_fig)
                    import matplotlib.pyplot as plt

                    plt.close(comparison_fig)
            wandb.log(log_dict)
        self.current_epoch += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            shoebox, dials, masks, metadata, counts = batch
            self.val_predictions = pl_module(shoebox, dials, masks, metadata, counts)


# NOTE: correct
class IntensityPlotter(Callback):
    def __init__(self, num_profiles=5, plot_every_n_epochs=5, d_vectors=None):
        super().__init__()
        self.train_predictions = {}
        self.val_predictions = {}
        self.num_profiles = num_profiles
        self.tracked_refl_ids = None
        self.all_seen_ids = set()
        self.tracked_predictions = {
            "qp": {},
            "counts": {},
            "qbg": {},
            "rates": {},
        }
        self.epoch_predictions = None
        self.plot_every_n_epochs = plot_every_n_epochs
        self.current_epoch = 0  # Track current epoch
        self.d_vectors = d_vectors

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_predictions = {
            "qp": [],
            "counts": [],
            "refl_ids": [],
            "qI": [],
            "dials_I_prf_value": [],
            "weighted_sum_mean": [],
            "thresholded_mean": [],
            "qbg": [],
            "rates": [],
        }
        # Clear tracked predictions at start of epoch
        self.tracked_predictions = {
            "qp": {},
            "counts": {},
            "qbg": {},
            "rates": {},
            "qI": {},
            "dials_I_prf_value": {},
        }

    def update_tracked_predictions(
        self, qp_preds, qbg_preds, rates, count_preds, refl_ids, dials_I, qI
    ):
        current_refl_ids = refl_ids.cpu().numpy()

        # Update all seen IDs and set tracked IDs if not set
        self.all_seen_ids.update(current_refl_ids)
        if self.tracked_refl_ids is None:
            self.tracked_refl_ids = sorted(list(self.all_seen_ids))[: self.num_profiles]
            print(
                f"Selected {self.num_profiles} reflection IDs to track: {self.tracked_refl_ids}"
            )

        # Get indices of tracked reflections in current batch
        qp_images = qp_preds.mean.reshape(-1, 3, 21, 21)[..., 1, :, :]
        count_images = count_preds.reshape(-1, 3, 21, 21)[..., 1, :, :]
        rate_images = rates.mean(1).reshape(-1, 3, 21, 21)[..., 1, :, :]
        bg_mean = qbg_preds.mean
        qI_mean = qI.mean
        dials_I_prf_value = dials_I

        for ref_id in self.tracked_refl_ids:
            matches = np.where(current_refl_ids == ref_id)[0]
            if len(matches) > 0:
                idx = matches[0]

                self.tracked_predictions["qp"][ref_id] = qp_images[idx].cpu()
                self.tracked_predictions["counts"][ref_id] = count_images[idx].cpu()
                self.tracked_predictions["rates"][ref_id] = rate_images[idx].cpu()
                self.tracked_predictions["qbg"][ref_id] = bg_mean[idx].cpu()
                self.tracked_predictions["qI"][ref_id] = qI_mean[idx].cpu()
                self.tracked_predictions["dials_I_prf_value"][
                    ref_id
                ] = dials_I_prf_value[idx]

    def create_comparison_grid(
        self,
        cmap="cividis",
    ):
        if not self.tracked_refl_ids:
            return None

        # Import needed for colorbar positioning
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import numpy as np

        # Create figure with proper subplot layout
        fig, axes = plt.subplots(
            3, self.num_profiles, figsize=(5 * self.num_profiles, 8)
        )

        # Handle case where only one column
        if self.num_profiles == 1:
            axes = axes.reshape(-1, 1)

        # Plot each column
        for i, refl_id in enumerate(self.tracked_refl_ids):
            # Get data for this column
            counts_data = self.tracked_predictions["counts"][refl_id]
            profile_data = self.tracked_predictions["qp"][refl_id]
            rates_data = self.tracked_predictions["rates"][refl_id]

            # Calculate shared min/max for rows 1 and 3
            vmin_13 = min(counts_data.min().item(), rates_data.min().item())
            vmax_13 = max(counts_data.max().item(), rates_data.max().item())

            # Row 1: Input counts
            im0 = axes[0, i].imshow(counts_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13)
            axes[0, i].set_title(
                f"reflection ID: {refl_id}\n DIALS I: {self.tracked_predictions['dials_I_prf_value'][refl_id]:.2f}"
            )
            axes[0, i].set_ylabel("raw image", labelpad=5)

            # Turn off axes but keep the labels
            axes[0, i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Row 2: QP prediction (with its own scale)
            im1 = axes[1, i].imshow(profile_data, cmap=cmap)
            axes[1, i].set_ylabel("profile", labelpad=5)
            axes[1, i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Row 3: Rates (same scale as row 1)
            im2 = axes[2, i].imshow(rates_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13)
            axes[2, i].set_title(
                f"Bg: {self.tracked_predictions['qbg'][refl_id]:.2f}\n qI: {self.tracked_predictions['qI'][refl_id]:.2f}"
            )

            axes[2, i].set_ylabel("rate = I*pij + Bg", labelpad=5)
            axes[2, i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Add colorbars
            # First row colorbar (same as third row)
            divider0 = make_axes_locatable(axes[0, i])
            cax0 = divider0.append_axes("right", size="5%", pad=0.05)
            cbar0 = plt.colorbar(im0, cax=cax0)
            cbar0.ax.tick_params(labelsize=8)

            # Second row colorbar (independent)
            divider1 = make_axes_locatable(axes[1, i])
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            cbar1 = plt.colorbar(im1, cax=cax1)
            cbar1.ax.tick_params(labelsize=8)

            # Third row colorbar (same as first row)
            divider2 = make_axes_locatable(axes[2, i])
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            cbar2 = plt.colorbar(im2, cax=cax2)
            cbar2.ax.tick_params(labelsize=8)

        plt.tight_layout()

        return fig

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            shoebox, dials, masks, metadata, counts = batch
            predictions = pl_module(shoebox, dials, masks, metadata, counts)

            # Only update tracked predictions if we're going to plot this epoch
            if self.current_epoch % self.plot_every_n_epochs == 0:
                self.update_tracked_predictions(
                    predictions["qp"],
                    predictions["qbg"],
                    predictions["rates"],
                    predictions["counts"],
                    predictions["refl_ids"],
                    predictions["dials_I_prf_value"],
                    predictions["qI"],
                )

            # Accumulate predictions
            for key in self.epoch_predictions.keys():
                if key in predictions:
                    self.epoch_predictions[key].append(predictions[key])

            self.train_predictions = predictions  # Keep last batch for other metrics

    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_predictions:
            # Original scatter plot data
            data = [
                [qi, prf, weighted_sum, thresholded_mean, simpson_idx]
                for qi, prf, weighted_sum, thresholded_mean, simpson_idx in zip(
                    torch.log(self.train_predictions["qI"].mean.flatten()),
                    torch.log(self.train_predictions["dials_I_prf_value"].flatten()),
                    torch.log(self.train_predictions["weighted_sum_mean"].flatten()),
                    torch.log(self.train_predictions["thresholded_mean"].flatten()),
                    torch.linalg.norm(self.train_predictions["qp"].mean, dim=-1).pow(2),
                )
            ]
            table = wandb.Table(
                data=data,
                columns=[
                    "qI",
                    "dials_I_prf_value",
                    "weighted_sum_mean",
                    "thresholded_mean",
                    "simpson_idx",
                ],
            )

            # Create log dictionary with metrics that we want to log every epoch
            log_dict = {
                "train_qI_vs_prf": wandb.plot.scatter(
                    table,
                    "qI",
                    "dials_I_prf_value",
                ),
                "train_weighted_sum_vs_prf": wandb.plot.scatter(
                    table,
                    "weighted_sum_mean",
                    "dials_I_prf_value",
                ),
                "train_thresholded_vs_prf": wandb.plot.scatter(
                    table,
                    "thresholded_mean",
                    "dials_I_prf_value",
                ),
                "corrcoef qI": torch.corrcoef(
                    torch.vstack(
                        [
                            self.train_predictions["qI"].mean.flatten(),
                            self.train_predictions["dials_I_prf_value"].flatten(),
                        ]
                    )
                )[0, 1],
                "corrcoef_weighted": torch.corrcoef(
                    torch.vstack(
                        [
                            self.train_predictions["weighted_sum_mean"].flatten(),
                            self.train_predictions["dials_I_prf_value"].flatten(),
                        ]
                    )
                )[0, 1],
                "corrcoef_masked": torch.corrcoef(
                    torch.vstack(
                        [
                            self.train_predictions["thresholded_mean"].flatten(),
                            self.train_predictions["dials_I_prf_value"].flatten(),
                        ]
                    )
                )[0, 1],
                "max_qI": torch.max(self.train_predictions["qI"].mean.flatten()),
                "mean_qI": torch.mean(self.train_predictions["qI"].mean.flatten()),
                "mean_bg": torch.mean(self.train_predictions["qbg"].mean),
            }

            # Only create and log comparison grid on specified epochs
            if self.current_epoch % self.plot_every_n_epochs == 0:
                comparison_fig = self.create_comparison_grid()
                if comparison_fig is not None:
                    log_dict["profile_comparisons"] = wandb.Image(comparison_fig)
                    plt.close(comparison_fig)

            wandb.log(log_dict)

        # Increment epoch counter
        self.current_epoch += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            shoebox, dials, masks, metadata, counts = batch
            self.val_predictions = pl_module(shoebox, dials, masks, metadata, counts)


# %%
class UNetPlotter(Callback):
    def __init__(self, num_profiles=5, plot_every_n_epochs=5, d_vectors=None):
        super().__init__()
        self.train_predictions = {}
        self.val_predictions = {}
        self.num_profiles = num_profiles
        self.tracked_refl_ids = None
        self.all_seen_ids = set()
        self.tracked_predictions = {
            "profile": {},
            "counts": {},
            "qbg": {},
            "rates": {},
        }
        self.epoch_predictions = None
        self.plot_every_n_epochs = plot_every_n_epochs
        self.current_epoch = 0
        self.d_vectors = d_vectors

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_predictions = {
            "profile": [],
            "counts": [],
            "refl_ids": [],
            "intensity_mean": [],
            "intensity_var": [],
            "dials_I_prf_value": [],
            "qbg": [],
            "rates": [],
        }

        self.tracked_predictions = {
            "profile": {},
            "counts": {},
            "qbg": {},
            "rates": {},
            "intensity_mean": {},
            "intensity_var": {},
            "qbg_var": {},
            "dials_I_prf_value": {},
        }

    def update_tracked_predictions(
        self,
        profile_preds,
        qbg_preds,
        rates,
        count_preds,
        refl_ids,
        dials_I,
        intensity_mean,
        intensity_var,
    ):
        current_refl_ids = refl_ids.cpu().numpy()

        # Update all seen IDs and set tracked IDs if not set
        self.all_seen_ids.update(current_refl_ids)
        if self.tracked_refl_ids is None:
            self.tracked_refl_ids = sorted(list(self.all_seen_ids))[: self.num_profiles]
            print(
                f"Selected {self.num_profiles} reflection IDs to track: {self.tracked_refl_ids}"
            )

        profile_images = profile_preds.reshape(-1, 3, 21, 21)[..., 1, :, :]
        count_images = count_preds.reshape(-1, 3, 21, 21)[..., 1, :, :]
        rate_images = rates.mean(1).reshape(-1, 3, 21, 21)[..., 1, :, :]
        bg_mean = qbg_preds.mean
        bg_var = qbg_preds.variance
        dials_I_prf_value = dials_I

        for ref_id in self.tracked_refl_ids:
            matches = np.where(current_refl_ids == ref_id)[0]
            if len(matches) > 0:
                idx = matches[0]

                self.tracked_predictions["profile"][ref_id] = profile_images[idx].cpu()
                self.tracked_predictions["counts"][ref_id] = count_images[idx].cpu()
                self.tracked_predictions["rates"][ref_id] = rate_images[idx].cpu()
                self.tracked_predictions["qbg"][ref_id] = bg_mean[idx].cpu()
                self.tracked_predictions["intensity_mean"][ref_id] = intensity_mean[
                    idx
                ].cpu()
                self.tracked_predictions["intensity_var"][ref_id] = intensity_var[
                    idx
                ].cpu()
                self.tracked_predictions["qbg_var"][ref_id] = bg_var[idx].cpu()
                self.tracked_predictions["dials_I_prf_value"][
                    ref_id
                ] = dials_I_prf_value[idx]

        torch.cuda.empty_cache()

    def create_comparison_grid(
        self,
        cmap="cividis",
    ):
        if not self.tracked_refl_ids:
            return None

        # Create figure with proper subplot layout
        fig, axes = plt.subplots(
            3, self.num_profiles, figsize=(5 * self.num_profiles, 8)
        )

        # Handle case where only one column
        if self.num_profiles == 1:
            axes = axes.reshape(-1, 1)

        # Plot each column
        for i, refl_id in enumerate(self.tracked_refl_ids):
            # Get data for this column
            counts_data = self.tracked_predictions["counts"][refl_id]
            profile_data = self.tracked_predictions["profile"][refl_id]
            rates_data = self.tracked_predictions["rates"][refl_id]

            vmin_13 = min(counts_data.min().item(), rates_data.min().item())
            vmax_13 = max(counts_data.max().item(), rates_data.max().item())

            # Row 1: Input counts
            im0 = axes[0, i].imshow(counts_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13)
            axes[0, i].set_title(
                f"reflection ID: {refl_id}\n DIALS I_prf: {self.tracked_predictions['dials_I_prf_value'][refl_id]:.2f}"
            )
            axes[0, i].set_ylabel("raw image", labelpad=5)

            # Turn off axes but keep the labels
            axes[0, i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Row 2: Profile prediction (with its own scale)
            im1 = axes[1, i].imshow(profile_data, cmap=cmap)
            axes[1, i].set_ylabel("profile", labelpad=5)
            axes[1, i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Row 3: Rates (same scale as row 1)
            im2 = axes[2, i].imshow(rates_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13)
            axes[2, i].set_title(
                f"Bg: {self.tracked_predictions['qbg'][refl_id]:.2f}\n I: {self.tracked_predictions['intensity_mean'][refl_id]:.2f}"
            )

            axes[2, i].set_ylabel("rate = I*pij + Bg", labelpad=5)
            axes[2, i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Add colorbars
            # First row colorbar (same as third row)
            divider0 = make_axes_locatable(axes[0, i])
            cax0 = divider0.append_axes("right", size="5%", pad=0.05)
            cbar0 = plt.colorbar(im0, cax=cax0)
            cbar0.ax.tick_params(labelsize=8)

            # Second row colorbar (independent)
            divider1 = make_axes_locatable(axes[1, i])
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            cbar1 = plt.colorbar(im1, cax=cax1)
            cbar1.ax.tick_params(labelsize=8)

            # Third row colorbar (same as first row)
            divider2 = make_axes_locatable(axes[2, i])
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            cbar2 = plt.colorbar(im2, cax=cax2)
            cbar2.ax.tick_params(labelsize=8)

        plt.tight_layout()

        return fig

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            # 1) Forward pass (no intensities yet)
            shoebox, dials, masks, metadata, counts = batch
            base_output = pl_module(shoebox, dials, masks, metadata, counts)

            # 2) Call calculate_intensities with the relevant fields
            intensities = pl_module.calculate_intensities(
                counts=base_output["counts"],
                qbg=base_output["qbg"],
                qp=base_output["qp"],
                dead_pixel_mask=base_output["masks"],
            )

            predictions = {
                **base_output,
                "weighted_sum_mean": intensities["weighted_sum_mean"],
                "weighted_sum_var": intensities["weighted_sum_var"],
                "thresholded_mean": intensities["thresholded_mean"],
                "thresholded_var": intensities["thresholded_var"],
            }

            if self.current_epoch % self.plot_every_n_epochs == 0:
                self.update_tracked_predictions(
                    predictions["qp"].mean,
                    predictions["qbg"],
                    predictions["rates"],
                    predictions["counts"],
                    predictions["refl_ids"],
                    predictions["dials_I_prf_value"],
                    predictions["intensity_mean"],
                    predictions["intensity_var"],
                )

            # Create CPU tensor versions to avoid keeping GPU memory
            self.train_predictions = {}
            for key in [
                "intensity_mean",
                "intensity_var",
                "dials_I_prf_value",
                "weighted_sum_mean",
                "thresholded_mean",
                "profile",
                "qbg",
            ]:
                if key in predictions:
                    if key == "profile":
                        self.train_predictions[key] = predictions[key].detach().cpu()
                    elif hasattr(predictions[key], "sample"):
                        self.train_predictions[key] = (
                            predictions[key].mean.detach().cpu()
                        )
                    else:
                        self.train_predictions[key] = predictions[key].detach().cpu()

            # Clean up
            del base_output, predictions
            torch.cuda.empty_cache()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_predictions:
            try:
                # Create data for scatter plots
                data = []

                I_flat = self.train_predictions["intensity_mean"].flatten() + 1e-8
                I_var_flat = self.train_predictions["intensity_var"].flatten() + 1e-8
                dials_flat = (
                    self.train_predictions["dials_I_prf_value"].flatten() + 1e-8
                )
                weighted_sum_flat = (
                    self.train_predictions["weighted_sum_mean"].flatten() + 1e-8
                )
                thresholded_flat = (
                    self.train_predictions["thresholded_mean"].flatten() + 1e-8
                )

                # Calculate simpson index from profile
                if "profile" in self.train_predictions:
                    profile_flat = self.train_predictions["profile"]
                    simpson_flat = torch.sum(profile_flat**2, dim=-1).flatten()
                else:
                    simpson_flat = torch.ones_like(I_flat)

                # Create data points with safe log transform
                for i in range(len(I_flat)):
                    try:
                        data.append(
                            [
                                float(torch.log(I_flat[i])),
                                float(torch.log(I_var_flat[i])),
                                float(torch.log(dials_flat[i])),
                                float(torch.log(weighted_sum_flat[i])),
                                float(torch.log(thresholded_flat[i])),
                                float(simpson_flat[i]),
                            ]
                        )
                    except Exception as e:
                        # Skip any problematic data points
                        pass

                # Create table
                table = wandb.Table(
                    data=data,
                    columns=[
                        "intensity_mean",
                        "intensity_var",
                        "dials_I_prf_value",
                        "weighted_sum_mean",
                        "thresholded_mean",
                        "simpson_idx",
                    ],
                )

                # Calculate correlation coefficients safely
                corr_I = (
                    torch.corrcoef(torch.vstack([I_flat, dials_flat]))[0, 1]
                    if len(I_flat) > 1
                    else 0
                )

                corr_masked = (
                    torch.corrcoef(torch.vstack([thresholded_flat, dials_flat]))[0, 1]
                    if len(thresholded_flat) > 1
                    else 0
                )

                # Create log dictionary
                log_dict = {
                    "train_I_vs_prf": wandb.plot.scatter(
                        table, "intensity_mean", "dials_I_prf_value"
                    ),
                    "train_thresholded_vs_prf": wandb.plot.scatter(
                        table, "thresholded_mean", "dials_I_prf_value"
                    ),
                    "corrcoef_I": corr_I,
                    "corrcoef_masked": corr_masked,
                    "max_I": torch.max(I_flat),
                    "mean_I": torch.mean(I_flat),
                    "mean_I_var": torch.mean(I_var_flat),
                    "min_I_var": torch.min(I_var_flat),
                    "max_I_var": torch.max(I_var_flat),
                }

                # Add mean background if available
                if "qbg" in self.train_predictions:
                    log_dict["mean_bg"] = torch.mean(self.train_predictions["qbg"])
                    log_dict["min_bg"] = torch.min(self.train_predictions["qbg"])
                    log_dict["max_bg"] = torch.max(self.train_predictions["qbg"])

                # Only create and log comparison grid on specified epochs
                if self.current_epoch % self.plot_every_n_epochs == 0:
                    comparison_fig = self.create_comparison_grid()

                    if comparison_fig is not None:
                        log_dict["profile_comparisons"] = wandb.Image(comparison_fig)
                        plt.close(comparison_fig)

                # Log metrics
                wandb.log(log_dict)

            except Exception as e:
                print(f"Error in on_train_epoch_end: {e}")

            # Clear memory
            self.train_predictions = {}
            torch.cuda.empty_cache()

        # Increment epoch counter
        self.current_epoch += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only track the last validation batch to save memory

        with torch.no_grad():
            shoebox, dials, masks, metadata, counts = batch
            base_output = pl_module(shoebox, dials, masks, metadata, counts)

            # Store only minimal data needed for metrics
            self.val_predictions = {}
            for key in [
                "intensity_mean",
                "intensity_var",
                "dials_I_prf_value",
                "weighted_sum_mean",
                "thresholded_mean",
            ]:
                if key in base_output:
                    if hasattr(base_output[key], "sample"):
                        self.val_predictions[key] = base_output[key].mean.detach().cpu()
                    else:
                        self.val_predictions[key] = base_output[key].detach().cpu()
                elif key in base_output:
                    self.val_predictions[key] = base_output[key].detach().cpu()

            # Clean up
            del base_output
            torch.cuda.empty_cache()


# %%
class tempMVNPlotter(Callback):
    def __init__(self, num_profiles=5, plot_every_n_epochs=5, d_vectors=None):
        super().__init__()
        self.train_predictions = {}
        self.val_predictions = {}
        self.num_profiles = num_profiles
        self.tracked_refl_ids = None
        self.all_seen_ids = set()
        self.tracked_predictions = {
            "profile": {},
            "counts": {},
            "qbg": {},
            "rates": {},
        }
        self.epoch_predictions = None
        self.plot_every_n_epochs = plot_every_n_epochs
        self.current_epoch = 0
        self.d_vectors = d_vectors

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_predictions = {
            "profile": [],
            "counts": [],
            "refl_ids": [],
            "qI": [],
            "dials_I_prf_value": [],
            "weighted_sum_mean": [],
            "thresholded_mean": [],
            "qbg": [],
            "rates": [],
        }

        self.tracked_predictions = {
            "profile": {},
            "counts": {},
            "qbg": {},
            "rates": {},
            "qI": {},
            # "qI_var": {},
            "qbg_var": {},
            "dials_I_prf_value": {},
        }

    def update_tracked_predictions(
        self, profile_preds, qbg_preds, rates, count_preds, refl_ids, dials_I, qI
    ):
        current_refl_ids = refl_ids.cpu().numpy()

        # Update all seen IDs and set tracked IDs if not set
        self.all_seen_ids.update(current_refl_ids)
        if self.tracked_refl_ids is None:
            self.tracked_refl_ids = sorted(list(self.all_seen_ids))[: self.num_profiles]
            print(
                f"Selected {self.num_profiles} reflection IDs to track: {self.tracked_refl_ids}"
            )

        profile_images = profile_preds.reshape(-1, 3, 21, 21)[..., 1, :, :]
        count_images = count_preds.reshape(-1, 3, 21, 21)[..., 1, :, :]
        rate_images = rates.mean(1).reshape(-1, 3, 21, 21)[..., 1, :, :]
        bg_mean = qbg_preds.mean
        bg_var = qbg_preds.variance
        qI_mean = qI.mean
        # qI_var = qI.variance
        dials_I_prf_value = dials_I

        for ref_id in self.tracked_refl_ids:
            matches = np.where(current_refl_ids == ref_id)[0]
            if len(matches) > 0:
                idx = matches[0]

                self.tracked_predictions["profile"][ref_id] = profile_images[idx].cpu()
                self.tracked_predictions["counts"][ref_id] = count_images[idx].cpu()
                self.tracked_predictions["rates"][ref_id] = rate_images[idx].cpu()
                self.tracked_predictions["qbg"][ref_id] = bg_mean[idx].cpu()
                self.tracked_predictions["qI"][ref_id] = qI_mean[idx].cpu()
                # self.tracked_predictions["qI_var"][ref_id] = qI_var[idx].cpu()
                self.tracked_predictions["qbg_var"][ref_id] = bg_var[idx].cpu()
                self.tracked_predictions["dials_I_prf_value"][
                    ref_id
                ] = dials_I_prf_value[idx]

        torch.cuda.empty_cache()

    def create_comparison_grid(
        self,
        cmap="cividis",
    ):
        if not self.tracked_refl_ids:
            return None

        # Create figure with proper subplot layout
        fig, axes = plt.subplots(
            3, self.num_profiles, figsize=(5 * self.num_profiles, 8)
        )

        # Handle case where only one column
        if self.num_profiles == 1:
            axes = axes.reshape(-1, 1)

        # Plot each column
        for i, refl_id in enumerate(self.tracked_refl_ids):
            # Get data for this column
            counts_data = self.tracked_predictions["counts"][refl_id]
            profile_data = self.tracked_predictions["profile"][refl_id]
            rates_data = self.tracked_predictions["rates"][refl_id]

            vmin_13 = min(counts_data.min().item(), rates_data.min().item())
            vmax_13 = max(counts_data.max().item(), rates_data.max().item())

            # Row 1: Input counts
            im0 = axes[0, i].imshow(counts_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13)
            axes[0, i].set_title(
                f"reflection ID: {refl_id}\n DIALS I_prf: {self.tracked_predictions['dials_I_prf_value'][refl_id]:.2f}"
            )
            axes[0, i].set_ylabel("raw image", labelpad=5)

            # Turn off axes but keep the labels
            axes[0, i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Row 2: Profile prediction (with its own scale)
            im1 = axes[1, i].imshow(profile_data, cmap=cmap)
            axes[1, i].set_ylabel("profile", labelpad=5)
            axes[1, i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Row 3: Rates (same scale as row 1)
            im2 = axes[2, i].imshow(rates_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13)
            axes[2, i].set_title(
                f"Bg: {self.tracked_predictions['qbg'][refl_id]:.2f}\n qI: {self.tracked_predictions['qI'][refl_id]:.2f}"
            )

            axes[2, i].set_ylabel("rate = I*pij + Bg", labelpad=5)
            axes[2, i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Add colorbars
            # First row colorbar (same as third row)
            divider0 = make_axes_locatable(axes[0, i])
            cax0 = divider0.append_axes("right", size="5%", pad=0.05)
            cbar0 = plt.colorbar(im0, cax=cax0)
            cbar0.ax.tick_params(labelsize=8)

            # Second row colorbar (independent)
            divider1 = make_axes_locatable(axes[1, i])
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            cbar1 = plt.colorbar(im1, cax=cax1)
            cbar1.ax.tick_params(labelsize=8)

            # Third row colorbar (same as first row)
            divider2 = make_axes_locatable(axes[2, i])
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            cbar2 = plt.colorbar(im2, cax=cax2)
            cbar2.ax.tick_params(labelsize=8)

        plt.tight_layout()

        return fig

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            # 1) Forward pass (no intensities yet)
            shoebox, dials, masks, metadata, counts = batch
            base_output = pl_module(shoebox, dials, masks, metadata, counts)

            # 2) Call calculate_intensities with the relevant fields
            intensities = pl_module.calculate_intensities(
                counts=base_output["counts"],
                qbg=base_output["qbg"],
                profile=base_output["profile"],
                dead_pixel_mask=base_output["masks"],
            )

            # 3) Merge intensities into a new dictionary
            #    so that "weighted_sum_mean", "thresholded_mean", etc. are available
            predictions = {
                **base_output,
                "weighted_sum_mean": intensities["weighted_sum_mean"],
                "weighted_sum_var": intensities["weighted_sum_var"],
                "thresholded_mean": intensities["thresholded_mean"],
                "thresholded_var": intensities["thresholded_var"],
            }

            # 4) (Optional) Only update tracked predictions if we’re going to plot this epoch
            if self.current_epoch % self.plot_every_n_epochs == 0:
                self.update_tracked_predictions(
                    predictions["profile"],
                    predictions["qbg"],
                    predictions["rates"],
                    predictions["counts"],
                    predictions["refl_ids"],
                    predictions["dials_I_prf_value"],
                    predictions["qI"],
                )

            # Store only a minimal version of the last batch predictions
            # Create CPU tensor versions to avoid keeping GPU memory
            self.train_predictions = {}
            for key in [
                "qI",
                # "qI_var",
                "dials_I_prf_value",
                "weighted_sum_mean",
                "thresholded_mean",
                "profile",
                "qbg",
            ]:
                if key in predictions:
                    if key == "profile":
                        self.train_predictions[key] = predictions[key].detach().cpu()
                    elif hasattr(predictions[key], "sample"):
                        self.train_predictions[key] = (
                            predictions[key].mean.detach().cpu()
                        )
                    else:
                        self.train_predictions[key] = predictions[key].detach().cpu()

            # Clean up
            del base_output, intensities, predictions
            torch.cuda.empty_cache()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_predictions:
            try:
                # Create data for scatter plots
                data = []

                qI_flat = (
                    self.train_predictions["qI"].flatten() + 1e-8
                )  # Add epsilon before log
                # qI_var_flat = (
                # self.train_predictions["qI_var"].flatten() + 1e-8
                # )  # Add epsilon before log
                dials_flat = (
                    self.train_predictions["dials_I_prf_value"].flatten() + 1e-8
                )
                weighted_sum_flat = (
                    self.train_predictions["weighted_sum_mean"].flatten() + 1e-8
                )
                thresholded_flat = (
                    self.train_predictions["thresholded_mean"].flatten() + 1e-8
                )

                # Calculate simpson index from profile
                if "profile" in self.train_predictions:
                    profile_flat = self.train_predictions["profile"]
                    simpson_flat = torch.sum(profile_flat**2, dim=-1).flatten()
                else:
                    simpson_flat = torch.ones_like(qI_flat)

                # Create data points with safe log transform
                for i in range(len(qI_flat)):
                    try:
                        data.append(
                            [
                                float(torch.log(qI_flat[i])),
                                # float(torch.log(qI_var_flat[i])),
                                float(torch.log(dials_flat[i])),
                                float(torch.log(weighted_sum_flat[i])),
                                float(torch.log(thresholded_flat[i])),
                                float(simpson_flat[i]),
                            ]
                        )
                    except Exception as e:
                        # Skip any problematic data points
                        pass

                # Create table
                table = wandb.Table(
                    data=data,
                    columns=[
                        "qI",
                        # "qI_var",
                        "dials_I_prf_value",
                        "weighted_sum_mean",
                        "thresholded_mean",
                        "simpson_idx",
                    ],
                )

                # Calculate correlation coefficients safely
                corr_qI = (
                    torch.corrcoef(torch.vstack([qI_flat, dials_flat]))[0, 1]
                    if len(qI_flat) > 1
                    else 0
                )
                corr_weighted = (
                    torch.corrcoef(torch.vstack([weighted_sum_flat, dials_flat]))[0, 1]
                    if len(weighted_sum_flat) > 1
                    else 0
                )
                corr_masked = (
                    torch.corrcoef(torch.vstack([thresholded_flat, dials_flat]))[0, 1]
                    if len(thresholded_flat) > 1
                    else 0
                )

                # Create log dictionary
                log_dict = {
                    "train_qI_vs_prf": wandb.plot.scatter(
                        table, "qI", "dials_I_prf_value"
                    ),
                    "train_weighted_sum_vs_prf": wandb.plot.scatter(
                        table, "weighted_sum_mean", "dials_I_prf_value"
                    ),
                    "train_thresholded_vs_prf": wandb.plot.scatter(
                        table, "thresholded_mean", "dials_I_prf_value"
                    ),
                    "corrcoef_qI": corr_qI,
                    "corrcoef_weighted": corr_weighted,
                    "corrcoef_masked": corr_masked,
                    "max_qI": torch.max(qI_flat),
                    "mean_qI": torch.mean(qI_flat),
                    # "mean_qI_var": torch.mean(qI_var_flat),
                    # "min_qI_var": torch.min(qI_var_flat),
                    # "max_qI_var": torch.max(qI_var_flat),
                }

                # Add mean background if available
                if "qbg" in self.train_predictions:
                    log_dict["mean_bg"] = torch.mean(self.train_predictions["qbg"])
                    log_dict["min_bg"] = torch.min(self.train_predictions["qbg"])
                    log_dict["max_bg"] = torch.max(self.train_predictions["qbg"])

                # Only create and log comparison grid on specified epochs
                if self.current_epoch % self.plot_every_n_epochs == 0:
                    comparison_fig = self.create_comparison_grid()
                    if comparison_fig is not None:
                        log_dict["profile_comparisons"] = wandb.Image(comparison_fig)
                        plt.close(comparison_fig)

                # Log metrics
                wandb.log(log_dict)

            except Exception as e:
                print(f"Error in on_train_epoch_end: {e}")

            # Clear memory
            self.train_predictions = {}
            torch.cuda.empty_cache()

        # Increment epoch counter
        self.current_epoch += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only track the last validation batch to save memory

        with torch.no_grad():
            shoebox, dials, masks, metadata, counts = batch
            base_output = pl_module(shoebox, dials, masks, metadata, counts)

            intensities = pl_module.calculate_intensities(
                counts=base_output["counts"],
                qbg=base_output["qbg"],
                profile=base_output["profile"],
                dead_pixel_mask=base_output["masks"],
            )

            # Store only minimal data needed for metrics
            self.val_predictions = {}
            for key in [
                "qI",
                "dials_I_prf_value",
                "weighted_sum_mean",
                "thresholded_mean",
            ]:
                if key in base_output:
                    if hasattr(base_output[key], "sample"):
                        self.val_predictions[key] = base_output[key].mean.detach().cpu()
                    else:
                        self.val_predictions[key] = base_output[key].detach().cpu()
                elif key in intensities:
                    self.val_predictions[key] = intensities[key].detach().cpu()

            # Clean up
            del base_output, intensities
            torch.cuda.empty_cache()


# %%
class MVNPlotter(Callback):
    def __init__(self, num_profiles=5, plot_every_n_epochs=5, d_vectors=None):
        super().__init__()
        self.train_predictions = {}
        self.val_predictions = {}
        self.num_profiles = num_profiles
        self.tracked_refl_ids = None
        self.all_seen_ids = set()
        self.tracked_predictions = {
            "profile": {},
            "counts": {},
            "qbg": {},
            "rates": {},
        }
        self.epoch_predictions = None
        self.plot_every_n_epochs = plot_every_n_epochs
        self.current_epoch = 0
        self.d_vectors = d_vectors

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_predictions = {
            "profile": [],
            "counts": [],
            "refl_ids": [],
            "intensity_mean": [],
            "dials_I_prf_value": [],
            "weighted_sum_mean": [],
            "thresholded_mean": [],
            "qbg": [],
            "rates": [],
        }

        self.tracked_predictions = {
            "profile": {},
            "counts": {},
            "qbg": {},
            "rates": {},
            "intensity_mean": {},
            # "qI_var": {},
            "qbg_var": {},
            "dials_I_prf_value": {},
        }

    def update_tracked_predictions(
        self,
        profile_preds,
        qbg_preds,
        rates,
        count_preds,
        refl_ids,
        dials_I,
        intensity_mean,
    ):
        current_refl_ids = refl_ids.cpu().numpy()

        # Update all seen IDs and set tracked IDs if not set
        self.all_seen_ids.update(current_refl_ids)
        if self.tracked_refl_ids is None:
            self.tracked_refl_ids = sorted(list(self.all_seen_ids))[: self.num_profiles]
            print(
                f"Selected {self.num_profiles} reflection IDs to track: {self.tracked_refl_ids}"
            )

        profile_images = profile_preds.reshape(-1, 3, 21, 21)[..., 1, :, :]
        count_images = count_preds.reshape(-1, 3, 21, 21)[..., 1, :, :]
        rate_images = rates.mean(1).reshape(-1, 3, 21, 21)[..., 1, :, :]
        bg_mean = qbg_preds.mean
        bg_var = qbg_preds.variance
        # qI_var = qI.variance
        dials_I_prf_value = dials_I

        for ref_id in self.tracked_refl_ids:
            matches = np.where(current_refl_ids == ref_id)[0]
            if len(matches) > 0:
                idx = matches[0]

                self.tracked_predictions["profile"][ref_id] = profile_images[idx].cpu()
                self.tracked_predictions["counts"][ref_id] = count_images[idx].cpu()
                self.tracked_predictions["rates"][ref_id] = rate_images[idx].cpu()
                self.tracked_predictions["qbg"][ref_id] = bg_mean[idx].cpu()
                self.tracked_predictions["intensity_mean"][ref_id] = intensity_mean[idx]
                # self.tracked_predictions["qI_var"][ref_id] = qI_var[idx].cpu()
                self.tracked_predictions["qbg_var"][ref_id] = bg_var[idx].cpu()
                self.tracked_predictions["dials_I_prf_value"][
                    ref_id
                ] = dials_I_prf_value[idx]

        torch.cuda.empty_cache()

    def create_comparison_grid(
        self,
        cmap="cividis",
    ):
        if not self.tracked_refl_ids:
            return None

        # Create figure with proper subplot layout
        fig, axes = plt.subplots(
            3, self.num_profiles, figsize=(5 * self.num_profiles, 8)
        )

        # Handle case where only one column
        if self.num_profiles == 1:
            axes = axes.reshape(-1, 1)

        # Plot each column
        for i, refl_id in enumerate(self.tracked_refl_ids):
            # Get data for this column
            counts_data = self.tracked_predictions["counts"][refl_id]
            profile_data = self.tracked_predictions["profile"][refl_id]
            rates_data = self.tracked_predictions["rates"][refl_id]

            vmin_13 = min(counts_data.min().item(), rates_data.min().item())
            vmax_13 = max(counts_data.max().item(), rates_data.max().item())

            # Row 1: Input counts
            im0 = axes[0, i].imshow(counts_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13)
            axes[0, i].set_title(
                f"reflection ID: {refl_id}\n DIALS I_prf: {self.tracked_predictions['dials_I_prf_value'][refl_id]:.2f}"
            )
            axes[0, i].set_ylabel("raw image", labelpad=5)

            # Turn off axes but keep the labels
            axes[0, i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Row 2: Profile prediction (with its own scale)
            im1 = axes[1, i].imshow(profile_data, cmap=cmap)
            axes[1, i].set_ylabel("profile", labelpad=5)
            axes[1, i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Row 3: Rates (same scale as row 1)
            im2 = axes[2, i].imshow(rates_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13)
            axes[2, i].set_title(
                f"Bg: {self.tracked_predictions['qbg'][refl_id]:.2f}\n intensity_mean: {self.tracked_predictions['intensity_mean'][refl_id]:.2f}"
            )

            axes[2, i].set_ylabel("rate = I*pij + Bg", labelpad=5)
            axes[2, i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Add colorbars
            # First row colorbar (same as third row)
            divider0 = make_axes_locatable(axes[0, i])
            cax0 = divider0.append_axes("right", size="5%", pad=0.05)
            cbar0 = plt.colorbar(im0, cax=cax0)
            cbar0.ax.tick_params(labelsize=8)

            # Second row colorbar (independent)
            divider1 = make_axes_locatable(axes[1, i])
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            cbar1 = plt.colorbar(im1, cax=cax1)
            cbar1.ax.tick_params(labelsize=8)

            # Third row colorbar (same as first row)
            divider2 = make_axes_locatable(axes[2, i])
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            cbar2 = plt.colorbar(im2, cax=cax2)
            cbar2.ax.tick_params(labelsize=8)

        plt.tight_layout()

        return fig

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            # 1) Forward pass (no intensities yet)
            shoebox, dials, masks, metadata, counts = batch
            base_output = pl_module(shoebox, dials, masks, metadata, counts)

            # 2) Call calculate_intensities with the relevant fields
            intensities = pl_module.calculate_intensities(
                counts=base_output["counts"],
                qbg=base_output["qbg"],
                profile=base_output["profile"],
                dead_pixel_mask=base_output["masks"],
            )

            # 3) Merge intensities into a new dictionary
            #    so that "weighted_sum_mean", "thresholded_mean", etc. are available
            predictions = {
                **base_output,
                "weighted_sum_mean": intensities["weighted_sum_mean"],
                "weighted_sum_var": intensities["weighted_sum_var"],
                "thresholded_mean": intensities["thresholded_mean"],
                "thresholded_var": intensities["thresholded_var"],
            }

            # 4) (Optional) Only update tracked predictions if we’re going to plot this epoch
            if self.current_epoch % self.plot_every_n_epochs == 0:
                self.update_tracked_predictions(
                    predictions["profile"],
                    predictions["qbg"],
                    predictions["rates"],
                    predictions["counts"],
                    predictions["refl_ids"],
                    predictions["dials_I_prf_value"],
                    predictions["intensity_mean"],
                )

            # Store only a minimal version of the last batch predictions
            # Create CPU tensor versions to avoid keeping GPU memory
            self.train_predictions = {}
            for key in [
                "intensity_mean",
                # "qI_var",
                "dials_I_prf_value",
                "weighted_sum_mean",
                "thresholded_mean",
                "profile",
                "qbg",
            ]:
                if key in predictions:
                    if key == "profile":
                        self.train_predictions[key] = predictions[key].detach().cpu()
                    elif hasattr(predictions[key], "sample"):
                        self.train_predictions[key] = (
                            predictions[key].mean.detach().cpu()
                        )
                    else:
                        self.train_predictions[key] = predictions[key].detach().cpu()

            # Clean up
            del base_output, intensities, predictions
            torch.cuda.empty_cache()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_predictions:
            try:
                # Create data for scatter plots
                data = []

                qI_flat = (
                    self.train_predictions["intensity_mean"].flatten() + 1e-8
                )  # Add epsilon before log
                # qI_var_flat = (
                # self.train_predictions["qI_var"].flatten() + 1e-8
                # )  # Add epsilon before log
                dials_flat = (
                    self.train_predictions["dials_I_prf_value"].flatten() + 1e-8
                )
                weighted_sum_flat = (
                    self.train_predictions["weighted_sum_mean"].flatten() + 1e-8
                )
                thresholded_flat = (
                    self.train_predictions["thresholded_mean"].flatten() + 1e-8
                )

                # Calculate simpson index from profile
                if "profile" in self.train_predictions:
                    profile_flat = self.train_predictions["profile"]
                    simpson_flat = torch.sum(profile_flat**2, dim=-1).flatten()
                else:
                    simpson_flat = torch.ones_like(qI_flat)

                # Create data points with safe log transform
                for i in range(len(qI_flat)):
                    try:
                        data.append(
                            [
                                float(torch.log(qI_flat[i])),
                                # float(torch.log(qI_var_flat[i])),
                                float(torch.log(dials_flat[i])),
                                float(torch.log(weighted_sum_flat[i])),
                                float(torch.log(thresholded_flat[i])),
                                float(simpson_flat[i]),
                            ]
                        )
                    except Exception as e:
                        # Skip any problematic data points
                        pass

                # Create table
                table = wandb.Table(
                    data=data,
                    columns=[
                        "intensity_mean",
                        # "qI_var",
                        "dials_I_prf_value",
                        "weighted_sum_mean",
                        "thresholded_mean",
                        "simpson_idx",
                    ],
                )

                # Calculate correlation coefficients safely
                corr_qI = (
                    torch.corrcoef(torch.vstack([qI_flat, dials_flat]))[0, 1]
                    if len(qI_flat) > 1
                    else 0
                )
                corr_weighted = (
                    torch.corrcoef(torch.vstack([weighted_sum_flat, dials_flat]))[0, 1]
                    if len(weighted_sum_flat) > 1
                    else 0
                )
                corr_masked = (
                    torch.corrcoef(torch.vstack([thresholded_flat, dials_flat]))[0, 1]
                    if len(thresholded_flat) > 1
                    else 0
                )

                # Create log dictionary
                log_dict = {
                    "train_qI_vs_prf": wandb.plot.scatter(
                        table, "intensity_mean", "dials_I_prf_value"
                    ),
                    "train_weighted_sum_vs_prf": wandb.plot.scatter(
                        table, "weighted_sum_mean", "dials_I_prf_value"
                    ),
                    "train_thresholded_vs_prf": wandb.plot.scatter(
                        table, "thresholded_mean", "dials_I_prf_value"
                    ),
                    "corrcoef_qI": corr_qI,
                    "corrcoef_weighted": corr_weighted,
                    "corrcoef_masked": corr_masked,
                    "max_qI": torch.max(qI_flat),
                    "mean_qI": torch.mean(qI_flat),
                    # "mean_qI_var": torch.mean(qI_var_flat),
                    # "min_qI_var": torch.min(qI_var_flat),
                    # "max_qI_var": torch.max(qI_var_flat),
                }

                # Add mean background if available
                if "qbg" in self.train_predictions:
                    log_dict["mean_bg"] = torch.mean(self.train_predictions["qbg"])
                    log_dict["min_bg"] = torch.min(self.train_predictions["qbg"])
                    log_dict["max_bg"] = torch.max(self.train_predictions["qbg"])

                # Only create and log comparison grid on specified epochs
                if self.current_epoch % self.plot_every_n_epochs == 0:
                    comparison_fig = self.create_comparison_grid()
                    if comparison_fig is not None:
                        log_dict["profile_comparisons"] = wandb.Image(comparison_fig)
                        plt.close(comparison_fig)

                # Log metrics
                wandb.log(log_dict)

            except Exception as e:
                print(f"Error in on_train_epoch_end: {e}")

            # Clear memory
            self.train_predictions = {}
            torch.cuda.empty_cache()

        # Increment epoch counter
        self.current_epoch += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only track the last validation batch to save memory

        with torch.no_grad():
            shoebox, dials, masks, metadata, counts = batch
            base_output = pl_module(shoebox, dials, masks, metadata, counts)

            intensities = pl_module.calculate_intensities(
                counts=base_output["counts"],
                qbg=base_output["qbg"],
                profile=base_output["profile"],
                dead_pixel_mask=base_output["masks"],
            )

            # Store only minimal data needed for metrics
            self.val_predictions = {}
            for key in [
                "intensity_mean",
                "dials_I_prf_value",
                "weighted_sum_mean",
                "thresholded_mean",
            ]:
                if key in base_output:
                    if hasattr(base_output[key], "sample"):
                        self.val_predictions[key] = base_output[key].mean.detach().cpu()
                    else:
                        self.val_predictions[key] = base_output[key].detach().cpu()
                elif key in intensities:
                    self.val_predictions[key] = intensities[key].detach().cpu()

            # Clean up
            del base_output, intensities
            torch.cuda.empty_cache()


# %%
class IntegratedPlotter(Callback):
    def __init__(self, num_profiles=5, plot_every_n_epochs=5):
        super().__init__()
        self.train_predictions = {}
        self.val_predictions = {}
        self.num_profiles = num_profiles
        self.tracked_refl_ids = None
        self.all_seen_ids = set()
        self.tracked_predictions = {
            "qp": {},
            "counts": {},
            "qbg": {},
            "rates": {},
            "qI": {},
            "dials_I_prf_value": {},
        }
        self.plot_every_n_epochs = plot_every_n_epochs
        self.current_epoch = 0

    def on_train_epoch_start(self, trainer, pl_module):
        # Clear tracked predictions at start of epoch
        self.tracked_predictions = {
            "qp": {},
            "counts": {},
            "qbg": {},
            "rates": {},
            "qI": {},
            "dials_I_prf_value": {},
            "weighted_sum_mean": {},
            "thresholded_mean": {},
            "signal_prob": {},
        }

    def update_tracked_predictions(self, predictions):
        """Update tracked predictions with model outputs"""
        try:
            # Extract required data
            refl_ids = predictions["refl_ids"]
            counts_data = predictions["counts"]
            dials_I_prf = predictions["dials_I_prf_value"]
            rates_data = predictions["rates"]
            qbg_data = predictions["qbg"]
            qI_data = predictions["qI"]
            profile_data = predictions["qp"]

            # Get optional data if available
            weighted_sum_mean = predictions.get("weighted_sum_mean")
            thresholded_mean = predictions.get("thresholded_mean")
            signal_prob = predictions.get("signal_prob")

            # Convert reflection IDs to numpy
            current_refl_ids = refl_ids.cpu().numpy()

            # Update all seen IDs and select IDs to track if not already done
            self.all_seen_ids.update(current_refl_ids)
            if self.tracked_refl_ids is None:
                self.tracked_refl_ids = sorted(list(self.all_seen_ids))[
                    : self.num_profiles
                ]
                print(
                    f"Selected {len(self.tracked_refl_ids)} reflection IDs to track: {self.tracked_refl_ids}"
                )

            # Process images
            profile_images = profile_data.mean.reshape(-1, 3, 21, 21)[:, 1, :, :]
            count_images = counts_data.reshape(-1, 3, 21, 21)[:, 1, :, :]

            # Handle rates based on dimensions
            if rates_data.dim() > 3:  # Has Monte Carlo dimension
                rate_images = rates_data.mean(1).reshape(-1, 3, 21, 21)[:, 1, :, :]
            else:
                rate_images = rates_data.reshape(-1, 3, 21, 21)[:, 1, :, :]

            # Get means
            bg_mean = qbg_data.mean if hasattr(qbg_data, "mean") else qbg_data
            qI_mean = qI_data.mean if hasattr(qI_data, "mean") else qI_data

            # Store data for each tracked reflection ID
            for refl_id in self.tracked_refl_ids:
                matches = np.where(current_refl_ids == refl_id)[0]
                if len(matches) > 0:
                    idx = matches[0]

                    # Store core data
                    self.tracked_predictions["qp"][refl_id] = profile_images[idx].cpu()
                    self.tracked_predictions["counts"][refl_id] = count_images[
                        idx
                    ].cpu()
                    self.tracked_predictions["rates"][refl_id] = rate_images[idx].cpu()
                    self.tracked_predictions["qbg"][refl_id] = bg_mean[idx].cpu()
                    self.tracked_predictions["qI"][refl_id] = qI_mean[idx].cpu()
                    self.tracked_predictions["dials_I_prf_value"][
                        refl_id
                    ] = dials_I_prf[idx].cpu()

                    # Store optional data if available
                    if weighted_sum_mean is not None:
                        self.tracked_predictions["weighted_sum_mean"][
                            refl_id
                        ] = weighted_sum_mean[idx].cpu()

                    if thresholded_mean is not None:
                        self.tracked_predictions["thresholded_mean"][
                            refl_id
                        ] = thresholded_mean[idx].cpu()

                    if signal_prob is not None:
                        self.tracked_predictions["signal_prob"][refl_id] = signal_prob[
                            idx
                        ].cpu()

        except Exception as e:
            print(f"Error in update_tracked_predictions: {e}")
            import traceback

            traceback.print_exc()

    def create_comparison_grid(self, cmap="cividis"):
        """Create visualization grid for profiles"""
        if not self.tracked_refl_ids:
            return None

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            3, self.num_profiles, figsize=(5 * self.num_profiles, 8)
        )
        if self.num_profiles == 1:
            axes = axes.reshape(-1, 1)

        for i, refl_id in enumerate(self.tracked_refl_ids):
            try:
                # Get data for this column
                counts_data = self.tracked_predictions["counts"][refl_id]
                profile_data = self.tracked_predictions["qp"][refl_id]
                rates_data = self.tracked_predictions["rates"][refl_id]
                bg_value = float(self.tracked_predictions["qbg"][refl_id])
                intensity_val = float(self.tracked_predictions["qI"][refl_id])
                dials_val = float(
                    self.tracked_predictions["dials_I_prf_value"][refl_id]
                )

                # Signal probability (if available)
                sig_prob = 0.0
                if refl_id in self.tracked_predictions.get("signal_prob", {}):
                    sig_prob = float(self.tracked_predictions["signal_prob"][refl_id])

                # Calculate shared min/max for rows 1 and 3
                vmin_13 = min(counts_data.min().item(), rates_data.min().item())
                vmax_13 = max(counts_data.max().item(), rates_data.max().item())

                # Row 1: Input counts
                im0 = axes[0, i].imshow(
                    counts_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13
                )
                axes[0, i].set_title(f"ID: {refl_id}\nDIALS: {dials_val:.2f}")
                axes[0, i].set_ylabel("raw image", labelpad=5)
                axes[0, i].tick_params(
                    left=False, bottom=False, labelleft=False, labelbottom=False
                )

                # Row 2: Profile prediction
                im1 = axes[1, i].imshow(profile_data, cmap=cmap)
                axes[1, i].set_ylabel("profile", labelpad=5)
                axes[1, i].tick_params(
                    left=False, bottom=False, labelleft=False, labelbottom=False
                )

                # Row 3: Rates
                im2 = axes[2, i].imshow(
                    rates_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13
                )
                axes[2, i].set_title(f"Bg: {bg_value:.2f} | I: {intensity_val:.2f}")
                axes[2, i].set_ylabel("rate = I*p + Bg", labelpad=5)
                axes[2, i].tick_params(
                    left=False, bottom=False, labelleft=False, labelbottom=False
                )

                # Add colorbars
                for ax, im in zip(axes[:, i], [im0, im1, im2]):
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)

            except Exception as e:
                print(f"Error plotting reflection {refl_id}: {e}")
                for row in range(3):
                    axes[row, i].text(
                        0.5,
                        0.5,
                        f"Error plotting\nreflection {refl_id}",
                        ha="center",
                        va="center",
                        transform=axes[row, i].transAxes,
                    )

        plt.tight_layout()
        return fig

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Process batch data and update tracking"""
        with torch.no_grad():
            try:
                # Get predictions from model
                shoebox, dials, masks, metadata, counts = batch
                predictions = pl_module(shoebox, dials, masks, metadata, counts)

                # Print available keys for debugging
                # if batch_idx == 0:
                # print(f"Available keys in predictions: {list(predictions.keys())}")

                # Only update tracked predictions if we're going to plot this epoch
                if self.current_epoch % self.plot_every_n_epochs == 0:
                    # Create CPU copy of predictions
                    cpu_predictions = {}
                    for key in predictions.keys():
                        if isinstance(predictions[key], torch.Tensor):
                            cpu_predictions[key] = predictions[key].detach().cpu()
                        elif hasattr(predictions[key], "mean"):
                            cpu_predictions[key] = predictions[
                                key
                            ]  # Keep distribution object
                        else:
                            cpu_predictions[key] = predictions[key]

                    # Update tracked predictions
                    self.update_tracked_predictions(cpu_predictions)

                # Store last batch predictions for correlation plots
                if batch_idx % 10 == 0:  # Only keep occasional batches to save memory
                    self.train_predictions = predictions

            except Exception as e:
                print(f"Error in on_train_batch_end: {e}")
                import traceback

                traceback.print_exc()

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.train_predictions:
            print("No train predictions available for plotting")
            return

        try:
            # Create basic correlation data
            data = []

            # Extract data for scatter plot
            if hasattr(self.train_predictions["qI"], "mean"):
                intensity_data = self.train_predictions["qI"].mean
            else:
                intensity_data = self.train_predictions["qI"]

            dials_data = self.train_predictions["dials_I_prf_value"]

            # Safety check for NaN/Inf values
            intensity_data = torch.nan_to_num(
                intensity_data, nan=0.0, posinf=1e6, neginf=0.0
            )
            dials_data = torch.nan_to_num(dials_data, nan=0.0, posinf=1e6, neginf=0.0)

            # Calculate correlation
            stacked = torch.vstack(
                [
                    intensity_data.flatten().detach().cpu(),
                    dials_data.flatten().detach().cpu(),
                ]
            )
            corr = torch.corrcoef(stacked)[0, 1].item()

            # Log simple metrics
            log_dict = {
                "corrcoef_intensity": corr,
                "epoch": self.current_epoch,
            }

            # Create and log comparison grid on specified epochs
            if self.current_epoch % self.plot_every_n_epochs == 0:
                try:
                    comparison_fig = self.create_comparison_grid()
                    if comparison_fig is not None:
                        log_dict["profile_comparisons"] = wandb.Image(comparison_fig)
                        import matplotlib.pyplot as plt

                        plt.close(comparison_fig)
                except Exception as e:
                    print(f"Error creating comparison grid: {e}")

            # Log to wandb
            wandb.log(log_dict)

        except Exception as e:
            print(f"Error in on_train_epoch_end: {e}")
            import traceback

            traceback.print_exc()

        # Increment epoch counter
        self.current_epoch += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Store validation predictions"""
        with torch.no_grad():
            shoebox, dials, masks, metadata, counts = batch
            self.val_predictions = pl_module(shoebox, dials, masks, metadata, counts)
