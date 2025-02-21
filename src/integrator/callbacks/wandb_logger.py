import wandb
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np


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
        }
        # Clear tracked predictions at start of epoch
        self.tracked_predictions = {
            "qp": {},
            "counts": {},
        }

    def update_tracked_predictions(self, qp_preds, count_preds, refl_ids):
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

        for ref_id in self.tracked_refl_ids:
            matches = np.where(current_refl_ids == ref_id)[0]
            if len(matches) > 0:
                idx = matches[0]
                self.tracked_predictions["qp"][ref_id] = qp_images[idx].cpu()
                self.tracked_predictions["counts"][ref_id] = count_images[idx].cpu()

    def create_comparison_grid(self):
        if not self.tracked_refl_ids:
            return None

        # Create figure with proper subplot layout
        fig, axes = plt.subplots(
            2, self.num_profiles, figsize=(4 * self.num_profiles, 8)
        )

        # Handle case where only one column
        if self.num_profiles == 1:
            axes = axes.reshape(-1, 1)

        # Plot each pair of images
        for i, refl_id in enumerate(self.tracked_refl_ids):
            if refl_id not in self.tracked_predictions["counts"]:
                # Skip if we don't have predictions for this reflection
                axes[0, i].text(0.5, 0.5, "Missing", ha="center", va="center")
                axes[1, i].text(0.5, 0.5, "Missing", ha="center", va="center")
                axes[0, i].set_title(f"ID: {refl_id} (missing)")
                axes[1, i].set_title(f"ID: {refl_id} (missing)")
                axes[0, i].axis("off")
                axes[1, i].axis("off")
                continue

            # Plot Input counts (top row)
            count_im = axes[0, i].imshow(
                self.tracked_predictions["counts"][refl_id], cmap="viridis"
            )
            axes[0, i].set_title(f"Input (ID: {refl_id})")
            axes[0, i].axis("off")

            # Plot QP prediction (bottom row)
            qp_im = axes[1, i].imshow(
                self.tracked_predictions["qp"][refl_id], cmap="viridis"
            )
            axes[1, i].set_title(f"Prediction (ID: {refl_id})")
            axes[1, i].axis("off")

        plt.tight_layout()
        return fig

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            shoebox, dials, masks, metadata, counts = batch
            predictions = pl_module(shoebox, dials, masks, metadata, counts)

            # Only update tracked predictions if we're going to plot this epoch
            if self.current_epoch % self.plot_every_n_epochs == 0:
                self.update_tracked_predictions(
                    predictions["qp"], predictions["counts"], predictions["refl_ids"]
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
                "corrcoef": torch.corrcoef(
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
