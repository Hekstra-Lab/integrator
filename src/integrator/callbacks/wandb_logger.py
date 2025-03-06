import wandb
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np


class UNetPlotter(Callback):
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
        }
        # Clear tracked predictions at start of epoch
        self.tracked_predictions = {
            "qp": {},
            "counts": {},
            "qbg": {},
            "rates": {},
            "intensity_mean": {},
            "dials_I_prf_value": {},
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
                    f"Bg: {bg_value:.2f} | I: {intensity_val:.2f}\nP(sig): "
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
            intensity_flat = intensity_data.flatten()
            dials_flat = dials_data.flatten()

            for i in range(len(intensity_flat)):
                try:
                    data.append(
                        [
                            float(torch.log(intensity_flat[i] + 1e-8)),
                            float(torch.log(dials_flat[i] + 1e-8)),
                        ]
                    )
                except Exception as e:
                    print(f"Error creating data point {i}: {e}")
            table = wandb.Table(
                data=data,
                columns=["log_intensity", "log_dials_I_prf"],
            )
            try:
                stacked = torch.vstack([intensity_flat, dials_flat])
                corr = torch.corrcoef(stacked)[0, 1].item()
            except Exception as e:
                print(f"Error calculating correlation: {e}")
                corr = float("nan")
            log_dict = {
                "train_intensity_vs_prf": wandb.plot.scatter(
                    table,
                    "log_intensity",
                    "log_dials_I_prf",
                ),
                "corrcoef_intensity": corr,
            }
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


class tempIntensityPlotter(Callback):
    def __init__(self, num_profiles=5, plot_every_n_epochs=5, d_vectors=None):
        super().__init__()
        self.train_predictions = {}
        self.val_predictions = {}
        self.num_profiles = num_profiles
        self.tracked_refl_ids = None
        self.all_seen_ids = set()
        self.tracked_predictions = {
            "qp": {},  # Profile
            "counts": {},  # Input counts
            "qbg": {},  # Background distribution
            "rates": {},  # Rate = I*p + bg
            "qz": {},  # Signal existence probability
            "q_I": {},  # Signal intensity when signal exists
            "q_I_nosignal": {},  # Intensity when no signal exists
            "signal_prob": {},  # Probability of signal presence
            "intensity_mean": {},  # Overall intensity (weighted)
            "intensity_var": {},  # Intensity variance
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
            "qI": [],  # For backward compatibility
            "q_z": [],  # Signal existence
            "q_I": [],  # Signal intensity
            "q_I_nosignal": [],  # No-signal intensity
            "signal_prob": [],  # Probability of signal
            "intensity_mean": [],  # Combined mean intensity
            "intensity_var": [],  # Combined variance
            "dials_I_prf_value": [],
            "weighted_sum_mean": [],
            "thresholded_mean": [],
            "qbg": [],
            "rates": [],
            "z_samples": [],  # Binary samples of signal presence
        }

        # Clear tracked predictions at start of epoch
        self.tracked_predictions = {
            "qp": {},
            "counts": {},
            "qbg": {},
            "rates": {},
            "q_z": {},
            "q_I": {},
            "q_I_nosignal": {},
            "signal_prob": {},
            "intensity_mean": {},
            "intensity_var": {},
            "dials_I_prf_value": {},
        }

    def update_tracked_predictions(self, outputs, refl_ids):
        """Update tracked predictions with model outputs"""
        try:
            current_refl_ids = refl_ids.cpu().numpy()

            # Update all seen IDs and set tracked IDs if not set
            self.all_seen_ids.update(current_refl_ids)
            if self.tracked_refl_ids is None:
                self.tracked_refl_ids = sorted(list(self.all_seen_ids))[
                    : self.num_profiles
                ]
                print(
                    f"Selected {self.num_profiles} reflection IDs to track: {self.tracked_refl_ids}"
                )

            # Safely reshape tensors with error handling
            try:
                qp_images = outputs["qp"].mean.reshape(-1, 3, 21, 21)[..., 1, :, :]
            except Exception as e:
                print(f"Error reshaping qp: {e}")
                # Provide a fallback empty tensor
                qp_images = torch.zeros(outputs["counts"].shape[0], 21, 21)

            try:
                count_images = outputs["counts"].reshape(-1, 3, 21, 21)[..., 1, :, :]
            except Exception as e:
                print(f"Error reshaping counts: {e}")
                # Fallback shape
                count_images = torch.zeros(outputs["counts"].shape[0], 21, 21)

            try:
                # Handle various possible rate tensor shapes
                rates = outputs["rates"]
                if len(rates.shape) == 3:  # [batch_size, mc_samples, num_components]
                    rate_images = rates.mean(1).reshape(-1, 3, 21, 21)[..., 1, :, :]
                else:
                    # Adapt to the actual shape
                    print(
                        f"Note: rates has shape {rates.shape}, using alternative reshape"
                    )
                    rate_images = rates.mean(1) if len(rates.shape) > 2 else rates
                    rate_images = rate_images.reshape(-1, 3, 21, 21)[..., 1, :, :]
            except Exception as e:
                print(f"Error reshaping rates: {e}")
                rate_images = torch.zeros_like(count_images)

            # Safely extract scalar values with proper error handling

            def safe_extract(output_dict, key, idx=None):
                """Safely extract values from outputs dict with error handling"""
                try:
                    if key not in output_dict:
                        print(f"Key {key} not found in outputs")
                        return 0.0

                    value = output_dict[key]

                    # Handle distribution objects
                    if hasattr(value, "mean") and callable(value.mean):
                        value = value.mean()
                    elif hasattr(value, "probs") and callable(getattr(value, "probs")):
                        value = value.probs

                    # Special handling for 2D tensors with odd shapes
                    if torch.is_tensor(value) and len(value.shape) == 2:
                        if value.shape[1] == 1:  # If column vector [batch, 1]
                            value = value.squeeze(1)
                        elif (
                            value.shape[0] == value.shape[1]
                        ):  # If square [batch, batch]
                            print(
                                f"Found square tensor for {key} with shape {value.shape}, using first column"
                            )
                            value = value[:, 0]

                    # Handle tensor vs scalar
                    if idx is not None and hasattr(value, "__getitem__"):
                        if torch.is_tensor(value[idx]):
                            return value[idx].detach().cpu().item()
                        else:
                            return float(value[idx])

                    # Handle final conversion
                    if torch.is_tensor(value):
                        return value.detach().cpu().item()
                    else:
                        return float(value)

                except Exception as e:
                    print(
                        f"Error extracting {key}: {e}, type: {type(output_dict[key])}"
                    )
                    return 0.0

            # Store values for tracked reflection IDs
            for ref_id in self.tracked_refl_ids:
                matches = np.where(current_refl_ids == ref_id)[0]
                if len(matches) > 0:
                    idx = matches[0]

                    # Store tensors with error handling
                    try:
                        self.tracked_predictions["qp"][ref_id] = (
                            qp_images[idx].detach().cpu()
                        )
                        self.tracked_predictions["counts"][ref_id] = (
                            count_images[idx].detach().cpu()
                        )
                        self.tracked_predictions["rates"][ref_id] = (
                            rate_images[idx].detach().cpu()
                        )

                        # Store scalar values
                        self.tracked_predictions["qbg"][ref_id] = safe_extract(
                            outputs, "qbg", idx
                        )
                        self.tracked_predictions["q_z"][ref_id] = (
                            safe_extract(outputs, "qI", idx)
                            if "qI" in outputs
                            else safe_extract(outputs, "signal_prob", idx)
                        )
                        self.tracked_predictions["q_I"][ref_id] = safe_extract(
                            outputs, "q_I", idx
                        )
                        self.tracked_predictions["q_I_nosignal"][ref_id] = safe_extract(
                            outputs, "q_I_nosignal", idx
                        )
                        self.tracked_predictions["signal_prob"][ref_id] = safe_extract(
                            outputs, "signal_prob", idx
                        )
                        self.tracked_predictions["intensity_mean"][
                            ref_id
                        ] = safe_extract(outputs, "intensity_mean", idx)
                        self.tracked_predictions["intensity_var"][
                            ref_id
                        ] = safe_extract(outputs, "intensity_var", idx)
                        self.tracked_predictions["dials_I_prf_value"][
                            ref_id
                        ] = safe_extract(outputs, "dials_I_prf_value", idx)
                    except Exception as e:
                        print(f"Error storing predictions for reflection {ref_id}: {e}")

        except Exception as e:
            print(f"Error in update_tracked_predictions: {e}")

    def create_comparison_grid(self, cmap="cividis"):
        """Create visualization grid with profiles, counts, and rates - with robust error handling"""
        try:
            if not self.tracked_refl_ids:
                return None

            # Import needed for colorbar positioning
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            import numpy as np
            import matplotlib.pyplot as plt

            # Create figure with proper subplot layout
            fig, axes = plt.subplots(
                3, self.num_profiles, figsize=(5 * self.num_profiles, 8)
            )

            # Handle case where only one column
            if self.num_profiles == 1:
                axes = axes.reshape(-1, 1)

            # Plot each column
            for i, refl_id in enumerate(self.tracked_refl_ids):
                try:
                    # Skip if this reflection ID wasn't properly tracked
                    if (
                        refl_id not in self.tracked_predictions["counts"]
                        or refl_id not in self.tracked_predictions["qp"]
                        or refl_id not in self.tracked_predictions["rates"]
                    ):
                        print(f"Missing data for reflection ID {refl_id}, skipping")
                        continue

                    # Get data for this column
                    counts_data = self.tracked_predictions["counts"][refl_id]
                    profile_data = self.tracked_predictions["qp"][refl_id]
                    rates_data = self.tracked_predictions["rates"][refl_id]

                    # Get mixture model parameters for this reflection with safe defaults
                    signal_prob = float(
                        self.tracked_predictions.get("signal_prob", {}).get(
                            refl_id, 0.0
                        )
                    )
                    signal_intensity = float(
                        self.tracked_predictions.get("q_I", {}).get(refl_id, 0.0)
                    )
                    bg_value = float(
                        self.tracked_predictions.get("qbg", {}).get(refl_id, 0.0)
                    )
                    intensity = float(
                        self.tracked_predictions.get("intensity_mean", {}).get(
                            refl_id, 0.0
                        )
                    )
                    dials_intensity = self.tracked_predictions["dials_I_prf_value"][
                        refl_id
                    ]

                    # Safely calculate min/max with error handling
                    try:
                        vmin_counts = counts_data.min().item()
                        vmax_counts = counts_data.max().item()

                        vmin_rates = rates_data.min().item()
                        vmax_rates = rates_data.max().item()

                        vmin_13 = min(vmin_counts, vmin_rates)
                        vmax_13 = max(vmax_counts, vmax_rates)

                        # Add small buffer to avoid empty range
                        if vmin_13 == vmax_13:
                            vmin_13 -= 0.1
                            vmax_13 += 0.1
                    except:
                        # Fallback values if min/max calculation fails
                        vmin_13, vmax_13 = 0, 1

                    # Row 1: Input counts
                    im0 = axes[0, i].imshow(
                        counts_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13
                    )
                    axes[0, i].set_title(
                        f"reflection ID: {refl_id}\nDIALS I: {dials_intensity:.2f}"
                    )
                    axes[0, i].set_ylabel("raw image", labelpad=5)
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
                    im2 = axes[2, i].imshow(
                        rates_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13
                    )
                    axes[2, i].set_title(
                        f"Bg: {bg_value:.2f} | P(signal): {signal_prob:.2f}\nI: {intensity:.2f} | I(signal): {signal_intensity:.2f}"
                    )
                    axes[2, i].set_ylabel("rate = I*pij + Bg", labelpad=5)
                    axes[2, i].tick_params(
                        left=False, bottom=False, labelleft=False, labelbottom=False
                    )

                    # Add colorbars
                    try:
                        # First row colorbar
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
                    except Exception as e:
                        print(f"Error adding colorbars for reflection {refl_id}: {e}")

                except Exception as e:
                    print(f"Error plotting reflection {refl_id}: {e}")
                    # Draw empty plots for this column to avoid breaking the grid
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

        except Exception as e:
            print(f"Error creating comparison grid: {e}")
            # Return a simple error figure instead of None
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                f"Error creating visualization:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            try:
                # Process this batch
                shoebox, dials, masks, metadata, counts = batch
                predictions = pl_module(shoebox, dials, masks, metadata, counts)

                # Debug output to understand shapes
                for key, value in predictions.items():
                    if isinstance(value, torch.Tensor):
                        print(f"Output tensor '{key}' has shape {value.shape}")
                    elif hasattr(value, "batch_shape"):
                        print(
                            f"Output distribution '{key}' has batch shape {value.batch_shape}"
                        )

                # Only update tracked predictions if we're going to plot this epoch
                if self.current_epoch % self.plot_every_n_epochs == 0:
                    self.update_tracked_predictions(
                        predictions, predictions["refl_ids"]
                    )

                # Accumulate predictions
                for key in self.epoch_predictions.keys():
                    if key in predictions:
                        self.epoch_predictions[key].append(predictions[key])

                self.train_predictions = (
                    predictions  # Keep last batch for other metrics
                )
            except Exception as e:
                print(f"Error in on_train_batch_end: {e}")
                import traceback

                traceback.print_exc()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_predictions:
            # Ensure all tensors are on CPU and handle potential shape issues
            try:
                # Print summary of what we have
                print(
                    "Available keys in train_predictions:",
                    list(self.train_predictions.keys()),
                )

                # Extract and prepare data carefully
                intensity_data = None
                dials_data = None
                signal_prob_data = None

                # Get intensity data and handle the 2D shape [5, 5]
                if "intensity_mean" in self.train_predictions:
                    intensity_data = self.train_predictions["intensity_mean"]
                    if hasattr(intensity_data, "detach"):
                        intensity_data = intensity_data.detach().cpu()

                    # Fix for the [5, 5] shape - take the first column
                    if (
                        len(intensity_data.shape) == 2
                        and intensity_data.shape[0] == intensity_data.shape[1]
                    ):
                        print(
                            f"Found square intensity tensor with shape {intensity_data.shape}, using first column"
                        )
                        intensity_data = intensity_data[:, 0]

                # Get dials data
                if "dials_I_prf_value" in self.train_predictions:
                    dials_data = self.train_predictions["dials_I_prf_value"]
                    if hasattr(dials_data, "detach"):
                        dials_data = dials_data.detach().cpu()

                # Get signal probability data and handle the [5, 1] shape
                if "signal_prob" in self.train_predictions:
                    signal_prob_data = self.train_predictions["signal_prob"]
                    if hasattr(signal_prob_data, "detach"):
                        signal_prob_data = signal_prob_data.detach().cpu()

                    # If shape is [batch, 1], squeeze it to [batch]
                    if (
                        len(signal_prob_data.shape) > 1
                        and signal_prob_data.shape[1] == 1
                    ):
                        signal_prob_data = signal_prob_data.squeeze(1)

                # Skip if we don't have the required data
                if intensity_data is None or dials_data is None:
                    print("Missing required data for visualization, skipping")
                    return

                # Debug information
                print(f"Adjusted tensor shapes:")
                print(f"Intensity data shape: {intensity_data.shape}")
                print(f"Dials data shape: {dials_data.shape}")
                if signal_prob_data is not None:
                    print(f"Signal probability data shape: {signal_prob_data.shape}")

                # Match shapes if necessary
                intensity_flat = intensity_data.flatten()
                dials_flat = dials_data.flatten()

                # Handle signal probability default if missing
                if signal_prob_data is None:
                    signal_flat = torch.ones_like(intensity_flat) * 0.5  # Default 0.5
                else:
                    signal_flat = signal_prob_data.flatten()

                # Create data for wandb
                data = []
                for i in range(len(intensity_flat)):
                    try:
                        if (
                            i < len(intensity_flat)
                            and i < len(dials_flat)
                            and i < len(signal_flat)
                        ):
                            data.append(
                                [
                                    float(torch.log(intensity_flat[i] + 1e-8)),
                                    float(torch.log(dials_flat[i] + 1e-8)),
                                    float(signal_flat[i]),
                                ]
                            )
                    except Exception as e:
                        print(f"Error creating data point {i}: {e}")

                # Create table for wandb
                table = wandb.Table(
                    data=data,
                    columns=[
                        "log_intensity",
                        "log_dials_I_prf",
                        "signal_probability",
                    ],
                )

                # Calculate correlation safely
                try:
                    # Make sure shapes match
                    stacked = torch.vstack([intensity_flat, dials_flat])
                    corr = torch.corrcoef(stacked)[0, 1].item()
                except Exception as e:
                    print(f"Error calculating correlation: {e}")
                    corr = float("nan")  # Use NaN rather than 0 to indicate failure

                # Log all available metrics
                log_dict = {
                    "train_intensity_vs_prf": wandb.plot.scatter(
                        table, "log_intensity", "log_dials_I_prf", "signal_probability"
                    ),
                    "corrcoef_intensity": corr,
                }

                # Add optional metrics if available
                try:
                    if signal_prob_data is not None:
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

            except Exception as e:
                print(f"Error in logging: {e}")
                import traceback

                traceback.print_exc()
                log_dict = {}

            # Plot grid if needed
            try:
                if self.current_epoch % self.plot_every_n_epochs == 0:
                    comparison_fig = self.create_comparison_grid()
                    if comparison_fig is not None:
                        log_dict["profile_comparisons"] = wandb.Image(comparison_fig)
                        plt.close(comparison_fig)
            except Exception as e:
                print(f"Error creating visualization: {e}")

            # Log what we have
            try:
                wandb.log(log_dict)
            except Exception as e:
                print(f"Error logging to wandb: {e}")

        # Increment epoch counter
        self.current_epoch += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            shoebox, dials, masks, metadata, counts = batch
            self.val_predictions = pl_module(shoebox, dials, masks, metadata, counts)


# class IntensityPlotter(Callback):
# def __init__(self, num_profiles=5, plot_every_n_epochs=5, d_vectors=None):
# super().__init__()
# self.train_predictions = {}
# self.val_predictions = {}
# self.num_profiles = num_profiles
# self.tracked_refl_ids = None
# self.all_seen_ids = set()
# self.tracked_predictions = {
# "qp": {},
# "counts": {},
# "qbg": {},
# "rates": {},
# }
# self.epoch_predictions = None
# self.plot_every_n_epochs = plot_every_n_epochs
# self.current_epoch = 0  # Track current epoch
# self.d_vectors = d_vectors

# def on_train_epoch_start(self, trainer, pl_module):
# self.epoch_predictions = {
# "qp": [],
# "counts": [],
# "refl_ids": [],
# "qI": [],
# "dials_I_prf_value": [],
# "weighted_sum_mean": [],
# "thresholded_mean": [],
# "qbg": [],
# "rates": [],
# }
# # Clear tracked predictions at start of epoch
# self.tracked_predictions = {
# "qp": {},
# "counts": {},
# "qbg": {},
# "rates": {},
# "qI": {},
# "dials_I_prf_value": {},
# }

# def update_tracked_predictions(
# self, qp_preds, qbg_preds, rates, count_preds, refl_ids, dials_I, qI
# ):
# current_refl_ids = refl_ids.cpu().numpy()

# # Update all seen IDs and set tracked IDs if not set
# self.all_seen_ids.update(current_refl_ids)
# if self.tracked_refl_ids is None:
# self.tracked_refl_ids = sorted(list(self.all_seen_ids))[: self.num_profiles]
# print(
# f"Selected {self.num_profiles} reflection IDs to track: {self.tracked_refl_ids}"
# )

# # Get indices of tracked reflections in current batch
# qp_images = qp_preds.mean.reshape(-1, 3, 21, 21)[..., 1, :, :]
# count_images = count_preds.reshape(-1, 3, 21, 21)[..., 1, :, :]
# rate_images = rates.mean(1).reshape(-1, 3, 21, 21)[..., 1, :, :]
# bg_mean = qbg_preds.mean
# qI_mean = qI.mean
# dials_I_prf_value = dials_I

# for ref_id in self.tracked_refl_ids:
# matches = np.where(current_refl_ids == ref_id)[0]
# if len(matches) > 0:
# idx = matches[0]

# self.tracked_predictions["qp"][ref_id] = qp_images[idx].cpu()
# self.tracked_predictions["counts"][ref_id] = count_images[idx].cpu()
# self.tracked_predictions["rates"][ref_id] = rate_images[idx].cpu()
# self.tracked_predictions["qbg"][ref_id] = bg_mean[idx].cpu()
# self.tracked_predictions["qI"][ref_id] = qI_mean[idx].cpu()
# self.tracked_predictions["dials_I_prf_value"][
# ref_id
# ] = dials_I_prf_value[idx]

# def create_comparison_grid(
# self,
# cmap="cividis",
# ):
# if not self.tracked_refl_ids:
# return None

# # Import needed for colorbar positioning
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import numpy as np

# # Create figure with proper subplot layout
# fig, axes = plt.subplots(
# 3, self.num_profiles, figsize=(5 * self.num_profiles, 8)
# )

# # Handle case where only one column
# if self.num_profiles == 1:
# axes = axes.reshape(-1, 1)

# # Plot each column
# for i, refl_id in enumerate(self.tracked_refl_ids):
# # Get data for this column
# counts_data = self.tracked_predictions["counts"][refl_id]
# profile_data = self.tracked_predictions["qp"][refl_id]
# rates_data = self.tracked_predictions["rates"][refl_id]

# # Calculate shared min/max for rows 1 and 3
# vmin_13 = min(counts_data.min().item(), rates_data.min().item())
# vmax_13 = max(counts_data.max().item(), rates_data.max().item())

# # Row 1: Input counts
# im0 = axes[0, i].imshow(counts_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13)
# axes[0, i].set_title(
# f"reflection ID: {refl_id}\n DIALS I: {self.tracked_predictions['dials_I_prf_value'][refl_id]:.2f}"
# )
# axes[0, i].set_ylabel("raw image", labelpad=5)

# # Turn off axes but keep the labels
# axes[0, i].tick_params(
# left=False, bottom=False, labelleft=False, labelbottom=False
# )

# # Row 2: QP prediction (with its own scale)
# im1 = axes[1, i].imshow(profile_data, cmap=cmap)
# axes[1, i].set_ylabel("profile", labelpad=5)
# axes[1, i].tick_params(
# left=False, bottom=False, labelleft=False, labelbottom=False
# )

# # Row 3: Rates (same scale as row 1)
# # im2 = axes[2, i].imshow(rates_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13)
# # axes[2, i].set_title(
# # f"Bg: {self.tracked_predictions['qbg'][refl_id]:.2f}\n qI: {self.tracked_predictions['qI'][refl_id]:.2f}"
# # )

# # axes[2, i].set_ylabel("rate = I*pij + Bg", labelpad=5)
# # axes[2, i].tick_params(
# # left=False, bottom=False, labelleft=False, labelbottom=False
# # )

# # Add colorbars
# # First row colorbar (same as third row)
# divider0 = make_axes_locatable(axes[0, i])
# cax0 = divider0.append_axes("right", size="5%", pad=0.05)
# cbar0 = plt.colorbar(im0, cax=cax0)
# cbar0.ax.tick_params(labelsize=8)

# # Second row colorbar (independent)
# divider1 = make_axes_locatable(axes[1, i])
# cax1 = divider1.append_axes("right", size="5%", pad=0.05)
# cbar1 = plt.colorbar(im1, cax=cax1)
# cbar1.ax.tick_params(labelsize=8)

# # Third row colorbar (same as first row)
# divider2 = make_axes_locatable(axes[2, i])
# cax2 = divider2.append_axes("right", size="5%", pad=0.05)
# # cbar2 = plt.colorbar(im2, cax=cax2)
# # cbar2.ax.tick_params(labelsize=8)

# plt.tight_layout()

# return fig

# def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
# with torch.no_grad():
# shoebox, dials, masks, metadata, counts = batch
# predictions = pl_module(shoebox, dials, masks, metadata, counts)

# # Only update tracked predictions if we're going to plot this epoch
# if self.current_epoch % self.plot_every_n_epochs == 0:
# self.update_tracked_predictions(
# predictions["qp"],
# predictions["qbg"],
# predictions["rates"],
# predictions["counts"],
# predictions["refl_ids"],
# predictions["dials_I_prf_value"],
# predictions["qI"],
# )

# # Accumulate predictions
# for key in self.epoch_predictions.keys():
# if key in predictions:
# self.epoch_predictions[key].append(predictions[key])

# self.train_predictions = predictions  # Keep last batch for other metrics

# def on_train_epoch_end(self, trainer, pl_module):
# if self.train_predictions:
# # Original scatter plot data
# data = [
# [qi, prf]
# for qi, prf, in zip(
# torch.log(self.train_predictions["qI"].mean.flatten()),
# torch.log(self.train_predictions["dials_I_prf_value"].flatten()),
# # torch.log(self.train_predictions["weighted_sum_mean"].flatten()),
# # torch.log(self.train_predictions["thresholded_mean"].flatten()),
# # torch.linalg.norm(self.train_predictions["qp"].mean, dim=-1).pow(2),
# )
# ]
# table = wandb.Table(
# data=data,
# columns=[
# "qI",
# "dials_I_prf_value",
# ],
# )

# # Create log dictionary with metrics that we want to log every epoch
# log_dict = {
# "train_qI_vs_prf": wandb.plot.scatter(
# table,
# "qI",
# "dials_I_prf_value",
# ),
# # "train_weighted_sum_vs_prf": wandb.plot.scatter(
# # table,
# # "weighted_sum_mean",
# # "dials_I_prf_value",
# # ),
# # "train_thresholded_vs_prf": wandb.plot.scatter(
# # table,
# # "thresholded_mean",
# # "dials_I_prf_value",
# # ),
# "corrcoef qI": torch.corrcoef(
# torch.vstack(
# [
# self.train_predictions["qI"].mean.flatten(),
# self.train_predictions["dials_I_prf_value"].flatten(),
# ]
# )
# )[0, 1],
# # "corrcoef_weighted": torch.corrcoef(
# # torch.vstack(
# # [
# # self.train_predictions["weighted_sum_mean"].flatten(),
# # self.train_predictions["dials_I_prf_value"].flatten(),
# # ]
# # )
# # )[0, 1],
# # "corrcoef_masked": torch.corrcoef(
# # torch.vstack(
# # [
# # self.train_predictions["thresholded_mean"].flatten(),
# # self.train_predictions["dials_I_prf_value"].flatten(),
# # ]
# # )
# # )[0, 1],
# "max_qI": torch.max(self.train_predictions["qI"].mean.flatten()),
# "mean_qI": torch.mean(self.train_predictions["qI"].mean.flatten()),
# "mean_bg": torch.mean(self.train_predictions["qbg"].mean),
# }

# # Only create and log comparison grid on specified epochs
# if self.current_epoch % self.plot_every_n_epochs == 0:
# comparison_fig = self.create_comparison_grid()
# if comparison_fig is not None:
# log_dict["profile_comparisons"] = wandb.Image(comparison_fig)
# plt.close(comparison_fig)

# wandb.log(log_dict)

# # Increment epoch counter
# self.current_epoch += 1

# def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
# with torch.no_grad():
# shoebox, dials, masks, metadata, counts = batch
# self.val_predictions = pl_module(shoebox, dials, masks, metadata, counts)


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


class MVNPlotter(Callback):
    def __init__(self, num_profiles=5, plot_every_n_epochs=5, d_vectors=None):
        super().__init__()
        self.train_predictions = {}
        self.val_predictions = {}
        self.num_profiles = num_profiles
        self.tracked_refl_ids = None
        self.all_seen_ids = set()
        self.tracked_predictions = {
            "profile": {},  # Changed from "qp" to "profile"
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
            "profile": [],  # Changed from "qp" to "profile"
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
            "profile": {},  # Changed from "qp" to "profile"
            "counts": {},
            "qbg": {},
            "rates": {},
            "qI": {},
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

        # Get indices of tracked reflections in current batch
        # The profile is already a tensor, no need to use .mean
        profile_images = profile_preds.reshape(-1, 3, 21, 21)[..., 1, :, :]
        count_images = count_preds.reshape(-1, 3, 21, 21)[..., 1, :, :]
        rate_images = rates.mean(1).reshape(-1, 3, 21, 21)[..., 1, :, :]
        bg_mean = qbg_preds.mean
        qI_mean = qI.mean
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
            profile_data = self.tracked_predictions["profile"][refl_id]
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
                qp=base_output["qp"],
                dead_pixel_mask=base_output["masks"],
            )

            # 3) Merge intensities into a new dictionary
            #    so that "weighted_sum_mean", "thresholded_mean", etc. are available
            predictions = {
                **base_output,
                "weighted_sum_mean": intensities["weighted_sum_intensity_mean"],
                "weighted_sum_var": intensities["weighted_sum_intensity_var"],
                "thresholded_mean": intensities["thresholded_mean"],
                "thresholded_var": intensities["thresholded_var"],
            }

            # 4) (Optional) Only update tracked predictions if we’re going to plot this epoch
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

            # 5) Accumulate predictions for epoch-level plotting
            for key in self.epoch_predictions.keys():
                if key in predictions:
                    self.epoch_predictions[key].append(predictions[key])

            # 6) Keep last batch predictions for scatter plots, correlation, etc.
            self.train_predictions = predictions

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
                    # For Simpson index, calculate directly from profile
                    torch.sum(self.train_predictions["profile"] ** 2, dim=-1),
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
            base_output = pl_module(shoebox, dials, masks, metadata, counts)

            intensities = pl_module.calculate_intensities(
                counts=base_output["counts"],
                qbg=base_output["qbg"],
                qp=base_output["qp"],
                dead_pixel_mask=base_output["masks"],
            )

            predictions = {
                **base_output,
                "weighted_sum_mean": intensities["weighted_sum_intensity_mean"],
                "weighted_sum_var": intensities["weighted_sum_intensity_var"],
                "thresholded_mean": intensities["thresholded_mean"],
                "thresholded_var": intensities["thresholded_var"],
            }

            # Store them (or do any validation-specific logic)
            self.val_predictions = predictions


# %%

# plotter.train_predictions["q_I"].mean
# plotter.train_predictions["dials_I_prf_value"]


class IntegratedPlotter(Callback):
    def __init__(self, num_profiles=5, plot_every_n_epochs=5, d_vectors=None):
        """
        Integrated plotter that supports both UNetPlotter and MVNPlotter functionality
        based on the profile type (probabilistic vs deterministic).
        """
        super().__init__()
        self.train_predictions = {}
        self.val_predictions = {}
        self.num_profiles = num_profiles
        self.tracked_refl_ids = None
        self.all_seen_ids = set()
        self.profile_type = None  # Will be set during first batch

        # Initialize tracked predictions with keys for both plotters
        self.tracked_predictions = {
            "qp": {},  # For probabilistic profile
            "profile": {},  # For deterministic profile
            "counts": {},  # Both use this
            "qbg": {},  # Both use this
            "rates": {},  # Both use this
            "qI": {},  # Both use this
            "dials_I_prf_value": {},  # Both use this
        }

        self.epoch_predictions = None
        self.plot_every_n_epochs = plot_every_n_epochs
        self.current_epoch = 0
        self.d_vectors = d_vectors

    def _detect_profile_type(self, predictions):
        """Use hasattr to detect if profile is probabilistic (has rsample) or deterministic"""
        if "profile" in predictions:
            # Already has a deterministic profile key
            return "deterministic"
        elif "qp" in predictions and hasattr(predictions["qp"], "rsample"):
            # Has a probabilistic profile with rsample method
            return "probabilistic"
        elif "qp" in predictions:
            # Has qp key but not rsample method
            return "deterministic"
        else:
            # Default - assume probabilistic
            return "probabilistic"

    def on_train_epoch_start(self, trainer, pl_module):
        # Initialize data collection for this epoch
        self.epoch_predictions = {
            "qp": [],
            "profile": [],
            "counts": [],
            "refl_ids": [],
            "qI": [],
            "dials_I_prf_value": [],
            "weighted_sum_mean": [],
            "thresholded_mean": [],
            "qbg": [],
            "rates": [],
            "intensity_mean": [],
            "signal_prob": [],
        }

        # Clear tracked predictions at start of epoch
        self.tracked_predictions = {
            "qp": {},
            "profile": {},
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
        """
        Update tracked predictions for a set of reflection IDs.
        Works with both deterministic and probabilistic profile types.
        """
        # Detect profile type if not already done
        if self.profile_type is None:
            self.profile_type = self._detect_profile_type(predictions)

        # Get essential data from predictions
        refl_ids = predictions["refl_ids"]
        counts_data = predictions["counts"]
        dials_I_prf = predictions["dials_I_prf_value"]
        rates_data = predictions["rates"]
        qbg_data = predictions["qbg"]
        qI_data = predictions["qI"]

        # Handle profile data based on profile type
        if self.profile_type == "deterministic":
            # Use profile key if available, otherwise use qp
            if "profile" in predictions:
                profile_data = predictions["profile"]
            else:
                profile_data = predictions["qp"]  # Deterministic qp
        else:
            # Probabilistic profile (UNet)
            profile_data = predictions["qp"]

        # Get weighted sum and thresholded means if available
        weighted_sum_mean = None
        if "weighted_sum_mean" in predictions:
            weighted_sum_mean = predictions["weighted_sum_mean"]
        elif "weighted_sum_intensity_mean" in predictions:
            weighted_sum_mean = predictions["weighted_sum_intensity_mean"]

        thresholded_mean = predictions.get("thresholded_mean")
        signal_prob = predictions.get("signal_prob")

        # Convert reflection IDs to numpy array for indexing
        current_refl_ids = refl_ids.cpu().numpy()

        # Update all seen IDs and set tracked IDs if not set
        self.all_seen_ids.update(current_refl_ids)
        if self.tracked_refl_ids is None:
            self.tracked_refl_ids = sorted(list(self.all_seen_ids))[: self.num_profiles]

        # Process profile data based on profile type
        if self.profile_type == "deterministic":
            # Direct reshape for deterministic profiles
            profile_images = profile_data.reshape(-1, 3, 21, 21)[..., 1, :, :]
        else:
            # Use mean attribute for probabilistic profiles
            profile_images = profile_data.mean.reshape(-1, 3, 21, 21)[..., 1, :, :]

        # Process other data
        count_images = counts_data.reshape(-1, 3, 21, 21)[..., 1, :, :]

        # Handle rates based on dimensions
        if rates_data.dim() > 3:
            rate_images = rates_data.mean(1).reshape(-1, 3, 21, 21)[..., 1, :, :]
        else:
            rate_images = rates_data.reshape(-1, 3, 21, 21)[..., 1, :, :]

        # Extract background mean
        if hasattr(qbg_data, "mean"):
            bg_mean = qbg_data.mean
        else:
            bg_mean = qbg_data

        # Extract intensity mean
        if hasattr(qI_data, "mean"):
            qI_mean = qI_data.mean
        else:
            qI_mean = qI_data

        # Store tracked predictions for each reflection ID
        for refl_id in self.tracked_refl_ids:
            matches = np.where(current_refl_ids == refl_id)[0]
            if len(matches) > 0:
                idx = matches[0]

                # Store data appropriately based on profile type
                if self.profile_type == "deterministic":
                    self.tracked_predictions["profile"][refl_id] = profile_images[
                        idx
                    ].cpu()
                else:
                    self.tracked_predictions["qp"][refl_id] = profile_images[idx].cpu()

                # Store common data
                self.tracked_predictions["counts"][refl_id] = count_images[idx].cpu()
                self.tracked_predictions["rates"][refl_id] = rate_images[idx].cpu()
                self.tracked_predictions["qbg"][refl_id] = bg_mean[idx].cpu()
                self.tracked_predictions["qI"][refl_id] = qI_mean[idx].cpu()
                self.tracked_predictions["dials_I_prf_value"][refl_id] = dials_I_prf[
                    idx
                ].cpu()

                # Store optional data
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

    def create_unet_comparison_grid(self, cmap="cividis"):
        """Create UNetPlotter-style grid for probabilistic profiles"""
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

                # Shared min/max for counts and rates
                vmin_13 = counts_data.min().item()
                vmax_13 = counts_data.max().item()

                # Row 1: Raw counts image
                im0 = axes[0, i].imshow(
                    counts_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13
                )
                axes[0, i].set_title(f"ID: {refl_id}\nDIALS: {dials_val:.2f}")
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
            except Exception:
                pass  # Silently handle errors

        plt.tight_layout()
        return fig

    def create_mvn_comparison_grid(self, cmap="cividis"):
        """Create MVNPlotter-style grid for deterministic profiles"""
        if not self.tracked_refl_ids:
            return None

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt

        # Create figure with proper subplot layout
        fig, axes = plt.subplots(
            3, self.num_profiles, figsize=(5 * self.num_profiles, 8)
        )

        # Handle case where only one column
        if self.num_profiles == 1:
            axes = axes.reshape(-1, 1)

        # Plot each column
        for i, refl_id in enumerate(self.tracked_refl_ids):
            try:
                # Get data for this column
                counts_data = self.tracked_predictions["counts"][refl_id]
                profile_data = self.tracked_predictions["profile"][refl_id]
                rates_data = self.tracked_predictions["rates"][refl_id]
                dials_val = float(
                    self.tracked_predictions["dials_I_prf_value"][refl_id]
                )
                bg_value = float(self.tracked_predictions["qbg"][refl_id])
                intensity_val = float(self.tracked_predictions["qI"][refl_id])

                # Signal probability (if available)
                sig_prob = 0.0
                if refl_id in self.tracked_predictions.get("signal_prob", {}):
                    sig_prob = float(self.tracked_predictions["signal_prob"][refl_id])

                # Calculate shared min/max for rows 1 and 3
                vmin_13 = counts_data.min().item()
                vmax_13 = counts_data.max().item()

                # Row 1: Input counts
                im0 = axes[0, i].imshow(
                    counts_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13
                )
                axes[0, i].set_title(
                    f"reflection ID: {refl_id}\n DIALS I: {dials_val:.2f}"
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
                im2 = axes[2, i].imshow(
                    rates_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13
                )
                axes[2, i].set_title(
                    f"Bg: {bg_value:.2f} | I: {intensity_val:.2f}\nP(sig): {sig_prob:.2f}"
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
            except Exception:
                pass  # Silently handle errors

        plt.tight_layout()
        return fig

    def create_comparison_grid(self, cmap="cividis"):
        """Create comparison grid based on detected profile type"""
        if self.profile_type == "deterministic":
            return self.create_mvn_comparison_grid(cmap)
        else:
            return self.create_unet_comparison_grid(cmap)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            # Process batch
            shoebox, dials, masks, metadata, counts = batch
            predictions = pl_module(shoebox, dials, masks, metadata, counts)

            # Only update tracked predictions if we're going to plot this epoch
            if self.current_epoch % self.plot_every_n_epochs == 0:
                self.update_tracked_predictions(predictions)

            # Accumulate predictions for the epoch
            for key in self.epoch_predictions.keys():
                if key in predictions:
                    self.epoch_predictions[key].append(predictions[key])

            # Store last batch predictions for metrics
            self.train_predictions = predictions

    def _create_probabilistic_correlation_data(self):
        """Create correlation data for probabilistic profiles (UNet style)"""
        import wandb

        try:
            data = []
            # Get necessary data
            intensity_data = self.train_predictions.get(
                "intensity_mean", self.train_predictions["qI"].mean
            )
            dials_data = self.train_predictions["dials_I_prf_value"]
            signal_prob_data = self.train_predictions.get(
                "signal_prob", torch.ones_like(intensity_data)
            )

            # Flatten data
            intensity_flat = intensity_data.flatten()
            dials_flat = dials_data.flatten()
            signal_flat = signal_prob_data.flatten()

            # Create data points with log transform
            for i in range(len(intensity_flat)):
                try:
                    # Make sure to catch any potential log(negative) errors
                    log_intensity = (
                        float(torch.log(intensity_flat[i] + 1e-8))
                        if intensity_flat[i] > -1e-8
                        else float(torch.log(1e-8))
                    )
                    log_dials = (
                        float(torch.log(dials_flat[i] + 1e-8))
                        if dials_flat[i] > -1e-8
                        else float(torch.log(1e-8))
                    )

                    data.append([log_intensity, log_dials, float(signal_flat[i])])
                except Exception:
                    pass  # Skip this data point on error

            # Create wandb table
            table = wandb.Table(
                data=data,
                columns=["log_intensity", "log_dials_I_prf", "signal_probability"],
            )

            # Calculate correlation
            try:
                stacked = torch.vstack([intensity_flat, dials_flat])
                corr = torch.corrcoef(stacked)[0, 1].item()
            except Exception:
                corr = float("nan")

            # Create log dictionary
            log_dict = {
                "train_intensity_vs_prf": wandb.plot.scatter(
                    table, "log_intensity", "log_dials_I_prf", "signal_probability"
                ),
                "corrcoef_intensity": corr,
            }

            # Add mean values
            try:
                log_dict["mean_signal_prob"] = torch.mean(signal_flat).item()
            except Exception:
                pass

            try:
                log_dict["mean_intensity"] = torch.mean(intensity_flat).item()
            except Exception:
                pass

            try:
                bg_mean = self.train_predictions["qbg"].mean
                if callable(bg_mean):
                    bg_mean = bg_mean()
                log_dict["mean_bg"] = torch.mean(bg_mean.detach().cpu()).item()
            except Exception:
                pass

            return log_dict
        except Exception:
            # If anything fails, return a minimal dictionary
            return {"profile_type": "probabilistic"}

    def _create_deterministic_correlation_data(self):
        """Create correlation data for deterministic profiles (MVN style)"""
        import wandb

        # Get profile data
        if "profile" in self.train_predictions:
            profile = self.train_predictions["profile"]
        else:
            profile = self.train_predictions["qp"]  # Deterministic profile in qp

        # Get necessary data exactly like MVNPlotter
        qI_data = self.train_predictions["qI"].mean.flatten()
        dials_data = self.train_predictions["dials_I_prf_value"].flatten()
        weighted_sum = self.train_predictions["weighted_sum_mean"].flatten()
        thresholded = self.train_predictions["thresholded_mean"].flatten()

        # Calculate Simpson index like original MVNPlotter
        simpson_idx = torch.sum(profile**2, dim=-1).flatten()

        # Create data points with log transform
        data = []
        for i in range(len(qI_data)):
            try:
                # Make sure to catch any potential log(negative) errors
                log_qi = (
                    float(torch.log(qI_data[i] + 1e-8))
                    if qI_data[i] > -1e-8
                    else float(torch.log(1e-8))
                )
                log_dials = (
                    float(torch.log(dials_data[i] + 1e-8))
                    if dials_data[i] > -1e-8
                    else float(torch.log(1e-8))
                )
                log_weighted = (
                    float(torch.log(weighted_sum[i] + 1e-8))
                    if weighted_sum[i] > -1e-8
                    else float(torch.log(1e-8))
                )
                log_thresholded = (
                    float(torch.log(thresholded[i] + 1e-8))
                    if thresholded[i] > -1e-8
                    else float(torch.log(1e-8))
                )

                data.append(
                    [
                        log_qi,
                        log_dials,
                        log_weighted,
                        log_thresholded,
                        float(simpson_idx[i]),
                    ]
                )
            except Exception:
                pass  # Skip this data point

        # Create table
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

        # Create log dict with scatter plots exactly like MVNPlotter
        log_dict = {
            "train_qI_vs_prf": wandb.plot.scatter(table, "qI", "dials_I_prf_value"),
            "train_weighted_sum_vs_prf": wandb.plot.scatter(
                table, "weighted_sum_mean", "dials_I_prf_value"
            ),
            "train_thresholded_vs_prf": wandb.plot.scatter(
                table, "thresholded_mean", "dials_I_prf_value"
            ),
        }

        # Calculate and log correlations
        try:
            log_dict["corrcoef_qI"] = torch.corrcoef(
                torch.vstack([qI_data, dials_data])
            )[0, 1].item()
        except Exception:
            log_dict["corrcoef_qI"] = float("nan")

        try:
            log_dict["corrcoef_weighted"] = torch.corrcoef(
                torch.vstack([weighted_sum, dials_data])
            )[0, 1].item()
        except Exception:
            log_dict["corrcoef_weighted"] = float("nan")

        try:
            log_dict["corrcoef_masked"] = torch.corrcoef(
                torch.vstack([thresholded, dials_data])
            )[0, 1].item()
        except Exception:
            log_dict["corrcoef_masked"] = float("nan")

        # Log summary statistics
        try:
            log_dict["max_qI"] = torch.max(qI_data).item()
            log_dict["mean_qI"] = torch.mean(qI_data).item()

            bg_mean = self.train_predictions["qbg"].mean
            if callable(bg_mean):
                bg_mean = bg_mean()
            log_dict["mean_bg"] = torch.mean(bg_mean).item()
        except Exception:
            pass

        return log_dict

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.train_predictions:
            return

        import wandb

        try:
            # Create correlation data based on profile type
            if self.profile_type == "deterministic":
                log_dict = self._create_deterministic_correlation_data()
            else:
                log_dict = self._create_probabilistic_correlation_data()

            # Create and log comparison grid on specified epochs
            if self.current_epoch % self.plot_every_n_epochs == 0:
                try:
                    comparison_fig = self.create_comparison_grid()
                    if comparison_fig is not None:
                        log_dict["profile_comparisons"] = wandb.Image(comparison_fig)

                        # Clean up figure
                        import matplotlib.pyplot as plt

                        plt.close(comparison_fig)
                except Exception:
                    pass

            # Add profile type to logging
            log_dict["profile_type"] = (
                1.0 if self.profile_type == "deterministic" else 0.0
            )

            # Log metrics to wandb
            wandb.log(log_dict)
        except Exception:
            # Log minimal data if correlation fails
            wandb.log({"epoch": self.current_epoch})

        # Increment epoch counter
        self.current_epoch += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            shoebox, dials, masks, metadata, counts = batch
            self.val_predictions = pl_module(shoebox, dials, masks, metadata, counts)
