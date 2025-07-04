import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning.callbacks import Callback

import wandb


def create_comparison_grid(
    num_profiles,
    refl_ids,
    pred_dict,
    cmap="cividis",
):
    """

    Args:
        num_profiles (int): number of shoeboxes to plot
        refl_ids (list): list of tracked shoebox ids
        pred_dict (dict): dictionary of tracked shoeboxes
        cmap (str): string name of color map

    Returns: a matplotlib figure

    """
    if not refl_ids:
        return None

    # Create figure with proper subplot layout
    fig, axes = plt.subplots(3, num_profiles, figsize=(5 * num_profiles, 8))

    # Plot each column
    for i, refl_id in enumerate(refl_ids):
        id_str = str(refl_id)
        # Get data for this column
        counts_data = pred_dict[id_str]["counts"]
        profile_data = pred_dict[id_str]["profile"]
        rates_data = pred_dict[id_str]["rates"]

        vmin_13 = min(counts_data.min().item(), rates_data.min().item())
        vmax_13 = max(counts_data.max().item(), rates_data.max().item())

        # Row 1: Input counts
        im0 = axes[0, i].imshow(counts_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13)
        axes[0, i].set_title(
            f"""reflection ID: {id_str}\n
            DIALS I_prf: {pred_dict[id_str]["dials_I_prf_var"]:.2f}\n
            DIALS var: {pred_dict[id_str]["dials_I_prf_var"]:.2f}\n
            DIALS bg mean: {pred_dict[id_str]["dials_bg_mean"]:.2f}"""
        )
        axes[0, i].set_ylabel("raw image", labelpad=5)

        # Turn off axes but keep the labels
        axes[0, i].tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )

        # row 2: predicted profile
        im1 = axes[1, i].imshow(profile_data, cmap=cmap)
        axes[1, i].set_title(
            f""""x_c: {pred_dict[id_str]["x_c"]:.2f}\n
            y_c: {pred_dict[id_str]["y_c"]:.2f}\n
            z_c: {pred_dict[id_str]["z_c"]:.2f}"""
        )
        axes[1, i].set_ylabel(
            "profile",
            labelpad=5,
        )
        axes[1, i].tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )

        # row 3: Rates (same scale as row 1)
        im2 = axes[2, i].imshow(rates_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13)
        axes[2, i].set_title(
            f"""Bg: {float(pred_dict[id_str]["bg_mean"]):.2f}\n
            I: {pred_dict[id_str]["intensity_mean"]:.2f}\n
            I_var: {pred_dict[id_str]["intensity_var"]:.2f}\n
            I_std: {np.sqrt(pred_dict[id_str]["intensity_var"]):.2f}\n"""
        )

        axes[2, i].set_ylabel(
            "rate = I*pij + Bg",
            labelpad=5,
        )
        axes[2, i].tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
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


# %%
class Plotter(Callback):
    def __init__(
        self,
        num_profiles=5,
        plot_every_n_epochs=5,
        d=3,
        h=21,
        w=21,
        d_vectors=None,
    ):
        super().__init__()
        self.d = d
        self.h = h
        self.w = w
        self.train_predictions = {}
        self.val_predictions = {}
        self.num_profiles = num_profiles
        self.tracked_refl_ids = None
        self.tracked_refl_ids_val = None
        self.all_seen_ids = set()
        self.all_seen_ids_val = set()
        self.epoch_predictions = None
        self.plot_every_n_epochs = plot_every_n_epochs
        self.current_epoch = 0
        self.d_vectors = d_vectors
        self.validation_tracked_predictions = dict()
        self.tracked_predictions = dict()

    def update_tracked_predictions(
        self,
        profile_preds,
        qbg_preds,
        rates,
        count_preds,
        refl_ids,
        dials_I,
        dials_I_var,
        intensity_mean,
        intensity_var,
        x_c,
        y_c,
        z_c,
        dials_bg_mean,
        dials_bg_sum_value,
        d,
        renyi_entropy,
    ):
        current_refl_ids = refl_ids

        # Update all seen IDs and set tracked IDs if not set
        # this will only happen once
        if self.tracked_refl_ids is None:
            self.tracked_refl_ids = current_refl_ids[: self.num_profiles]
            print(
                f"Selected {self.num_profiles} reflection IDs to track: {self.tracked_refl_ids}"
            )

        profile_images = profile_preds.reshape(-1, self.d, self.h, self.w)[
            ..., (self.d - 1) // 2, :, :
        ]
        count_images = count_preds.reshape(-1, self.d, self.h, self.w)[
            ..., (self.d - 1) // 2, :, :
        ]
        rate_images = rates.mean(1).reshape(-1, self.d, self.h, self.w)[
            ..., (self.d - 1) // 2, :, :
        ]
        bg_mean = qbg_preds.mean
        bg_var = qbg_preds.variance
        dials_I_prf_value = dials_I
        x_c = x_c
        y_c = y_c

        for ref_id in self.tracked_refl_ids:
            id_str = str(ref_id)
            matches = np.where(np.array(current_refl_ids) == ref_id)[0]

            if len(matches) > 0:
                idx = matches[0]

                self.tracked_predictions[id_str] = {
                    "profile": profile_images[idx].cpu(),
                    "counts": count_images[idx].cpu(),
                    "rates": rate_images[idx].cpu(),
                    "bg_mean": bg_mean[idx].cpu(),
                    "intensity_mean": intensity_mean[idx].cpu(),
                    "intensity_var": intensity_var[idx].cpu(),
                    "bg_var": bg_var[idx].cpu(),
                    "dials_I_prf_value": dials_I_prf_value[idx],
                    "dials_I_prf_var": dials_I_var[idx],
                    "dials_bg_mean": dials_bg_mean[idx].cpu(),
                    "x_c": x_c[idx].cpu(),
                    "y_c": y_c[idx].cpu(),
                    "z_c": z_c[idx].cpu(),
                    "d": d[idx].cpu(),
                    "renyi_entropy": renyi_entropy[idx].cpu(),
                }

        torch.cuda.empty_cache()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            # 1) Forward pass (no intensities yet)
            shoebox, dials, masks, counts = batch
            base_output = pl_module(shoebox, dials, masks, counts)

            renyi_entropy = -torch.log(base_output["qp"].mean.pow(2).sum(-1))

            predictions = {
                **base_output,
            }

            if self.current_epoch % self.plot_every_n_epochs == 0:
                self.update_tracked_predictions(
                    # predictions["qp"].mean,
                    predictions["profile"],
                    predictions["qbg"],
                    predictions["rates"],
                    predictions["counts"],
                    predictions["refl_ids"],
                    predictions["dials_I_prf_value"],
                    predictions["dials_I_prf_var"],
                    predictions["intensity_mean"],
                    predictions["intensity_var"],
                    predictions["x_c"],
                    predictions["y_c"],
                    predictions["z_c"],
                    predictions["dials_bg_mean"],
                    predictions["dials_bg_sum_value"],
                    predictions["d"],
                    renyi_entropy,
                )

            # Create CPU tensor versions to avoid keeping GPU memory
            self.train_predictions = {}
            for key in [
                "intensity_mean",
                "intensity_var",
                "dials_I_prf_value",
                "dials_I_prf_var",
                "profile",
                "qbg",
                "x_c",
                "y_c",
                "z_c",
                "dials_bg_mean",
                "dials_bg_sum_value",
                "d",
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
                dials_var_flat = (
                    self.train_predictions["dials_I_prf_var"].flatten() + 1e-8
                )
                dials_bg_flat = self.train_predictions["dials_bg_mean"].flatten() + 1e-8
                qbg_flat = self.train_predictions["qbg"].flatten() + 1e-8

                renyi_entropy_flat = -torch.log(
                    self.train_predictions["profile"].pow(2).sum(-1)
                )

                x_c_flat = self.train_predictions["x_c"].flatten()
                y_c_flat = self.train_predictions["y_c"].flatten()
                z_c_flat = self.train_predictions["z_c"].flatten()
                d_flat = 1 / self.train_predictions["d"].flatten().pow(2)
                d_ = self.train_predictions["d"]

                # Create data points with safe log transform
                for i in range(len(I_flat)):
                    try:
                        data.append(
                            [
                                float(torch.log(I_flat[i])),
                                float(torch.log(I_var_flat[i])),
                                float(torch.log(dials_flat[i])),
                                float(torch.log(dials_var_flat[i])),
                                dials_bg_flat[i],
                                qbg_flat[i],
                                renyi_entropy_flat[i],
                                x_c_flat[i],
                                y_c_flat[i],
                                d_flat[i],
                                d_[i],
                            ]
                        )
                    except Exception as e:
                        print("Caught exception in on_train_epoch_end!")
                        print("Type of exception:", type(e))
                        print("Exception object:", e)
                        traceback.print_exc(file=sys.stdout)

                df = pd.DataFrame(
                    data,
                    columns=[
                        "mean(qI)",
                        "var(qI)",
                        "DIALS intensity.prf.value",
                        "DIALS intensity.prf.variance",
                        "DIALS background.mean",
                        "mean(qbg)",
                        "Renyi entropy",
                        "x_c",
                        "y_c",
                        "d",
                        "d_",
                    ],
                )

                # Create table
                table = wandb.Table(dataframe=df)

                # Calculate correlation coefficients
                corr_I = (
                    torch.corrcoef(torch.vstack([I_flat, dials_flat]))[0, 1]
                    if len(I_flat) > 1
                    else 0
                )

                # plot renyi entropy
                renyi_vs_d = px.scatter(
                    df,
                    x="d",
                    y="Renyi entropy",
                    hover_data=["mean(qI)", "DIALS intensity.prf.value"],
                )

                layout_updates = {
                    "xaxis_title": "Resolution (Å)",
                    "showlegend": False,
                    "hovermode": "closest",
                    "plot_bgcolor": "white",
                    "xaxis": dict(
                        showgrid=True,
                        gridcolor="lightgrey",
                        tickmode="array",
                        ticktext=[
                            f"{d:.1f}"
                            for d in np.linspace(df["d_"].min(), df["d_"].max(), 6)
                        ],
                        tickvals=1
                        / np.linspace(df["d_"].min(), df["d_"].max(), 6) ** 2,
                        tickangle=90,
                    ),
                    "yaxis": dict(showgrid=True, gridcolor="lightgrey"),
                }
                renyi_vs_d.update_layout(**layout_updates)

                corr_bg = (
                    torch.corrcoef(torch.vstack([dials_bg_flat, dials_flat]))[0, 1]
                    if len(dials_bg_flat) > 1
                    else 0
                )

                # Create log dictionary
                log_dict = {
                    "Train: qI vs DIALS I prf": wandb.plot.scatter(
                        table, "mean(qI)", "DIALS intensity.prf.value"
                    ),
                    "Renyi entropy vs d": wandb.Html(renyi_vs_d.to_html()),
                    "Train: Bg vs DIALS bg": wandb.plot.scatter(
                        table, "mean(qbg)", "DIALS background.mean"
                    ),
                    "Correlation Coefficient: qI": corr_I,
                    "Correlation Coefficient: bg": corr_bg,
                    "Max mean(I)": torch.max(I_flat),
                    "Mean mean(I)": torch.mean(I_flat),
                    "Mean var(I) ": torch.mean(I_var_flat),
                    "Min var(I)": torch.min(I_var_flat),
                    "Max var(I)": torch.max(I_var_flat),
                }

                log_dict["mean(qbg.mean)"] = torch.mean(self.train_predictions["qbg"])
                log_dict["min(qbg.mean)"] = torch.min(self.train_predictions["qbg"])
                log_dict["max(qbg.mean)"] = torch.max(self.train_predictions["qbg"])

                # plot every n user-specified epochs
                if self.current_epoch % self.plot_every_n_epochs == 0:
                    comparison_fig = create_comparison_grid(
                        num_profiles=self.num_profiles,
                        refl_ids=self.tracked_refl_ids,
                        pred_dict=self.tracked_predictions,
                    )

                    if comparison_fig is not None:
                        log_dict["Tracked Profiles"] = wandb.Image(comparison_fig)
                        plt.close(comparison_fig)

                # Log metrics
                wandb.log(log_dict)

            except Exception as e:
                print("Caught exception in on_train_epoch_end!")
                print("Type of exception:", type(e))
                print("Exception object:", e)
                traceback.print_exc(file=sys.stdout)

            # Clear memory
            self.train_predictions = {}
            torch.cuda.empty_cache()

        # Increment epoch counter
        self.current_epoch += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only track the last validation batch to save memory

        with torch.no_grad():
            shoebox, dials, masks, counts = batch
            base_output = pl_module(shoebox, dials, masks, counts)

            renyi_entropy = -torch.log(base_output["qp"].mean.pow(2).sum(-1))

            self.val_predictions = {}
            for key in [
                "intensity_mean",
                "intensity_var",
                "dials_I_prf_value",
                "dials_I_prf_var",
                "profile",
                "qbg",
                "x_c",
                "y_c",
                "z_c",
                "dials_bg_mean",
                "dials_bg_sum_value",
                "d",
            ]:
                if key in base_output:
                    if hasattr(base_output[key], "sample"):
                        self.val_predictions[key] = base_output[key].mean.detach().cpu()
                    else:
                        self.val_predictions[key] = base_output[key].detach().cpu()
                elif key in base_output:
                    self.val_predictions[key] = base_output[key].detach().cpu()

            self.all_seen_ids.update(base_output["refl_ids"])
            if self.tracked_refl_ids_val is None:
                self.tracked_refl_ids_val = list(self.all_seen_ids)[: self.num_profiles]

            profile_images = base_output["profile"].reshape(
                -1,
                self.d,
                self.h,
                self.w,
            )[..., (self.d - 1) // 2, :, :]
            count_images = base_output["counts"].reshape(-1, self.d, self.h, self.w)[
                ..., (self.d - 1) // 2, :, :
            ]
            rate_images = (
                base_output["rates"]
                .mean(1)
                .reshape(-1, self.d, self.h, self.w)[..., (self.d - 1) // 2, :, :]
            )
            bg_mean = base_output["qbg"].mean
            bg_var = base_output["qbg"].variance
            dials_I_prf_value = base_output["dials_I_prf_value"]
            x_c = base_output["x_c"]
            y_c = base_output["y_c"]

            current_refl_ids = base_output["refl_ids"]

            for ref_id in self.tracked_refl_ids_val:
                id_str = str(ref_id)
                matches = np.where(np.array(current_refl_ids) == ref_id)[0]
                if len(matches) > 0:
                    idx = matches[0]

                    self.validation_tracked_predictions[id_str] = {
                        "profile": profile_images[idx].cpu(),
                        "counts": count_images[idx].cpu(),
                        "rates": rate_images[idx].cpu(),
                        "bg_mean": bg_mean[idx].cpu(),
                        "bg_var": base_output["qbg"].variance[idx].cpu(),
                        "intensity_mean": base_output["intensity_mean"][idx].cpu(),
                        "intensity_var": base_output["intensity_var"][idx].cpu(),
                        "dials_I_prf_value": base_output["dials_I_prf_value"][idx],
                        "dials_I_prf_var": base_output["dials_I_prf_var"][idx],
                        "dials_bg_mean": base_output["dials_bg_mean"][idx].cpu(),
                        "x_c": base_output["x_c"][idx].cpu(),
                        "y_c": base_output["y_c"][idx].cpu(),
                        "z_c": base_output["z_c"][idx].cpu(),
                        "d": base_output["d"][idx].cpu(),
                        "renyi_entropy": renyi_entropy[idx].cpu(),
                    }

            # Clean up
            del base_output
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_predictions:
            try:
                data = []

                I_flat = self.val_predictions["intensity_mean"].flatten() + 1e-8

                I_var_flat = self.val_predictions["intensity_var"].flatten() + 1e-8

                dials_flat = self.val_predictions["dials_I_prf_value"].flatten() + 1e-8
                dials_var_flat = (
                    self.val_predictions["dials_I_prf_var"].flatten() + 1e-8
                )
                dials_bg_flat = self.val_predictions["dials_bg_mean"].flatten() + 1e-8
                qbg_flat = self.val_predictions["qbg"].flatten() + 1e-8

                renyi_entropy_flat = -torch.log(
                    self.val_predictions["profile"].pow(2).sum(-1)
                )

                x_c_flat = self.val_predictions["x_c"].flatten()
                y_c_flat = self.val_predictions["y_c"].flatten()
                z_c_flat = self.val_predictions["z_c"].flatten()
                d_flat = 1 / self.val_predictions["d"].flatten().pow(2)
                d_ = self.val_predictions["d"]

                # Create data points with safe log transform
                for i in range(len(I_flat)):
                    try:
                        data.append(
                            [
                                float(torch.log(I_flat[i])),
                                float(torch.log(I_var_flat[i])),
                                float(torch.log(dials_flat[i])),
                                float(torch.log(dials_var_flat[i])),
                                dials_bg_flat[i],
                                qbg_flat[i],
                                renyi_entropy_flat[i],
                                x_c_flat[i],
                                y_c_flat[i],
                                d_flat[i],
                                d_[i],
                            ]
                        )
                    except Exception as e:
                        print("Caught exception in on_train_epoch_end!")
                        print("Type of exception:", type(e))
                        print("Exception object:", e)
                        traceback.print_exc(file=sys.stdout)

                df = pd.DataFrame(
                    data,
                    columns=[
                        "val: mean(qI)",
                        "val: var(qI)",
                        "DIALS intensity.prf.value",
                        "DIALS intensity.prf.variance",
                        "DIALS background.mean",
                        "val: mean(qbg)",
                        "val: Renyi entropy",
                        "x_c",
                        "y_c",
                        "d",
                        "d_",
                    ],
                )

                # Create table
                table = wandb.Table(dataframe=df)

                # Calculate correlation coefficients
                corr_I = (
                    torch.corrcoef(torch.vstack([I_flat, dials_flat]))[0, 1]
                    if len(I_flat) > 1
                    else 0
                )

                renyi_vs_d = px.scatter(
                    df,
                    x="d",
                    y="val: Renyi entropy",
                    hover_data=["val: mean(qI)", "DIALS intensity.prf.value"],
                )

                layout_updates = {
                    "xaxis_title": "Resolution (Å)",
                    "showlegend": False,
                    "hovermode": "closest",
                    "plot_bgcolor": "white",
                    "xaxis": dict(
                        showgrid=True,
                        gridcolor="lightgrey",
                        tickmode="array",
                        ticktext=[
                            f"{d:.1f}"
                            for d in np.linspace(df["d_"].min(), df["d_"].max(), 6)
                        ],
                        tickvals=1
                        / np.linspace(df["d_"].min(), df["d_"].max(), 6) ** 2,
                        tickangle=90,
                    ),
                    "yaxis": dict(showgrid=True, gridcolor="lightgrey"),
                }
                renyi_vs_d.update_layout(**layout_updates)

                corr_bg = (
                    torch.corrcoef(torch.vstack([dials_bg_flat, dials_flat]))[0, 1]
                    if len(dials_bg_flat) > 1
                    else 0
                )

                # Create log dictionary
                log_dict = {
                    "Val: qI vs DIALS I prf": wandb.plot.scatter(
                        table, "val: mean(qI)", "DIALS intensity.prf.value"
                    ),
                    "Val: Renyi entropy vs d": wandb.Html(renyi_vs_d.to_html()),
                    "Val: Bg vs DIALS bg": wandb.plot.scatter(
                        table, "val: mean(qbg)", "DIALS background.mean"
                    ),
                    "Val: Correlation Coefficient qI": corr_I,
                    "Val: Correlation Coefficient bg": corr_bg,
                    "Val: Max mean(I)": torch.max(I_flat),
                    "Val: Mean mean(I)": torch.mean(I_flat),
                    "Val: Mean var(I) ": torch.mean(I_var_flat),
                    "Val: Min var(I)": torch.min(I_var_flat),
                    "Val: Max var(I)": torch.max(I_var_flat),
                    "val: mean(qbg.mean)": torch.mean(self.val_predictions["qbg"]),
                    "val: min(qbg.mean)": torch.min(self.val_predictions["qbg"]),
                    "val: max(qbg.mean)": torch.max(self.val_predictions["qbg"]),
                }

                # plot input shoebox and predicted profile
                comparison_fig = create_comparison_grid(
                    num_profiles=self.num_profiles,
                    refl_ids=self.tracked_refl_ids_val,
                    pred_dict=self.validation_tracked_predictions,
                )

                log_dict["Val: Tracked Profiles"] = wandb.Image(comparison_fig)
                plt.close(comparison_fig)

                # Log metrics
                wandb.log(log_dict)

            except Exception as e:
                print("Caught exception in on_val_epoch_end")
                print("Type of exception:", type(e))
                print("Exception object:", e)
                traceback.print_exc(file=sys.stdout)

            # Clear memory
            self.val_predictions = {}
            torch.cuda.empty_cache()


# %%
class Plotter2(Callback):
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
            "qI": {},
            "dials_I_prf_value": {},
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
            "dials_I_prf_var": [],
            "qbg": [],
            "rates": [],
            "x_c": [],
            "y_c": [],
            "z_c": [],
            "dials_bg_mean": [],
            "dials_bg_sum_value": [],
            "d": [],
            "renyi_entropy": [],
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
            "dials_I_prf_var": {},
            "metadata": {},
            "x_c": {},
            "y_c": {},
            "z_c": {},
            "dials_bg_mean": {},
            "dials_bg_sum_value": {},
            "d": {},
            "renyi_entropy": {},
        }

    def update_tracked_predictions(
        self,
        profile_preds,
        qbg_preds,
        rates,
        count_preds,
        refl_ids,
        dials_I,
        dials_I_var,
        intensity_mean,
        intensity_var,
        x_c,
        y_c,
        z_c,
        dials_bg_mean,
        dials_bg_sum_value,
        d,
        renyi_entropy,
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
        x_c = x_c
        y_c = y_c

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
                self.tracked_predictions["dials_I_prf_value"][ref_id] = (
                    dials_I_prf_value[idx]
                )
                self.tracked_predictions["dials_I_prf_var"][ref_id] = dials_I_var[idx]
                self.tracked_predictions["dials_bg_mean"][ref_id] = dials_bg_mean[
                    idx
                ].cpu()
                self.tracked_predictions["x_c"][ref_id] = x_c[idx].cpu()
                self.tracked_predictions["y_c"][ref_id] = y_c[idx].cpu()
                self.tracked_predictions["z_c"][ref_id] = z_c[idx].cpu()
                self.tracked_predictions["d"][ref_id] = d[idx].cpu()
                self.tracked_predictions["renyi_entropy"][ref_id] = renyi_entropy[
                    idx
                ].cpu()

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

        # Plot each column
        for i, refl_id in enumerate(self.tracked_refl_ids):
            # Get data for this column
            counts_data = self.tracked_predictions["counts"][refl_id]
            profile_data = self.tracked_predictions["profile"][refl_id]
            rates_data = self.tracked_predictions["rates"][refl_id]
            bg_value = self.tracked_predictions["qbg"][refl_id]
            intensity_value = self.tracked_predictions["intensity_mean"][refl_id]

            vmin_13 = min(counts_data.min().item(), rates_data.min().item())
            vmax_13 = max(counts_data.max().item(), rates_data.max().item())

            # Row 1: Input counts
            im0 = axes[0, i].imshow(counts_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13)
            axes[0, i].set_title(
                f"reflection ID: {refl_id}\n DIALS I_prf: {self.tracked_predictions['dials_I_prf_value'][refl_id]:.2f}\nDIALS var: {self.tracked_predictions['dials_I_prf_var'][refl_id]:.2f}\nDIALS bg mean: {self.tracked_predictions['dials_bg_mean'][refl_id]:.2f}"
            )
            axes[0, i].set_ylabel("raw image", labelpad=5)
            # Turn off axes but keep the labels
            axes[0, i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Row 2: Profile prediction (with its own scale)
            im1 = axes[1, i].imshow(profile_data, cmap=cmap)
            axes[1, i].set_title(
                f"x_c: {self.tracked_predictions['x_c'][refl_id]:.2f} y_c: {self.tracked_predictions['y_c'][refl_id]:.2f} z_c: {self.tracked_predictions['z_c'][refl_id]:.2f}"
            )
            axes[1, i].set_ylabel("profile", labelpad=5)

            axes[1, i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Row 3: Rates (same scale as row 1)
            im2 = axes[2, i].imshow(rates_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13)
            axes[2, i].set_title(
                f"Bg: {float(self.tracked_predictions['qbg'][refl_id]):.2f}\n I: {self.tracked_predictions['intensity_mean'][refl_id]:.2f}\n I_var: {self.tracked_predictions['intensity_var'][refl_id]:.2f}"
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
            counts, masks, reference = batch
            base_output = pl_module(counts, masks, reference)

            # 2) Call calculate_intensities
            intensities = pl_module.calculate_intensities(
                counts=base_output["counts"],
                qbg=base_output["qbg"],
                qp=base_output["qp"],
                masks=base_output["masks"],
            )

            renyi_entropy = -torch.log(base_output["qp"].mean.pow(2).sum(-1))

            predictions = {
                **base_output,
                "kabsch_sum_mean": intensities["kabsch_sum_mean"],
                "kabsch_sum_var": intensities["kabsch_sum_var"],
                "profile_masking_mean": intensities["profile_masking_mean"],
                "profile_masking_var": intensities["profile_masking_var"],
            }

            if self.current_epoch % self.plot_every_n_epochs == 0:
                self.update_tracked_predictions(
                    # predictions["qp"].mean,
                    predictions["profile"],
                    predictions["qbg"],
                    predictions["rates"],
                    predictions["counts"],
                    predictions["refl_ids"],
                    predictions["dials_I_prf_value"],
                    predictions["dials_I_prf_var"],
                    predictions["intensity_mean"],
                    predictions["intensity_var"],
                    predictions["x_c"],
                    predictions["y_c"],
                    predictions["z_c"],
                    predictions["dials_bg_mean"],
                    predictions["dials_bg_sum_value"],
                    predictions["d"],
                    renyi_entropy,
                )

            # Create CPU tensor versions to avoid keeping GPU memory
            self.train_predictions = {}
            for key in [
                "intensity_mean",
                "intensity_var",
                "dials_I_prf_value",
                "dials_I_prf_var",
                "kabsch_sum_mean",
                "kabsch_sum_var",
                "profile_masking_mean",
                "profile_masking_var",
                "profile",
                "qbg",
                "x_c",
                "y_c",
                "z_c",
                "dials_bg_mean",
                "dials_bg_sum_value",
                "d",
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
                dials_var_flat = (
                    self.train_predictions["dials_I_prf_var"].flatten() + 1e-8
                )
                kabsch_sum_flat = (
                    self.train_predictions["kabsch_sum_mean"].flatten() + 1e-8
                )
                kabsch_sum_flat_var = (
                    self.train_predictions["kabsch_sum_var"].flatten() + 1e-8
                )
                profile_masking_flat = (
                    self.train_predictions["profile_masking_mean"].flatten() + 1e-8
                )
                profile_masking_flat_var = (
                    self.train_predictions["profile_masking_var"].flatten() + 1e-8
                )
                dials_bg_flat = self.train_predictions["dials_bg_mean"].flatten() + 1e-8
                qbg_flat = self.train_predictions["qbg"].flatten() + 1e-8

                renyi_entropy_flat = -torch.log(
                    self.train_predictions["profile"].pow(2).sum(-1)
                )

                x_c_flat = self.train_predictions["x_c"].flatten()
                y_c_flat = self.train_predictions["y_c"].flatten()
                z_c_flat = self.train_predictions["z_c"].flatten()
                d_flat = 1 / self.train_predictions["d"].flatten().pow(2)
                d_ = self.train_predictions["d"]

                # Create data points with safe log transform
                for i in range(len(I_flat)):
                    try:
                        data.append(
                            [
                                float(torch.log(I_flat[i])),
                                float(torch.log(I_var_flat[i])),
                                float(torch.log(dials_flat[i])),
                                float(torch.log(dials_var_flat[i])),
                                float(torch.log(kabsch_sum_flat[i])),
                                float(torch.log(kabsch_sum_flat_var[i])),
                                float(torch.log(profile_masking_flat[i])),
                                float(torch.log(profile_masking_flat_var[i])),
                                dials_bg_flat[i],
                                qbg_flat[i],
                                renyi_entropy_flat[i],
                                x_c_flat[i],
                                y_c_flat[i],
                                d_flat[i],
                                d_[i],
                            ]
                        )
                    except Exception as e:
                        print("Caught exception in on_train_epoch_end!")
                        print("Type of exception:", type(e))
                        print("Exception object:", e)
                        traceback.print_exc(file=sys.stdout)

                df = pd.DataFrame(
                    data,
                    columns=[
                        "mean(qI)",
                        "var(qI)",
                        "DIALS intensity.prf.value",
                        "DIALS intensity.prf.variance",
                        "mean(Kabsch sum)",
                        "var(Kabsch sum)",
                        "mean(Profile Masking)",
                        "var(Profile Masking)",
                        "DIALS background.mean",
                        "mean(qbg)",
                        "Renyi entropy",
                        "x_c",
                        "y_c",
                        "d",
                        "d_",
                    ],
                )

                # Create table
                table = wandb.Table(dataframe=df)

                # Calculate correlation coefficients
                corr_I = (
                    torch.corrcoef(torch.vstack([I_flat, dials_flat]))[0, 1]
                    if len(I_flat) > 1
                    else 0
                )

                corr_masked = (
                    torch.corrcoef(torch.vstack([profile_masking_flat, dials_flat]))[
                        0, 1
                    ]
                    if len(profile_masking_flat) > 1
                    else 0
                )

                positional_renyi = px.scatter_3d(
                    df,
                    x="x_c",
                    y="y_c",
                    z="Renyi entropy",
                    hover_data=["mean(qI)", "DIALS intensity.prf.value"],
                )

                positional_renyi.update_layout(
                    scene=dict(
                        xaxis=dict(range=[0, 4500]),
                        yaxis=dict(range=[0, 4500]),
                        zaxis=dict(range=[0, renyi_entropy_flat.max() + 0.5]),
                    )
                )

                renyi_vs_d = px.scatter(
                    df,
                    x="d",
                    y="Renyi entropy",
                    hover_data=["mean(qI)", "DIALS intensity.prf.value"],
                )

                layout_updates = {
                    "xaxis_title": "Resolution (Å)",
                    "showlegend": False,
                    "hovermode": "closest",
                    "plot_bgcolor": "white",
                    "xaxis": dict(
                        showgrid=True,
                        gridcolor="lightgrey",
                        tickmode="array",
                        ticktext=[
                            f"{d:.1f}"
                            for d in np.linspace(df["d_"].min(), df["d_"].max(), 6)
                        ],
                        tickvals=1
                        / np.linspace(df["d_"].min(), df["d_"].max(), 6) ** 2,
                        tickangle=90,
                    ),
                    "yaxis": dict(showgrid=True, gridcolor="lightgrey"),
                }
                renyi_vs_d.update_layout(**layout_updates)

                corr_bg = (
                    torch.corrcoef(torch.vstack([dials_bg_flat, dials_flat]))[0, 1]
                    if len(dials_bg_flat) > 1
                    else 0
                )

                # Create log dictionary
                log_dict = {
                    "Train: qI vs DIALS I prf": wandb.plot.scatter(
                        table, "mean(qI)", "DIALS intensity.prf.value"
                    ),
                    "Train: Profile Masking vs DIALS I prf": wandb.plot.scatter(
                        table, "mean(Profile Masking)", "DIALS intensity.prf.value"
                    ),
                    "Renyi entropy vs detector position": wandb.Html(
                        positional_renyi.to_html()
                    ),
                    "Renyi entropy vs d": wandb.Html(renyi_vs_d.to_html()),
                    "Train: Kabsch sum vs DIALS I prf": wandb.plot.scatter(
                        table, "mean(Kabsch sum)", "DIALS intensity.prf.value"
                    ),
                    "Train: Bg vs DIALS bg": wandb.plot.scatter(
                        table, "mean(qbg)", "DIALS background.mean"
                    ),
                    "Correlation Coefficient: qI": corr_I,
                    "Correlation Coefficient: profile masking": corr_masked,
                    "Correlation Coefficient: bg": corr_bg,
                    "Max mean(I)": torch.max(I_flat),
                    "Mean mean(I)": torch.mean(I_flat),
                    "Mean var(I) ": torch.mean(I_var_flat),
                    "Min var(I)": torch.min(I_var_flat),
                    "Max var(I)": torch.max(I_var_flat),
                }

                # Add mean background if available
                if "qbg" in self.train_predictions:
                    log_dict["mean(qbg.mean)"] = torch.mean(
                        self.train_predictions["qbg"]
                    )
                    log_dict["min(qbg.mean)"] = torch.min(self.train_predictions["qbg"])
                    log_dict["max(qbg.mean)"] = torch.max(self.train_predictions["qbg"])

                # Only create and log comparison grid on specified epochs
                if self.current_epoch % self.plot_every_n_epochs == 0:
                    comparison_fig = self.create_comparison_grid()

                    if comparison_fig is not None:
                        log_dict["Tracked Profiles"] = wandb.Image(comparison_fig)
                        plt.close(comparison_fig)

                # Log metrics
                wandb.log(log_dict)

            except Exception as e:
                print("Caught exception in on_train_epoch_end!")
                print("Type of exception:", type(e))
                print("Exception object:", e)
                traceback.print_exc(file=sys.stdout)

            # Clear memory
            self.train_predictions = {}
            torch.cuda.empty_cache()

        # Increment epoch counter
        self.current_epoch += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only track the last validation batch to save memory

        with torch.no_grad():
            counts, masks, reference = batch
            base_output = pl_module(counts, masks, reference)

            # Store only minimal data needed for metrics
            self.val_predictions = {}
            for key in [
                "intensity_mean",
                "intensity_var",
                "dials_I_prf_value",
                "kabsch_sum_mean",
                "kabsch_sum_var",
                "profile_masking_mean",
                "profile_masking_var",
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
                self.tracked_predictions["dials_I_prf_value"][ref_id] = (
                    dials_I_prf_value[idx]
                )

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
                    except Exception:
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
                        log_dict["Tracked Profiles"] = wandb.Image(comparison_fig)
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
