import sys
import traceback
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from torch import Tensor

import wandb


def to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x


def plot_symlog_qi_vs_dials(
    qi_mean,
    dials_prf,
    title="qI mean vs DIALS I_prf",
):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(qi_mean, dials_prf, s=10, alpha=0.6)

    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("symlog", linthresh=1e-3)

    ax.set_xlabel("symlog mean(qI)", fontsize=12)
    ax.set_ylabel("symlog DIALS I_prf", fontsize=12)
    ax.set_title(title)

    ax.grid(True, which="both", alpha=0.3)

    return fig


def create_comparison_grid(
    n_profiles,
    refl_ids,
    pred_dict,
    cmap="cividis",
):
    """

    Args:
        n_profiles (int): number of shoeboxes to plot
        refl_ids (list): list of tracked shoebox ids
        pred_dict (dict): dictionary of tracked shoeboxes
        cmap (str): string name of color map

    Returns: a matplotlib figure

    """
    # if not refl_ids:
    #     return None

    # Create figure with proper subplot layout
    fig, axes = plt.subplots(3, n_profiles, figsize=(5 * n_profiles, 8))

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
        im0 = axes[0, i].imshow(
            counts_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13
        )
        axes[0, i].set_title(
            f"reflection ID: {id_str}\n"
            f"DIALS I_prf: {pred_dict[id_str]['intensity.prf.value']:.2f}\n"
            f"DIALS var: {pred_dict[id_str]['intensity.prf.variance']:.2f}\n"
            f"DIALS bg mean: {pred_dict[id_str]['background.mean']:.2f}"
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
        im1 = axes[1, i].imshow(profile_data.detach(), cmap=cmap)
        axes[1, i].set_title(
            f"xyzcal.px.0: {pred_dict[id_str]['xyzcal.px.0']:.2f}\n"
            f"xyzcal.px.1: {pred_dict[id_str]['xyzcal.px.1']:.2f}\n"
            f"xyzcal.px.2: {pred_dict[id_str]['xyzcal.px.2']:.2f}"
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
        im2 = axes[2, i].imshow(
            rates_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13
        )
        axes[2, i].set_title(
            f"Bg: {float(pred_dict[id_str]['bg_mean'].detach()):.2f}\n"
            f"I: {pred_dict[id_str]['qi_mean'].detach():.2f}\n"
            f"I_var: {pred_dict[id_str]['qi_var'].detach():.2f}\n"
            f"I_std: {np.sqrt(pred_dict[id_str]['qi_var'].detach()):.2f}"
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


def _plot_avg_fano(df):
    fig, ax = plt.subplots()

    labels = df["intensity_bin"].to_list()
    y = df["avg_fano"].to_list()
    x = np.linspace(0, len(y), len(y))

    ax.scatter(x, y, color="black")
    ax.set_xticks(ticks=x, labels=labels, rotation=55)
    ax.set_xlabel("intensity bin")
    ax.set_ylabel("avg var/mean ratio")
    ax.set_title("Average variance/mean per intensity bin")
    ax.grid()
    plt.tight_layout()
    return fig


def _plot_avg_cv(df):
    fig, ax = plt.subplots()

    labels = df["intensity_bin"].to_list()
    y = df["avg_cv"].to_list()
    x = np.linspace(0, len(y), len(y))

    ax.scatter(x, y, color="black")
    ax.set_xticks(ticks=x, labels=labels, rotation=55)
    ax.set_xlabel("intensity bin")
    ax.set_ylabel("avg coefficient of variation")
    ax.set_title("Average variance/mean per intensity bin")
    ax.grid()
    plt.tight_layout()
    return fig


def _plot_avg_isigi(df):
    fig, ax = plt.subplots()

    labels = df["intensity_bin"].to_list()
    y = df["avg_isigi"].to_list()
    x = np.linspace(0, len(y), len(y))

    ax.scatter(x, y, color="black")
    ax.set_xticks(ticks=x, labels=labels, rotation=55)
    ax.set_xlabel("intensity bin")
    ax.set_ylabel("mean i/sigi")
    ax.set_title("Average signal-to-noise per intensity bin")
    ax.grid()
    plt.tight_layout()
    return fig


def _fano(
    outputs: Any,
    mean_key: str,
    var_key: str,
) -> Tensor:
    return to_cpu(outputs[var_key]) / (to_cpu(outputs[mean_key]) + 1e-8)


def _cv(
    outputs: Any,
    mean_key: str,
    var_key: str,
) -> Tensor:
    return to_cpu(outputs[var_key].sqrt()) / (to_cpu(outputs[mean_key]) + 1e-8)


def _get_agg_df(bin_labels):
    return pl.DataFrame(
        data={
            "intensity_bin": bin_labels,
            "fano_sum": pl.zeros(len(bin_labels), eager=True),
            "isigi_sum": pl.zeros(len(bin_labels), eager=True),
            "cv_sum": pl.zeros(len(bin_labels), eager=True),
            "n": pl.zeros(len(bin_labels), eager=True),
        },
        schema={
            "intensity_bin": pl.Categorical,
            "fano_sum": pl.Float32,
            "isigi_sum": pl.Float32,
            "cv_sum": pl.Float32,
            "n": pl.Int32,
        },
    )


class LogFano(Callback):
    def __init__(self):
        super().__init__()

        edges = [0, 10, 25, 50, 100, 300, 600, 1000, 1500, 2500, 5000, 10000]
        bin_edges = zip(edges[:-1], edges[1:], strict=False)

        bin_labels = []
        for l, r in bin_edges:
            bin_labels.append(f"{l} - {r}")

        # add end conditions
        bin_labels.insert(0, f"<{bin_labels[0].split()[0]}")
        bin_labels.append(f">{bin_labels[-1].split()[1]}")

        self.bin_edges = edges
        self.bin_labels = bin_labels

        # dataframe to merge and get all intensity bins
        self.base_df = pl.DataFrame(
            {"intensity_bin": bin_labels},
            schema={"intensity_bin": pl.Categorical},
        )

        # columns to aggregate
        self.numeric_cols = ["fano_sum", "n", "isigi_sum", "cv_sum"]

        # initialize an empty dataframe to aggregate data across steps
        self.agg_df = _get_agg_df(self.bin_labels)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        out = to_cpu(outputs["model_output"])
        fano = to_cpu(_fano(out, "qi_mean", "qi_var"))
        cv = to_cpu(_cv(out, "qi_mean", "qi_var"))

        # aggregate
        df = pl.DataFrame(
            {
                "refl_ids": to_cpu(out["refl_ids"]),
                "qi_mean": to_cpu(out["qi_mean"]),
                "qi_var": to_cpu(out["qi_var"]),
                "fano": to_cpu(fano),
                "cv": to_cpu(cv),
            }
        )

        # bin by intensity
        df = df.with_columns(
            pl.col("qi_mean")
            .cut(self.bin_edges, labels=self.bin_labels)
            .alias("intensity_bin")
        )

        # signal-to-noise expression
        isigi = pl.col("qi_mean") / pl.col("qi_var").sqrt()

        # group by intensity bin and get mean
        avg_df = df.group_by(pl.col("intensity_bin")).agg(
            fano_sum=pl.col("fano").sum(),
            cv_sum=pl.col("cv").sum(),
            isigi_sum=isigi.sum(),
            n=pl.len(),
        )

        merged_df = self.base_df.join(
            avg_df,
            how="left",
            on="intensity_bin",
        ).fill_null(0)

        self.agg_df = self.agg_df.with_columns(
            [pl.col(c) + merged_df[c] for c in self.numeric_cols]
        )

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module,
    ):
        # get avg variance/mean ratio per intensity bin
        epoch_df = self.agg_df.with_columns(
            (pl.col("fano_sum") / pl.col("n")).alias("avg_fano"),
            (pl.col("isigi_sum") / pl.col("n")).alias("avg_isigi"),
            (pl.col("cv_sum") / pl.col("n")).alias("avg_cv"),
        )

        # plot average Fano factor
        fig = _plot_avg_fano(epoch_df)
        wandb.log({"train: avg var/mean": wandb.Image(fig)})
        plt.close(fig)

        # plot average Coefficient of variation
        fig = _plot_avg_cv(epoch_df)
        wandb.log({"train: avg CV": wandb.Image(fig)})
        plt.close(fig)

        # plot average signal-to-noise
        fig = _plot_avg_isigi(epoch_df)
        wandb.log({"train: avg signal-to-noise": wandb.Image(fig)})
        plt.close(fig)

        log_dir = trainer.logger.experiment.dir

        csv_fname = (
            log_dir + f"/log_fano_csv_epoch_{trainer.current_epoch}.csv"
        )
        epoch_df.write_csv(csv_fname)

        # reset agg_df
        self.agg_df = _get_agg_df(self.bin_labels)


class PlotterLD(Callback):
    def __init__(
        self,
        n_profiles=5,
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
        self.preds_train = {}
        self.preds_validation = {}
        self.n_profiles = n_profiles
        self.tracked_ids_train = None
        self.tracked_ids_val = None
        self.epoch_preds = None
        self.plot_every_n_epochs = plot_every_n_epochs
        self.current_epoch = 0
        self.d_vectors = d_vectors
        self.tracked_shoeboxes_val = dict()
        self.tracked_shoeboxes_train = dict()

    def update_tracked_shoeboxes(
        self,
        preds,
        tracked_ids,
        tracked_shoeboxes,
    ):
        current_refl_ids = preds["refl_ids"].int()

        if tracked_ids is None:
            tracked_ids = current_refl_ids[: self.n_profiles]
            print(
                f"Selected {self.n_profiles} refl_ids to track: {tracked_ids}"
            )

        if self.d == 1:
            # 2D shoebox
            count_images = preds["counts"].reshape(-1, self.h, self.w)
            profile_images = preds["profile"].reshape(-1, self.h, self.w)
            rate_images = preds["rates"].mean(1).reshape(-1, self.h, self.w)
        else:
            # 3D shoebox
            count_images = preds["counts"].reshape(-1, self.d, self.h, self.w)[
                :, self.d // 2
            ]
            profile_images = preds["profile"].reshape(
                -1, self.d, self.h, self.w
            )[:, self.d // 2]
            rate_images = (
                preds["rates"]
                .mean(1)
                .reshape(-1, self.d, self.h, self.w)[:, self.d // 2]
            )

        for ref_id in tracked_ids:
            id_str = str(ref_id)
            matches = np.where(np.array(current_refl_ids) == ref_id)[0]

            if len(matches) > 0:
                idx = matches[0]

                tracked_shoeboxes[id_str] = {
                    "profile": profile_images[idx].cpu(),
                    "counts": count_images[idx].cpu(),
                    "rates": rate_images[idx].cpu(),
                    "bg_mean": preds["qbg_mean"][idx].cpu(),
                    "bg_var": preds["qbg_var"][idx].cpu(),
                    "qi_mean": preds["qi_mean"][idx].cpu(),
                    "qi_var": preds["qi_var"][idx].cpu(),
                    "dials_I_prf_value": preds["dials_I_prf_value"][idx].cpu(),
                    "dials_I_prf_var": preds["dials_I_prf_var"][idx].cpu(),
                    "dials_bg_mean": preds["dials_bg_mean"][idx].cpu(),
                    "x_c": preds["x_c"][idx].cpu(),
                    "y_c": preds["y_c"][idx].cpu(),
                    "z_c": preds["z_c"][idx].cpu(),
                }

        torch.cuda.empty_cache()
        return tracked_ids, tracked_shoeboxes

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        with torch.no_grad():
            # get forward outputs
            base_output = outputs["model_output"]

            # additional metrics to log

            if self.current_epoch % self.plot_every_n_epochs == 0:
                self.tracked_ids_train, self.tracked_shoeboxes_train = (
                    self.update_tracked_shoeboxes(
                        base_output,
                        self.tracked_ids_train,
                        self.tracked_shoeboxes_train,
                    )
                )

            # Create CPU tensor versions to avoid keeping GPU memory
            self.preds_train = {}
            for key in [
                "qi_mean",
                "qi_var",
                "dials_I_prf_value",
                "dials_I_prf_var",
                "profile",
                "qbg_mean",
                "x_c",
                "y_c",
                "z_c",
                "dials_bg_mean",
                "dials_bg_sum_value",
            ]:
                if key in base_output:
                    if key == "profile":
                        self.preds_train[key] = to_cpu(base_output[key])
                    elif hasattr(base_output[key], "sample"):
                        self.preds_train[key] = to_cpu(base_output[key].mean)
                    else:
                        self.preds_train[key] = to_cpu(base_output[key])

            # Clean up
            torch.cuda.empty_cache()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.preds_train:
            try:
                # Create data for scatter plots
                data = []

                i_flat = self.preds_train["qi_mean"].flatten() + 1e-8

                i_var_flat = self.preds_train["qi_var"].flatten() + 1e-8

                dials_flat = (
                    self.preds_train["dials_I_prf_value"].flatten() + 1e-8
                )
                dials_var_flat = (
                    self.preds_train["dials_I_prf_var"].flatten() + 1e-8
                )
                dials_bg_flat = (
                    self.preds_train["dials_bg_mean"].flatten() + 1e-8
                )
                qbg_flat = self.preds_train["qbg_mean"].flatten() + 1e-8

                x_c_flat = self.preds_train["x_c"].flatten()
                y_c_flat = self.preds_train["y_c"].flatten()
                z_c_flat = self.preds_train["z_c"].flatten()

                # Create data points with safe log transform
                for i in range(len(i_flat)):
                    try:
                        data.append(
                            [
                                i_flat[i],
                                i_var_flat[i],
                                dials_flat[i],
                                dials_var_flat[i],
                                dials_bg_flat[i],
                                qbg_flat[i],
                                x_c_flat[i],
                                y_c_flat[i],
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
                        "x_c",
                        "y_c",
                    ],
                )

                # Create table
                table = wandb.Table(dataframe=df)

                # Calculate correlation coefficients
                corr_I = (
                    torch.corrcoef(torch.vstack([i_flat, dials_flat]))[0, 1]
                    if len(i_flat) > 1
                    else 0
                )

                corr_bg = (
                    torch.corrcoef(torch.vstack([dials_bg_flat, dials_flat]))[
                        0, 1
                    ]
                    if len(dials_bg_flat) > 1
                    else 0
                )

                # Create log dictionary
                log_dict = {
                    "Train: qi vs DIALS I prf": wandb.plot.scatter(
                        table, "mean(qI)", "DIALS intensity.prf.value"
                    ),
                    "Train: Bg vs DIALS bg": wandb.plot.scatter(
                        table, "mean(qbg)", "DIALS background.mean"
                    ),
                    "Correlation Coefficient: qi": corr_I,
                    "Correlation Coefficient: bg": corr_bg,
                    "Max mean(I)": torch.max(i_flat),
                    "Mean mean(I)": torch.mean(i_flat),
                    "Mean var(I) ": torch.mean(i_var_flat),
                    "Min var(I)": torch.min(i_var_flat),
                    "Max var(I)": torch.max(i_var_flat),
                }

                log_dict["mean(qbg.mean)"] = torch.mean(
                    self.preds_train["qbg_mean"]
                )
                log_dict["min(qbg.mean)"] = torch.min(
                    self.preds_train["qbg_mean"]
                )
                log_dict["max(qbg.mean)"] = torch.max(
                    self.preds_train["qbg_mean"]
                )

                # plot every n user-specified epochs
                if self.current_epoch % self.plot_every_n_epochs == 0:
                    comparison_fig = create_comparison_grid(
                        n_profiles=self.n_profiles,
                        refl_ids=self.tracked_ids_train,
                        pred_dict=self.tracked_shoeboxes_train,
                    )

                    if comparison_fig is not None:
                        log_dict["Tracked Profiles"] = wandb.Image(
                            comparison_fig
                        )
                        plt.close(comparison_fig)

                # Log metrics
                wandb.log(log_dict)

                fig = plot_symlog_qi_vs_dials(
                    to_cpu(i_flat).numpy(), dials_flat.cpu().numpy()
                )
                wandb.log({"train: qi_vs_dials_symlog": wandb.Image(fig)})
                plt.close(fig)

            except Exception as e:
                print("Caught exception in on_train_epoch_end!")
                print("Type of exception:", type(e))
                print("Exception object:", e)
                traceback.print_exc(file=sys.stdout)

            # Clear memory
            self.preds_train = {}
            torch.cuda.empty_cache()

        # Increment epoch counter
        self.current_epoch += 1

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        with torch.no_grad():
            # get forward outputs
            base_output = outputs["model_output"]

            # updated tracked shoeboxes
            self.tracked_ids_val, self.tracked_shoeboxes_val = (
                self.update_tracked_shoeboxes(
                    base_output,
                    tracked_ids=self.tracked_ids_val,
                    tracked_shoeboxes=self.tracked_shoeboxes_val,
                )
            )

            self.preds_validation = {}
            for key in [
                "qi_mean",
                "qi_var",
                "dials_I_prf_value",
                "dials_I_prf_var",
                "profile",
                "qbg_mean",
                "x_c",
                "y_c",
                "z_c",
                "dials_bg_mean",
                "dials_bg_sum_value",
            ]:
                if key in base_output:
                    if hasattr(base_output[key], "sample"):
                        self.preds_validation[key] = to_cpu(
                            base_output[key].mean
                        )
                    else:
                        self.preds_validation[key] = to_cpu(base_output[key])
                elif key in base_output:
                    self.preds_validation[key] = to_cpu(base_output[key])

            # Clean up
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.preds_validation:
            try:
                data = []

                i_flat = self.preds_validation["qi_mean"].flatten() + 1e-8

                i_var_flat = self.preds_validation["qi_var"].flatten() + 1e-8

                dials_flat = (
                    self.preds_validation["dials_I_prf_value"].flatten() + 1e-8
                )
                dials_var_flat = (
                    self.preds_validation["dials_I_prf_var"].flatten() + 1e-8
                )
                dials_bg_flat = (
                    self.preds_validation["dials_bg_mean"].flatten() + 1e-8
                )
                qbg_flat = self.preds_validation["qbg_mean"].flatten() + 1e-8

                x_c_flat = self.preds_validation["x_c"].flatten()
                y_c_flat = self.preds_validation["y_c"].flatten()
                z_c_flat = self.preds_validation["z_c"].flatten()

                # Create data points with safe log transform
                for i in range(len(i_flat)):
                    try:
                        data.append(
                            [
                                i_flat[i],
                                i_var_flat[i],
                                dials_flat[i],
                                dials_var_flat[i],
                                dials_bg_flat[i],
                                qbg_flat[i],
                                x_c_flat[i],
                                y_c_flat[i],
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
                        "validation: mean(qI)",
                        "validation: var(qI)",
                        "DIALS intensity.prf.value",
                        "DIALS intensity.prf.variance",
                        "DIALS background.mean",
                        "validation: mean(qbg)",
                        "x_c",
                        "y_c",
                    ],
                )

                # Create table
                table = wandb.Table(dataframe=df)

                # Calculate correlation coefficients
                corr_I = (
                    torch.corrcoef(torch.vstack([i_flat, dials_flat]))[0, 1]
                    if len(i_flat) > 1
                    else 0
                )

                corr_bg = (
                    torch.corrcoef(torch.vstack([dials_bg_flat, dials_flat]))[
                        0, 1
                    ]
                    if len(dials_bg_flat) > 1
                    else 0
                )

                # Create log dictionary
                log_dict = {
                    "Validation: qi vs DIALS I prf": wandb.plot.scatter(
                        table,
                        "validation: mean(qI)",
                        "DIALS intensity.prf.value",
                    ),
                    "Validation: Bg vs DIALS bg": wandb.plot.scatter(
                        table, "validation: mean(qbg)", "DIALS background.mean"
                    ),
                    "validation: Correlation Coefficient qi": corr_I,
                    "validation: Correlation Coefficient bg": corr_bg,
                    "validation: Max mean(I)": torch.max(i_flat),
                    "validation: Mean mean(I)": torch.mean(i_flat),
                    "validation: Mean var(I) ": torch.mean(i_var_flat),
                    "validation: Min var(I)": torch.min(i_var_flat),
                    "validation: Max var(I)": torch.max(i_var_flat),
                    "validation: mean(qbg.mean)": torch.mean(
                        self.preds_validation["qbg_mean"]
                    ),
                    "validation: min(qbg.mean)": torch.min(
                        self.preds_validation["qbg_mean"]
                    ),
                    "validation: max(qbg.mean)": torch.max(
                        self.preds_validation["qbg_mean"]
                    ),
                }

                # plot input shoebox and predicted profile
                comparison_fig = create_comparison_grid(
                    n_profiles=self.n_profiles,
                    refl_ids=self.tracked_ids_val,
                    pred_dict=self.tracked_shoeboxes_val,
                )

                log_dict["validation: Tracked Profiles"] = wandb.Image(
                    comparison_fig
                )
                plt.close(comparison_fig)

                fig = plot_symlog_qi_vs_dials(
                    i_flat.cpu().numpy(), dials_flat.cpu().numpy()
                )
                wandb.log({"validation: qi_vs_dials_symlog": wandb.Image(fig)})
                plt.close(fig)

                # Log metrics
                wandb.log(log_dict)

            except Exception as e:
                print("Caught exception in on_val_epoch_end")
                print("Type of exception:", type(e))
                print("Exception object:", e)
                traceback.print_exc(file=sys.stdout)

            # Clear memory
            self.preds_validation = {}
            torch.cuda.empty_cache()


class Plotter(Callback):
    def __init__(
        self,
        n_profiles: int = 5,
        plot_every_n_epochs: int = 5,
        d: int = 3,
        h: int = 21,
        w: int = 21,
    ):
        super().__init__()
        self.d = d
        self.h = h
        self.w = w
        self.preds_train = {}
        self.preds_validation = {}
        self.n_profiles = n_profiles
        self.tracked_ids_train = None
        self.tracked_ids_val = None
        self.epoch_preds = None
        self.plot_every_n_epochs = plot_every_n_epochs
        self.current_epoch = 0
        self.tracked_shoeboxes_val = dict()
        self.tracked_shoeboxes_train = dict()

    def update_tracked_shoeboxes(
        self,
        preds,
        tracked_ids,
        tracked_shoeboxes,
    ):
        current_refl_ids = preds["refl_ids"].int()

        if tracked_ids is None:
            # tracked_ids = current_refl_ids[: self.n_profiles]
            tracked_ids = to_cpu(current_refl_ids[: self.n_profiles]).tolist()

            print(
                f"Selected {self.n_profiles} refl_ids to track: {tracked_ids}"
            )

        profile_images = preds["profile"].reshape(-1, self.d, self.h, self.w)[
            ..., (self.d - 1) // 2, :, :
        ]
        count_images = preds["counts"].reshape(-1, self.d, self.h, self.w)[
            ..., (self.d - 1) // 2, :, :
        ]
        rate_images = (
            preds["rates"]
            .mean(1)
            .reshape(-1, self.d, self.h, self.w)[..., (self.d - 1) // 2, :, :]
        )

        for ref_id in tracked_ids:
            id_str = str(int(ref_id))
            matches = torch.where(current_refl_ids == ref_id)[0]

            if len(matches) > 0:
                idx = matches[0].item()

                tracked_shoeboxes[id_str] = {
                    "profile": profile_images[idx].cpu(),
                    "counts": count_images[idx].cpu(),
                    "rates": rate_images[idx].cpu(),
                    "bg_mean": preds["qbg_mean"][idx].cpu(),
                    "bg_var": preds["qbg_var"][idx].cpu(),
                    "qi_mean": preds["qi_mean"][idx].cpu(),
                    "qi_var": preds["qi_var"][idx].cpu(),
                    "intensity.prf.value": preds["intensity.prf.value"][idx],
                    "intensity.prf.variance": preds["intensity.prf.variance"][
                        idx
                    ],
                    "background.mean": preds["background.mean"][idx].cpu(),
                    "xyzcal.px.0": preds["xyzcal.px.0"][idx].cpu(),
                    "xyzcal.px.1": preds["xyzcal.px.1"][idx].cpu(),
                    "xyzcal.px.2": preds["xyzcal.px.2"][idx].cpu(),
                    "d": preds["d"][idx].cpu(),
                }

        torch.cuda.empty_cache()
        return tracked_ids, tracked_shoeboxes

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        with torch.no_grad():
            # get forward outputs
            base_output = outputs["model_output"]

            # additional metrics to log

            if self.current_epoch % self.plot_every_n_epochs == 0:
                self.tracked_ids_train, self.tracked_shoeboxes_train = (
                    self.update_tracked_shoeboxes(
                        base_output,
                        self.tracked_ids_train,
                        self.tracked_shoeboxes_train,
                    )
                )

            # Create CPU tensor versions to avoid keeping GPU memory
            self.preds_train = {}
            for key in [
                "qi_mean",
                "qi_var",
                "intensity.prf.value",
                "intensity.prf.variance",
                "profile",
                "qbg_mean",
                "xyzcal.px.0",
                "xyzcal.px.1",
                "xyzcal.px.2",
                "background.mean",
                "dials_bg_sum_value",
                "d",
            ]:
                if key in base_output:
                    if key == "profile":
                        self.preds_train[key] = to_cpu(base_output[key])
                    elif hasattr(base_output[key], "sample"):
                        self.preds_train[key] = to_cpu(base_output[key].mean)

                    else:
                        self.preds_train[key] = to_cpu(base_output[key])

            # Clean up
            torch.cuda.empty_cache()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.preds_train:
            try:
                # Create data for scatter plots
                data = []

                i_flat = self.preds_train["qi_mean"].flatten() + 1e-8

                i_var_flat = self.preds_train["qi_var"].flatten() + 1e-8

                dials_flat = (
                    self.preds_train["intensity.prf.value"].flatten() + 1e-8
                )
                dials_var_flat = (
                    self.preds_train["intensity.prf.variance"].flatten() + 1e-8
                )
                dials_bg_flat = (
                    self.preds_train["background.mean"].flatten() + 1e-8
                )
                qbg_flat = self.preds_train["qbg_mean"].flatten() + 1e-8

                x_c_flat = self.preds_train["xyzcal.px.0"].flatten()
                y_c_flat = self.preds_train["xyzcal.px.1"].flatten()
                z_c_flat = self.preds_train["xyzcal.px.2"].flatten()
                d_flat = 1 / self.preds_train["d"].flatten().pow(2)
                d_ = self.preds_train["d"]

                # Create data points with safe log transform
                for i in range(len(i_flat)):
                    try:
                        data.append(
                            [
                                float(torch.log(i_flat[i])),
                                float(torch.log(i_var_flat[i])),
                                float(torch.log(dials_flat[i])),
                                float(torch.log(dials_var_flat[i])),
                                dials_bg_flat[i],
                                qbg_flat[i],
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
                        "xyzcal.px.0",
                        "xyzcal.px.1",
                        "d",
                        "d_",
                    ],
                )

                # Create table
                table = wandb.Table(dataframe=df)

                # Calculate correlation coefficients
                corr_I = (
                    torch.corrcoef(torch.vstack([i_flat, dials_flat]))[0, 1]
                    if len(i_flat) > 1
                    else 0
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
                            for d in np.linspace(
                                df["d_"].min(), df["d_"].max(), 6
                            )
                        ],
                        tickvals=1
                        / np.linspace(df["d_"].min(), df["d_"].max(), 6) ** 2,
                        tickangle=90,
                    ),
                    "yaxis": dict(showgrid=True, gridcolor="lightgrey"),
                }

                corr_bg = (
                    torch.corrcoef(torch.vstack([dials_bg_flat, dials_flat]))[
                        0, 1
                    ]
                    if len(dials_bg_flat) > 1
                    else 0
                )

                # Create log dictionary
                log_dict = {
                    "Train: qi vs DIALS I prf": wandb.plot.scatter(
                        table, "mean(qI)", "DIALS intensity.prf.value"
                    ),
                    "Train: Bg vs DIALS bg": wandb.plot.scatter(
                        table, "mean(qbg)", "DIALS background.mean"
                    ),
                    "Correlation Coefficient: qi": corr_I,
                    "Correlation Coefficient: bg": corr_bg,
                    "Max mean(I)": torch.max(i_flat),
                    "Mean mean(I)": torch.mean(i_flat),
                    "Mean var(I) ": torch.mean(i_var_flat),
                    "Min var(I)": torch.min(i_var_flat),
                    "Max var(I)": torch.max(i_var_flat),
                }

                log_dict["mean(qbg.mean)"] = torch.mean(
                    self.preds_train["qbg_mean"]
                )
                log_dict["min(qbg.mean)"] = torch.min(
                    self.preds_train["qbg_mean"]
                )
                log_dict["max(qbg.mean)"] = torch.max(
                    self.preds_train["qbg_mean"]
                )

                # plot every n user-specified epochs
                if self.current_epoch % self.plot_every_n_epochs == 0:
                    comparison_fig = create_comparison_grid(
                        n_profiles=self.n_profiles,
                        refl_ids=self.tracked_ids_train,
                        pred_dict=self.tracked_shoeboxes_train,
                    )

                    if comparison_fig is not None:
                        log_dict["Tracked Profiles"] = wandb.Image(
                            comparison_fig
                        )
                        plt.close(comparison_fig)

                # Log metrics
                wandb.log(log_dict)

            except Exception as e:
                print("Caught exception in on_train_epoch_end!")
                print("Type of exception:", type(e))
                print("Exception object:", e)
                traceback.print_exc(file=sys.stdout)

            # Clear memory
            self.preds_train = {}
            torch.cuda.empty_cache()

        # Increment epoch counter
        self.current_epoch += 1

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        with torch.no_grad():
            # get forward outputs
            base_output = outputs["model_output"]

            # additional metrics to log

            # updated tracked shoeboxes
            self.tracked_ids_val, self.tracked_shoeboxes_val = (
                self.update_tracked_shoeboxes(
                    base_output,
                    tracked_ids=self.tracked_ids_val,
                    tracked_shoeboxes=self.tracked_shoeboxes_val,
                )
            )

            self.preds_validation = {}
            for key in [
                "qi_mean",
                "qi_var",
                "intensity.prf.value",
                "intensity.prf.variance",
                "profile",
                "qbg_mean",
                "xyzcal.px.0",
                "xyzcal.px.1",
                "xyzcal.px.2",
                "background.mean",
                "dials_bg_sum_value",
                "d",
            ]:
                if key in base_output:
                    if hasattr(base_output[key], "sample"):
                        self.preds_validation[key] = to_cpu(
                            base_output[key].mean
                        )

                    else:
                        self.preds_validation[key] = to_cpu(base_output[key])

                elif key in base_output:
                    self.preds_validation[key] = to_cpu(base_output[key])

            # Clean up
            del base_output
            torch.cuda.empty_cache()

    def on_validation_epoch_end(
        self,
        trainer,
        pl_module,
    ):
        if self.preds_validation:
            try:
                data = []

                i_flat = self.preds_validation["qi_mean"].flatten() + 1e-8

                i_var_flat = self.preds_validation["qi_var"].flatten() + 1e-8

                dials_flat = (
                    self.preds_validation["intensity.prf.value"].flatten()
                    + 1e-8
                )
                dials_var_flat = (
                    self.preds_validation["intensity.prf.variance"].flatten()
                    + 1e-8
                )
                dials_bg_flat = (
                    self.preds_validation["background.mean"].flatten() + 1e-8
                )
                qbg_flat = self.preds_validation["qbg_mean"].flatten() + 1e-8

                x_c_flat = self.preds_validation["xyzcal.px.0"].flatten()
                y_c_flat = self.preds_validation["xyzcal.px.1"].flatten()
                z_c_flat = self.preds_validation["xyzcal.px.2"].flatten()
                d_flat = 1 / self.preds_validation["d"].flatten().pow(2)
                d_ = self.preds_validation["d"]

                # Create data points with safe log transform
                for i in range(len(i_flat)):
                    try:
                        data.append(
                            [
                                float(torch.log(i_flat[i])),
                                float(torch.log(i_var_flat[i])),
                                float(torch.log(dials_flat[i])),
                                float(torch.log(dials_var_flat[i])),
                                dials_bg_flat[i],
                                qbg_flat[i],
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
                        "validation: mean(qI)",
                        "validation: var(qI)",
                        "DIALS intensity.prf.value",
                        "DIALS intensity.prf.variance",
                        "DIALS background.mean",
                        "validation: mean(qbg)",
                        "xyzcal.px.0",
                        "xyzcal.px.1",
                        "d",
                        "d_",
                    ],
                )

                # Create table
                table = wandb.Table(dataframe=df)

                # Calculate correlation coefficients
                corr_I = (
                    torch.corrcoef(torch.vstack([i_flat, dials_flat]))[0, 1]
                    if len(i_flat) > 1
                    else 0
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
                            for d in np.linspace(
                                df["d_"].min(), df["d_"].max(), 6
                            )
                        ],
                        tickvals=1
                        / np.linspace(df["d_"].min(), df["d_"].max(), 6) ** 2,
                        tickangle=90,
                    ),
                    "yaxis": dict(showgrid=True, gridcolor="lightgrey"),
                }

                corr_bg = (
                    torch.corrcoef(torch.vstack([dials_bg_flat, dials_flat]))[
                        0, 1
                    ]
                    if len(dials_bg_flat) > 1
                    else 0
                )

                # Create log dictionary
                log_dict = {
                    "validation: qi vs DIALS I prf": wandb.plot.scatter(
                        table,
                        "validation: mean(qI)",
                        "DIALS intensity.prf.value",
                    ),
                    "validation: Bg vs DIALS bg": wandb.plot.scatter(
                        table, "validation: mean(qbg)", "DIALS background.mean"
                    ),
                    "validation: Correlation Coefficient qi": corr_I,
                    "validation: Correlation Coefficient bg": corr_bg,
                    "validation: Max mean(I)": torch.max(i_flat),
                    "validation: Mean mean(I)": torch.mean(i_flat),
                    "validation: Mean var(I) ": torch.mean(i_var_flat),
                    "validation: Min var(I)": torch.min(i_var_flat),
                    "validation: Max var(I)": torch.max(i_var_flat),
                    "validation: mean(qbg.mean)": torch.mean(
                        self.preds_validation["qbg_mean"]
                    ),
                    "validation: min(qbg.mean)": torch.min(
                        self.preds_validation["qbg_mean"]
                    ),
                    "validation: max(qbg.mean)": torch.max(
                        self.preds_validation["qbg_mean"]
                    ),
                }

                # plot input shoebox and predicted profile
                comparison_fig = create_comparison_grid(
                    n_profiles=self.n_profiles,
                    refl_ids=self.tracked_ids_val,
                    pred_dict=self.tracked_shoeboxes_val,
                )

                log_dict["validation: Tracked Profiles"] = wandb.Image(
                    comparison_fig
                )
                plt.close(comparison_fig)

                # Log metrics
                wandb.log(log_dict)

            except Exception as e:
                print("Caught exception in on_val_epoch_end")
                print("Type of exception:", type(e))
                print("Exception object:", e)
                traceback.print_exc(file=sys.stdout)

            # Clear memory
            self.preds_validation = {}
            torch.cuda.empty_cache()


# -
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
            "qbg_mean": {},
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
            "qi_mean": [],
            "dials_I_prf_value": [],
            "weighted_sum_mean": [],
            "thresholded_mean": [],
            "qbg_mean": [],
            "rates": [],
        }

        self.tracked_predictions = {
            "profile": {},
            "counts": {},
            "qbg_mean": {},
            "rates": {},
            "qi_mean": {},
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
        qi_mean,
    ):
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

        profile_images = profile_preds.reshape(-1, 3, 21, 21)[..., 1, :, :]
        count_images = count_preds.reshape(-1, 3, 21, 21)[..., 1, :, :]
        rate_images = rates.mean(1).reshape(-1, 3, 21, 21)[..., 1, :, :]
        bg_mean = qbg_preds.mean
        bg_var = qbg_preds.variance
        # qI_var = qi.variance
        dials_I_prf_value = dials_I

        for ref_id in self.tracked_refl_ids:
            matches = np.where(current_refl_ids == ref_id)[0]
            if len(matches) > 0:
                idx = matches[0]

                self.tracked_predictions["profile"][ref_id] = profile_images[
                    idx
                ].cpu()
                self.tracked_predictions["counts"][ref_id] = count_images[
                    idx
                ].cpu()
                self.tracked_predictions["rates"][ref_id] = rate_images[
                    idx
                ].cpu()
                self.tracked_predictions["qbg_mean"][ref_id] = bg_mean[
                    idx
                ].cpu()
                self.tracked_predictions["qi_mean"][ref_id] = qi_mean[idx]
                # self.tracked_predictions["qI_var"][ref_id] = qi_var[idx].cpu()
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
            im0 = axes[0, i].imshow(
                counts_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13
            )
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
            im2 = axes[2, i].imshow(
                rates_data, cmap=cmap, vmin=vmin_13, vmax=vmax_13
            )
            axes[2, i].set_title(
                f"Bg: {self.tracked_predictions['qbg_mean'][refl_id]:.2f}\n qi_mean: {self.tracked_predictions['qi_mean'][refl_id]:.2f}"
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

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        with torch.no_grad():
            # 1) Forward pass (no intensities yet)
            shoebox, dials, masks, metadata, counts = batch
            base_output = pl_module(shoebox, dials, masks, metadata, counts)

            # 2) Call calculate_intensities with the relevant fields
            intensities = pl_module.calculate_intensities(
                counts=base_output["counts"],
                qbg=base_output["qbg_mean"],
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
                    predictions["qbg_mean"],
                    predictions["rates"],
                    predictions["counts"],
                    predictions["refl_ids"],
                    predictions["dials_I_prf_value"],
                    predictions["qi_mean"],
                )

            # Store only a minimal version of the last batch predictions
            # Create CPU tensor versions to avoid keeping GPU memory
            self.train_predictions = {}
            for key in [
                "qi_mean",
                # "qI_var",
                "dials_I_prf_value",
                "weighted_sum_mean",
                "thresholded_mean",
                "profile",
                "qbg_mean",
            ]:
                if key in predictions:
                    if key == "profile":
                        self.train_predictions[key] = (
                            predictions[key].detach().cpu()
                        )
                    elif hasattr(predictions[key], "sample"):
                        self.train_predictions[key] = (
                            predictions[key].mean.detach().cpu()
                        )
                    else:
                        self.train_predictions[key] = (
                            predictions[key].detach().cpu()
                        )

            # Clean up
            del base_output, intensities, predictions
            torch.cuda.empty_cache()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_predictions:
            try:
                # Create data for scatter plots
                data = []

                qI_flat = (
                    self.train_predictions["qi_mean"].flatten() + 1e-8
                )  # Add epsilon before log
                # qI_var_flat = (
                # self.train_predictions["qI_var"].flatten() + 1e-8
                # )  # Add epsilon before log
                dials_flat = (
                    self.train_predictions["dials_I_prf_value"].flatten()
                    + 1e-8
                )
                weighted_sum_flat = (
                    self.train_predictions["weighted_sum_mean"].flatten()
                    + 1e-8
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
                        "qi_mean",
                        # "qI_var",
                        "dials_I_prf_value",
                        "weighted_sum_mean",
                        "thresholded_mean",
                        "simpson_idx",
                    ],
                )

                # Calculate correlation coefficients safely
                corr_qi = (
                    torch.corrcoef(torch.vstack([qI_flat, dials_flat]))[0, 1]
                    if len(qI_flat) > 1
                    else 0
                )
                corr_weighted = (
                    torch.corrcoef(
                        torch.vstack([weighted_sum_flat, dials_flat])
                    )[0, 1]
                    if len(weighted_sum_flat) > 1
                    else 0
                )
                corr_masked = (
                    torch.corrcoef(
                        torch.vstack([thresholded_flat, dials_flat])
                    )[0, 1]
                    if len(thresholded_flat) > 1
                    else 0
                )

                # Create log dictionary
                log_dict = {
                    "train_qI_vs_prf": wandb.plot.scatter(
                        table, "qi_mean", "dials_I_prf_value"
                    ),
                    "train_weighted_sum_vs_prf": wandb.plot.scatter(
                        table, "weighted_sum_mean", "dials_I_prf_value"
                    ),
                    "train_thresholded_vs_prf": wandb.plot.scatter(
                        table, "thresholded_mean", "dials_I_prf_value"
                    ),
                    "corrcoef_qi": corr_qi,
                    "corrcoef_weighted": corr_weighted,
                    "corrcoef_masked": corr_masked,
                    "max_qi": torch.max(qI_flat),
                    "mean_qi": torch.mean(qI_flat),
                    # "mean_qI_var": torch.mean(qI_var_flat),
                    # "min_qI_var": torch.min(qI_var_flat),
                    # "max_qI_var": torch.max(qI_var_flat),
                }

                # Add mean background if available
                if "qbg_mean" in self.train_predictions:
                    log_dict["mean_bg"] = torch.mean(
                        self.train_predictions["qbg_mean"]
                    )
                    log_dict["min_bg"] = torch.min(
                        self.train_predictions["qbg_mean"]
                    )
                    log_dict["max_bg"] = torch.max(
                        self.train_predictions["qbg_mean"]
                    )

                # Only create and log comparison grid on specified epochs
                if self.current_epoch % self.plot_every_n_epochs == 0:
                    comparison_fig = self.create_comparison_grid()
                    if comparison_fig is not None:
                        log_dict["Tracked Profiles"] = wandb.Image(
                            comparison_fig
                        )
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

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        _batch_idx,
    ):
        # Only track the last validation batch to save memory

        with torch.no_grad():
            base_output = outputs

            intensities = pl_module.calculate_intensities(
                counts=base_output["counts"],
                qbg=base_output["qbg_mean"],
                profile=base_output["profile"],
                dead_pixel_mask=base_output["masks"],
            )

            # Store only minimal data needed for metrics
            self.val_predictions = {}
            for key in [
                "qi_mean",
                "dials_I_prf_value",
                "weighted_sum_mean",
                "thresholded_mean",
            ]:
                if key in base_output:
                    if hasattr(base_output[key], "sample"):
                        self.val_predictions[key] = (
                            base_output[key].mean.detach().cpu()
                        )
                    else:
                        self.val_predictions[key] = (
                            base_output[key].detach().cpu()
                        )
                elif key in intensities:
                    self.val_predictions[key] = intensities[key].detach().cpu()

            # Clean up
            del base_output, intensities
            torch.cuda.empty_cache()


# %%
if __name__ == "__main__":
    import polars as pl
    import torch

    from integrator.utils import (
        create_data_loader,
        create_integrator,
        load_config,
    )
    from utils import CONFIGS

    cfg = list(CONFIGS.glob("*"))[0]

    cfg = load_config(cfg)

    integrator = create_integrator(cfg)
    data = create_data_loader(cfg)

    # load a batch
    counts, sbox, mask, meta = next(iter(data.train_dataloader()))

    # get training step output
    training_step_out = integrator.training_step((counts, sbox, mask, meta), 0)

    out = training_step_out["model_output"]

    # %%

    # %%

    # setting up bin edges
    edges = [0, 50, 100, 300, 600, 1000, 1500, 2500]
    bin_edges = zip(edges[:-1], edges[1:], strict=False)

    bin_labels = []
    for l, r in bin_edges:
        bin_labels.append(f"{l} - {r}")

    # add end conditions
    bin_labels.insert(0, f"<{bin_labels[0].split()[0]}")
    bin_labels.append(f">{bin_labels[-1].split()[0]}")

    # %%
    fano = _fano(out, "qi_mean", "qi_var")

    df = pl.DataFrame(
        {
            "refl_ids": out["refl_ids"],
            "qi_mean": out["qi_mean"].detach().cpu(),
            "qi_var": out["qi_var"].detach().cpu(),
            "fano": fano.detach().cpu(),
        }
    )

    df = df.with_columns(
        pl.col("qi_mean").cut(edges, labels=bin_labels).alias("intensity_bin")
    )

    avg_df = df.group_by(pl.col("intensity_bin")).agg(
        fano_mean=pl.col("fano").mean(), n=pl.len()
    )

    base_df = pl.DataFrame(
        {"intensity_bin": bin_labels}, schema={"intensity_bin": pl.Categorical}
    )

    merged_df = base_df.join(avg_df, how="left", on="intensity_bin").fill_null(
        0
    )

    agg_df = pl.DataFrame(
        data={
            "intensity_bin": bin_labels,
            "fano_mean": pl.zeros(len(bin_labels), eager=True),
            "n": pl.zeros(len(bin_labels), eager=True),
        },
        schema={
            "intensity_bin": pl.Categorical,
            "fano_mean": pl.Float32,
            "n": pl.Float32,
        },
    )
    numeric_cols = ["fano_mean", "n"]

    result = agg_df.with_columns(
        [pl.col(c) + merged_df[c] for c in numeric_cols]
    )
    epoch_df = result.with_columns(
        (pl.col("fano_mean") / pl.col("n")).alias("avg_fano")
    )

    # %%
    def plot_avg_fano(df):
        fig, ax = plt.subplots()

        labels = df["intensity_bin"].to_list()
        y = df["avg_fano"].to_list()
        x = np.linspace(0, len(y), len(y))

        ax.scatter(x, y, color="black")
        ax.set_xticks(ticks=x, labels=labels)
        ax.set_xlabel("intensity bin")
        ax.set_ylabel("avg var/mean ratio")
        ax.set_title("Average variance/mean per intensity bin")
        ax.grid()
        return fig

    fig = plot_avg_fano(epoch_df)
    pass
