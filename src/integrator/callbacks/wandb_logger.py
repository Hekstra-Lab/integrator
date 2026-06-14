import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning.callbacks import Callback
from torch import Tensor

from .run_logger import get_run_logger


def to_cpu(x) -> torch.Tensor:
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
    """Plot a 3-row grid (counts, profile, rate) per tracked shoebox.

    Args:
        n_profiles (int): number of shoeboxes to plot
        refl_ids (list): list of tracked shoebox ids
        pred_dict (dict): dictionary of tracked shoeboxes
        cmap (str): name of the color map

    Returns:
        A matplotlib figure.
    """
    fig, axes = plt.subplots(3, n_profiles, figsize=(5 * n_profiles, 8))

    for i, refl_id in enumerate(refl_ids):
        id_str = str(refl_id)
        counts_data = pred_dict[id_str]["counts"]
        profile_data = pred_dict[id_str]["profile"]
        rates_data = pred_dict[id_str]["rates"]

        vmin_13 = min(counts_data.min().item(), rates_data.min().item())
        vmax_13 = max(counts_data.max().item(), rates_data.max().item())

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

        axes[0, i].tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )

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

        # First row colorbar (same scale as third row)
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
        self.bin_edges = torch.tensor(edges)

        self.qi_mean = []
        self.qi_var = []

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        if outputs is None:
            return

        out = outputs["forward_out"]

        self.qi_mean.append(out["qi_mean"].detach())
        self.qi_var.append(out["qi_var"].detach())

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.qi_mean:
            return

        qi = torch.cat(self.qi_mean).cpu()
        qv = torch.cat(self.qi_var).cpu()

        # clear buffers early to release memory
        self.qi_mean.clear()
        self.qi_var.clear()

        fano = qv / (qi + 1e-8)
        isigi = qi / (qv.sqrt() + 1e-8)
        cv = qv.sqrt() / (qi + 1e-8)

        bin_idx = torch.bucketize(qi, self.bin_edges)

        n_bins = len(self.bin_edges) + 1

        fano_sum = torch.zeros(n_bins)
        cv_sum = torch.zeros(n_bins)
        isigi_sum = torch.zeros(n_bins)
        counts = torch.zeros(n_bins)

        for i in range(n_bins):
            mask = bin_idx == i
            if mask.any():
                fano_sum[i] = fano[mask].sum()
                cv_sum[i] = cv[mask].sum()
                isigi_sum[i] = isigi[mask].sum()
                counts[i] = mask.sum()

        # avoid division by zero
        valid = counts > 0
        avg_fano = torch.zeros_like(fano_sum)
        avg_cv = torch.zeros_like(cv_sum)
        avg_isigi = torch.zeros_like(isigi_sum)

        avg_fano[valid] = fano_sum[valid] / counts[valid]
        avg_cv[valid] = cv_sum[valid] / counts[valid]
        avg_isigi[valid] = isigi_sum[valid] / counts[valid]

        rl = get_run_logger(self, trainer)
        rl.log_scalars(
            {
                "train/avg_fano": avg_fano[valid].mean().item(),
                "train/avg_cv": avg_cv[valid].mean().item(),
                "train/avg_isigi": avg_isigi[valid].mean().item(),
            }
        )

        fig = _plot_avg_fano(
            pl.DataFrame(
                {
                    "intensity_bin": list(range(n_bins)),
                    "avg_fano": avg_fano.numpy(),
                }
            )
        )
        rl.log_figure("train/fano_vs_bin", fig, step=trainer.current_epoch)


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
            # 3D shoebox: take the central slice along d
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

        return tracked_ids, tracked_shoeboxes

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        with torch.no_grad():
            forward_out = outputs["forward_out"]

            if self.current_epoch % self.plot_every_n_epochs == 0:
                self.tracked_ids_train, self.tracked_shoeboxes_train = (
                    self.update_tracked_shoeboxes(
                        forward_out,
                        self.tracked_ids_train,
                        self.tracked_shoeboxes_train,
                    )
                )

            # move to CPU to avoid holding GPU memory across the epoch
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
                if key in forward_out:
                    if key == "profile":
                        self.preds_train[key] = to_cpu(forward_out[key])
                    elif hasattr(forward_out[key], "sample"):
                        self.preds_train[key] = to_cpu(forward_out[key].mean)
                    else:
                        self.preds_train[key] = to_cpu(forward_out[key])

    def on_train_epoch_end(self, trainer, pl_module):
        if self.preds_train:
            try:
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

                df = pl.DataFrame(
                    {
                        "mean(qI)": i_flat.cpu().numpy(),
                        "var(qI)": i_var_flat.cpu().numpy(),
                        "DIALS intensity.prf.value": dials_flat.cpu().numpy(),
                        "DIALS intensity.prf.variance": (
                            dials_var_flat.cpu().numpy()
                        ),
                        "DIALS background.mean": dials_bg_flat.cpu().numpy(),
                        "mean(qbg)": qbg_flat.cpu().numpy(),
                        "x_c": x_c_flat.cpu().numpy(),
                        "y_c": y_c_flat.cpu().numpy(),
                    }
                )

                rl = get_run_logger(self, trainer)

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

                rl.log_scatter(
                    "Train: qi vs DIALS I prf",
                    df,
                    "mean(qI)",
                    "DIALS intensity.prf.value",
                    step=self.current_epoch,
                )
                rl.log_scatter(
                    "Train: Bg vs DIALS bg",
                    df,
                    "mean(qbg)",
                    "DIALS background.mean",
                    step=self.current_epoch,
                )

                log_dict = {
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

                rl.log_scalars(log_dict, step=self.current_epoch)

                if self.current_epoch % self.plot_every_n_epochs == 0:
                    comparison_fig = create_comparison_grid(
                        n_profiles=self.n_profiles,
                        refl_ids=self.tracked_ids_train,
                        pred_dict=self.tracked_shoeboxes_train,
                    )

                    if comparison_fig is not None:
                        rl.log_figure(
                            "Tracked Profiles",
                            comparison_fig,
                            step=self.current_epoch,
                        )

                fig = plot_symlog_qi_vs_dials(
                    to_cpu(i_flat).numpy(), dials_flat.cpu().numpy()
                )
                rl.log_figure(
                    "train: qi_vs_dials_symlog", fig, step=self.current_epoch
                )

            except Exception as e:
                print("Caught exception in on_train_epoch_end!")
                print("Type of exception:", type(e))
                print("Exception object:", e)
                traceback.print_exc(file=sys.stdout)

            self.preds_train = {}

        self.current_epoch += 1

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        with torch.no_grad():
            forward_out = outputs["forward_out"]

            self.tracked_ids_val, self.tracked_shoeboxes_val = (
                self.update_tracked_shoeboxes(
                    forward_out,
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
                if key in forward_out:
                    if hasattr(forward_out[key], "sample"):
                        self.preds_validation[key] = to_cpu(
                            forward_out[key].mean
                        )
                    else:
                        self.preds_validation[key] = to_cpu(forward_out[key])
                elif key in forward_out:
                    self.preds_validation[key] = to_cpu(forward_out[key])

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.preds_validation:
            try:
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

                df = pl.DataFrame(
                    {
                        "validation: mean(qI)": i_flat.cpu().numpy(),
                        "validation: var(qI)": i_var_flat.cpu().numpy(),
                        "DIALS intensity.prf.value": dials_flat.cpu().numpy(),
                        "DIALS intensity.prf.variance": (
                            dials_var_flat.cpu().numpy()
                        ),
                        "DIALS background.mean": dials_bg_flat.cpu().numpy(),
                        "validation: mean(qbg)": qbg_flat.cpu().numpy(),
                        "x_c": x_c_flat.cpu().numpy(),
                        "y_c": y_c_flat.cpu().numpy(),
                    }
                )

                rl = get_run_logger(self, trainer)

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

                rl.log_scatter(
                    "Validation: qi vs DIALS I prf",
                    df,
                    "validation: mean(qI)",
                    "DIALS intensity.prf.value",
                    step=self.current_epoch,
                )
                rl.log_scatter(
                    "Validation: Bg vs DIALS bg",
                    df,
                    "validation: mean(qbg)",
                    "DIALS background.mean",
                    step=self.current_epoch,
                )

                log_dict = {
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

                rl.log_scalars(log_dict, step=self.current_epoch)

                comparison_fig = create_comparison_grid(
                    n_profiles=self.n_profiles,
                    refl_ids=self.tracked_ids_val,
                    pred_dict=self.tracked_shoeboxes_val,
                )

                rl.log_figure(
                    "validation: Tracked Profiles",
                    comparison_fig,
                    step=self.current_epoch,
                )

                fig = plot_symlog_qi_vs_dials(
                    i_flat.cpu().numpy(), dials_flat.cpu().numpy()
                )
                rl.log_figure(
                    "validation: qi_vs_dials_symlog",
                    fig,
                    step=self.current_epoch,
                )

            except Exception as e:
                print("Caught exception in on_val_epoch_end")
                print("Type of exception:", type(e))
                print("Exception object:", e)
                traceback.print_exc(file=sys.stdout)

            self.preds_validation = {}


class LossTraceRecorder(Callback):
    """Record per-step loss components to CSV/parquet without slowing training.

    Accumulates scalar loss values in plain Python lists (no GPU tensors held),
    then flushes to disk once per epoch.

    Columns: step, loss, nll, kl, kl_prf, kl_i, kl_bg
    """

    _KEYS = ("loss", "nll", "kl", "kl_prf", "kl_i", "kl_bg")

    def __init__(
        self,
        out_dir: str | Path,
        use_parquet: bool = True,
    ):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.use_parquet = use_parquet
        self._rows: list[dict[str, float]] = []

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        lc = (
            outputs.get("loss_components")
            if isinstance(outputs, dict)
            else None
        )
        if lc is None:
            return
        row = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
        }
        for k in self._KEYS:
            v = lc.get(k)
            row[k] = float(v) if v is not None else float("nan")
        self._rows.append(row)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        lc = (
            outputs.get("loss_components")
            if isinstance(outputs, dict)
            else None
        )
        if lc is None:
            return
        row = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
        }
        for k in self._KEYS:
            v = lc.get(k)
            row[k] = float(v) if v is not None else float("nan")
        # tag so train/val are distinguishable if someone merges files
        row["split"] = "val"
        self._rows.append(row)

    def on_train_epoch_end(self, trainer, pl_module):
        self._flush(trainer, split="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._flush(trainer, split="val")

    def _flush(self, trainer, split: str):
        if not self._rows:
            return

        epoch = trainer.current_epoch
        suffix = "parquet" if self.use_parquet else "csv"
        fname = self.out_dir / f"loss_trace_{split}_epoch_{epoch:04d}.{suffix}"

        df = pl.DataFrame(self._rows)
        if self.use_parquet:
            df.write_parquet(fname)
        else:
            df.write_csv(fname)

        self._rows.clear()


class EpochMetricRecorder(Callback):
    def __init__(
        self,
        out_dir: str | Path,
        keys: list[str],
        split: str = "train",  # "train" or "val"
        every_n_epochs: int = 1,
        max_rows_per_epoch: int | None = None,
        use_parquet: bool = True,
    ):
        super().__init__()
        self.out_dir = out_dir
        self.keys = keys
        self.split = split
        self.every_n_epochs = every_n_epochs
        self.max_rows_per_epoch = max_rows_per_epoch
        self.use_parquet = use_parquet

        self.buffers: dict[str, list[torch.Tensor]] = {}
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    def _append(self, key, tensor):
        self.buffers.setdefault(key, []).append(tensor.detach())

    def _collect(self, outputs):
        out = outputs["forward_out"]
        for key in self.keys:
            if key in out:
                self._append(key, out[key])

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        if self.split == "train":
            self._collect(outputs)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        if self.split == "val":
            self._collect(outputs)

    def _flush(self, trainer):
        epoch = trainer.current_epoch

        if epoch % self.every_n_epochs != 0:
            self.buffers.clear()
            return

        if not self.buffers:
            return

        data = {}

        for key, chunks in self.buffers.items():
            x = torch.cat(chunks)

            if (
                self.max_rows_per_epoch
                and x.shape[0] > self.max_rows_per_epoch
            ):
                idx = torch.randperm(x.shape[0])[: self.max_rows_per_epoch]
                x = x[idx]

            data[key] = x.cpu().numpy()

        df = pl.DataFrame(data).select(
            pl.lit(epoch).alias("epoch"), pl.all()
        )

        suffix = "parquet" if self.use_parquet else "csv"
        fname = f"{self.out_dir}/{self.split}_epoch_{epoch:04d}.{suffix}"

        if self.use_parquet:
            df.write_parquet(fname)
        else:
            df.write_csv(fname)

        print(f"[Recorder] wrote {fname}")

        self.buffers.clear()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.split == "train":
            self._flush(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.split == "val":
            self._flush(trainer)


class Plotter(Callback):
    def __init__(
        self,
        n_profiles: int = 5,
        plot_every_n_epochs: int = 5,
        d: int = 3,
        h: int = 21,
        w: int = 21,
        eps: float = 1e-8,
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
        self.eps = eps

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

        return tracked_ids, tracked_shoeboxes

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if outputs is None:
            return

        if not isinstance(outputs, Mapping):
            raise TypeError("Outputs should be a dictionary type")

        if "forward_out" not in outputs:
            raise KeyError(
                "outputs dictionary should contain a 'forward_out' key"
            )

        with torch.no_grad():
            forward_out = outputs["forward_out"]

            if self.current_epoch % self.plot_every_n_epochs == 0:
                self.tracked_ids_train, self.tracked_shoeboxes_train = (
                    self.update_tracked_shoeboxes(
                        forward_out,
                        self.tracked_ids_train,
                        self.tracked_shoeboxes_train,
                    )
                )

            # move to CPU to avoid holding GPU memory across the epoch
            self.preds_train = {k: to_cpu(v) for k, v in forward_out.items()}

    def on_train_epoch_end(self, trainer, pl_module):
        if self.preds_train:
            try:
                i_flat = self.preds_train["qi_mean"].flatten() + self.eps
                i_var_flat = self.preds_train["qi_var"].flatten() + self.eps

                dials_flat = (
                    self.preds_train["intensity.prf.value"].flatten()
                    + self.eps
                )
                dials_var_flat = (
                    self.preds_train["intensity.prf.variance"].flatten()
                    + self.eps
                )
                dials_bg_flat = (
                    self.preds_train["background.mean"].flatten() + self.eps
                )
                qbg_flat = self.preds_train["qbg_mean"].flatten() + self.eps

                x_c_flat = self.preds_train["xyzcal.px.0"].flatten()
                y_c_flat = self.preds_train["xyzcal.px.1"].flatten()
                z_c_flat = self.preds_train["xyzcal.px.2"].flatten()

                d_flat = 1 / self.preds_train["d"].flatten().pow(2)
                d_ = self.preds_train["d"]

                df = pl.DataFrame(
                    {
                        "mean(qI)": i_flat.cpu().numpy(),
                        "var(qI)": i_var_flat.cpu().numpy(),
                        "DIALS intensity.prf.value": dials_flat.cpu().numpy(),
                        "DIALS intensity.prf.variance": (
                            dials_var_flat.cpu().numpy()
                        ),
                        "DIALS background.mean": dials_bg_flat.cpu().numpy(),
                        "mean(qbg)": qbg_flat.cpu().numpy(),
                        "xyzcal.px.0": x_c_flat.cpu().numpy(),
                        "xyzcal.px.1": y_c_flat.cpu().numpy(),
                        "d": d_flat.cpu().numpy(),
                        "d_": d_.flatten().cpu().numpy(),
                    }
                )

                rl = get_run_logger(self, trainer)

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

                rl.log_scatter(
                    "Train: qi vs DIALS I prf",
                    df,
                    "mean(qI)",
                    "DIALS intensity.prf.value",
                    step=self.current_epoch,
                )
                rl.log_scatter(
                    "Train: Bg vs DIALS bg",
                    df,
                    "mean(qbg)",
                    "DIALS background.mean",
                    step=self.current_epoch,
                )

                log_dict = {
                    "Train correlation coefficient: qi": corr_I,
                    "Train correlation coefficient: bg": corr_bg,
                    "Train max mean(I)": torch.max(i_flat),
                    "Train mean mean(I)": torch.mean(i_flat),
                    "Train mean var(I) ": torch.mean(i_var_flat),
                    "Train min var(I)": torch.min(i_var_flat),
                    "Train max var(I)": torch.max(i_var_flat),
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

                rl.log_scalars(log_dict, step=self.current_epoch)

                if self.current_epoch % self.plot_every_n_epochs == 0:
                    comparison_fig = create_comparison_grid(
                        n_profiles=self.n_profiles,
                        refl_ids=self.tracked_ids_train,
                        pred_dict=self.tracked_shoeboxes_train,
                    )

                    if comparison_fig is not None:
                        rl.log_figure(
                            "Tracked Profiles",
                            comparison_fig,
                            step=self.current_epoch,
                        )

            except Exception as e:
                print("Caught exception in on_train_epoch_end!")
                print("Type of exception:", type(e))
                print("Exception object:", e)
                traceback.print_exc(file=sys.stdout)

            self.preds_train = {}

        self.current_epoch += 1

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        with torch.no_grad():
            forward_out = outputs["forward_out"]

            self.tracked_ids_val, self.tracked_shoeboxes_val = (
                self.update_tracked_shoeboxes(
                    forward_out,
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
                if key in forward_out:
                    if hasattr(forward_out[key], "sample"):
                        self.preds_validation[key] = to_cpu(
                            forward_out[key].mean
                        )

                    else:
                        self.preds_validation[key] = to_cpu(forward_out[key])

                elif key in forward_out:
                    self.preds_validation[key] = to_cpu(forward_out[key])

            del forward_out

    def on_validation_epoch_end(
        self,
        trainer,
        pl_module,
    ):
        if self.preds_validation:
            try:
                i_flat = self.preds_validation["qi_mean"].flatten() + self.eps
                i_var_flat = (
                    self.preds_validation["qi_var"].flatten() + self.eps
                )

                dials_flat = (
                    self.preds_validation["intensity.prf.value"].flatten()
                    + self.eps
                )
                dials_var_flat = (
                    self.preds_validation["intensity.prf.variance"].flatten()
                    + self.eps
                )
                dials_bg_flat = (
                    self.preds_validation["background.mean"].flatten()
                    + self.eps
                )
                qbg_flat = (
                    self.preds_validation["qbg_mean"].flatten() + self.eps
                )

                x_c_flat = self.preds_validation["xyzcal.px.0"].flatten()
                y_c_flat = self.preds_validation["xyzcal.px.1"].flatten()
                z_c_flat = self.preds_validation["xyzcal.px.2"].flatten()
                d_flat = 1 / self.preds_validation["d"].flatten().pow(2)
                d_ = self.preds_validation["d"]

                df = pl.DataFrame(
                    {
                        "validation: mean(qI)": i_flat.cpu().numpy(),
                        "validation: var(qI)": i_var_flat.cpu().numpy(),
                        "DIALS intensity.prf.value": dials_flat.cpu().numpy(),
                        "DIALS intensity.prf.variance": (
                            dials_var_flat.cpu().numpy()
                        ),
                        "DIALS background.mean": dials_bg_flat.cpu().numpy(),
                        "validation: mean(qbg)": qbg_flat.cpu().numpy(),
                        "xyzcal.px.0": x_c_flat.cpu().numpy(),
                        "xyzcal.px.1": y_c_flat.cpu().numpy(),
                        "d": d_flat.cpu().numpy(),
                        "d_": d_.flatten().cpu().numpy(),
                    }
                )

                rl = get_run_logger(self, trainer)

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

                rl.log_scatter(
                    "validation: qi vs DIALS I prf",
                    df,
                    "validation: mean(qI)",
                    "DIALS intensity.prf.value",
                    step=self.current_epoch,
                )
                rl.log_scatter(
                    "validation: Bg vs DIALS bg",
                    df,
                    "validation: mean(qbg)",
                    "DIALS background.mean",
                    step=self.current_epoch,
                )

                log_dict = {
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

                rl.log_scalars(log_dict, step=self.current_epoch)

                comparison_fig = create_comparison_grid(
                    n_profiles=self.n_profiles,
                    refl_ids=self.tracked_ids_val,
                    pred_dict=self.tracked_shoeboxes_val,
                )

                rl.log_figure(
                    "validation: Tracked Profiles",
                    comparison_fig,
                    step=self.current_epoch,
                )

            except Exception as e:
                print("Caught exception in on_val_epoch_end")
                print("Type of exception:", type(e))
                print("Exception object:", e)
                traceback.print_exc(file=sys.stdout)

            self.preds_validation = {}
