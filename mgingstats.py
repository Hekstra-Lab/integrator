#!/usr/bin/env python
import argparse
import itertools
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import polars as plr
import seaborn as sns
import torch
from bs4 import BeautifulSoup
from dials.array_family import flex
from matplotlib.colors import Normalize, TwoSlopeNorm
from plotly.subplots import make_subplots

import wandb
from integrator.utils import load_config


########################################
# 1. PARSING FUNCTION
########################################
def parse_html(file_path):
    """Parse the merged-*.html file for resolution shell statistics."""
    with open(file_path) as file:
        soup = BeautifulSoup(file, "html.parser")
    table_div = soup.find("div", {"id": "collapse_merging_stats_1_03752"})
    if not table_div:
        raise ValueError(
            f"Could not find div with id='collapse_merging_stats_1_03752' in {file_path}"
        )
    table = table_div.find("table")
    if not table:
        raise ValueError(f"Could not find <table> inside that div in {file_path}")
    # Extract headers
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    # Extract data rows (skip the header row)
    rows = []
    for row in table.find_all("tr")[1:]:
        row_vals = [td.get_text(strip=True) for td in row.find_all("td")]
        rows.append(row_vals)

    rows_array = np.array(rows)
    # remove asterisks if any
    cleaned_array = np.char.replace(rows_array, "*", "")
    return headers, cleaned_array


########################################
# 2. DATA MANAGER CLASS
########################################


class DataManager:
    def __init__(self, predictions_path, reference_path=None):
        self.predictions_path = Path(predictions_path)
        self.html_files = sorted(
            list(self.predictions_path.glob("**/merged.html")),
            key=lambda x: int(x.parents[2].name.split("_")[1]),
        )
        self.csv_files = sorted(
            list(self.predictions_path.glob("**/peaks.csv")),
            key=lambda x: int(x.parents[3].name.split("_")[1]),
        )
        self.data_dict = defaultdict(lambda: defaultdict(dict))
        self.counting_methods = []

        for h, c in zip(self.html_files, self.csv_files):
            counting_method = h.parent.name.replace("dials_out_", "")
            if counting_method not in self.counting_methods:
                self.counting_methods.append(counting_method)
            epoch = h.parents[2].name
            headers, cleaned_array = parse_html(h)
            df = plr.read_csv(c)

            # Extract epoch number from epoch name (e.g., "epoch_99" -> "99")
            epoch_num = (
                re.findall(r"\d+", epoch)[0] if re.findall(r"\d+", epoch) else None
            )

            val = None
            if epoch_num:
                # Find the checkpoint file that matches this epoch
                ckpt_files = list(self.predictions_path.glob("**/*=*.ckpt"))
                for ckpt_file in ckpt_files:
                    if f"epoch={epoch_num}-" in ckpt_file.name:
                        # Extract val_loss from the matched checkpoint file
                        val_matches = re.findall(r"val_loss=(\d+\.\d+)", ckpt_file.name)
                        if val_matches:
                            val = float(val_matches[0])
                        break

            self.data_dict[counting_method][epoch]["merging_stats"] = cleaned_array
            self.data_dict[counting_method][epoch]["peaks"] = df
            self.data_dict[counting_method][epoch]["val"] = val

        if reference_path:
            self.reference_path = Path(reference_path)
            self.reference_peaks = plr.read_csv(self.reference_path / "peaks_ref.csv")
            self.reference_html = self.reference_path / "merged_reference.html"
            self.reference_data_dict = defaultdict(dict)
            headers, cleaned_array = parse_html(self.reference_html)
            self.reference_data_dict["merging_stats"] = cleaned_array
            self.reference_data_dict["peaks"] = self.reference_peaks

        self.metrics = {
            "cc_half": {
                "name": "CC_half",
                "display_name": "CC 1/2",
                "col_idx": -2,
                "color": "green",
            },
            "cc_anom": {
                "name": "CC_anom",
                "display_name": "CC anom",
                "col_idx": -1,
                "color": "red",
            },
            "r_pim": {
                "name": "R_pim",
                "display_name": "R pim",
                "col_idx": -4,
                "color": "blue",
            },
            "I_vs_sigI": {
                "name": "I_sigI",
                "display_name": "Mean I/sig(I)",
                "col_idx": -7,
                "color": "blue",
            },
        }

    def plot_peaks(self, counting_method, show=False, plot_reference=False):
        num_peaks = [
            self.data_dict[counting_method][epoch]["peaks"].height
            for epoch in self.data_dict[counting_method]
        ]
        x = np.arange(len(num_peaks))
        plt.plot(x, num_peaks, label=counting_method)
        if plot_reference and hasattr(self, "reference_peaks"):
            plt.plot(
                x,
                np.ones(len(x)) * self.reference_peaks.height,
                label="DIALS",
                linestyle="--",
                color="black",
            )
        if show:
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Number of peaks")
            plt.grid(alpha=0.3)
            plt.show()

    def plot_resolution(
        self,
        counting_method,
        epoch,
        metric,
        show=False,
        show_reference=False,
        show_legend=True,
    ):
        y_data = self.data_dict[counting_method][epoch]["merging_stats"][
            :,
            self.metrics[metric]["col_idx"],
        ].astype(float)
        ticks = self.data_dict[counting_method][epoch]["merging_stats"][:, 0]
        x_axis = np.arange(len(ticks))
        plt.plot(x_axis, y_data, label=epoch)

        if show_reference and hasattr(self, "reference_data_dict"):
            y_data_ref = self.reference_data_dict["merging_stats"][
                :, self.metrics[metric]["col_idx"]
            ].astype(float)
            plt.plot(x_axis, y_data_ref, label="DIALS", linestyle="--", color="black")

        if show:
            plt.xticks(x_axis, ticks, rotation=45, fontsize=12)
            plt.xlabel("Resolution (Å)", fontsize=24)
            plt.ylabel(self.metrics[metric]["display_name"], fontsize=24)
            plt.title(self.metrics[metric]["display_name"], fontsize=16)
            if show_legend:
                plt.legend(loc="best")
            plt.grid(alpha=0.3)
            plt.show()

    def plot_subplots_all_epochs(
        self, counting_method, metrics=None, show_reference=False, show_legend=True
    ):
        if metrics is None:
            metrics = ["I_vs_sigI", "cc_half", "cc_anom", "r_pim"]

        epochs = list(self.data_dict[counting_method].keys())

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        axs = axs.flatten()

        for idx, metric in enumerate(metrics):
            ax = axs[idx]

            # If you want to skip the last epoch, use epochs[:-1]
            # If there's only one epoch, epochs[:-1] might be empty, so handle that case carefully.
            if len(epochs) == 1:
                # For a single epoch, just plot it:
                selected_epochs = epochs
            else:
                selected_epochs = epochs

            for epoch in selected_epochs:
                y_data = self.data_dict[counting_method][epoch]["merging_stats"][
                    :, self.metrics[metric]["col_idx"]
                ].astype(float)
                ticks = self.data_dict[counting_method][epoch]["merging_stats"][:, 0]
                x_axis = np.arange(len(ticks))

                ax.plot(x_axis, y_data, label=f"{epoch}")

            # Reference (optional)
            if show_reference and hasattr(self, "reference_data_dict"):
                y_data_ref = self.reference_data_dict["merging_stats"][
                    :, self.metrics[metric]["col_idx"]
                ].astype(float)
                ax.plot(
                    x_axis,
                    y_data_ref,
                    label="DIALS (ref)",
                    linestyle="--",
                    color="black",
                )

            # If there was at least one epoch plotted, set x_ticks
            if len(epochs) > 0:
                ax.set_xticks(x_axis)
                ax.set_xticklabels(ticks, rotation=45, fontsize=10)
            ax.set_xlabel("Resolution (Å)", fontsize=12)
            ax.set_ylabel(self.metrics[metric]["display_name"], fontsize=12)
            ax.set_title(
                f"{self.metrics[metric]['display_name']} ({counting_method})",
                fontsize=24,
            )
            ax.grid(alpha=0.3)
            if show_legend:
                ax.legend(loc="best", fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.96])


def plot_method(method, metrics=["cc_half", "cc_anom", "r_pim", "I_vs_sigI"]):
    epochs = list(data.data_dict[method].keys())
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[data.metrics[metric]["display_name"] for metric in metrics],
        vertical_spacing=0.15,
    )

    axes = list(itertools.product(range(1, 3), range(1, 3)))

    # Create a consistent colormap
    colors = (
        px.colors.qualitative.Plotly
    )  # Or any other colormap like px.colors.sequential.Viridis

    # Track which epochs have been added to the legend
    added_to_legend = set()

    for ax, metric in zip(axes, metrics):
        for i, epoch in enumerate(epochs):
            y = data.data_dict[method][epoch]["merging_stats"][
                :, data.metrics[metric]["col_idx"]
            ].astype(float)
            ticks = data.data_dict[method][epoch]["merging_stats"][:, 0]

            # Determine if this trace should appear in the legend
            showlegend = epoch not in added_to_legend
            if showlegend:
                added_to_legend.add(epoch)

            fig.add_trace(
                go.Scatter(
                    x=ticks,
                    y=y,
                    mode="lines+markers",
                    name=f"{epoch}",
                    line=dict(
                        color=colors[i % len(colors)]
                    ),  # Consistent color per epoch
                    showlegend=showlegend,  # Only show in legend once
                ),
                row=ax[0],
                col=ax[1],
            )

        # Handle reference data (DIALS)
        # Only add to legend once (for the first subplot)
        y_ref = data.reference_data_dict["merging_stats"][
            :, data.metrics[metric]["col_idx"]
        ].astype(float)
        ticks_ref = data.reference_data_dict["merging_stats"][:, 0]

        fig.add_trace(
            go.Scatter(
                x=ticks,
                y=y_ref,
                mode="lines+markers",
                name="DIALS",
                line=dict(
                    dash="dash", color="black"
                ),  # Use a distinct style for reference
                showlegend="DIALS" not in added_to_legend,  # Only show in legend once
            ),
            row=ax[0],
            col=ax[1],
        )

        if "DIALS" not in added_to_legend:
            added_to_legend.add("DIALS")

        # Add axis labels for each subplot
        fig.update_xaxes(title_text="Resolution", row=ax[0], col=ax[1])
        fig.update_yaxes(
            title_text=data.metrics[metric]["display_name"], row=ax[0], col=ax[1]
        )

    # Update overall layout
    fig.update_layout(
        title=f"Method: {method}",
        height=1200,
        width=1800,
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="top",
            y=1.1,  # Position below the plots
            xanchor="center",
            x=0.5,
        ),
    )

    return fig


def create_peaks_tables(
    data_dict: Dict[str, Dict[str, Dict[str, Any]]],
    reference_peaks: pl.DataFrame,
    peak_value_column: str = "peakz",
) -> Dict[str, go.Figure]:
    methods = list(data_dict.keys())
    method_figures = {}

    for method in methods:
        # Get all epochs for this method
        method_epochs = []
        for epoch in data_dict[method].keys():
            if "peaks" in data_dict[method][epoch]:
                method_epochs.append(epoch)

        # Sort epochs in descending order
        method_epochs = sorted(
            method_epochs,
            key=lambda x: int(re.sub(r"\D", "", x)) if re.findall(r"\d+", x) else 0,
            reverse=True,
        )

        for epoch in method_epochs:
            val_loss = data_dict[method][epoch].get("val")

        if not method_epochs:
            continue

        # Rest of the seqid collection code (same as before)
        method_seqids = set()
        reference_seqids = reference_peaks["seqid"].to_list()
        for seqid in reference_seqids:
            method_seqids.add(seqid)

        for epoch in method_epochs:
            peaks_df = data_dict[method][epoch]["peaks"]
            epoch_seqids = peaks_df["seqid"].to_list()
            for seqid in epoch_seqids:
                method_seqids.add(seqid)

        unique_seqids = sorted(list(method_seqids))

        # Create seqid to residue mapping
        seqid_to_residue = {}
        for i, seqid in enumerate(reference_peaks["seqid"].to_list()):
            residue = reference_peaks["residue"].to_list()[i]
            seqid_to_residue[seqid] = residue

        for epoch in method_epochs:
            peaks_df = data_dict[method][epoch]["peaks"]
            for i, seqid in enumerate(peaks_df["seqid"].to_list()):
                residue = peaks_df["residue"].to_list()[i]
                seqid_to_residue[seqid] = residue

        # Create header row
        header_row = ["Epoch", "Val Loss"]
        for seqid in unique_seqids:
            residue = seqid_to_residue.get(seqid, "Unknown")
            header_row.append(f"{seqid}<br>{residue}")

        # Prepare table data
        table_data = []

        # Reference row (no val_loss)
        reference_row = ["Reference", "N/A"]  # Changed from "-" to "N/A" for clarity
        ref_seqid_list = reference_peaks["seqid"].to_list()
        ref_peakz_list = reference_peaks[peak_value_column].to_list()

        for seqid in unique_seqids:
            if seqid in ref_seqid_list:
                idx = ref_seqid_list.index(seqid)
                reference_row.append(f"{ref_peakz_list[idx]:.2f}")
            else:
                reference_row.append("-")

        table_data.append(reference_row)

        for epoch in method_epochs:
            peaks_df = data_dict[method][epoch]["peaks"]
            val = data_dict[method][epoch].get("val")  # Use "val" not "val_loss"

            # DEBUG: Print what we're getting

            # Format val_loss
            val_str = f"{val:.4f}" if val is not None else "Not Found"

            row = [f"{epoch}", val_str]

            epoch_seqid_list = peaks_df["seqid"].to_list()
            epoch_peakz_list = peaks_df[peak_value_column].to_list()

            for seqid in unique_seqids:
                if seqid in epoch_seqid_list:
                    idx = epoch_seqid_list.index(seqid)
                    row.append(f"{epoch_peakz_list[idx]:.2f}")
                else:
                    row.append("-")

            table_data.append(row)

        # Create table (same as before)
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=header_row,
                        line_color="darkslategray",
                        fill_color="#CDCDCD",
                        align="center",
                        font=dict(color="black", size=14),
                    ),
                    cells=dict(
                        values=list(map(list, zip(*table_data))),
                        line_color="darkslategray",
                        fill_color=[["white", "#F3F3F3"] * len(table_data)],
                        align="center",
                        font=dict(color="black", size=12),
                    ),
                )
            ]
        )

        fig.update_layout(
            title=f"Peaks Analysis for Method: {method} (Epochs in Descending Order)",
            margin=dict(l=20, r=20, t=60, b=20),
        )

        method_figures[method] = fig

    return method_figures


# -
########################################
# 3. MAIN SCRIPT
########################################

if __name__ == "__main__":
    # load data

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--path",
        type=str,
    )
    argparser.add_argument(
        "--reference_path",
        type=str,
        default="/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/reference_data/",
    )
    args = argparser.parse_args()

    reference_path = Path(args.reference_path)
    path = Path(args.path)

    config = load_config(list(path.glob("**/config_copy.yaml"))[0])
    p_prf_scale = config["components"]["loss"]["params"]["p_p_scale"]
    p_I_weight = config["components"]["loss"]["params"]["p_I_weight"]
    p_bg_weight = config["components"]["loss"]["params"]["p_bg_weight"]

    id = path.name.split("-")[-1]

    # Instantiate DataManager
    data = DataManager(
        path,
        reference_path=reference_path,
    )

    # initialize wandb
    run = wandb.init(project="data_analysis_v5", id=id)

    # function to normalize colorbar
    class CustomNorm(mpl.colors.Normalize):
        def __init__(self, vmin=0, vmax=1, cmap_min=0.25, cmap_max=1.0):
            super().__init__(vmin, vmax)
            self.cmap_min = cmap_min
            self.cmap_max = cmap_max

        def __call__(self, value, clip=None):
            # First normalize to [0, 1]
            normalized = super().__call__(value, clip)
            # Then map to [cmap_min, cmap_max]
            return self.cmap_min + normalized * (self.cmap_max - self.cmap_min)

    epochs = len(data.data_dict["posterior"].items())
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, data.metrics.keys()):
        # reference data
        ref_data = data.reference_data_dict["merging_stats"][
            :, data.metrics[metric]["col_idx"]
        ].astype(np.float32)

        x_labels = data.data_dict["posterior"]["epoch_1"]["merging_stats"][:, 0]
        x_ticks = np.linspace(0, len(x_labels), len(x_labels))

        # set up color map and count number of epochs
        # cmap = plt.cm.Greys
        cmap = sns.cubehelix_palette(
            start=0.5, rot=-0.55, dark=0, light=0.8, as_cmap=True
        )
        cmap_list = cmap(np.linspace(0.0, 1, epochs, retstep=2)[0])

        # plot
        i = 0
        for color, epoch in zip(cmap_list, data.data_dict["posterior"].items()):
            arr = epoch[1]["merging_stats"][:, data.metrics[metric]["col_idx"]].astype(
                np.float32
            )
            ax.plot(x_ticks, arr, color=color, linewidth=1.5)
            i += 1

        ax.plot(x_ticks, ref_data, color="red", linewidth=1.5, label="DIALS")
        plt.grid()
        # Use custom normalization
        norm = CustomNorm(vmin=0, vmax=epochs - 1, cmap_min=0.25, cmap_max=1.0)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Epoch")

        label = data.metrics[metric]["display_name"]
        ax.set_xticks(x_ticks, x_labels, rotation=55, ha="right", fontsize=12)
        ax.set_ylabel(f"{label}", fontsize=14)
        ax.set_xlabel(r"Resolution", fontsize=14)
        ax.grid(alpha=0.7)
        ax.legend()

    plt.suptitle(
        f"Merging stats for model {id}\np_prf scale: {p_prf_scale}\np_I_scale: {p_I_weight}\np_bg_weight: {p_bg_weight}",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.savefig(
        f"{path.as_posix()}/merging_stats_p_prf_{p_prf_scale}_{id}.png", dpi=600
    )
    wandb.log({"Posterior Merging Stats": wandb.Image(plt.gcf())})

    # log anomalous peak height tables
    peakz_df_list = []
    method_figures = create_peaks_tables(data.data_dict, data.reference_peaks)

    direct_method_tbl = method_figures["posterior"]
    wandb.log(
        {"Direct Method Anomalous Peaks": direct_method_tbl}
    )  # Log the table to W&B

    # Get metrics for all epochs
    d = dict()
    for h, e in zip(data.html_files, data.data_dict["posterior"].keys()):
        tbl = pd.read_html(h, header=0)[0]
        d[e] = {
            "Observations": tbl.loc[1].values[1:].astype(int),
            "MeanI/SigI": tbl.loc[5].values[1:].astype(float),
            "Rmerge": tbl.loc[6].values[1:].astype(float),
            "Rmeas": tbl.loc[7].values[1:].astype(float),
            "Rpim": tbl.loc[8].values[1:].astype(float),
            "CChalf": tbl.loc[9].values[1:].astype(float),
        }

    d_ref = {
        "Observations": pd.read_html(data.reference_html, header=0)[0]
        .loc[1]
        .values[1:]
        .astype(int),
        "MeanI/SigI": pd.read_html(data.reference_html, header=0)[0]
        .loc[5]
        .values[1:]
        .astype(float),
        "Rmerge": pd.read_html(data.reference_html, header=0)[0]
        .loc[6]
        .values[1:]
        .astype(float),
        "Rmeas": pd.read_html(data.reference_html, header=0)[0]
        .loc[7]
        .values[1:]
        .astype(float),
        "Rpim": pd.read_html(data.reference_html, header=0)[0]
        .loc[8]
        .values[1:]
        .astype(float),
        "CChalf": pd.read_html(data.reference_html, header=0)[0]
        .loc[9]
        .values[1:]
        .astype(float),
    }

    # -
    # NOTE: Function to plot tables across epoch
    def plot_table(metric, ref):
        df = pd.DataFrame(d).transpose()[metric]
        df_ref = pd.DataFrame(ref)[metric]
        arr = np.vstack(df.values)  # shape (N, 3)
        arr = np.vstack([df_ref.values, arr])
        epochs = [e.split("_")[-1] for e in df.keys().tolist()]
        epochs.insert(0, "reference")

        # Determine alternating row colors
        n_rows = len(epochs)
        fill_colors = [["#f9f9f9", "#e6e6e6"][(i % 2)] for i in range(n_rows)]

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Table(
                columnwidth=[5, 10, 10, 10],  # Adjust as needed
                header=dict(
                    values=["Epoch", "Overall", "Low-res", "High-res"],
                    fill_color="lightgrey",
                    align="center",
                    font=dict(color="black", size=12),
                ),
                cells=dict(
                    values=[epochs, arr[:, 0], arr[:, 1], arr[:, 2]],
                    fill_color=[fill_colors]
                    * 4,  # Apply same alternating pattern to all columns
                    align="center",
                    font=dict(size=12),
                ),
            )
        )

        fig.update_layout(
            title_text=f"{metric} Over Epochs\np_prf_scale: {p_prf_scale}\np_I_scale: {p_I_weight}\np_bg_weight: {p_bg_weight}",
            title_x=0.5,
            width=700,
        )

        return fig

    # plot and show figure
    fig = plot_table("Observations", d_ref)

    wandb.log({"Observations": wandb.Html(fig.to_html())})

    # -
    # NOTE: Code to get start/final r-free and r-work from phenix.logs
    pattern1 = re.compile(r"Start R-work")
    pattern2 = re.compile(r"Final R-work")
    log_files = list(path.glob("**/refine*.log"))
    log_files.insert(0, list(reference_path.glob("**/refine*.log"))[0])

    # Search files
    matches_start = {}
    matches_final = {}
    for log_file in log_files:
        with log_file.open("r") as f:
            lines = f.readlines()
        if re.search("epoch", log_file.parents[3].name):
            epoch = log_file.parents[3].name.split("_")[-1]
        else:
            epoch = "reference"

        matched_lines_start = [line.strip() for line in lines if pattern1.search(line)]
        matched_lines_final = [line.strip() for line in lines if pattern2.search(line)]
        if matched_lines_start:
            matches_start[epoch] = {
                "r_work": re.findall(r"\d+\.\d+", matched_lines_start[0])[0],
                "r_free": re.findall(r"\d+\.\d+", matched_lines_start[0])[1],
            }
        if matched_lines_final:
            matches_final[epoch] = {
                "r_work": re.findall(r"\d+\.\d+", matched_lines_final[0])[0],
                "r_free": re.findall(r"\d+\.\d+", matched_lines_final[0])[1],
            }

    df_start = pd.DataFrame(matches_start).transpose()

    def sort_key(index_val):
        if index_val == "reference":
            return (0, 0)
        else:
            return (1, int(index_val))

    df_start_sorted = df_start.iloc[
        sorted(range(len(df_start)), key=lambda i: sort_key(df_start.index[i]))
    ]

    df_final = pd.DataFrame(matches_final).transpose()

    df_final_sorted = df_final.iloc[
        sorted(range(len(df_final)), key=lambda i: sort_key(df_final.index[i]))
    ]

    rgap = (
        df_final_sorted["r_free"].astype(np.float64)
        - df_final_sorted["r_work"].astype(np.float64)
    ).to_numpy()

    rgap_str = np.vectorize(lambda x: f"{x:.2g}")(rgap)

    # row names
    epochs = df_start_sorted.index.tolist()

    arr_start = df_start_sorted.astype(float).values
    arr_final = df_final_sorted.astype(float).values

    fill_colors = [["#f9f9f9", "#e6e6e6"][(i % 2)] for i in range(len(epochs))]

    # Build the table
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Table(
            columnwidth=[5, 10, 10, 10],
            header=dict(
                values=[
                    "Epoch",
                    "Start R-work",
                    "Final R-work",
                    "Start R-free",
                    "Final R-free",
                    "Final (R-free - R-work)",
                ],
                fill_color="lightgrey",
                align="center",
                font=dict(color="black", size=12),
            ),
            cells=dict(
                values=[
                    epochs,
                    arr_start[:, 0],  # Start R-work
                    arr_final[:, 0],  # Final R-work
                    arr_start[:, 1],  # Start R-free
                    arr_final[:, 1],  # Final R-free
                    rgap_str,
                ],
                fill_color=[fill_colors],
                align="center",
                font=dict(size=12),
            ),
        )
    )

    fig.update_layout(
        title_text=f"R-values over epoch\np_prf_scale: {p_prf_scale}\np_I_scale: {p_I_weight}\np_bg_weight: {p_bg_weight}",
        title_x=0.5,
        width=700,
    )

    wandb.log({"R-vals": wandb.Html(fig.to_html())})

    plt.clf()

    # NOTE: Code to calculate difference in peak heights between reference and model
    plt.clf()
    ref_df = plr.DataFrame(
        {
            "residue": data.reference_peaks["residue"],
            "seqid": data.reference_peaks["seqid"],
            "peakz": data.reference_peaks["peakz"],
        }
    )

    dfs = []  # empty list to store all difference dataframes
    for epoch in data.data_dict["posterior"].keys():
        if "peaks" in data.data_dict["posterior"][epoch]:
            merged_df = ref_df.join(
                plr.DataFrame(
                    {
                        "residue": data.data_dict["posterior"][epoch]["peaks"][
                            "residue"
                        ],
                        "seqid": data.data_dict["posterior"][epoch]["peaks"]["seqid"],
                        "peakz": data.data_dict["posterior"][epoch]["peaks"]["peakz"],
                    }
                ),
                how="full",
                on="seqid",
            )

            diff_df = plr.DataFrame(
                {
                    "residue": merged_df["residue"],
                    "seqid": merged_df["seqid"],
                    f"{epoch}": merged_df["peakz_right"] - merged_df["peakz"],
                }
            ).drop_nulls()
            dfs.append(diff_df)

    diffs = plr.concat(dfs, how="align").sort(by="seqid")
    values = diffs.transpose(include_header=True)[2:, 1:]
    epochs = diffs.transpose(include_header=True)[2:, :]["column"]

    seqids = [
        f"{str(str1)}\n{str2}"
        for str1, str2 in zip(diffs[:, 1].to_list(), diffs[:, 0].to_list())
    ]

    total_diff = [x[0][:, -1].sum() for x in dfs]

    y_axis = np.linspace(1, len(epochs), len(epochs)) - 0.5

    # -
    # NOTE: code to plot a heatmap of peak differences

    # Convert to float matrix with NaNs
    arr = values.cast(plr.Float64).to_numpy()  # ensure numeric type

    # Create mask for NaNs
    mask = np.isnan(arr)

    # cmap = sns.color_palette('coolwarm_r',as_cmap=True)
    cmap = sns.color_palette("PRGn", as_cmap=True)

    # cmap = sns.diverging_palette(350, 145, s=80, as_cmap=True)

    x_axis = np.linspace(1, len(seqids), len(seqids)) - 0.5
    y_axis = np.linspace(1, len(epochs), len(epochs)) - 0.5

    # Define the norm: min, center (0), and max
    #    vmin = np.array(total_diff).min()
    #    vmax = np.array(total_diff).max()

    vmax = 5.0
    vmin = -5.0

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # -
    total_diff = [x[0][:, -1].sum() for x in dfs]
    y_axis = np.linspace(1, len(epochs), len(epochs)) - 0.5

    best = epochs.to_list()[np.argmax(total_diff)]

    plt.figure(figsize=(4, 9))
    sns.heatmap(
        np.array(total_diff).reshape(-1, 1),
        norm=norm,
        cmap=cmap,
        annot=True,
        linewidths=0.5,
        linecolor="black",
    )
    plt.yticks(ticks=y_axis, labels=epochs, rotation=0)
    plt.xticks(visible=False)
    plt.title(
        f"{best} has the largest positive difference\np_prf_scale: {p_prf_scale}\np_I_scale: {p_I_weight}\np_bg_weight: {p_bg_weight}"
    )
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    plt.savefig(
        f"{path.as_posix()}/overall_peak_heatmap_p_prf_{p_prf_scale}_{id}.png", dpi=600
    )
    wandb.log({"Largest peak difference": wandb.Image(plt.gcf())})

    # -
    # Reuse your colormap or modify saturation as needed
    vmin = arr[~np.isnan(arr)].min()
    vmax = arr[~np.isnan(arr)].max()

    # norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # to get a better colorbar
    if vmax < 0:
        norm = Normalize(vmin=vmin, vmax=vmax)
    elif vmin > 0:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.figure(figsize=(20, 12))

    annot_matrix = np.empty_like(arr, dtype=object)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if val == "" or val is None:
                annot_matrix[i, j] = ""
            else:
                try:
                    annot_matrix[i, j] = str(round(float(val), 2))
                except (ValueError, TypeError):
                    annot_matrix[i, j] = ""

    # Apply to heatmap
    ax = sns.heatmap(
        arr,
        mask=mask,
        cmap=cmap,
        norm=norm,
        cbar=True,
        annot=True,
        linewidths=0.5,
        fmt=".2f",
        linecolor="black",
    )  # Set axis titles and ticks

    plt.title(
        f"Peak differences\n(model - reference) > 0 => model scored higher\np_prf_scale: {p_prf_scale}\np_I_scale: {p_I_weight}\np_bg_weight: {p_bg_weight}",
        fontsize=16,
    )
    plt.xlabel("Residue", fontsize=24)
    plt.ylabel("Epoch", fontsize=24)
    plt.xticks(ticks=x_axis, labels=seqids)
    plt.yticks(ticks=y_axis, labels=epochs, rotation=0)

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.savefig(
        f"{path.as_posix()}/heatmap_peaks_p_prf_{p_prf_scale}_{id}.png", dpi=600
    )
    wandb.log({"Anomalous peak differences": wandb.Image(plt.gcf())})

    ref_tbl = flex.reflection_table.from_file(config["output"]["refl_file"])

    # -

    # -
    # NOTE: Code to plot weighted cc_half

    plt.clf()
    epochs = diffs.transpose(include_header=True)[2:, :]["column"]

    vals = []
    for epoch in data.data_dict["posterior"].keys():
        chalf = data.data_dict["posterior"][epoch]["merging_stats"][:, -2].astype(
            np.float64
        )
        n_obs = data.data_dict["posterior"][epoch]["merging_stats"][:, 1].astype(
            np.float64
        )

        vals.append((chalf * n_obs).sum() / n_obs.sum())

    print(epochs[1:], np.argmax(vals))
    best = epochs.to_list()[np.argmax(vals)]

    ref_obs = data.reference_data_dict["merging_stats"][:, 1].astype(float)
    ref_cchalf = data.reference_data_dict["merging_stats"][:, -2].astype(float)
    ref_cc_weighted = (ref_obs * ref_cchalf).sum() / ref_obs.sum()

    ## NOTE: Code to plot metrics vs validation loss

    # train metrics
    train_df = pd.read_csv(list(path.glob("**/avg_train_metrics.csv"))[0])
    train_avg_loss = train_df["avg_loss"].to_numpy()
    train_avg_kl = train_df["avg_kl"].to_numpy()
    train_avg_nll = train_df["avg_nll"].to_numpy()
    train_epochs = train_df["epoch"].to_numpy()

    # val metrics
    val_df = pd.read_csv(list(path.glob("**/avg_val_metrics.csv"))[0])
    val_avg_loss = val_df["avg_loss"].to_numpy()
    val_avg_kl = val_df["avg_kl"].to_numpy()
    val_avg_nll = val_df["avg_nll"].to_numpy()
    val_epochs = val_df["epoch"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    x_axis = np.arange(len(epochs))
    val_loss = np.array(
        [
            data.data_dict["posterior"][epoch]["val"]
            for epoch in data.data_dict["posterior"].keys()
        ]
    )
    axes[0, 0].plot(x_axis, arr_final[1:, 0], color="black", label="Final R-work")
    axes[0, 0].plot(x_axis, arr_final[1:, 1], color="blue", label="Final R-free")
    axes[0, 0].axhline(
        y=arr_final[0, 0], color="black", label="Ref Final R-work", linestyle="--"
    )
    axes[0, 0].axhline(
        y=arr_final[0, 1], color="blue", label="Ref Final R-free", linestyle="--"
    )
    axes[0, 0].grid()
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_title(
        f"Model Final R-values vs Epoch\np_prf_scale: {p_prf_scale}\np_I_scale: {p_I_weight}\np_bg_weight: {p_bg_weight}"
    )
    axes[0, 0].set_ylabel("R-value")
    axes[0, 0].set_xticks(x_axis, epochs.to_list(), rotation=70)
    axes[0, 0].legend()

    axes[0, 1].plot(train_epochs, train_avg_loss, color="black", label="train_loss")
    axes[0, 1].plot(
        train_epochs, train_avg_nll, color="black", label="train_nll", linestyle="--"
    )
    axes[0, 1].plot(val_epochs, val_avg_loss, color="red", label="val_loss")
    axes[0, 1].plot(
        val_epochs, val_avg_nll, color="red", label="val_nll", linestyle="--"
    )
    axes[0, 1].set_title("Val loss vs Epoch")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("loss")
    axes[0, 1].set_xticks(val_epochs[1:], epochs, rotation=70)
    axes[0, 1].set_ylim(ymin=1200, ymax=1800)
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].grid()

    axes[1, 0].plot(x_axis, vals, color="black", label="Model")
    axes[1, 0].axhline(y=ref_cc_weighted, color="red", label="DIALS")
    axes[1, 0].set_title(
        f"Weigted Average: CC_half\nbest epoch: {best}\np_prf_scale: {p_prf_scale}\np_I_scale: {p_I_weight}\np_bg_weight: {p_bg_weight}"
    )
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("weighted average")
    axes[1, 0].set_xticks(x_axis, epochs.to_list(), rotation=70)
    axes[1, 0].grid()
    axes[1, 0].legend()
    plt.tight_layout()
    wandb.log({"metric subplots": wandb.Image(plt.gcf())})

    # -
    # NOTE: Code to plot reference vs model CC_star

    plt.clf()
    cc_star_ref = np.sqrt((2 * ref_cchalf) / (1 + ref_cchalf))
    x_axis = np.linspace(0, len(ref_cchalf), len(ref_cchalf))
    resolution = data.reference_data_dict["merging_stats"][:, 0]

    fig, ax = plt.subplots(1, 1)

    vals = []

    epochs = len(data.data_dict["posterior"].items())
    cmap = sns.cubehelix_palette(start=0.5, rot=-0.55, dark=0, light=0.8, as_cmap=True)
    cmap_list = cmap(np.linspace(0.0, 1, epochs, retstep=2)[0])

    for color, epoch in zip(cmap_list, data.data_dict["posterior"].keys()):
        cchalf = data.data_dict["posterior"][epoch]["merging_stats"][:, -2].astype(
            np.float64
        )
        cc_star = np.sqrt(2 * cchalf / (1 + cchalf))

        vals.append(cc_star)
        ax.plot(x_axis, cc_star, color=color)

    ax.plot(x_axis, cc_star_ref, color="red", label="CCstar")
    # plt.plot(x_axis, ref_cchalf, color="black", label="CChalf")

    norm = CustomNorm(vmin=0, vmax=epochs - 1, cmap_min=0.0, cmap_max=1.0)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Epoch")
    ax.legend()
    plt.title(
        f"CC_star DIALS vs Model\np_prf_scale: {p_prf_scale}\np_I_scale: {p_I_weight}\np_bg_weight: {p_bg_weight}"
    )
    plt.ylabel("CC_half")
    plt.xlabel("epoch")
    plt.xticks(x_axis, resolution, rotation=55)
    plt.tight_layout()
    plt.grid()

    wandb.log({"CCstar": wandb.Image(plt.gcf())})
    # -
    best_preds = torch.load(list(path.glob(f"**/{best}/*.pt"))[0], weights_only=False)

    best_preds.keys()

    sel = np.zeros(len(ref_tbl), dtype=bool)
    reflection_ids = np.concatenate(best_preds["refl_ids"]).astype(np.int32)
    sel[reflection_ids] = True

    temp = ref_tbl.select(flex.bool(sel))

    # -
    nn_preds = (np.concatenate(best_preds["intensity_mean"]),)

    plt.clf()

    plt.figure(figsize=(20, 12))
    plt.scatter(nn_preds, temp["intensity.sum.value"], color="black", s=5.0, alpha=0.2)

    plt.grid()
    plt.plot([0, 1e7], [0, 1e7], "r", alpha=0.3, linewidth=2.0)
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(0.1, 1e6)
    plt.xlim(0.1, 1e6)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title(
        f"DIALS summation algorithm vs NN Integrator {best}\np_prf_scale: {p_prf_scale}\np_I_scale: {p_I_weight}\np_bg_weight: {p_bg_weight}",
        fontsize=28,
    )
    plt.ylabel("DIALS I_sum", fontsize=26)
    plt.xlabel("I_NN", fontsize=26)
    plt.tight_layout()
    plt.savefig(
        f"{path.as_posix()}/corr_plot_sum_vs_nn_p_prf_{p_prf_scale}_{id}.png", dpi=600
    )

    wandb.log({"Correlation plot: DIALS summation vs NN": wandb.Image(plt.gcf())})

    # -
    plt.clf()
    plt.figure(figsize=(20, 12))
    plt.scatter(nn_preds, temp["intensity.prf.value"], color="black", s=5.0, alpha=0.2)

    plt.grid()
    plt.plot([0, 1e7], [0, 1e7], "r", alpha=0.3, linewidth=2.0)
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(0.1, 1e6)
    plt.xlim(0.1, 1e6)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title(
        f"DIALS profile fitting algorithm vs NN Integrator {best}\np_prf_scale: {p_prf_scale}\np_I_scale: {p_I_weight}\np_bg_weight: {p_bg_weight}",
        fontsize=30,
    )
    plt.ylabel("DIALS I_prf", fontsize=26)
    plt.xlabel("I_NN", fontsize=26)
    plt.tight_layout()
    plt.savefig(
        f"{path.as_posix()}/corr_plot_prf_vs_nn_p_prf_{p_prf_scale}_{id}.png", dpi=600
    )

    wandb.log(
        {
            "Correlation plot: DIALS profile fitting algorithm vs NN": wandb.Image(
                plt.gcf()
            )
        }
    )

    # -
    nn_background = np.concatenate(best_preds["qbg"])

    plt.clf()
    plt.figure(figsize=(20, 12))
    plt.scatter(nn_background, temp["background.mean"], s=5.0, alpha=0.2, color="black")
    plt.plot([0, 10], [0, 10], "r", alpha=0.3)
    plt.ylim(0, 10)
    plt.xlim(0, 10)
    plt.ylabel("DIALS Background mean", fontsize=26)
    plt.xlabel("NN background", fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title(
        f"DIALS background vs NN Integrator {best}\np_prf_scale: {p_prf_scale}\np_I_scale: {p_I_weight}\np_bg_weight: {p_bg_weight}",
        fontsize=30,
    )
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{path.as_posix()}/corr_plot_bg_p_prf_{p_prf_scale}_{id}.png", dpi=600)
    wandb.log({"Correlation plot: DIALS bg vs NN": wandb.Image(plt.gcf())})

    # -

    run.finish()
