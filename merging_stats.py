#!/usr/bin/env python

import numpy as np
import polars as plr
from pathlib import Path
from collections import defaultdict
import itertools
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import polars as pl
from typing import Dict, Any
import argparse

import matplotlib.pyplot as plt
import wandb
import pandas as pd
from bs4 import BeautifulSoup


########################################
# 1. PARSING FUNCTION
########################################
def parse_html(file_path):
    """Parse the merged-*.html file for resolution shell statistics."""
    with open(file_path, "r") as file:
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
            self.data_dict[counting_method][epoch]["merging_stats"] = cleaned_array
            self.data_dict[counting_method][epoch]["peaks"] = df

        if reference_path:
            self.reference_path = Path(reference_path)
            self.reference_peaks = plr.read_csv(
                self.reference_path / "peaks_reference.csv"
            )
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
            plt.xlabel("Resolution (Å)", fontsize=14)
            plt.ylabel(self.metrics[metric]["display_name"], fontsize=14)
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
                fontsize=14,
            )
            ax.grid(alpha=0.3)
            if show_legend:
                ax.legend(loc="best", fontsize=9)

        plt.tight_layout()


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
    """
    Parameters:
    -----------
    data_dict : Dict
        Nested dictionary with structure: method -> epoch -> 'peaks' -> polars DataFrame
    reference_peaks : pl.DataFrame
        Polars DataFrame containing the reference peaks
    peak_value_column : str
        Column name for the peak values (default: 'peakz')

    Returns:
    --------
    Dict[str, go.Figure]
        Dictionary mapping method names to their Plotly figures
    """
    # Get all methods
    methods = list(data_dict.keys())
    print(f"Found methods: {methods}")

    # Create a dictionary to store figures for each method
    method_figures = {}

    # Process each method separately
    for method in methods:
        print(f"Processing method: {method}")

        # Get all epochs for this method
        method_epochs = []
        for epoch in data_dict[method].keys():
            if "peaks" in data_dict[method][epoch]:
                method_epochs.append(epoch)
        method_epochs = sorted(method_epochs)
        print(f"  Found epochs for {method}: {method_epochs}")

        if not method_epochs:
            print(f"  No epochs with peaks found for method {method}, skipping")
            continue

        # Collect all unique seqids for this method and reference
        method_seqids = set()

        # Add reference seqids
        reference_seqids = reference_peaks["seqid"].to_list()
        for seqid in reference_seqids:
            method_seqids.add(seqid)

        # Add seqids from all epochs for this method
        for epoch in method_epochs:
            peaks_df = data_dict[method][epoch]["peaks"]
            epoch_seqids = peaks_df["seqid"].to_list()
            for seqid in epoch_seqids:
                method_seqids.add(seqid)

        unique_seqids = sorted(list(method_seqids))
        print(f"  Found {len(unique_seqids)} unique seqids for method {method}")

        # Create mapping of seqid to residue
        seqid_to_residue = {}

        # First from reference
        for i, seqid in enumerate(reference_peaks["seqid"].to_list()):
            residue = reference_peaks["residue"].to_list()[i]
            seqid_to_residue[seqid] = residue

        # Then from all epochs for this method
        for epoch in method_epochs:
            peaks_df = data_dict[method][epoch]["peaks"]
            for i, seqid in enumerate(peaks_df["seqid"].to_list()):
                residue = peaks_df["residue"].to_list()[i]
                seqid_to_residue[seqid] = residue

        # Create header row with seqid and residue
        header_row = ["Epoch"]
        for seqid in unique_seqids:
            residue = seqid_to_residue.get(seqid, "Unknown")
            header_row.append(f"{seqid}<br>{residue}")

        # Prepare data for the table
        table_data = []

        # Add reference row first
        reference_row = ["Reference"]
        ref_seqid_list = reference_peaks["seqid"].to_list()
        ref_peakz_list = reference_peaks[peak_value_column].to_list()

        for seqid in unique_seqids:
            if seqid in ref_seqid_list:
                idx = ref_seqid_list.index(seqid)
                reference_row.append(f"{ref_peakz_list[idx]:.2f}")
            else:
                reference_row.append("-")

        table_data.append(reference_row)

        # Add rows for each epoch in this method
        for epoch in method_epochs:
            peaks_df = data_dict[method][epoch]["peaks"]
            row = [f"{epoch}"]

            epoch_seqid_list = peaks_df["seqid"].to_list()
            epoch_peakz_list = peaks_df[peak_value_column].to_list()

            for seqid in unique_seqids:
                if seqid in epoch_seqid_list:
                    idx = epoch_seqid_list.index(seqid)
                    row.append(f"{epoch_peakz_list[idx]:.2f}")
                else:
                    row.append("-")

            table_data.append(row)

        # Create Plotly table for this method
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
                        values=list(
                            map(list, zip(*table_data))
                        ),  # Transpose for Plotly format
                        line_color="darkslategray",
                        fill_color=[["white", "#F3F3F3"] * len(table_data)],
                        align="center",
                        font=dict(color="black", size=12),
                    ),
                )
            ]
        )

        fig.update_layout(
            title=f"Peaks Analysis for Method: {method}",
            margin=dict(l=20, r=20, t=60, b=20),
            width=None,  # Full width
            height=None,  # Auto height
        )

        # Store this figure in the dictionary
        method_figures[method] = fig

    return method_figures


# %%
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
        default="/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/data/"
    )
    args = argparser.parse_args()

    reference_path = Path(args.reference_path)
    path = Path(args.path)

    id = path.name.split("-")[-1]

    # Instantiate DataManager
    data = DataManager(
        path,
        reference_path=reference_path,
    )
    print("Reference Data Dict Keys:", data.reference_data_dict.keys())

    # initialize wandb
    run = wandb.init(project="local-plots", id=id)

    # log DIALS HTML reports
    for h in data.html_files:
        method = h.parents[0].name.replace("dials_out_", "")
        epoch = h.parents[2].name.split("_")[1]

        wandb.log({f"DIALS {method}: epoch {epoch}": wandb.Html(h.read_text())})

    # log merging stats
    for method in data.counting_methods:
        # Plot each method
        fig = plot_method(method)
        wandb.log({f"{method} Merging Stats": wandb.Html(fig.to_html())})

    # log anomalous peak height tables
    peakz_df_list = []
    method_figures = create_peaks_tables(data.data_dict, data.reference_peaks)

    profile_masking_tbl = method_figures["thresholded"]
    direct_method_tbl = method_figures["posterior"]
    kabsch_sum_tbl = method_figures["weighted"]
    wandb.log(
        {"Profile Making Anomalous Peaks": profile_masking_tbl}
    )  # Log the table to W&B
    wandb.log(
        {"Direct Method Anomalous Peaks": direct_method_tbl}
    )  # Log the table to W&B
    wandb.log({"Kabsch Sum Anomalous Peaks": kabsch_sum_tbl})  # Log the table to W&B

    run.finish()
