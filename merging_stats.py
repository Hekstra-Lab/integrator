#!/usr/bin/env python

import numpy as np
import polars as plr
from pathlib import Path
from collections import defaultdict

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
                selected_epochs = epochs[:-1]

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
                ax.plot(x_axis, y_data_ref, label="DIALS (ref)", linestyle="--", color="black")

            # If there was at least one epoch plotted, set x_ticks
            if len(epochs) > 0:
                ax.set_xticks(x_axis)
                ax.set_xticklabels(ticks, rotation=45, fontsize=10)
            ax.set_xlabel("Resolution (Å)", fontsize=12)
            ax.set_ylabel(self.metrics[metric]["display_name"], fontsize=12)
            ax.set_title(f"{self.metrics[metric]['display_name']} ({counting_method})", fontsize=14)
            ax.grid(alpha=0.3)
            if show_legend:
                ax.legend(loc="best", fontsize=9)

        plt.tight_layout()

########################################
# 3. MAIN SCRIPT
########################################

if __name__ == "__main__":

    reference_path = Path('/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/data')
    path = Path("/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/lightning_logs/wandb/run-20250421_121835-mxp871sk")

    # Instantiate our DataManager
    data = DataManager(path, reference_path=reference_path)
    print("Reference Data Dict Keys:", data.reference_data_dict.keys())

    # Subplots for each method
    methods = ["thresholded", "weighted", "posterior"]
    for method in methods:
        data.plot_subplots_all_epochs(method, show_reference=True)
    plt.show()

    # Plot anomalous peaks
    plt.clf()
    data.plot_peaks("weighted")
    data.plot_peaks("thresholded")
    data.plot_peaks("posterior", show=True, plot_reference=True)

    # Plot merging stats
    metric_dict = {
        "metrics": ["I_vs_sigI", "cc_half", "cc_anom", "r_pim"],
        "ylabels": ["Mean I/sig(I)", "CC 1/2", "CC anom", "R pim"],
    }

    for metric in metric_dict["metrics"]:
        for method in methods:
            fig = plt.figure(figsize=(8, 5))
            for epoch in data.data_dict[method].keys():
                # skip last epoch if you want
                if epoch == list(data.data_dict[method].keys())[-1]:
                    continue
                else:
                    data.plot_resolution(method, epoch, metric, show=False)
            data.plot_resolution(method, epoch, metric, show=False, show_reference=True)
            plt.xlabel("Resolution (Å)")
            plt.ylabel(metric_dict["ylabels"][metric_dict["metrics"].index(metric)])
            plt.title(metric_dict["ylabels"][metric_dict["metrics"].index(metric)] + f" ({method})")
            plt.legend(loc="best")
            plt.grid(alpha=0.3)
            plt.show()

    # 4. Log everything to W&B
    run = wandb.init(project="local-plots")

    # Log subplots for each method
    data.plot_subplots_all_epochs("posterior", show_reference=True, show_legend=False)
    wandb.log({"Posterior Merging Stats": plt})
    plt.clf()

    data.plot_subplots_all_epochs("thresholded", show_reference=True, show_legend=False)
    wandb.log({"Thresholded Merging Stats": plt})
    plt.clf()

    data.plot_subplots_all_epochs("weighted", show_reference=True, show_legend=False)
    wandb.log({"Weighted Merging Stats": plt})
    plt.clf()

    # Plot anomalous peaks again for logging
    plt.clf()
    data.plot_peaks("weighted")
    data.plot_peaks("thresholded")
    data.plot_peaks("posterior", plot_reference=True)
    plt.grid(alpha=0.3)
    plt.ylabel("Number of peaks")
    plt.xlabel("Epoch")
    plt.title("Anomalous Peaks")
    wandb.log({"Anomalous Peaks": plt})
    plt.clf()

    # 5. Gather all anomalous peaks into a single table and log to W&B
    peakz_df_list = []

    for method, epochs_dict in data.data_dict.items():
        for epoch, results_dict in epochs_dict.items():
            # Polars DataFrame with columns: seqid, residue, peakz, etc.
            peak_pl = results_dict["peaks"].select(["seqid", "residue", "peakz"])
            peak_pd = peak_pl.to_pandas()

            # Label the row by epoch
            peak_pd["epoch"] = epoch
            
            # This line ensures "\n" is an actual newline *character*:
            peak_pd["seq_res"] = (
                peak_pd["seqid"].astype(str)
                + "\n"    # <-- This is a true newline character
                + peak_pd["residue"].astype(str)
            )

            peakz_df_list.append(peak_pd[["epoch", "seq_res", "peakz"]])

    # Add the reference, labeled as "reference"
    ref_pl = data.reference_peaks.select(["seqid", "residue", "peakz"])
    ref_pd = ref_pl.to_pandas()
    ref_pd["epoch"] = "reference"
    ref_pd["seq_res"] = (
        ref_pd["seqid"].astype(str)
        + "\n" 
        + ref_pd["residue"].astype(str)
    )
    peakz_df_list.append(ref_pd[["epoch", "seq_res", "peakz"]])

    # Combine all into one long DataFrame
    peakz_df_long = pd.concat(peakz_df_list, ignore_index=True)

    ###############################
    # 2) PIVOT INTO THE FINAL WIDE TABLE
    ###############################
    peakz_df_wide = peakz_df_long.pivot_table(
        index="epoch",
        columns="seq_res",
        values="peakz",
        aggfunc="first"
    )

    # 3. Log the original numeric version (with NaN) to wandb
    wandb_table = wandb.Table(dataframe=peakz_df_wide.reset_index())
    wandb.log({"peakz_wide_table": wandb_table})    # Log the table to W&B

    wandb_table = wandb.Table(dataframe=peakz_df_wide.reset_index())
    wandb.log({"Anomalous Peak Heights": wandb_table})

    run.finish()
