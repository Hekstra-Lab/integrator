from bs4 import BeautifulSoup
import polars as plr
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import wandb

# Function to parse html file
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
    cleaned_array = np.char.replace(rows_array, "*", "")  # remove asterisks if any
    return headers, cleaned_array

class DataManager:
    def __init__(
        self,
        predictions_path,
        reference_path=None,
    ):
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
        if plot_reference:
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

        if show_reference:
            y_data_ref = self.reference_data_dict["merging_stats"][
                :, self.metrics[metric]["col_idx"]
            ].astype(float)
            plt.plot(x_axis, y_data_ref, label="DIALS", linestyle="--", color="black")

        if show:
            plt.xticks(x_axis, ticks, rotation=45, fontsize=12)
            plt.xlabel("Resolution (Å)", fontsize=14)
            plt.ylabel(self.metrics[metric]["display_name"], fontsize=14)
            plt.title(self.metrics[metric]["display_name"], fontsize=16)
            plt.legend(loc="best") if show_legend else None
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

            for epoch in epochs[:-1]:  # Skip last epoch if desired
                y_data = self.data_dict[counting_method][epoch]["merging_stats"][
                    :, self.metrics[metric]["col_idx"]
                ].astype(float)
                ticks = self.data_dict[counting_method][epoch]["merging_stats"][:, 0]
                x_axis = np.arange(len(ticks))

                ax.plot(x_axis, y_data, label=f"{epoch}")

            if show_reference:
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

            ax.set_xticks(x_axis)
            ax.set_xticklabels(ticks, rotation=45, fontsize=10)
            ax.set_xlabel("Resolution (Å)", fontsize=12)
            ax.set_ylabel(self.metrics[metric]["display_name"], fontsize=12)
            ax.set_title(
                f"{self.metrics[metric]['display_name']} ({counting_method})",
                fontsize=14,
            )
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=9) if show_legend else None

        plt.tight_layout()

        # plt.show()

reference_path = Path('/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/data')
        
path = Path("/n/hekstra_lab/people/aldama/lightning_logs/wandb/run-20250312_123355-nkbi1e6d")
data = DataManager(path, reference_path=reference_path)
data.reference_data_dict.keys()

# %%
# Get subplots
methods = ["thresholded", "weighted", "posterior"]

for method in methods:
    data.plot_subplots_all_epochs(method, show_reference=True)
plt.show()

# Plot anomalous peaks
plt.clf()
data.plot_peaks("weighted")
data.plot_peaks("thresholded")
data.plot_peaks("posterior",show=True, plot_reference=True)

# %%

# Plot merging stats
methods = ["thresholded", "weighted", "posterior"]
metric_dict = {
    "metrics": ["I_vs_sigI", "cc_half", "cc_anom", "r_pim"],
    "ylabels": ["Mean I/sig(I)", "CC 1/2", "CC anom", "R pim"],
}

# Generate matplotlib figures
for metric in metric_dict["metrics"]:
    for method in methods:
        fig = plt.figure(figsize=(8, 5))
        for epoch in data.data_dict[method].keys():
            # skip last epoch
            if epoch == list(data.data_dict[method].keys())[-1]:
                continue
            else:
                data.plot_resolution(method, epoch, metric, show=False)
        data.plot_resolution(method, epoch, metric, show=False, show_reference=True)
        plt.xlabel("Resolution (Å)")
        plt.ylabel(metric_dict["ylabels"][metric_dict["metrics"].index(metric)])
        plt.title(
            metric_dict["ylabels"][metric_dict["metrics"].index(metric)]
            + f" ({method})"
        )
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.show()


# loggint onto wandb
run = wandb.init(project="local-plots")


data.plot_subplots_all_epochs("posterior", show_reference=True, show_legend=False)
wandb.log({"Posterior Merging Stats": plt})
plt.clf()

data.plot_subplots_all_epochs("thresholded", show_reference=True, show_legend=False)
wandb.log({"Thresholded Merging Stats": plt})
plt.clf()

data.plot_subplots_all_epochs("weighted", show_reference=True, show_legend=False)
wandb.log({"Weighted Merging Stats": plt})
plt.clf()


# Plot anomalous peaks
plt.clf()
data.plot_peaks("weighted")
data.plot_peaks("thresholded")
data.plot_peaks("posterior", plot_reference=True)
plt.grid(alpha=0.3)
plt.ylabel("Number of peaks")
plt.xlabel("Epoch")
plt.title("Anomalous Peaks")
wandb.log({"Anomalous Peaks": plt})


