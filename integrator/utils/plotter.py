import torch
import numpy as np
import matplotlib.pyplot as plt
from dials.array_family import flex
import pickle


# Class to perform analysis on the model output
class OutputPlotter:
    def __init__(self, refl_file, metadata_pkl_file):
        self.refl_tbl = flex.reflection_table().from_file(refl_file)
        with open(metadata_pkl_file, "rb") as f:
            self.metadata = pickle.load(f)

    def plot_loss(
        self,
        save=False,
        out_png_filename="loss.png",
        title="Loss vs. Epoch",
        ylabel="Loss",
        xlabel="Epoch",
        display=True,
    ):

        plt.clf()

        plt.plot(self.metadata["train_avg_loss"], label="train")
        plt.plot(self.metadata["test_avg_loss"], label="test")

        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.title(title)
        plt.grid(alpha=0.3)
        if save:
            plt.savefig(out_png_filename, dpi=600)
        if display:
            plt.display()
        plt.clf()

    def plot_uncertainty(
        self,
        save=False,
        out_png_filename="uncertainty_comparison.png",
        title="Network vs. DIALS Sig(I)",
        ylabel="DIALS Sig(I)",
        xlabel="Network Sig(I)",
        display=True,
    ):
        q_I_stddev = self.refl_tbl["intensity.sum.variance"].as_numpy_array()
        dials_I_stddev = self.refl_tbl["intensity.prf.variance"].as_numpy_array()

        # plot variance
        plt.clf()
        plt.plot([0, 1e6], [0, 1e6], "r", alpha=0.3)
        plt.scatter(q_I_stddev, dials_I_stddev, alpha=0.1, color="black")
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(0, 1e8)
        plt.xlim(0, 1e6)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.grid(alpha=0.3)
        if save:
            plt.savefig(out_png_filename, dpi=600)
        if display:
            plt.display()
        plt.clf()

    def plot_intensities(
        self,
        ylabel="DIALS intensity",
        xlabel="Network intensity",
        title="Predicted vs. DIALS intensity",
        save=False,
        out_png_filename="intensity_comparison.png",
        display=True,
    ):

        q_I_mean = self.refl_tbl["intensity.sum.value"].as_numpy_array()
        dials_I = self.refl_tbl["intensity.prf.value"].as_numpy_array()
        # equality line

        plt.clf()
        plt.plot([0, 1e6], [0, 1e6], "r", alpha=0.3)
        plt.scatter(q_I_mean, dials_I, alpha=0.1, color="black")
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(0, 1e8)
        plt.xlim(0, 1e6)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.grid(alpha=0.3)
        plt.title(title)

        if save:
            plt.savefig(out_png_filename, dpi=600)

        if display:
            plt.display()
        plt.clf()
