import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from dials.array_family import flex


# Class to perform analysis on the model output
class Plotter:
    def __init__(self, refl_file, output_dir, encoder_type, profile_type, batch_size):
        self.refl_tbl = flex.reflection_table().from_file(refl_file)
        self.output_dir = output_dir
        self.encoder_type = encoder_type
        self.profile_type = profile_type
        self.batch_size = batch_size

    def _get_output_path(self, filename):
        return os.path.join(self.output_dir, filename)

    def plot_uncertainty(
        self,
        save=False,
        out_png_filename="uncertainty_comparison.png",
        title=None,
        ylabel="DIALS Sig(I)",
        xlabel="Network Sig(I)",
        display=True,
    ):
        q_i_stddev = self.refl_tbl["intensity.sum.variance"].as_numpy_array()
        dials_I_stddev = self.refl_tbl["intensity.prf.variance"].as_numpy_array()

        # plot variance
        plt.clf()
        plt.plot([0, 1e6], [0, 1e6], "r", alpha=0.3)
        plt.scatter(q_i_stddev, dials_I_stddev, alpha=0.1, color="black")
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(1, 1e6)
        plt.xlim(1, 1e6)
        plt.gca().set_aspect("equal", adjustable="box")  # Set aspect ratio to be equal
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

        # Create title if not provided
        if title is None:
            title = (
                f"Network vs. DIALS Sig(I)\n"
                f"Encoder: {self.encoder_type}, Profile: {self.profile_type}, "
                f"Batch Size: {self.batch_size}"
            )
        plt.title(title, fontsize=10)
        plt.grid(alpha=0.3)
        if save:
            full_path = self._get_output_path(out_png_filename)
            plt.savefig(full_path, dpi=600)
        if display:
            plt.show()
        plt.clf()

    def plot_intensities(
        self,
        ylabel="DIALS intensity",
        xlabel="Network intensity",
        title=None,
        save=False,
        out_png_filename="intensity_comparison.png",
        display=True,
    ):
        q_i_mean = self.refl_tbl["intensity.sum.value"].as_numpy_array()
        dials_I = self.refl_tbl["intensity.prf.value"].as_numpy_array()
        # equality line

        plt.clf()
        plt.plot([0, 1e6], [0, 1e6], "r", alpha=0.3)
        plt.scatter(q_i_mean, dials_I, alpha=0.1, color="black")
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(1, 1e6)
        plt.xlim(1, 1e6)

        plt.gca().set_aspect("equal", adjustable="box")  # Set aspect ratio to be equal
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.grid(alpha=0.3)

        # Create title if not provided
        if title is None:
            title = (
                f"Predicted vs. DIALS intensity\n"
                f"Encoder: {self.encoder_type}, Profile: {self.profile_type}, "
                f"Batch Size: {self.batch_size}"
            )
        plt.title(title)

        if save:
            full_path = self._get_output_path(out_png_filename)
            plt.savefig(full_path, dpi=600)

        if display:
            plt.show()
        plt.clf()


def visualize_shoebox(shoebox, color_weight=2, cmap="gray"):
    """
    Visualizes a 3D tensor as a series of 2D slices in a 3D plot.

    Parameters:
    shoebox (numpy.ndarray): 3D tensor to visualize.
    color_weight (float): Weight to modify contrast.
    cmap (str): Colormap to use for visualization.
    """
    # Get the shape of the data
    z, y, x = shoebox.shape

    # matplotlib figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Normalize the data for color mapping
    norm_shoebox = (shoebox - shoebox.min()) / (shoebox.max() - shoebox.min())
    norm = mcolors.Normalize(vmin=shoebox.min(), vmax=shoebox.max())

    # Create a ScalarMappable for the colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(shoebox)

    # Loop through each slice in the stack
    for idx in range(z):
        X, Y = np.meshgrid(np.arange(x), np.arange(y))
        Z = np.full_like(X, idx)

        ax.plot_surface(
            X,
            Y,
            Z,
            facecolors=plt.cm.get_cmap(cmap)(norm(shoebox[idx, ...] * color_weight)),
            shade=True,
            alpha=0.8,
        )

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Add the colorbar
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label("Counts")

    plt.show()


if __name__ == "__main__":
    import matplotlib.colors as mcolors

    samples = torch.load("samples.pt")
    masks = torch.load("masks.pt")
    # %%

    visualize_shoebox = visualize_shoebox(shoebox)

    # %%

    shoebox = samples[..., -1][20000].reshape(3, 21, 21)

    plt.imshow(shoebox[1], cmap="gray_r", vmin=0, vmax=15)

    plt.axis("off")

    plt.show()

    bg_rate = 1.0

    I_rate = 50.0

    dxy = samples[0][..., 3:5, ...][: 21 * 21]

    mvn = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.zeros(2), covariance_matrix=torch.eye(2)
    )
    profile = (torch.exp(mvn.log_prob(dxy))).reshape(21, 21)
    bg = torch.distributions.Poisson(torch.tensor(bg_rate)).sample([21, 21])
    I = torch.distributions.Poisson(torch.tensor(I_rate)).sample([21, 21])
    rate = bg + I * profile
    vmax = rate.max()
    plt.imshow(bg, cmap="gray_r", vmin=0, vmax=vmax)
    plt.imshow(rate, cmap="gray_r", vmin=0, vmax=vmax)
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(bg, cmap="gray", vmin=0, vmax=vmax)
    ax[0].set_title("Bg")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].imshow(I * profile, cmap="gray", vmin=0, vmax=vmax)
    ax[1].set_title("I*Profile")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].imshow(rate, cmap="gray", vmin=0, vmax=vmax)
    ax[2].set_title("Rate = I * Profile + Bg")
    # turn off ticks on ax[2]
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    # plt.savefig('data_generation.png',dpi=600)

    plt.show()

    plt.imshow(profile, cmap="gray_r")
    plt.title("Profile")
    plt.show()

    poisson = torch.distributions.Poisson(torch.tensor([bg_rate, I_rate, I_rate]))

    poisson.sample()
