from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import reciprocalspaceship as rs
import torch

from integrator.utils import BaseParser


def get_preds(pt: str | Path):
    return torch.load(pt, weights_only=False)


def concatenate_preds(preds: dict):
    return np.concatenate(preds)


def plot_isigi_raw(
    df,
    im_name,
    savefig=False,
    title="I/Sig(I)",
    dpi=300,
):
    fig, ax = plt.subplots()

    x = 1 / (df["dHKL"] ** 2)
    y = df["I"] / df["SIGI"]

    ax.scatter(x, y, color="black", alpha=0.4, s=3.0)
    ax.set_xlabel("(1/d^2)")
    ax.set_ylabel("Mean(I)/Sig(I)")

    ax.grid(alpha=0.4)
    ax.set_title(title)

    if savefig:
        fig.savefig(im_name, dpi=dpi)
    plt.close(fig)


def plot_mean_isigi_raw(
    df,
    im_name,
    savefig=False,
    bins=None,
    title="Mean(Mean(I)/Sig(I))",
    dpi=300,
):
    fig, ax = plt.subplots()

    x = df.index
    y = df["isigi"]

    ax.plot(x, y, color="black")
    ax.set_xlabel("resolution bin")
    ax.set_ylabel("Mean(Mean(I)/Sig(I))")
    if bins is not None:
        ax.set_xticks(x, labels=bins, rotation=55)
    ax.grid(alpha=0.4)
    ax.set_title(title)
    fig.tight_layout()

    if savefig:
        fig.savefig(im_name, dpi=dpi)
    plt.close(fig)


def plot_intensity(
    intensity_x,
    intensity_y,
    im_name: str,
    title: str,
    savefig: bool,
    dpi=300,
    axis_scale: str = "symlog",
):
    # equality line
    max = np.max([intensity_x, intensity_y])
    fig, ax = plt.subplots()

    # plot intensities
    ax.plot(
        [0, max],
        [0, max],
        color="red",
        alpha=0.7,
        label="x=y",
    )
    ax.scatter(
        intensity_x,
        intensity_y,
        color="black",
        alpha=0.3,
        s=5.0,
    )

    ax.set_yscale(axis_scale)
    ax.set_xscale(axis_scale)
    ax.set_ylabel("laue-dials")
    ax.set_xlabel("model")
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend()

    if savefig:
        fig.savefig(im_name, dpi=dpi)
    plt.close(fig)


def main(args):
    # form path to lightning log
    root = Path(args.root)
    id = Path(args.wandb_id)
    path = list((root).glob(f"*{id}"))[0]

    # iterate of all preds and plots raw intensity correlation against laue-dials
    for p in list(path.glob("**/preds.pt")):
        # get some metadata
        epoch = p.parent.name
        wandb_id = p.parents[2].name.split("-")[-1]

        # load torch predictions
        preds = get_preds(p.as_posix())

        # get network and laue-dials intensity
        intensity_x = concatenate_preds(preds["intensity_mean"])
        intensity_y = concatenate_preds(preds["dials_I_prf_value"])

        # plots the intensity correlations
        plot_intensity(
            intensity_x=intensity_x,
            intensity_y=intensity_y,
            im_name=f"{p.parent.as_posix()}/I_ld_vs_nn.png",
            title=f"LD vs NN {epoch}\nwandb id: {wandb_id}",
            savefig=True,
        )

    # plot I/sigI as function of resolution
    for p in list(path.glob("**/preds.mtz")):
        # get some metadata
        epoch = p.parent.name
        wandb_id = p.parents[2].name.split("-")[-1]

        # load mtz file
        df = rs.read_mtz(p.as_posix()).compute_dHKL()
        df, bins = df.assign_resolution_bins()
        df["isigi"] = df["I"] / df["SIGI"]
        mean_df = df.groupby("bin").mean()

        plot_isigi_raw(
            df,
            im_name=f"{p.parent.as_posix()}/ISigI_nn_{epoch}.png",
            savefig=True,
            title=f"Mean(I)/Sig(I) {epoch}\nwandb_id: {wandb_id}",
        )
        plot_mean_isigi_raw(
            mean_df,
            im_name=f"{p.parent.as_posix()}/mean_isigi_nn_{epoch}.png",
            savefig=True,
            bins=bins,
            title=f"Mean(Mean(I)/Sig(I){epoch}\nwandb_id:{wandb_id}",
        )


if __name__ == "__main__":
    argparser = BaseParser()
    args = argparser.parse_args()
    main(args)
