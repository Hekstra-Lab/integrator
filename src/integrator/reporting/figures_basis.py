"""Figures for the learned profile basis and how it settles during training.

The profile surrogate decodes `p = softmax(W h + b)`, so `W`'s columns are
the profile modes and `b` is the mean profile in logit space. A linear
decoder has no canonical column order or sign, so everything here is
ordered by the final-epoch column norm and sign-aligned to the final
epoch before plotting.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from .figure_style import (
    add_colorbar,
    fmt_epoch,
    imshow_panel,
    middle_slice,
    paper_style,
)


def align_basis(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order and sign-align stacked decoder weights `(E, K, d)`.

    Columns are ordered by their norm in the final epoch and each epoch's
    column is sign-flipped to agree with the final one, so a column means
    the same thing across the whole filmstrip.

    Returns:
        Tuple of the aligned stack and the column order applied.
    """
    weights = np.asarray(weights, dtype=float)
    final = weights[-1]
    order = np.argsort(-np.linalg.norm(final, axis=0))
    aligned = weights[:, :, order]
    reference = aligned[-1]
    signs = np.sign((aligned * reference[None]).sum(axis=1))
    signs[signs == 0] = 1.0
    return aligned * signs[:, None, :], order


def _mode_images(w: np.ndarray, shape) -> list[np.ndarray]:
    return [middle_slice(w[:, i], shape) for i in range(w.shape[1])]


def plot_basis_atlas(
    weight: np.ndarray,
    bias: np.ndarray,
    shape,
    mode: str = "weight",
    scale: float = 3.0,
):
    """All basis modes as images, next to the mean profile.

    Args:
        weight: Decoder weight `(K, d)`.
        bias: Decoder bias `(K,)`.
        shape: Shoebox shape, `(H, W)` or `(D, H, W)`.
        mode: `weight` plots the raw columns; `effect` plots the profile
            change `softmax(b + scale·W_i) − softmax(b)`, which is what
            the mode does to an actual profile.
        scale: Latent amplitude used by `effect`.
    """
    import matplotlib.pyplot as plt

    weight = np.asarray(weight, dtype=float)
    bias = np.asarray(bias, dtype=float)

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    base = softmax(bias)
    if mode == "effect":
        panels = [
            (
                f"h{i}",
                middle_slice(softmax(bias + scale * weight[:, i]) - base, shape),
            )
            for i in range(weight.shape[1])
        ]
    else:
        panels = [
            (f"h{i}", img)
            for i, img in enumerate(_mode_images(weight, shape))
        ]

    n = len(panels) + 1
    ncols = min(6, n)
    nrows = math.ceil(n / ncols)
    with paper_style():
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(1.25 * ncols, 1.35 * nrows), squeeze=False
        )
        flat = axes.ravel()
        mean_im = imshow_panel(flat[0], middle_slice(base, shape))
        add_colorbar(mean_im, flat[0], label="p")
        flat[0].set_title("mean profile", fontsize=7)
        for ax, (title, img) in zip(flat[1:], panels, strict=False):
            im = imshow_panel(ax, img, symmetric=True)
            add_colorbar(im, ax)
            ax.set_title(title, fontsize=7)
        for ax in flat[n:]:
            ax.axis("off")
        label = "profile modes" if mode == "weight" else "mode effect on p"
        fig.suptitle(f"learned basis: {label}", fontsize=9)
    return fig


def plot_basis_filmstrip(
    snapshots: dict, n_epochs: int = 6, max_modes: int = 8
):
    """Basis modes (rows) against training epochs (columns)."""
    import matplotlib.pyplot as plt

    epochs = [float(e) for e in snapshots["epochs"]]
    shape = tuple(int(s) for s in snapshots["shape"])
    aligned, _ = align_basis(snapshots["weights"])
    idx = np.unique(
        np.linspace(0, len(epochs) - 1, min(n_epochs, len(epochs)))
        .round()
        .astype(int)
    )
    n_modes = min(max_modes, aligned.shape[2])

    with paper_style():
        fig, axes = plt.subplots(
            n_modes,
            len(idx),
            figsize=(1.05 * len(idx) + 1.2, 1.05 * n_modes),
            squeeze=False,
        )
        for r in range(n_modes):
            lim = float(np.abs(aligned[:, :, r]).max()) or 1.0
            last_im = None
            for c, e in enumerate(idx):
                last_im = imshow_panel(
                    axes[r][c],
                    middle_slice(aligned[e][:, r], shape),
                    symmetric=True,
                    vmax=lim,
                )
                if r == 0:
                    axes[r][c].set_title(
                        f"ep {fmt_epoch(epochs[e])}", fontsize=7
                    )
            if last_im is not None:
                add_colorbar(last_im, axes[r][-1], pad=0.08)
            axes[r][0].set_ylabel(
                f"h{r}", fontsize=7, rotation=0, ha="right", va="center",
                labelpad=8,
            )
        fig.suptitle("profile basis over training", fontsize=9)
    return fig


def plot_basis_convergence(diagnostics: pl.DataFrame):
    """Spectrum, effective rank, and step size of the decoder per epoch.

    The spectrum answers whether the chosen latent dimension is being
    used: modes whose singular value never lifts off the floor are dead
    capacity. The step size and the principal angle to the final basis
    say when the basis stops moving, which is the honest place to read
    off convergence.
    """
    import matplotlib.pyplot as plt

    frame = diagnostics.sort("epoch")
    epoch = frame["epoch"].to_numpy()
    sv_cols = sorted(
        (c for c in frame.columns if c.startswith("sv_")),
        key=lambda c: int(c.split("_")[1]),
    )
    with paper_style():
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
        cmap = plt.get_cmap("viridis")
        for i, col in enumerate(sv_cols):
            axes[0].plot(
                epoch,
                frame[col].to_numpy(),
                color=cmap(i / max(len(sv_cols) - 1, 1)),
                label=f"σ{i}" if i in (0, len(sv_cols) - 1) else None,
            )
        axes[0].set_yscale("log")
        axes[0].set_ylabel("singular value")
        axes[0].set_title("decoder spectrum")
        axes[0].legend()

        axes[1].plot(epoch, frame["eff_rank"].to_numpy(), color="#0173B2")
        axes[1].axhline(
            len(sv_cols), color="0.5", linestyle=":", linewidth=0.8
        )
        axes[1].set_ylabel("effective rank")
        axes[1].set_title("modes actually used")

        axes[2].plot(
            epoch, frame["rel_step"].to_numpy(), color="#DE8F05",
            label="‖ΔW‖/‖W‖",
        )
        axes[2].set_yscale("log")
        axes[2].set_ylabel("relative step")
        twin = axes[2].twinx()
        twin.plot(
            epoch,
            frame["angle_to_final_deg"].to_numpy(),
            color="#029E73",
            label="angle to final",
        )
        twin.set_ylabel("principal angle (°)")
        twin.spines["top"].set_visible(False)
        axes[2].set_title("basis convergence")
        lines = axes[2].get_lines() + twin.get_lines()
        axes[2].legend(lines, [ln.get_label() for ln in lines], loc="best")

        for ax in axes:
            ax.set_xlabel("epoch")
    return fig


def animate_basis(snapshots: dict, max_modes: int = 12):
    """Animate the basis atlas across training epochs."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    epochs = [float(e) for e in snapshots["epochs"]]
    shape = tuple(int(s) for s in snapshots["shape"])
    aligned, order = align_basis(snapshots["weights"])
    biases = np.asarray(snapshots["biases"], dtype=float)
    n_modes = min(max_modes, aligned.shape[2])
    lims = [
        float(np.abs(aligned[:, :, r]).max()) or 1.0 for r in range(n_modes)
    ]

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    # Fixed color scales across the movie: the mean profile grows out of a
    # flat prior, so scaling to frame 0 would saturate it immediately.
    mean_max = max(
        float(middle_slice(softmax(biases[f]), shape).max())
        for f in range(len(epochs))
    ) or 1.0

    n = n_modes + 1
    ncols = min(6, n)
    nrows = math.ceil(n / ncols)
    with paper_style():
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(1.4 * ncols, 1.35 * nrows), squeeze=False
        )
        flat = axes.ravel()

        mean_im = imshow_panel(
            flat[0], middle_slice(softmax(biases[0]), shape), vmax=mean_max
        )
        add_colorbar(mean_im, flat[0], label="p")
        flat[0].set_title("mean profile", fontsize=7)
        images = []
        for r in range(n_modes):
            im = imshow_panel(
                flat[r + 1],
                middle_slice(aligned[0][:, r], shape),
                symmetric=True,
                vmax=lims[r],
            )
            add_colorbar(im, flat[r + 1])
            images.append(im)
            flat[r + 1].set_title(f"h{r}", fontsize=7)
        for ax in flat[n:]:
            ax.axis("off")
        suptitle = fig.suptitle(
            f"basis, epoch {fmt_epoch(epochs[0])}", fontsize=9
        )

        def update(frame):
            suptitle.set_text(f"basis, epoch {fmt_epoch(epochs[frame])}")
            mean_im.set_data(middle_slice(softmax(biases[frame]), shape))
            for r, im in enumerate(images):
                im.set_data(middle_slice(aligned[frame][:, r], shape))
            return [mean_im, *images, suptitle]

        anim = FuncAnimation(
            fig, update, frames=len(epochs), blit=False, interval=250
        )
    return fig, anim
