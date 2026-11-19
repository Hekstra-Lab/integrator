"""Every component of the learned profile basis, and how it settled.

The profile model decodes a low-dimensional latent through one linear map,
so the model's whole vocabulary of shapes is the columns of that map. There
are usually twelve of them over a 3x31x31 box. Showing all of them is the
only way to see what the model can and cannot express, and the standing rule
on this project is that a profile model is never judged on metrics alone.

A column is **not itself a profile.** The profile is the simplex projection
of $Wh + b$, so a column may be negative anywhere and only the combination
is a distribution. The columns are therefore drawn on a diverging scale
about zero and the bias -- the shape the model falls back to when the latent
is zero -- is drawn beside them on its own scale.

The companion panel is what the components did over training: each one's
norm, and its cosine against its own final value. The failure those two
catch, which an aggregate cannot, is one component collapsing while the rest
look settled -- a dead component is one the encoder stopped using, and it
shows as a norm going to zero while the mean cosine stays near one.

Depth is summed, not sliced. A component that puts its mass on a different
slice than the one picked would be invisible in a central slice, and the
profile the model multiplies by the counts is the whole box.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_profile_basis.py \\
            --data docs/figures/data/basis
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def figstyle():
    """figTeX, imported at the call; not a dependency of this project."""
    import figstyle as _figstyle

    return _figstyle


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA = ROOT / "docs" / "figures" / "data" / "basis"
FIGURES = ROOT / "docs" / "figures"
OUT = FIGURES / "m9"
# Which figure directory each dataset's arms are written to. An arm whose
# dataset is not listed falls back to OUT rather than inventing a directory,
# so a dataset missing from here files its figures under another dataset's
# name: `hewl_1118` was absent and its two arms were written into `m9/`.
DATASET_DIR = {"hewl_9b7c": "m9", "pdz2_1116": "pdz2", "hewl_1118": "poly1118"}
SNAPSHOTS = "basis_snapshots.npz"
COMPONENTS = "basis_components.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA,
                   help="directory holding <arm>/basis_snapshots.npz")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="override the per-dataset directory for every arm")
    p.add_argument("--arm", default=None, help="draw only this arm")
    p.add_argument("--prefix", default="basis")
    p.add_argument("--dataset", default=None,
                   help="draw only the arms of this dataset, as named in "
                        "the `dataset` column of basis_components.csv")
    p.add_argument("--template", default=None,
                   help="figTeX template to size for (article, paper, book, "
                        "slides, poster); default is figstyle's")
    return p.parse_args()


def arms_in(data_dir: Path) -> list[tuple[str, Path]]:
    return sorted(
        (p.parent.name, p) for p in data_dir.glob(f"*/{SNAPSHOTS}")
    )


def as_images(weight: np.ndarray, shape) -> np.ndarray:
    """Basis columns as 2D images, depth summed.

    `weight` is `(d*h*w, n_components)`; the result is
    `(n_components, h, w)`. A basis that carries no usable shape -- the
    recorder falls back to a flat `(K,)` where the run's config did not
    give a geometry -- comes back as `(n_components, 1, K)`, which still
    draws, as a strip, rather than failing.
    """
    n, k = weight.shape[1], weight.shape[0]
    dims = [int(s) for s in np.asarray(shape).ravel()]
    if len(dims) == 3 and int(np.prod(dims)) == k:
        d, h, w = dims
        return weight.T.reshape(n, d, h, w).sum(axis=1)
    # The geometric surrogate decodes onto a CANVAS larger than the
    # shoebox -- `nn.Linear(d, d*ch*cw)` -- and then shifts and crops, so
    # the configured h and w do not multiply out to the number of rows. The
    # canvas is square, so recover it from the row count instead of
    # dropping to a strip.
    depth = dims[0] if dims else 1
    if depth and k % depth == 0:
        side = int(round((k // depth) ** 0.5))
        if depth * side * side == k:
            return weight.T.reshape(n, depth, side, side).sum(axis=1)
    side = int(round(k ** 0.5))
    if side * side == k:
        return weight.T.reshape(n, side, side)
    return weight.T.reshape(n, 1, k)


def _image_axes(ax) -> None:
    """An image panel: no ticks, and `Panels.finish` leaves it alone."""
    ax._image = True
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def draw_basis(images: np.ndarray, bias: np.ndarray, shape, arm: str,
               epoch: float, out_dir: Path, prefix: str) -> Path:
    """Every component at one epoch, plus the bias, on one grid.

    The components share one colour bar on a symmetric norm, so zero is the
    midpoint of the diverging ramp and a component with little amplitude
    reads as pale rather than being rescaled to look like the others. The
    bias is a different quantity -- the decoded shape at zero latent -- and
    is drawn on the sequential ramp without a bar of its own: only its
    shape is read from it, never its values.

    The headings are the component index and nothing else; which run drew
    the basis belongs in the filename, where it already is.
    """
    from matplotlib.colors import Normalize

    n = images.shape[0]
    bias_image = as_images(np.asarray(bias).reshape(-1, 1), shape)[0]
    # six across once there are more than twelve cells: sixteen components
    # and the bias at four across left the bias alone on a fifth row and
    # the figure at figTeX's 0.6 text-height cap
    cols = min(4, n + 1) if n + 1 <= 12 else 6
    rows = (n + 1 + cols - 1) // cols
    aspect = images.shape[1] / images.shape[2]
    P = figstyle().panels(f"{rows}x{cols}", images=aspect, shared=False)
    flat = list(P.flat)

    span = float(np.abs(images).max()) or 1.0
    norm = Normalize(vmin=-span, vmax=span)
    mappable = None
    for i in range(n):
        ax = flat[i]
        mappable = ax.imshow(images[i], cmap=figstyle().diverging_cmap(),
                             norm=norm, interpolation="nearest")
        P.head(ax, i, f"{i}")
        _image_axes(ax)
    ax = flat[n]
    ax.imshow(bias_image, cmap=figstyle().cmap(), interpolation="nearest")
    P.head(ax, n, "bias")
    _image_axes(ax)
    for ax in flat[n + 1:]:
        # ticks removed as well: figTeX's clip check measures the default
        # 0-1 tick labels of a hidden axes
        _image_axes(ax)
        ax.set_visible(False)
    # named by a short title over the bar rather than a rotated label beside
    # it: figTeX's image geometry reserves 8 tick-sizes for the bar, which
    # holds the tick numbers but not a rotated label as well, and in final
    # mode `weight, depth summed` ran 6.8 pt past the right edge of the
    # article figure
    cb = P.colorbar(mappable)
    cb.ax.set_title("weight", fontsize=P.v["f"], loc="center")
    P.finish()
    path = out_dir / f"{prefix}_components_{arm}"
    figstyle().save(P.fig, path)
    plt.close(P.fig)
    logger.info("  %s: epoch %d, %d components, symmetric span %.3g",
                arm, int(epoch), n, span)
    return path


def draw_history(frame: pl.DataFrame, arm: str, out_dir: Path,
                 prefix: str) -> Path | None:
    """Per-component norm and cosine to final, against epoch.

    Two quantities on two scales, so neither axis is shared: `panels`
    shares x by default on a grid, and sharing y would put a cosine on a
    norm's axis.
    """
    part = frame.filter(pl.col("method") == arm)
    if not len(part):
        return None
    components = sorted(part["component"].unique().to_list())
    _, ramp = figstyle().epoch_ramp(len(components))
    P = figstyle().panels("2", shared=False)
    axes = list(P.flat)
    for i, component in enumerate(components):
        line = part.filter(pl.col("component") == component).sort("epoch")
        axes[0].plot(line["epoch"], line["norm"], lw=0.8, color=ramp[i])
        axes[1].plot(line["epoch"], line["cos_to_final"], lw=0.8,
                     color=ramp[i])
    axes[0].set_ylabel("column norm")
    axes[1].set_ylabel("cosine to its final value")
    for ax in axes:
        ax.set_xlabel("epoch")
    P.head(axes[0], 0, "amplitude per component")
    P.head(axes[1], 1, "convergence per component")
    P.finish()
    path = out_dir / f"{prefix}_history_{arm}"
    figstyle().save(P.fig, path)
    plt.close(P.fig)
    return path


def main() -> int:
    args = parse_args()
    found = arms_in(args.data)
    if not found:
        raise SystemExit(
            f"no <arm>/{SNAPSHOTS} under {args.data}; the cluster side is "
            "scripts/figures/dump_basis.py, pulled by "
            "`bash scripts/figures/pull_figdata.sh basis`"
        )
    if args.arm:
        found = [(a, p) for a, p in found if a == args.arm]
        if not found:
            raise SystemExit(f"no snapshots for arm {args.arm!r}")

    if args.template:
        figstyle().use(args.template)
    else:
        figstyle().use()
    history = None
    csv_path = args.data / COMPONENTS
    if csv_path.is_file():
        history = pl.read_csv(csv_path, infer_schema_length=None)
    else:
        logger.warning("no %s; drawing the basis without its history",
                       csv_path)

    # an arm belongs to the dataset its component rows say it does, so the
    # figure lands beside that dataset's other figures rather than all of
    # them landing in the monochromatic directory
    where = {}
    if history is not None and "dataset" in history.columns:
        for arm, dataset in history.select("method", "dataset").unique().rows():
            where[arm] = FIGURES / DATASET_DIR.get(str(dataset), OUT.name)

    if args.dataset:
        if history is None or "dataset" not in history.columns:
            raise SystemExit(f"--dataset needs the dataset column of {csv_path}")
        keep = set(history.filter(pl.col("dataset") == args.dataset)["method"])
        found = [(a, p) for a, p in found if a in keep]
        if not found:
            raise SystemExit(f"no arms of dataset {args.dataset!r}")

    written = 0
    for arm, path in found:
        out_dir = args.out_dir or where.get(arm, OUT)
        out_dir.mkdir(parents=True, exist_ok=True)
        blob = np.load(path)
        weights, epochs = blob["weights"], blob["epochs"]
        shape = blob["shape"] if "shape" in blob else ()
        bias = (blob["biases"][-1] if "biases" in blob
                else np.zeros(weights.shape[1]))
        images = as_images(weights[-1], shape)
        out = draw_basis(images, bias, shape, arm, float(epochs[-1]),
                         out_dir, args.prefix)
        logger.info("Saved %s (%d components, %d epochs)",
                    out, images.shape[0], len(epochs))
        written += 1
        if history is not None:
            out = draw_history(history, arm, out_dir, args.prefix)
            if out is not None:
                logger.info("Saved %s", out)
                written += 1
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
