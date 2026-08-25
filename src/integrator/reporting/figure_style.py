"""Shared style, colors, and output helpers for the training figures.

Every figure module here writes through `save_figure`, so a single call
site controls the file formats and the paper resolution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REGIMES = ("weak", "medium", "strong")

REGIME_COLORS = {
    "weak": "#0173B2",
    "medium": "#DE8F05",
    "strong": "#029E73",
}

SEQUENTIAL_CMAP = "cividis"
DIVERGING_CMAP = "RdBu_r"

PAPER_RC = {
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
    "image.interpolation": "nearest",
    "figure.constrained_layout.use": True,
}


def paper_style():
    """Return an rc context manager with the publication defaults."""
    import matplotlib.pyplot as plt

    return plt.rc_context(PAPER_RC)


def save_figure(
    fig,
    out_dir: str | Path,
    name: str,
    formats: tuple[str, ...] = ("png", "pdf"),
    close: bool = True,
) -> list[Path]:
    """Write `fig` as `<out_dir>/<name>.<ext>` for each requested format."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        written.append(path)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    logger.info("wrote %s.{%s}", out_dir / name, ",".join(formats))
    return written


def save_animation(
    anim,
    out_dir: str | Path,
    name: str,
    fps: int = 4,
    formats: tuple[str, ...] = ("gif",),
) -> list[Path]:
    """Write a `FuncAnimation` as GIF and/or MP4.

    MP4 needs ffmpeg; when it is missing the MP4 is skipped with a warning
    rather than failing the whole figure run.
    """
    from matplotlib import animation

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out_dir / f"{name}.{ext}"
        if ext == "gif":
            writer = animation.PillowWriter(fps=fps)
        elif ext == "mp4":
            if not animation.FFMpegWriter.isAvailable():
                logger.warning("ffmpeg not available; skipping %s", path)
                continue
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        else:
            raise ValueError(f"unsupported animation format {ext!r}")
        anim.save(path, writer=writer, dpi=130)
        written.append(path)
        logger.info("wrote %s", path)
    return written


def middle_slice(vec: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reshape a flat shoebox vector and return its central z-slice."""
    if len(shape) == 2:
        return np.asarray(vec).reshape(shape)
    d, h, w = shape
    return np.asarray(vec).reshape(d, h, w)[d // 2]


def imshow_panel(ax, img, cmap=SEQUENTIAL_CMAP, symmetric=False, vmax=None):
    """Draw a shoebox image with no ticks and a sensible color range."""
    img = np.asarray(img, dtype=float)
    if symmetric:
        lim = vmax if vmax is not None else float(np.abs(img).max()) or 1.0
        im = ax.imshow(img, cmap=DIVERGING_CMAP, vmin=-lim, vmax=lim)
    else:
        hi = vmax if vmax is not None else float(img.max())
        im = ax.imshow(img, cmap=cmap, vmin=0.0, vmax=hi if hi > 0 else 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def add_colorbar(im, ax, label=None, size="6%", pad=0.04):
    """Attach a colorbar to `ax` sized to match the panel.

    Uses an axes divider so the bar is exactly the panel height regardless
    of the panel's aspect ratio, which the `fraction=` trick only gets
    right for square axes.  Without a scale bar a per-panel autoscaled
    image reads as if a faint, near-uniform profile had a confident peak,
    so the color range has to be shown.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cax = make_axes_locatable(ax).append_axes("right", size=size, pad=pad)
    cb = ax.figure.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5, length=2, pad=1)
    cb.outline.set_visible(False)
    if label:
        cb.set_label(label, fontsize=6)
    return cb


def fmt_epoch(value) -> str:
    """Label a possibly fractional epoch: `5` for whole ones, `0.44` within."""
    value = float(value)
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}"
    return f"{value:.2f}"


def regime_color(regime: str) -> str:
    """Color for a `weak`/`medium`/`strong` regime label."""
    return REGIME_COLORS.get(regime, "#666666")
