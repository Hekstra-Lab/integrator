"""Anomalous, refinement, and scatter plots for the merging checkpoint eval.

Self-contained port of the refltorch plotting conventions (refltorch is not
importable from the integrator envs): a y=x scatter primitive, R-work/R-free vs
epoch, metric-vs-epoch curves, and the anomalous peak height at known anomalous
atom sites (the ANOM/PHANOM difference map sampled at the PDB positions, in
sigma -- the real anomalous metric, vs find_peaks' top peak). matplotlib only.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]


def _palette(keys) -> dict:
    keys = list(dict.fromkeys(keys))
    return {k: _COLORS[i % len(_COLORS)] for i, k in enumerate(keys)}


def save_figure(fig, path, dpi: int = 200) -> Path:
    """Save with the project's standard options and close the figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def plot_scatter_identity(
    x,
    y,
    *,
    xlabel=None,
    ylabel=None,
    title=None,
    log=False,
    figsize=(5, 5),
    s=4,
    alpha=0.3,
):
    """Scatter x against y with a y=x reference line (model-vs-reference)."""
    import numpy as np

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if log:
        keep &= (x > 0) & (y > 0)
    x, y = x[keep], y[keep]
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=s, alpha=alpha, c="black", edgecolors="none")
    if len(x):
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], c="red", alpha=0.6, lw=1, label="y = x")
        if len(x) > 2:
            xl = np.log(x) if log else x
            yl = np.log(y) if log else y
            cc = float(np.corrcoef(xl, yl)[0, 1])
            ax.set_title(
                (title + "  " if title else "")
                + ("log-CC" if log else "CC")
                + f"={cc:.3f}  n={len(x)}"
            )
    elif title:
        ax.set_title(title)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    return fig, ax


def plot_r_values_vs_epoch(rows, *, ref_vals=None, title=None, figsize=(6, 4)):
    """R-work (solid) and R-free (dashed) vs epoch, one color per variant.

    `rows`: list of dicts with epoch, variant, r_work, r_free.
    `ref_vals`: optional {r_work, r_free} drawn as horizontal reference lines.
    """
    variants = sorted({r["variant"] for r in rows if r.get("variant")})
    pal = _palette(variants)
    fig, ax = plt.subplots(figsize=figsize)
    for v in variants:
        pts = sorted(
            (r["epoch"], r.get("r_work"), r.get("r_free"))
            for r in rows
            if r["variant"] == v
        )
        ep = [p[0] for p in pts]
        for idx, ls, name in ((1, "-", "r_work"), (2, "--", "r_free")):
            ys = [p[idx] for p in pts]
            xy = [(e, y) for e, y in zip(ep, ys) if isinstance(y, (int, float))]
            if xy:
                ax.plot(
                    [a for a, _ in xy], [b for _, b in xy], ls,
                    marker="o", ms=3, color=pal[v], label=f"{v} {name}",
                )
    if ref_vals:
        if ref_vals.get("r_work") is not None:
            ax.axhline(ref_vals["r_work"], color="#888", lw=1, label="ref r_work")
        if ref_vals.get("r_free") is not None:
            ax.axhline(
                ref_vals["r_free"], color="#888", lw=1, ls="--",
                label="ref r_free",
            )
    ax.set_xlabel("epoch")
    ax.set_ylabel("R value")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    return fig, ax


def plot_metric_over_epoch(
    series,
    *,
    ref_value=None,
    ref_label="DIALS",
    x_label="epoch",
    y_label="value",
    title=None,
    figsize=(6, 4),
):
    """Per-model metric curves over epoch, with an optional reference line.

    `series`: iterable of (label, x_epochs, y_values, color). `ref_value`: a
    scalar (e.g. the DIALS anomalous peakz for this site) drawn as a horizontal
    line. Mirrors refltorch.plots.metric_plots.plot_metric_over_epoch.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for label, x, y, color in series:
        ax.plot(x, y, marker="o", ms=3, color=color, label=label)
    if ref_value is not None:
        ax.axhline(ref_value, color="red", lw=1, ls="--", label=ref_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig, ax
