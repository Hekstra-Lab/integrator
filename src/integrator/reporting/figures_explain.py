"""Explanatory figures aimed at an audience outside the field.

These carry no new information over the diagnostic figures; they trade
density for legibility, which is the right trade for a talk's first
slides or a paper's opening figure.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .figure_style import (
    REGIMES,
    imshow_panel,
    middle_slice,
    paper_style,
    regime_color,
)

REGIME_WORDS = {
    "weak": "faint spot",
    "medium": "ordinary spot",
    "strong": "bright spot",
}


def plot_decomposition(
    counts: np.ndarray,
    rate: np.ndarray,
    profile: np.ndarray,
    intensity: float,
    background: float,
    shape,
    title: str = "what the model does to one spot",
):
    """The generative model as a picture: counts ≈ I · p + B.

    Args:
        counts: Observed counts for one shoebox `(K,)`.
        rate: Fitted Poisson rate for the same shoebox `(K,)`.
        profile: Posterior mean profile `(K,)`, sums to one.
        intensity: Posterior mean intensity.
        background: Posterior mean background per pixel.
        shape: Shoebox shape.
        title: Figure title.
    """
    import matplotlib.pyplot as plt

    obs = middle_slice(counts, shape)
    fit = middle_slice(rate, shape)
    signal = middle_slice(intensity * np.asarray(profile), shape)
    bg = np.full_like(signal, float(background))
    vmax = float(max(obs.max(), fit.max())) or 1.0

    with paper_style():
        fig = plt.figure(figsize=(7.4, 2.0))
        grid = fig.add_gridspec(1, 7, width_ratios=[1, 0.3, 1, 0.3, 1, 0.3, 1])
        panels = [
            (0, obs, "measured counts", vmax),
            (2, fit, "model", vmax),
            (4, signal, f"intensity × shape\nI = {intensity:,.0f}", vmax),
            (6, bg, f"background\nB = {background:.1f}/px", vmax),
        ]
        for col, img, label, limit in panels:
            ax = fig.add_subplot(grid[0, col])
            imshow_panel(ax, img, vmax=limit)
            ax.set_title(label, fontsize=8)
        for col, symbol in ((1, "≈"), (3, "="), (5, "+")):
            ax = fig.add_subplot(grid[0, col])
            ax.axis("off")
            ax.text(0.5, 0.55, symbol, fontsize=16, ha="center", va="center")
        fig.suptitle(title, fontsize=10)
    return fig


def plot_posterior_gallery(
    scalars: pl.DataFrame,
    arrays: dict,
    epoch: int | None = None,
):
    """One example per regime with the posterior over its intensity drawn.

    The point for a general audience: the model does not return a
    number, it returns a belief, and that belief is wide exactly where
    the measurement is weak.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gamma as gamma_dist

    epochs = [int(e) for e in arrays["epochs"]]
    epoch = epochs[-1] if epoch is None else int(epoch)
    frame = scalars.filter(pl.col("epoch") == epoch)
    shape = tuple(int(s) for s in arrays["shape"])
    counts = arrays["counts"]

    picks = []
    for regime in REGIMES:
        sub = frame.filter(pl.col("regime") == regime)
        if sub.is_empty():
            continue
        picks.append(sub.sort("dials_snr")[len(sub) // 2])

    with paper_style():
        fig, axes = plt.subplots(
            2, len(picks), figsize=(2.1 * len(picks), 3.4),
            gridspec_kw={"height_ratios": [1.0, 0.85]}, squeeze=False,
        )
        for col, row in enumerate(picks):
            slot = int(row["slot"][0])
            regime = row["regime"][0]
            color = regime_color(regime)
            obs = middle_slice(counts[slot], shape)
            imshow_panel(axes[0][col], obs)
            axes[0][col].set_title(
                f"{REGIME_WORDS.get(regime, regime)}\n"
                f"peak {obs.max():.0f} counts",
                fontsize=8,
                color=color,
            )

            mean = float(row["qi_mean"][0])
            sd = max(float(row["qi_sd"][0]), 1e-6)
            k = max((mean / sd) ** 2, 1e-3)
            theta = sd**2 / max(mean, 1e-6)
            lo = gamma_dist.ppf(0.001, k, scale=theta)
            hi = gamma_dist.ppf(0.999, k, scale=theta)
            xs = np.linspace(max(lo, 0.0), hi, 400)
            ax = axes[1][col]
            ax.fill_between(
                xs, gamma_dist.pdf(xs, k, scale=theta), color=color,
                alpha=0.35, linewidth=0,
            )
            ax.plot(xs, gamma_dist.pdf(xs, k, scale=theta), color=color)
            ax.axvline(
                float(row["dials_i"][0]), color="0.3", linestyle="--",
                linewidth=0.8,
            )
            ax.set_yticks([])
            ax.set_xlabel("intensity")
            ax.set_title(
                f"I = {mean:,.0f} ± {sd:,.0f}", fontsize=8, color=color
            )
        axes[1][0].set_ylabel("belief")
        fig.suptitle(
            "the model reports a distribution, not a number "
            "(dashed = conventional estimate)",
            fontsize=9,
        )
    return fig


def plot_regime_shrinkage(scalars: pl.DataFrame):
    """Model intensity against the conventional estimate, by regime.

    Weak reflections are where the two disagree, and the direction of
    that disagreement is the prior doing its job.
    """
    import matplotlib.pyplot as plt

    last = scalars["epoch"].max()
    frame = scalars.filter(pl.col("epoch") == last)
    with paper_style():
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        for regime in REGIMES:
            sub = frame.filter(pl.col("regime") == regime)
            if sub.is_empty():
                continue
            ax.errorbar(
                sub["dials_i"].to_numpy(),
                sub["qi_mean"].to_numpy(),
                yerr=sub["qi_sd"].to_numpy(),
                fmt="o",
                markersize=4,
                color=regime_color(regime),
                elinewidth=0.8,
                capsize=1.5,
                label=regime,
            )
        lim = [
            min(frame["dials_i"].min(), frame["qi_mean"].min()),
            max(frame["dials_i"].max(), frame["qi_mean"].max()),
        ]
        ax.plot(lim, lim, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("conventional intensity (DIALS)")
        ax.set_ylabel("model posterior intensity")
        ax.set_title(f"epoch {last}", fontsize=8)
        ax.legend()
    return fig
