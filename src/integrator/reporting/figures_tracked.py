"""Figures showing how predictions for fixed shoeboxes evolve over training.

All functions take the dumps written by
`integrator.reporting.figure_data.TrackedRecorder`, so they work the same
whether the dump came from a training callback or from replaying saved
checkpoints.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .figure_style import (
    REGIMES,
    add_colorbar,
    fmt_epoch,
    imshow_panel,
    middle_slice,
    paper_style,
    regime_color,
)


def pick_epochs(epochs, n: int = 6) -> list[float]:
    """`n` recorded frames spanning the run by value, first and last included.

    Selecting by epoch *value* (not by position) keeps the static filmstrip
    spanning the whole run even though the dense early sampling packs many
    frames into the first epochs.
    """
    epochs = [float(e) for e in epochs]
    if len(epochs) <= n:
        return epochs
    lo, hi = min(epochs), max(epochs)
    picked: list[float] = []
    for target in np.linspace(lo, hi, n):
        nearest = min(epochs, key=lambda e: abs(e - target))
        if nearest not in picked:
            picked.append(nearest)
    return picked


def _slot_order(selection: dict) -> list[int]:
    """Slots ordered weak, medium, strong so figure rows group by regime."""
    regimes = selection["regime"]
    return sorted(
        range(len(regimes)),
        key=lambda s: (REGIMES.index(regimes[s]), -selection["snr"][s]),
    )


def _row_label(selection: dict, slot: int) -> str:
    snr = selection["snr"][slot]
    return f"{selection['regime'][slot]}\nI/σ={snr:.1f}"


def _balanced_rows(selection: dict, max_rows: int) -> list[int]:
    """Slots spread across the regimes, so the movie shows weak *and* strong.

    Taking the first `max_rows` of the regime-sorted order would fill up on
    weak reflections and never reach strong; splitting the budget evenly
    keeps the contrast the movie is meant to show.
    """
    ordered = _slot_order(selection)
    regime = selection["regime"]
    per = max(1, max_rows // len(REGIMES))
    rows: list[int] = []
    for name in REGIMES:
        rows += [s for s in ordered if regime[s] == name][:per]
    return rows[:max_rows]


def plot_tracked_trajectories(scalars: pl.DataFrame):
    """Posterior intensity trajectories, one panel per regime.

    Each line is one tracked shoebox: the posterior mean with a +/-1 sd
    band, against the DIALS value as a dashed reference. Strong
    reflections lock in within a few epochs; weak ones drift toward the
    prior and keep a wide band.
    """
    import matplotlib.pyplot as plt

    with paper_style():
        fig, axes = plt.subplots(
            1, len(REGIMES), figsize=(7.2, 2.4), sharex=True
        )
        for ax, regime in zip(np.atleast_1d(axes), REGIMES, strict=False):
            sub = scalars.filter(pl.col("regime") == regime)
            color = regime_color(regime)
            for slot in sorted(sub["slot"].unique().to_list()):
                one = sub.filter(pl.col("slot") == slot).sort("epoch")
                epoch = one["epoch"].to_numpy()
                mean = one["qi_mean"].to_numpy()
                sd = one["qi_sd"].to_numpy()
                ax.plot(epoch, mean, color=color, alpha=0.9)
                ax.fill_between(
                    epoch, mean - sd, mean + sd, color=color, alpha=0.15,
                    linewidth=0,
                )
                dials = float(one["dials_i"][0])
                ax.axhline(
                    dials, color="0.35", linestyle="--", linewidth=0.7
                )
            ax.set_title(regime)
            ax.set_xlabel("epoch")
            positive = float(
                min(sub["qi_mean"].min(), sub["dials_i"].min())
            )
            if positive > 0:
                ax.set_yscale("log")
            else:
                ax.set_yscale("symlog", linthresh=1.0)
        np.atleast_1d(axes)[0].set_ylabel("posterior intensity")
        fig.suptitle(
            "tracked reflections: q(I) mean ± sd (dashed = DIALS)",
            fontsize=9,
        )
    return fig


def plot_tracked_uncertainty(scalars: pl.DataFrame):
    """Relative posterior width and residual fit quality against epoch.

    Left: sd/mean of q(I), the quantity that should stay large for weak
    reflections and collapse for strong ones. Right: rms Pearson
    residual of the pixel fit, which is the honest check that the rate
    model is tracking the counts.
    """
    import matplotlib.pyplot as plt

    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4), sharex=True)
        frame = scalars.with_columns(
            (pl.col("qi_sd") / pl.col("qi_mean").abs().clip(1e-6))
            .alias("rel_sd")
        )
        for regime in REGIMES:
            sub = frame.filter(pl.col("regime") == regime)
            if sub.is_empty():
                continue
            color = regime_color(regime)
            for key, ax in (("rel_sd", axes[0]), ("z_rms", axes[1])):
                per_slot = sub.pivot(
                    values=key, index="epoch", on="slot", aggregate_function="first"
                ).sort("epoch")
                epoch = per_slot["epoch"].to_numpy()
                values = per_slot.drop("epoch").to_numpy()
                ax.plot(epoch, values, color=color, alpha=0.25, linewidth=0.8)
                ax.plot(
                    epoch,
                    np.nanmean(values, axis=1),
                    color=color,
                    linewidth=1.8,
                    label=regime,
                )
        axes[0].set_ylabel("sd(I) / mean(I)")
        axes[0].set_yscale("log")
        axes[1].set_ylabel("rms Pearson residual")
        axes[1].axhline(1.0, color="0.35", linestyle="--", linewidth=0.7)
        for ax in axes:
            ax.set_xlabel("epoch")
        axes[1].legend(title=None)
    return fig


def plot_tracked_filmstrip(
    selection: dict,
    scalars: pl.DataFrame,
    arrays: dict,
    field: str = "rate",
    n_epochs: int = 6,
):
    """Grid of tracked shoeboxes (rows) against training epochs (columns).

    Args:
        selection: Contents of `tracked_selection.json`.
        scalars: The `tracked_scalars` frame.
        arrays: The `tracked_arrays` npz contents.
        field: `rate` (predicted counts), `profile`, or `residual`.
        n_epochs: How many epochs to show, evenly spaced.
    """
    import matplotlib.pyplot as plt

    epochs = [float(e) for e in arrays["epochs"]]
    shown = pick_epochs(epochs, n_epochs)
    cols = [epochs.index(e) for e in shown]
    shape = tuple(int(s) for s in arrays["shape"])
    counts = arrays["counts"]
    stack = arrays["rates"] if field != "profile" else arrays["profiles"]
    if field == "residual":
        stack = counts[None, ...] - arrays["rates"]
    order = _slot_order(selection)
    # The join key is a float coordinate; round both sides so a float32 npz
    # value and a float64 parquet value for the same frame still match.
    qi_lookup = {
        (round(float(row["epoch"]), 4), row["slot"]): row["qi_mean"]
        for row in scalars.iter_rows(named=True)
    }
    cbar_label = {"rate": "counts", "profile": "p", "residual": "resid"}[field]

    n_rows, n_cols = len(order), len(cols) + 1
    with paper_style():
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(1.05 * n_cols + 1.4, 1.05 * n_rows),
            squeeze=False,
        )
        for r, slot in enumerate(order):
            obs = middle_slice(counts[slot], shape)
            vmax = float(obs.max()) or 1.0
            imshow_panel(axes[r][0], obs, vmax=vmax)
            axes[r][0].set_ylabel(
                _row_label(selection, slot),
                fontsize=6,
                rotation=0,
                ha="right",
                va="center",
                labelpad=16,
            )
            # One scale for the whole row so the epoch panels are comparable
            # and a single colorbar tells the truth about a faint profile.
            if field == "profile":
                field_max = max(
                    (
                        float(middle_slice(stack[ei][slot], shape).max())
                        for ei in cols
                    ),
                    default=1.0,
                ) or 1.0
            else:
                field_max = vmax
            last_im = None
            for c, epoch_idx in enumerate(cols, start=1):
                img = middle_slice(stack[epoch_idx][slot], shape)
                if field == "residual":
                    last_im = imshow_panel(
                        axes[r][c], img, symmetric=True, vmax=field_max
                    )
                else:
                    last_im = imshow_panel(axes[r][c], img, vmax=field_max)
                posterior = qi_lookup.get((round(epochs[epoch_idx], 4), slot))
                if posterior is not None:
                    axes[r][c].text(
                        0.04,
                        0.96,
                        f"{posterior:,.0f}",
                        transform=axes[r][c].transAxes,
                        fontsize=5.5,
                        color="white",
                        ha="left",
                        va="top",
                    )
            if last_im is not None:
                add_colorbar(last_im, axes[r][-1], label=cbar_label, pad=0.08)
            if r == 0:
                axes[r][0].set_title("observed", fontsize=7)
                for c, epoch_idx in enumerate(cols, start=1):
                    axes[r][c].set_title(
                        f"ep {fmt_epoch(epochs[epoch_idx])}", fontsize=7
                    )
        if field == "rate":
            fig.supxlabel(
                "inset number = posterior mean intensity", fontsize=6,
                color="0.35",
            )
        titles = {
            "rate": "predicted rate  I·p + B",
            "profile": "profile  p",
            "residual": "residual  counts − rate",
        }
        fig.suptitle(titles.get(field, field), fontsize=9)
    return fig


def animate_tracked(selection: dict, arrays: dict, max_rows: int = 6):
    """Animate observed / rate / profile / residual across training epochs."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    epochs = [float(e) for e in arrays["epochs"]]
    shape = tuple(int(s) for s in arrays["shape"])
    counts, rates = arrays["counts"], arrays["rates"]
    profiles = arrays["profiles"]
    order = _balanced_rows(selection, max_rows)
    panels = ("observed", "rate", "profile", "residual")
    n_frames = len(epochs)

    with paper_style():
        fig, axes = plt.subplots(
            len(order),
            len(panels),
            figsize=(1.15 * len(panels) + 1.4, 1.15 * len(order)),
            squeeze=False,
        )
        images = {}
        for r, slot in enumerate(order):
            obs = middle_slice(counts[slot], shape)
            vmax = float(obs.max()) or 1.0
            # Hold the color scale fixed across the movie: for the profile the
            # frame-0 max is the untrained near-uniform profile, so scaling to
            # it would saturate the panel the moment the peak sharpens. A fixed
            # scale lets the peak visibly grow out of the flat prior.
            prof_max = max(
                float(middle_slice(profiles[f][slot], shape).max())
                for f in range(n_frames)
            ) or 1.0
            resid_max = max(
                float(
                    np.abs(
                        middle_slice(counts[slot] - rates[f][slot], shape)
                    ).max()
                )
                for f in range(n_frames)
            ) or 1.0
            obs_im = imshow_panel(axes[r][0], obs, vmax=vmax)
            images[(r, "rate")] = imshow_panel(
                axes[r][1], middle_slice(rates[0][slot], shape), vmax=vmax
            )
            images[(r, "profile")] = imshow_panel(
                axes[r][2], middle_slice(profiles[0][slot], shape),
                vmax=prof_max,
            )
            images[(r, "residual")] = imshow_panel(
                axes[r][3],
                middle_slice(counts[slot] - rates[0][slot], shape),
                symmetric=True,
                vmax=resid_max,
            )
            axes[r][0].set_ylabel(
                _row_label(selection, slot),
                fontsize=6,
                rotation=0,
                ha="right",
                va="center",
                labelpad=16,
            )
            # Fixed scales, so one colorbar per panel stays valid every frame.
            add_colorbar(obs_im, axes[r][0], label="counts")
            add_colorbar(images[(r, "profile")], axes[r][2], label="p")
            add_colorbar(images[(r, "residual")], axes[r][3], label="resid")
        for c, name in enumerate(panels):
            axes[0][c].set_title(name, fontsize=7)
        suptitle = fig.suptitle(f"epoch {fmt_epoch(epochs[0])}", fontsize=9)

        def update(frame):
            artists = [suptitle]
            suptitle.set_text(f"epoch {fmt_epoch(epochs[frame])}")
            for r, slot in enumerate(order):
                rate = middle_slice(rates[frame][slot], shape)
                images[(r, "rate")].set_data(rate)
                images[(r, "profile")].set_data(
                    middle_slice(profiles[frame][slot], shape)
                )
                images[(r, "residual")].set_data(
                    middle_slice(counts[slot] - rates[frame][slot], shape)
                )
                artists += [
                    images[(r, "rate")],
                    images[(r, "profile")],
                    images[(r, "residual")],
                ]
            return artists

        anim = FuncAnimation(
            fig, update, frames=len(epochs), blit=False, interval=250
        )
    return fig, anim
