#!/usr/bin/env python
"""Compare the profiles different arms predict for the same reflections.

Reads the `tracked_arrays.npz` written by `make_training_figures.py` for
several runs and puts them side by side. This is only meaningful because
`select_tracked` picks its reflections from the DIALS metadata and the raw
counts, never from the model, so every arm on a dataset tracks the same
shoeboxes and the panels can be read across a row.

Three figures and a table:

* `fig_family_profiles` -- one row per tracked reflection, the observed
  shoebox then each arm's predicted profile, on a colour scale shared
  across the row so breadth differences are visible rather than
  normalized away.
* `fig_family_radial` -- azimuthal average about the box centre, one panel
  per regime, which separates breadth from the speckle in a single
  shoebox.
* `fig_family_neff` -- `n_eff = 1/sum(pi^2)` per arm and regime, the scalar
  the radial curves are summarizing.
* `family_profile_stats.csv` -- the per-reflection numbers behind all three.

Usage:

    uv run python scripts/compare_profile_families.py FIG_DIR [FIG_DIR ...] \
        --out-dir DEST [--labels a,b,c] [--epoch -1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REGIME_ORDER = ("weak", "medium", "strong")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "fig_dirs",
        nargs="+",
        type=Path,
        help="directories holding tracked_arrays.npz",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--labels",
        default=None,
        help="comma-separated arm names; default: directory basenames",
    )
    p.add_argument(
        "--epoch",
        type=int,
        default=-1,
        help="index into the stored epochs, -1 for the last",
    )
    p.add_argument(
        "--grid-per-regime",
        type=int,
        default=2,
        help="reflections per regime in the image grids; the radial curves, "
        "n_eff and the CSV always use all of them",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def load_arm(fig_dir: Path, epoch: int) -> dict | None:
    """Read one arm's tracked profiles, or None when the dump is absent."""
    npz = fig_dir / "tracked_arrays.npz"
    if not npz.is_file():
        print(f"  {fig_dir.name}: no tracked_arrays.npz, skipping")
        return None
    z = np.load(npz, allow_pickle=True)
    profiles = z["profiles"]
    if profiles.ndim != 3:
        print(f"  {fig_dir.name}: profiles have shape {profiles.shape}, skipping")
        return None

    sel_path = fig_dir / "tracked_selection.json"
    sel = json.loads(sel_path.read_text()) if sel_path.is_file() else {}

    return {
        "dir": fig_dir,
        "profile": profiles[epoch],
        "counts": z["counts"],
        "masks": z["masks"],
        "shape": tuple(int(v) for v in z["shape"]),
        "epoch": float(z["epochs"][epoch]),
        "refl_id": [int(v) for v in sel.get("refl_id", [])],
        "regime": list(sel.get("regime", [])),
        "snr": [float(v) for v in sel.get("snr", [])],
    }


def align(arms: list[dict]) -> list[dict]:
    """Keep the arms whose tracked reflections match the first arm's."""
    ref = arms[0]
    kept = [ref]
    for arm in arms[1:]:
        if arm["profile"].shape != ref["profile"].shape:
            print(
                f"  {arm['dir'].name}: {arm['profile'].shape} against "
                f"{ref['profile'].shape}, dropping"
            )
            continue
        if arm["refl_id"] and ref["refl_id"] and arm["refl_id"] != ref["refl_id"]:
            print(f"  {arm['dir'].name}: different tracked reflections, dropping")
            continue
        kept.append(arm)
    return kept


def normalize(profile: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Renormalize to sum one over the valid pixels."""
    p = np.clip(np.asarray(profile, dtype=float), 0.0, None)
    if mask is not None:
        p = p * np.asarray(mask, dtype=float)
    return p / np.clip(p.sum(-1, keepdims=True), 1e-12, None)


def n_eff(profile: np.ndarray) -> np.ndarray:
    """Kish effective pixel count, `1/sum(pi^2)`; lower is more peaked."""
    p = normalize(profile)
    return 1.0 / np.clip((p**2).sum(-1), 1e-12, None)


def radial_bins(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Integer radius of every pixel from the box centre, and the radii present.

    Shoeboxes are cut around the predicted centroid, so the box centre is the
    one reference every arm shares; measuring about each profile's own centre
    of mass would hide a profile that is merely misplaced.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - (h - 1) / 2.0, xx - (w - 1) / 2.0).ravel()
    r_int = np.round(r).astype(int)
    return r_int, np.unique(r_int)


def radial_mean(profile: np.ndarray, r_int: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Azimuthal average of a normalized profile at each integer radius."""
    p = normalize(profile)
    return np.stack(
        [p[..., r_int == r].mean(-1) for r in radii], axis=-1
    )


def tail_mass(profile: np.ndarray, r_int: np.ndarray, r_cut: float = 5.0) -> np.ndarray:
    """Profile mass beyond `r_cut` from the box centre.

    This separates two things `n_eff` conflates. A softmax profile can hold a
    sharp core and still spread most of its MASS over a flat pedestal, because
    a pedestal contributes almost nothing to `sum(pi^2)` while contributing
    directly to `I_hat = sum(w pi (c - b)) / sum(w pi^2)`, where it multiplies
    background pixels.
    """
    return normalize(profile)[..., r_int > r_cut].sum(-1)


def r_containing(profile: np.ndarray, r_int: np.ndarray, frac: float) -> np.ndarray:
    """Smallest radius whose enclosed mass reaches `frac`."""
    p = normalize(profile)
    order = np.argsort(r_int)
    cum = np.cumsum(p[..., order], axis=-1)
    hit = np.argmax(cum >= frac, axis=-1)
    return r_int[order][hit].astype(float)


def grid_rows(arms: list[dict], per_regime: int) -> list[int]:
    """Evenly spaced reflections within each regime, for the image grids."""
    ref = arms[0]
    n = ref["profile"].shape[0]
    regimes = ref["regime"] or ["all"] * n
    if per_regime <= 0:
        return list(range(n))
    rows: list[int] = []
    for regime in [r for r in REGIME_ORDER if r in regimes] or sorted(set(regimes)):
        members = [i for i, g in enumerate(regimes) if g == regime]
        take = min(per_regime, len(members))
        picks = np.linspace(0, len(members) - 1, take).round().astype(int)
        rows.extend(members[p] for p in dict.fromkeys(picks))
    return rows


def figure_profiles(arms: list[dict], out: Path, dpi: int, rows: list[int]) -> None:
    """Observed shoebox and every arm's profile, one row per reflection."""
    import matplotlib.pyplot as plt

    ref = arms[0]
    shape = ref["shape"]
    n_refl = len(rows)
    ncol = len(arms) + 1

    fig, axes = plt.subplots(
        n_refl, ncol, figsize=(1.45 * ncol, 1.5 * n_refl), squeeze=False
    )
    obs = normalize(ref["counts"] * ref["masks"])

    for row_ix, i in enumerate(rows):
        ax = axes[row_ix][0]
        ax.imshow(
            (ref["counts"][i] * ref["masks"][i]).reshape(shape),
            cmap="cividis",
            origin="lower",
        )
        label = ref["regime"][i] if i < len(ref["regime"]) else ""
        snr = ref["snr"][i] if i < len(ref["snr"]) else float("nan")
        ax.set_ylabel(f"{label}\n$\\sqrt{{I}}$={snr:.1f}", fontsize=6)
        if row_ix == 0:
            ax.set_title("observed", fontsize=7)

        # One scale for the whole row: each arm's profile is a distribution
        # over the same pixels, so rescaling per panel would erase exactly
        # the breadth difference the figure exists to show.
        row = np.stack([normalize(a["profile"][i]) for a in arms])
        vmax = float(row.max())

        for j, arm in enumerate(arms, start=1):
            p = normalize(arm["profile"][i])
            axes[row_ix][j].imshow(
                p.reshape(shape), cmap="cividis", origin="lower", vmin=0.0, vmax=vmax
            )
            axes[row_ix][j].text(
                0.04,
                0.93,
                f"{n_eff(arm['profile'][i]):.0f}",
                transform=axes[row_ix][j].transAxes,
                fontsize=5.5,
                color="w",
                va="top",
            )
            if row_ix == 0:
                axes[row_ix][j].set_title(arm["label"], fontsize=6, rotation=30,
                                          ha="left")

        for ax in axes[row_ix]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"predicted profile by arm, epoch {ref['epoch']:.0f}; "
        f"shared scale per row, inset is $n_{{eff}}$",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

    # The residual against the observed shoebox, same layout.
    fig, axes = plt.subplots(
        n_refl, len(arms), figsize=(1.45 * len(arms), 1.5 * n_refl), squeeze=False
    )
    for row_ix, i in enumerate(rows):
        row = np.stack([normalize(a["profile"][i]) - obs[i] for a in arms])
        lim = float(np.abs(row).max())
        for j, arm in enumerate(arms):
            axes[row_ix][j].imshow(
                row[j].reshape(shape),
                cmap="RdBu_r",
                origin="lower",
                vmin=-lim,
                vmax=lim,
            )
            axes[row_ix][j].set_xticks([])
            axes[row_ix][j].set_yticks([])
            if row_ix == 0:
                axes[row_ix][j].set_title(arm["label"], fontsize=6, rotation=30,
                                          ha="left")
        axes[row_ix][0].set_ylabel(
            ref["regime"][i] if i < len(ref["regime"]) else "", fontsize=6
        )
    # The observed shoebox is 97% background on Laue data, so a broad profile
    # scores a LOWER residual here than a correct one. Read this panel for
    # where the mass sits, never as a goodness-of-fit.
    fig.suptitle("predicted minus observed (observed includes background)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    resid = out.with_name(out.name.replace("profiles", "residual"))
    fig.savefig(resid, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {resid}")


def figure_radial(arms: list[dict], out: Path, dpi: int) -> None:
    """Azimuthal average per regime, every arm on one axis."""
    import matplotlib.pyplot as plt

    ref = arms[0]
    r_int, radii = radial_bins(ref["shape"])
    regimes = ref["regime"] or ["all"] * ref["profile"].shape[0]
    present = [r for r in REGIME_ORDER if r in regimes] or sorted(set(regimes))

    fig, axes = plt.subplots(
        1, len(present), figsize=(3.6 * len(present), 3.2), squeeze=False
    )
    obs_curve = radial_mean(ref["counts"] * ref["masks"], r_int, radii)

    for k, regime in enumerate(present):
        ax = axes[0][k]
        rows = [i for i, g in enumerate(regimes) if g == regime]
        ax.plot(
            radii,
            obs_curve[rows].mean(0),
            color="0.3",
            ls="--",
            lw=1.2,
            label="observed (signal+bg)",
        )
        for arm in arms:
            curve = radial_mean(arm["profile"], r_int, radii)
            ax.plot(radii, curve[rows].mean(0), lw=1.2, label=arm["label"])
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-8)
        ax.set_xlabel("radius from box centre (px)")
        if k == 0:
            ax.set_ylabel("mean $\\pi$ per pixel")
        ax.set_title(regime, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)

    axes[0][-1].legend(fontsize=6, frameon=False)
    fig.suptitle("radial profile by arm", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def figure_neff(arms: list[dict], out: Path, dpi: int) -> None:
    """Peakedness and pedestal mass per arm, split by regime."""
    import matplotlib.pyplot as plt

    ref = arms[0]
    r_int, _ = radial_bins(ref["shape"])
    regimes = ref["regime"] or ["all"] * ref["profile"].shape[0]
    present = [r for r in REGIME_ORDER if r in regimes] or sorted(set(regimes))
    x = np.arange(len(arms))

    fig, axes = plt.subplots(
        1, 2, figsize=(2.0 + 1.1 * len(arms), 3.6), squeeze=False
    )
    panels = (
        (axes[0][0], lambda a, rows: np.median(n_eff(a["profile"][rows])),
         "median $n_{eff}$  (lower = more peaked)"),
        (axes[0][1], lambda a, rows: np.median(tail_mass(a["profile"][rows], r_int)),
         "profile mass beyond r = 5 px"),
    )
    for ax, stat, ylabel in panels:
        for regime in present:
            rows = [i for i, g in enumerate(regimes) if g == regime]
            ax.plot(
                x,
                [float(stat(a, rows)) for a in arms],
                marker="o",
                ms=4,
                lw=1.2,
                label=regime,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [a["label"] for a in arms], rotation=40, ha="right", fontsize=7
        )
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def write_stats(arms: list[dict], out: Path) -> None:
    """Per-reflection scalars for every arm."""
    import polars as pl

    ref = arms[0]
    r_int, _ = radial_bins(ref["shape"])
    obs = normalize(ref["counts"] * ref["masks"])
    rows = []
    for arm in arms:
        p = normalize(arm["profile"])
        rows.append(
            pl.DataFrame(
                {
                    "arm": [arm["label"]] * p.shape[0],
                    "slot": np.arange(p.shape[0]),
                    "refl_id": ref["refl_id"] or list(range(p.shape[0])),
                    "regime": ref["regime"] or ["all"] * p.shape[0],
                    "sqrt_i": ref["snr"] or [float("nan")] * p.shape[0],
                    "n_eff": n_eff(arm["profile"]),
                    "tail_frac": tail_mass(arm["profile"], r_int),
                    "peak": p.max(-1),
                    "r50": r_containing(arm["profile"], r_int, 0.50),
                    "r90": r_containing(arm["profile"], r_int, 0.90),
                    "resid_rms": np.sqrt(((p - obs) ** 2).mean(-1)),
                }
            )
        )
    df = pl.concat(rows)
    df.write_csv(out)
    print(f"  wrote {out}")

    summary = (
        df.group_by("arm")
        .agg(
            pl.col("n_eff").median().alias("n_eff_median"),
            pl.col("tail_frac").median().alias("tail_frac_median"),
            pl.col("r50").median().alias("r50_median"),
            pl.col("resid_rms").mean().alias("resid_rms_mean"),
        )
        .sort("n_eff_median")
    )
    print()
    print(summary)


def main() -> None:
    args = parse_args()
    labels = args.labels.split(",") if args.labels else None

    arms: list[dict] = []
    for i, d in enumerate(args.fig_dirs):
        arm = load_arm(d, args.epoch)
        if arm is None:
            continue
        arm["label"] = labels[i] if labels and i < len(labels) else d.name
        arms.append(arm)

    if len(arms) < 2:
        raise SystemExit("need at least two arms with tracked_arrays.npz")

    arms = align(arms)
    print(f"  comparing {len(arms)} arms over {arms[0]['profile'].shape[0]} reflections")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = grid_rows(arms, args.grid_per_regime)
    figure_profiles(arms, args.out_dir / "fig_family_profiles.png", args.dpi, rows)
    figure_radial(arms, args.out_dir / "fig_family_radial.png", args.dpi)
    figure_neff(arms, args.out_dir / "fig_family_neff.png", args.dpi)
    write_stats(arms, args.out_dir / "family_profile_stats.csv")


if __name__ == "__main__":
    main()
