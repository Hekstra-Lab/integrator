"""Choose a shoebox window from the bounding boxes DIALS actually used.

The integrator needs one fixed window for every reflection, while DIALS sizes
a bounding box per reflection from the predicted spot extent. So the question
is what fixed window covers those boxes, and the answer is a distribution
rather than a number: a window large enough for the worst reflection wastes
most of its volume on background for the typical one.

The rule here is explicit. Take the smallest odd window whose per-axis extent
covers `--coverage` of reflections, then report what that choice clips and
what it costs in memory. Odd, because the window is centred on the predicted
position and an even one has no centre pixel.

Clipping is not free but it is also not fatal: a clipped reflection loses the
tail of its profile, which matters most for the strong reflections whose
tails carry real signal. The coverage default of 0.99 is a starting point to
be checked against the table this prints, not a law.

Usage:
    dials.python scripts/sbgrid/shoebox_size.py --data-dir <dataset dir>
    dials.python scripts/sbgrid/shoebox_size.py --refl a.refl b.refl --coverage 0.995
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Choose a shoebox window")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="dataset dir; finds every integrated.refl under dials_reference",
    )
    p.add_argument("--refl", nargs="*", type=Path, default=None)
    p.add_argument(
        "--coverage",
        type=float,
        default=0.99,
        help="fraction of reflections whose bbox must fit on each axis",
    )
    p.add_argument(
        "--max-window",
        type=int,
        default=61,
        help="refuse to recommend anything larger than this on x or y",
    )
    p.add_argument(
        "--update-card",
        action="store_true",
        help="write the recommendation into dataset_card.json",
    )
    return p.parse_args()


def find_refl(data_dir: Path) -> list[Path]:
    return sorted((data_dir / "dials_reference").glob("*/integrated.refl"))


def bbox_extents(paths: list[Path]) -> np.ndarray:
    """Per-axis bbox extents, over the reflections DIALS actually integrated.

    Unintegrated reflections carry a predicted box that was never filled, so
    including them would size the window for spots that contributed nothing.
    """
    from dials.array_family import flex

    rows = []
    for path in paths:
        table = flex.reflection_table.from_file(str(path))
        n_all = len(table)
        table = table.select(table.get_flags(table.flags.integrated_sum))
        bbox = np.asarray(table["bbox"]).reshape(-1, 6)
        rows.append(bbox)
        print(f"  {path.parent.name}: {len(table):,} of {n_all:,} integrated")
    stacked = np.vstack(rows)
    return np.column_stack(
        [
            stacked[:, 1] - stacked[:, 0],  # x
            stacked[:, 3] - stacked[:, 2],  # y
            stacked[:, 5] - stacked[:, 4],  # z
        ]
    )


def odd_at_least(value: float) -> int:
    """The smallest odd integer at or above `value`."""
    n = int(np.ceil(value))
    return n + 1 if n % 2 == 0 else n


def choose(extents: np.ndarray, coverage: float, max_window: int) -> dict:
    """The smallest odd window covering `coverage` of reflections per axis."""
    quantiles = np.quantile(extents, coverage, axis=0)
    window = [odd_at_least(q) for q in quantiles]
    window[0] = min(window[0], max_window)
    window[1] = min(window[1], max_window)

    w, h, d = window
    clipped = (
        (extents[:, 0] > w) | (extents[:, 1] > h) | (extents[:, 2] > d)
    ).mean()
    # how much of the volume DIALS used falls outside the window
    used = extents.prod(axis=1).astype(float)
    covered = (
        np.minimum(extents[:, 0], w)
        * np.minimum(extents[:, 1], h)
        * np.minimum(extents[:, 2], d)
    ).astype(float)
    return {
        "w": int(w),
        "h": int(h),
        "d": int(d),
        "coverage_requested": coverage,
        "clipped_fraction": float(clipped),
        "volume_loss_fraction": float(1 - covered.sum() / used.sum()),
        "n_reflections": int(len(extents)),
    }


def report(extents: np.ndarray, chosen: dict) -> None:
    print("\nbbox extents over the integrated reflections:")
    print(f"  {'axis':5s}{'p50':>7s}{'p90':>7s}{'p99':>7s}{'p99.9':>8s}{'max':>7s}")
    for i, axis in enumerate("xyz"):
        q = np.percentile(extents[:, i], [50, 90, 99, 99.9, 100])
        print(f"  {axis:5s}" + "".join(f"{v:7.0f}" for v in q))

    w, h, d = chosen["w"], chosen["h"], chosen["d"]
    print(f"\nrecommended window: {d} x {h} x {w}  (d x h x w)")
    # the per-axis coverage is applied independently, so the fraction clipped
    # on *any* axis is higher than 1 - coverage; the printed number is the
    # one that matters
    print(f"  clips {chosen['clipped_fraction'] * 100:.2f}% of reflections "
          f"(on any axis; per-axis target was "
          f"{(1 - chosen['coverage_requested']) * 100:.1f}%)")
    print(f"  loses {chosen['volume_loss_fraction'] * 100:.3f}% of the volume")
    pixels = w * h * d
    for dtype, size in (("uint16", 2), ("int32", 4)):
        gb = pixels * chosen["n_reflections"] * size / 1e9
        print(f"  {gb:6.2f} GB of counts as {dtype}")

    print("\n  alternatives:")
    print(f"    {'window':>14s}{'clipped %':>11s}{'volume loss %':>15s}")
    for scale in (-4, -2, 0, 2, 4):
        cw, ch = w + scale, h + scale
        if cw < 3 or ch < 3:
            continue
        clip = (
            (extents[:, 0] > cw)
            | (extents[:, 1] > ch)
            | (extents[:, 2] > d)
        ).mean()
        covered = (
            np.minimum(extents[:, 0], cw)
            * np.minimum(extents[:, 1], ch)
            * np.minimum(extents[:, 2], d)
        ).astype(float)
        loss = 1 - covered.sum() / extents.prod(axis=1).astype(float).sum()
        mark = "  <- chosen" if scale == 0 else ""
        print(
            f"    {d:>4d} x {ch:2d} x {cw:2d}{clip * 100:>11.2f}"
            f"{loss * 100:>15.3f}{mark}"
        )


def main():
    args = parse_args()
    paths = args.refl or (find_refl(args.data_dir) if args.data_dir else [])
    if not paths:
        raise SystemExit("no integrated.refl found; pass --refl or --data-dir")

    print(f"reading {len(paths)} reflection table(s):")
    extents = bbox_extents(paths)
    chosen = choose(extents, args.coverage, args.max_window)
    report(extents, chosen)

    if args.update_card and args.data_dir:
        card_path = args.data_dir / "dataset_card.json"
        card = json.loads(card_path.read_text())
        card["shoebox"] = chosen
        card_path.write_text(json.dumps(card, indent=2))
        print(f"\nwrote the recommendation into {card_path}")
        print(
            f"  integrator.make_shoeboxes --d {chosen['d']} "
            f"--h {chosen['h']} --w {chosen['w']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
