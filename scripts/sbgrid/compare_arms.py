"""Put the DIALS reference and the integrator side by side.

Both arms end at the same three numbers -- merging statistics, R-factors, and
anomalous peak heights -- because they were run through the same scaling,
merging, refinement and peak search. Only the integration step differs, so a
difference in these numbers is a difference in integration.

Each arm is a directory holding `merging_stats.csv` and a `refine/`
subdirectory, which is the layout both `01_reference.sh` and
`05_integrator_pipeline.sh` produce.

Usage:
    python scripts/sbgrid/compare_arms.py \
        --reference <data>/dials_reference \
        --integrator <run>/predictions/epoch_0039/scaled
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

# the merging-stats columns worth showing side by side, and whether more is
# better; r_pim is the one where less is
STAT_COLUMNS = (
    ("cc_half", True),
    ("cc_anom", True),
    ("r_pim", False),
    ("i_over_sigma", True),
)
RFACTOR = re.compile(r"Final R-work\s*=\s*([\d.]+),\s*R-free\s*=\s*([\d.]+)")


def parse_args():
    p = argparse.ArgumentParser(description="Compare two processing arms")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--integrator", type=Path, required=True)
    p.add_argument("--labels", nargs=2, default=("DIALS", "integrator"))
    p.add_argument(
        "--refine-subdir",
        nargs=2,
        default=("refine", "refine"),
        help="refinement subdirectory within each arm; an arm refined more "
        "than once keeps each attempt in its own directory",
    )
    p.add_argument("--out", type=Path, default=None, help="write the peak table")
    p.add_argument(
        "--peak-tol",
        type=float,
        default=1.0,
        help="angstroms within which two arms' peaks are the same site",
    )
    return p.parse_args()


def merging_stats(arm: Path) -> pl.DataFrame | None:
    path = arm / "merging_stats.csv"
    return pl.read_csv(path) if path.exists() else None


def rfactors(arm: Path, subdir: str = "refine") -> tuple[float, float] | None:
    """The last R-work/R-free phenix printed.

    The last, not the first: a refinement prints one pair per macro-cycle and
    only the final one describes the model that was written out.
    """
    log = arm / subdir / "phenix_refine.log"
    if not log.exists():
        return None
    found = RFACTOR.findall(log.read_text(errors="replace"))
    if not found:
        return None
    work, free = found[-1]
    return float(work), float(free)


def peaks(arm: Path, subdir: str = "refine") -> pl.DataFrame | None:
    path = arm / subdir / "peaks.csv"
    if not path.exists():
        return None
    frame = pl.read_csv(path)
    return frame.select(
        pl.col("chain"),
        pl.col("seqid"),
        pl.col("residue"),
        pl.col("peakz"),
        pl.col("coordx"),
        pl.col("coordy"),
        pl.col("coordz"),
    )


def shrinkage(mtz: Path, edges) -> list[dict]:
    """Merged-intensity mean, sd and sd/mean per shell, acentrics only.

    For acentric reflections Wilson statistics make the intensity
    exponentially distributed, so sd/mean is exactly 1.0. Below that the
    posterior is compressing the spread of true intensities toward the prior
    mean, which is what costs CC-half: the half-dataset correlation is taken
    across reflections, so squeezing them together destroys it however well
    each individual one is measured.

    Centrics are excluded rather than pooled. Their intensities follow a
    chi-squared with one degree of freedom, giving sd/mean = sqrt(2), so a
    mixed set has no single ideal value to compare against -- roughly 7% of
    reflections here, enough to bias the ratio upward and flatter a posterior
    that is over-shrinking.
    """
    import reciprocalspaceship as rs

    ds = rs.read_mtz(str(mtz))
    ds["d"] = ds.compute_dHKL()["dHKL"]
    ds = ds.label_centrics()
    ds = ds[~ds["CENTRIC"]]
    column = "IMEAN" if "IMEAN" in ds.columns else "I"
    rows = []
    for hi, lo in edges:
        shell = ds[(ds["d"] <= hi) & (ds["d"] > lo)]
        values = shell[column].to_numpy().astype(float)
        if values.size < 20 or values.mean() == 0:
            continue
        rows.append(
            {
                "hi": hi,
                "lo": lo,
                "mean": float(values.mean()),
                "sd": float(values.std()),
                "ratio": float(values.std() / values.mean()),
                "n": int(values.size),
            }
        )
    return rows


def normalized(rows: list[dict], key: str) -> list[float]:
    """Each shell's value against that arm's own average across the shells.

    Comparing one arm's raw mean to another's would mostly measure the
    difference in their overall scale, which is arbitrary: two merges of the
    same data sit on whatever scale their scaling model chose. Dividing by
    the arm's own average leaves the falloff *shape*, which is what a
    resolution-dependent prior distorts, and lets the arms be compared
    without ever putting them on a common scale.
    """
    values = [r[key] for r in rows]
    average = sum(values) / len(values) if values else 1.0
    return [v / average if average else float("nan") for v in values]


def show_shrinkage(arm_a: Path, arm_b: Path, stats: pl.DataFrame, labels):
    """How far each arm sits from the Wilson ideal, and which way it fails.

    sd/mean alone says a shell is compressed but not why. Splitting it into
    the mean and the spread separates two failures that need different fixes:
    a mean pulled up toward the prior with the spread intact points at the
    prior's resolution dependence, while a spread collapsing onto a mean that
    is already right points at its strength.
    """
    edges = [(r["d_max"], r["d_min"]) for r in stats.iter_rows(named=True)]
    try:
        rows_a = shrinkage(arm_a / "merged.mtz", edges)
        rows_b = shrinkage(arm_b / "merged.mtz", edges)
    except (OSError, KeyError) as error:
        print(f"\nsd/mean unavailable: {error}")
        return
    if len(rows_a) != len(rows_b):
        print("\nsd/mean skipped: the arms binned to different shells")
        return

    mean_a, mean_b = normalized(rows_a, "mean"), normalized(rows_b, "mean")
    sd_a, sd_b = normalized(rows_a, "sd"), normalized(rows_b, "sd")

    print("\nsd/mean of merged I, acentrics only  (Wilson ideal = 1.0)")
    print("  mean r / sd r are normalized within each arm, so they are free of")
    print("  the arbitrary overall scale of either merge")
    print(f"    {'shell':>14s}{labels[0]:>10s}{labels[1]:>10s}"
          f"{'ratio':>8s}{'mean r':>8s}{'sd r':>7s}{'n':>7s}")
    # flagged on the ratio between the arms, not on the absolute value: both
    # arms fall below 1.0 at low resolution, where the deviation belongs to
    # the data rather than to either integrator, and an absolute threshold
    # reports that shared behaviour as if one arm were at fault
    for i, (ra, rb) in enumerate(zip(rows_a, rows_b, strict=True)):
        ratio = rb["ratio"] / ra["ratio"] if ra["ratio"] else float("nan")
        m = mean_b[i] / mean_a[i] if mean_a[i] else float("nan")
        d = sd_b[i] / sd_a[i] if sd_a[i] else float("nan")
        flag = ""
        if ratio < 0.9:
            flag = "  <- variance collapse" if d < m else "  <- mean inflation"
        shell = f"{ra['hi']:.2f}-{ra['lo']:.2f}"
        print(f"    {shell:>14s}{ra['ratio']:>10.3f}{rb['ratio']:>10.3f}"
              f"{ratio:>8.3f}{m:>8.3f}{d:>7.3f}{ra['n']:>7d}{flag}")


def show_shells(a: pl.DataFrame, b: pl.DataFrame, labels) -> None:
    print("\nmerging statistics by resolution shell")
    for column, higher_is_better in STAT_COLUMNS:
        if column not in a.columns or column not in b.columns:
            continue
        arrow = "higher better" if higher_is_better else "lower better"
        print(f"\n  {column}  ({arrow})")
        print(
            f"    {'shell':>14s}{labels[0]:>12s}{labels[1]:>12s}{'diff':>10s}"
        )
        for row_a, row_b in zip(a.iter_rows(named=True), b.iter_rows(named=True),
                                strict=False):
            shell = f"{row_a['d_max']:.2f}-{row_a['d_min']:.2f}"
            va, vb = row_a[column], row_b[column]
            if va is None or vb is None:
                continue
            delta = vb - va
            better = (delta > 0) == higher_is_better
            mark = "" if abs(delta) < 1e-9 else ("  +" if better else "  -")
            print(f"    {shell:>14s}{va:>12.3f}{vb:>12.3f}{delta:>10.3f}{mark}")


def match_peaks(a: pl.DataFrame, b: pl.DataFrame, tol: float):
    """Pair peaks by position, nearest first.

    Not by (chain, residue, name): `rs.find_peaks` leaves the atom name blank,
    so a ligand with several anomalous atoms -- 7LVC's NAP has seven above
    5 sigma -- collapses to one key, and joining on it multiplies the peaks
    together instead of pairing them. Position is what actually identifies a
    peak, and the two arms place the same peak within a fraction of an
    angstrom.

    Greedy nearest-first, so each peak is used once: the closest pair over
    the whole set is matched, both are removed, and the search repeats.
    """
    xyz = ["coordx", "coordy", "coordz"]
    pa, pb = a.select(xyz).to_numpy(), b.select(xyz).to_numpy()
    if len(pa) == 0 or len(pb) == 0:
        return [], list(range(len(pa))), list(range(len(pb)))

    distance = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    pairs = []
    free_a, free_b = set(range(len(pa))), set(range(len(pb)))
    while free_a and free_b:
        best = min(
            ((i, j) for i in free_a for j in free_b), key=lambda ij: distance[ij]
        )
        if distance[best] > tol:
            break
        pairs.append(best)
        free_a.discard(best[0])
        free_b.discard(best[1])
    return pairs, sorted(free_a), sorted(free_b)


def show_peaks(a: pl.DataFrame, b: pl.DataFrame, labels, out: Path | None,
               tol: float):
    pairs, only_a, only_b = match_peaks(a, b, tol)
    rows_a, rows_b = a.to_dicts(), b.to_dicts()

    def site(row):
        return f"{row['chain']}/{row['seqid']} {row['residue']}"

    print(f"\nanomalous peaks (sigma), matched within {tol:.1f} A")
    print(f"  {'site':>16s}{labels[0]:>12s}{labels[1]:>12s}{'diff':>10s}")
    records, wins = [], 0
    for i, j in sorted(pairs, key=lambda ij: -rows_a[ij[0]]["peakz"]):
        za, zb = rows_a[i]["peakz"], rows_b[j]["peakz"]
        wins += zb > za
        print(f"  {site(rows_a[i]):>16s}{za:>12.1f}{zb:>12.1f}{zb - za:>+10.1f}")
        records.append({"site": site(rows_a[i]), labels[0]: za, labels[1]: zb})
    for i in only_a:
        print(f"  {site(rows_a[i]):>16s}{rows_a[i]['peakz']:>12.1f}"
              f"{'--':>12s}{'':>10s}  only in {labels[0]}")
        records.append({"site": site(rows_a[i]), labels[0]: rows_a[i]["peakz"],
                        labels[1]: None})
    for j in only_b:
        print(f"  {site(rows_b[j]):>16s}{'--':>12s}"
              f"{rows_b[j]['peakz']:>12.1f}{'':>10s}  only in {labels[1]}")
        records.append({"site": site(rows_b[j]), labels[0]: None,
                        labels[1]: rows_b[j]["peakz"]})

    if pairs:
        deltas = [rows_b[j]["peakz"] - rows_a[i]["peakz"] for i, j in pairs]
        print(f"\n  {len(pairs)} peaks matched; {labels[1]} higher at {wins}")
        print(f"  mean difference {sum(deltas) / len(deltas):+.2f} sigma")
    if out:
        pl.DataFrame(records).write_csv(out)
        print(f"  wrote {out}")


def main():
    args = parse_args()
    labels = tuple(args.labels)

    a, b = merging_stats(args.reference), merging_stats(args.integrator)
    if a is not None and b is not None:
        show_shells(a, b, labels)
        show_shrinkage(args.reference, args.integrator, a, labels)
    else:
        missing = [
            str(arm)
            for arm, frame in ((args.reference, a), (args.integrator, b))
            if frame is None
        ]
        print(f"no merging_stats.csv under {missing}")

    print("\nrefinement")
    print(f"  {'':>16s}{labels[0]:>12s}{labels[1]:>12s}")
    sub_a, sub_b = args.refine_subdir
    ra = rfactors(args.reference, sub_a)
    rb = rfactors(args.integrator, sub_b)
    if ra and rb:
        for i, name in enumerate(("R-work", "R-free")):
            print(f"  {name:>16s}{ra[i]:>12.4f}{rb[i]:>12.4f}")
    else:
        print(f"  {'':>16s}{str(ra):>12s}{str(rb):>12s}   (incomplete)")

    pa = peaks(args.reference, sub_a)
    pb = peaks(args.integrator, sub_b)
    if pa is not None and pb is not None:
        show_peaks(pa, pb, labels, args.out, args.peak_tol)
        print(f"\n  peaks above 5 sigma: {labels[0]} {len(pa)}, "
              f"{labels[1]} {len(pb)}")
    else:
        print("\nno peaks.csv in one of the arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
