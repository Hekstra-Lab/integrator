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


def rfactors(arm: Path) -> tuple[float, float] | None:
    """The last R-work/R-free phenix printed.

    The last, not the first: a refinement prints one pair per macro-cycle and
    only the final one describes the model that was written out.
    """
    log = arm / "refine" / "phenix_refine.log"
    if not log.exists():
        return None
    found = RFACTOR.findall(log.read_text(errors="replace"))
    if not found:
        return None
    work, free = found[-1]
    return float(work), float(free)


def peaks(arm: Path) -> pl.DataFrame | None:
    path = arm / "refine" / "peaks.csv"
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
    else:
        missing = [
            str(arm)
            for arm, frame in ((args.reference, a), (args.integrator, b))
            if frame is None
        ]
        print(f"no merging_stats.csv under {missing}")

    print("\nrefinement")
    print(f"  {'':>16s}{labels[0]:>12s}{labels[1]:>12s}")
    ra, rb = rfactors(args.reference), rfactors(args.integrator)
    if ra and rb:
        for i, name in enumerate(("R-work", "R-free")):
            print(f"  {name:>16s}{ra[i]:>12.4f}{rb[i]:>12.4f}")
    else:
        print(f"  {'':>16s}{str(ra):>12s}{str(rb):>12s}   (incomplete)")

    pa, pb = peaks(args.reference), peaks(args.integrator)
    if pa is not None and pb is not None:
        show_peaks(pa, pb, labels, args.out, args.peak_tol)
        print(f"\n  peaks above 5 sigma: {labels[0]} {len(pa)}, "
              f"{labels[1]} {len(pb)}")
    else:
        print("\nno peaks.csv in one of the arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
