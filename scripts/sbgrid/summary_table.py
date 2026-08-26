"""Assemble the DIALS-vs-integrator comparison table across datasets.

Built from the artifacts each pipeline already writes, so it reflects whatever
has finished rather than whatever was true when someone last typed it out.
Point it at dataset directories; arms that have not run yet are simply absent.

Three decisions are baked in, each of which cost us a wrong conclusion first:

`--dials-arm sum` by default. DIALS' `intensity_choice` defaults to `combine`,
which is profile-dominated, and profile fitting borrows a profile across
neighbouring reflections. That correlates their errors and inflates CC-half
where signal is low -- half of an apparent 0.070 deficit on 821 turned out to
be the benchmark rather than the integrator. Compare against summation.

I/sigma is never printed without CC-half beside it. On 821 the arm with the
highest I/sigma had the lowest CC-half: I/sigma measures the model's
confidence, not its accuracy.

sd/mean is computed on acentrics only, where the Wilson value is exactly 1.0.
Centrics are chi-squared with one degree of freedom, sd/mean = sqrt(2), so a
pooled set has no single ideal to compare against.

Usage:
    python scripts/sbgrid/summary_table.py --dataset <dir> [<dir> ...]
    python scripts/sbgrid/summary_table.py --dataset <dir> --format markdown
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import polars as pl

RFACTOR = re.compile(r"Final R-work\s*=\s*([\d.]+),\s*R-free\s*=\s*([\d.]+)")
# the summary table's row labels, mapped to the names used here
SUMMARY_ROWS = {
    "Mean I/σ(I)": "i_over_sigma",
    "CC½": "cc_half",
    "Rpim": "r_pim",
    "Completeness": "completeness",
    "Multiplicity": "multiplicity",
    "Unique reflections": "n_unique",
}


def parse_args():
    p = argparse.ArgumentParser(description="Cross-dataset comparison table")
    p.add_argument("--dataset", nargs="+", type=Path, required=True)
    p.add_argument(
        "--dials-arm",
        default="sum",
        choices=("sum", "profile", "combine"),
        help="which DIALS intensities to compare against; sum is the "
        "no-borrowing benchmark and the honest default",
    )
    p.add_argument(
        "--outer-criterion",
        type=float,
        default=1.0,
        help="the outer shell reported is the weakest one whose I/sigma on the "
        "DIALS reference reaches this; 0 disables and uses the last shell",
    )
    p.add_argument("--format", default="text", choices=("text", "markdown", "csv"))
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def summary(html: Path) -> dict | None:
    """The Overall / Low / High block DIALS writes above the per-shell table."""
    if not html.exists():
        return None
    import pandas as pd

    for table in pd.read_html(str(html)):
        labels = table.iloc[:, 0].astype(str)
        if not labels.str.contains("Resolution").any():
            continue
        out = {}
        for row, name in SUMMARY_ROWS.items():
            hit = table[labels == row]
            if hit.empty:
                continue
            for column, suffix in (("Overall", ""), ("High resolution", "_outer")):
                if column not in table.columns:
                    continue
                value = str(hit[column].iloc[0]).replace("%", "")
                try:
                    out[name + suffix] = float(value)
                except ValueError:
                    pass
        hit = table[labels == "Resolution (Å)"]
        if not hit.empty:
            out["resolution"] = str(hit["Overall"].iloc[0])
            out["outer_shell"] = str(hit["High resolution"].iloc[0])
        return out
    return None


def outer_shell(reference_csv: Path, criterion: float) -> tuple[float, float] | None:
    """The weakest shell still carrying signal, chosen on the reference arm.

    834 and 845 were integrated to 1.70 A but die near 1.85, so the last shell
    of the range is noise -- CC-half -0.012 there on 834. Reporting it as
    "outer" compares two arms on which fits noise better.

    The shell is chosen once, on the DIALS reference, and then applied to every
    arm. Letting each arm pick its own would let the metric under test select
    its own domain, which is exactly how a borrowing-confounded CC-half
    flatters whichever arm borrows more.
    """
    if criterion <= 0 or not reference_csv.exists():
        return None
    frame = pl.read_csv(reference_csv)
    keep = frame.filter(pl.col("i_over_sigma") >= criterion)
    if not len(keep):
        return None
    row = keep.row(-1, named=True)
    return float(row["d_max"]), float(row["d_min"])


def shell_stats(csv: Path, bounds: tuple[float, float] | None) -> dict:
    """Per-shell columns for the chosen outer shell, matched by d_min."""
    if bounds is None or not csv.exists():
        return {}
    frame = pl.read_csv(csv)
    hit = frame.filter((pl.col("d_min") - bounds[1]).abs() < 1e-6)
    if not len(hit):
        return {}
    row = hit.row(0, named=True)
    return {
        "cc_half_outer": row.get("cc_half"),
        "i_over_sigma_outer": row.get("i_over_sigma"),
        "r_pim_outer": row.get("r_pim"),
    }


def rfactors(refine_dir: Path) -> dict:
    log = refine_dir / "phenix_refine.log"
    if not log.exists():
        return {}
    found = RFACTOR.findall(log.read_text(errors="replace"))
    if not found:
        return {}
    work, free = found[-1]
    return {"r_work": float(work), "r_free": float(free)}


def peak_stats(refine_dir: Path) -> dict:
    path = refine_dir / "peaks.csv"
    if not path.exists():
        return {}
    frame = pl.read_csv(path)
    if not len(frame):
        return {"n_peaks": 0}
    return {"n_peaks": len(frame), "top_peak": float(frame["peakz"].max())}


def arm_row(name: str, merged_html: Path, refine_dir: Path,
            bounds: tuple[float, float] | None = None) -> dict | None:
    stats = summary(merged_html)
    if stats is None:
        return None
    row = {"arm": name, **stats}
    # replace the html's last-shell columns with the chosen outer shell
    row.update(shell_stats(merged_html.parent / "merging_stats.csv", bounds))
    row.update(rfactors(refine_dir))
    row.update(peak_stats(refine_dir))
    return row


def dials_arm(dataset: Path, choice: str) -> tuple[Path, Path]:
    """The reference merge for the requested intensity choice."""
    reference = dataset / "dials_reference"
    if choice == "combine":
        return reference / "merged.html", reference / "refine"
    directory = reference / f"choice_{choice}"
    # the alternative merges are diagnostics and carry no refinement of their
    # own; R-factors come from the refined combine merge either way
    return directory / "merged.html", reference / "refine"


def integrator_arms(dataset: Path) -> list[tuple[str, Path, Path]]:
    found = []
    for run in sorted((dataset / "integrator").glob("*/run_paths.yaml")):
        import yaml

        meta = yaml.safe_load(run.read_text())
        predictions = Path(meta.get("predictions_dir", ""))
        for scaled in sorted(predictions.glob("epoch_*/scaled")):
            refine = scaled / "refine_sharedfree"
            if not refine.exists():
                refine = scaled / "refine"
            found.append((run.parent.name, scaled / "merged.html", refine))
    return found


def main():
    args = parse_args()
    rows, missing, shells = [], [], {}
    for dataset in args.dataset:
        card = {}
        card_path = dataset / "dataset_card.json"
        if card_path.exists():
            card = json.loads(card_path.read_text())
        label = f"{dataset.name} ({card.get('pdb_id', '?')})"

        html, refine = dials_arm(dataset, args.dials_arm)
        bounds = outer_shell(html.parent / "merging_stats.csv", args.outer_criterion)
        if bounds:
            shells[label] = f"{bounds[0]:.2f}-{bounds[1]:.2f}"
        row = arm_row(f"DIALS ({args.dials_arm})", html, refine, bounds)
        if row:
            rows.append({"dataset": label, **row})
        else:
            # say so rather than dropping the dataset: an absent row reads as
            # "nothing to report" when it means "this merge was never made"
            missing.append(
                f"{label}: no DIALS '{args.dials_arm}' merge at {html.parent}"
                + ("  -- run cannon/07_intensity_choice.sh"
                   if args.dials_arm != "combine" else "")
            )
        arms = integrator_arms(dataset)
        if not arms:
            missing.append(f"{label}: no integrator arm has finished the pipeline")
        for name, merged, refined in arms:
            row = arm_row(name, merged, refined, bounds)
            if row:
                rows.append({"dataset": label, **row})
            else:
                missing.append(f"{label}/{name}: no merged.html at {merged.parent}")

    if not rows:
        raise SystemExit(
            "no finished arms found under the given datasets:\n  "
            + "\n  ".join(missing)
        )

    order = ["dataset", "arm", "resolution", "cc_half", "cc_half_outer",
             "i_over_sigma", "i_over_sigma_outer", "r_pim", "r_pim_outer",
             "completeness", "multiplicity", "r_work", "r_free",
             "n_peaks", "top_peak"]
    frame = pl.DataFrame(rows, infer_schema_length=None)
    frame = frame.select([c for c in order if c in frame.columns])

    if args.format == "csv":
        text = frame.write_csv()
    elif args.format == "markdown":
        header = "| " + " | ".join(frame.columns) + " |"
        rule = "|" + "|".join("---" for _ in frame.columns) + "|"
        body = [
            "| " + " | ".join("" if v is None else str(v) for v in r) + " |"
            for r in frame.iter_rows()
        ]
        text = "\n".join([header, rule, *body])
    else:
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
            text = str(frame)

    print(f"DIALS arm: {args.dials_arm}"
          + ("  (the no-borrowing benchmark)" if args.dials_arm == "sum" else
             "  -- profile borrowing inflates CC-half at low signal"))
    if args.dials_arm != "combine":
        # the alternative merges are diagnostics and are not refined, so the
        # DIALS row's merging columns and its refinement columns come from
        # different merges. Same intensities, different weighting of them.
        print(f"  note: DIALS merging columns are from the '{args.dials_arm}' "
              "merge; its r_work/r_free/peaks are from the refined 'combine' "
              "merge, which is the only DIALS arm that is refined")
    print(text)
    if shells:
        print("  outer shell per dataset (weakest with reference "
              f"I/sigma >= {args.outer_criterion:g}, applied to every arm):")
        for label, bounds in shells.items():
            print(f"    {label}: {bounds} A")
    for line in missing:
        print(f"  missing: {line}")
    if args.out:
        args.out.write_text(text)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
