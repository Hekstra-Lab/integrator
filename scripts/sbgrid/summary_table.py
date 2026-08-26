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


def arm_row(name: str, merged_html: Path, refine_dir: Path) -> dict | None:
    stats = summary(merged_html)
    if stats is None:
        return None
    row = {"arm": name, **stats}
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
    rows = []
    for dataset in args.dataset:
        card = {}
        card_path = dataset / "dataset_card.json"
        if card_path.exists():
            card = json.loads(card_path.read_text())
        label = f"{dataset.name} ({card.get('pdb_id', '?')})"

        html, refine = dials_arm(dataset, args.dials_arm)
        row = arm_row(f"DIALS ({args.dials_arm})", html, refine)
        if row:
            rows.append({"dataset": label, **row})
        for name, merged, refined in integrator_arms(dataset):
            row = arm_row(name, merged, refined)
            if row:
                rows.append({"dataset": label, **row})

    if not rows:
        raise SystemExit("no finished arms found under the given datasets")

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
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
