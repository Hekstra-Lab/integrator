#!/usr/bin/env python3
"""
Compare Gamma, LogNormal, and FoldedNormal qi prediction outputs.

This script searches each run folder for prediction parquet files,
loads them, summarizes key columns, and writes a CSV comparison table.

Expected run folders:

  mfx_gamma_qi_25files
  mfx_lognormal_qi_25files_mc_kl
  mfx_foldednormal_qi_25files_mc_kl
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RUNS_ROOT = Path(
    "/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/"
    "mfx101555026_cctbx/runs"
)

RUNS = {
    "gamma": RUNS_ROOT / "mfx_gamma_qi_25files",
    "lognormal": RUNS_ROOT / "mfx_lognormal_qi_25files_mc_kl",
    "foldednormal": RUNS_ROOT / "mfx_foldednormal_qi_25files_mc_kl",
}

OUT_CSV = RUNS_ROOT / "compare_qi_25files_summary.csv"


def find_prediction_file(run_dir: Path) -> Path:
    """
    Find the main prediction parquet file under a run folder.

    integrator.predict may create files such as:
      pred.parquet
      test_preds_all.parquet

    We prefer test_preds_all.parquet if it exists, otherwise pred.parquet.
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder does not exist: {run_dir}")

    candidates = list(run_dir.rglob("test_preds_all.parquet"))
    if candidates:
        return sorted(candidates)[-1]

    candidates = list(run_dir.rglob("pred.parquet"))
    if candidates:
        return sorted(candidates)[-1]

    raise FileNotFoundError(f"No prediction parquet found under: {run_dir}")


def summarize_prediction(label: str, parquet_path: Path) -> dict:
    """
    Load one prediction parquet and summarize useful columns.
    """
    df = pd.read_parquet(parquet_path)

    summary = {
        "model": label,
        "parquet_path": str(parquet_path),
        "n_rows": len(df),
    }

    # Summarize common prediction columns if they exist.
    cols = [
        "qi_mean",
        "qi_var",
        "qbg_mean",
        "qbg_var",
        "intensity.sum.value",
        "intensity.sum.variance",
        "d",
    ]

    for col in cols:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            summary[f"{col}_mean"] = x.mean()
            summary[f"{col}_std"] = x.std()
            summary[f"{col}_min"] = x.min()
            summary[f"{col}_max"] = x.max()
            summary[f"{col}_nan_count"] = x.isna().sum()
        else:
            summary[f"{col}_missing"] = True

    # Test-set count if available.
    if "is_test" in df.columns:
        summary["n_test"] = int(df["is_test"].sum())
        summary["n_train_flagged"] = int((~df["is_test"].astype(bool)).sum())

    return summary


def main() -> None:
    rows = []

    for label, run_dir in RUNS.items():
        print(f"\n=== {label} ===")
        print(f"run_dir: {run_dir}")

        parquet_path = find_prediction_file(run_dir)
        print(f"prediction file: {parquet_path}")

        row = summarize_prediction(label, parquet_path)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_CSV, index=False)

    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    print(f"\nWrote summary CSV:\n{OUT_CSV}")


if __name__ == "__main__":
    main()