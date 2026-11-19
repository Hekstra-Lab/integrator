"""Predicted intensities from several runs into one long table.

The comparison is model against model, so the reflections have to be
joined on their identity rather than on row order. `refl_ids` is that
identity and it is written into every prediction parquet; row order is
not, because two runs can predict in different orders and a positional
join would then compare one reflection with another and show the result
as scatter.

One row per (method, dataset, refl_id) with `qi_mean`, `qi_var` and `d`.
Small: four columns over a few hundred thousand reflections is the sort
of file `pull_figdata.sh` is for, where the parquets are gigabytes.

Usage:
    DATASET=hewl_1118 python tidy_intensities.py <out.csv> <run-dir>...
"""

from __future__ import annotations

import glob
import os
import sys

DATASET = os.environ.get("DATASET", "")
WANT = ("refl_ids", "qi_mean", "qi_var", "d")


def latest_parquet(run_dir: str) -> str | None:
    """The newest prediction parquet under a run directory.

    `integrator.predict` writes one per scored checkpoint, so the last one
    is the trained model rather than a mid-training snapshot.
    """
    found = sorted(
        glob.glob(f"{run_dir}/predictions/**/*.parquet", recursive=True)
        + glob.glob(f"{run_dir}/**/preds_epoch_*.parquet", recursive=True)
    )
    return found[-1] if found else None


def main() -> int:
    if not DATASET:
        raise SystemExit("set DATASET to the dataset these runs trained on")
    if len(sys.argv) < 3:
        raise SystemExit("usage: tidy_intensities.py <out.csv> <run-dir>...")

    import polars as pl

    out_path, runs = sys.argv[1], sys.argv[2:]
    frames = []
    for run in runs:
        arm = os.path.basename(run.rstrip("/"))
        path = latest_parquet(run)
        if path is None:
            print(f"  {arm}: no prediction parquet")
            continue
        df = pl.read_parquet(path)
        missing = [c for c in WANT if c not in df.columns]
        if missing:
            print(f"  {arm}: parquet lacks {missing}; has {df.columns[:8]}")
            continue
        sub = (df.select(WANT)
               .rename({"refl_ids": "refl_id"})
               .with_columns(pl.lit(arm).alias("method"),
                             pl.lit(DATASET).alias("dataset")))
        frames.append(sub)
        print(f"  {arm}: {sub.height:,} reflections from "
              f"{os.path.basename(path)}")
    if not frames:
        raise SystemExit("nothing to write")
    table = pl.concat(frames, how="vertical")
    table.write_csv(out_path)
    print(f"{out_path}: {table.height:,} rows, "
          f"{table['method'].n_unique()} arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
