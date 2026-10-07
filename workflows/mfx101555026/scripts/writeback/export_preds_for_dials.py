#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Integrator parquet predictions to NPZ for dials.python write-back."
    )
    parser.add_argument("--ckpt-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    single = args.ckpt_dir / "pred.parquet"
    if single.exists():
        pred_files = [single]
    else:
        pred_files = sorted(args.ckpt_dir.glob("preds_epoch_*.parquet"))

    if not pred_files:
        raise RuntimeError(f"No parquet files found in {args.ckpt_dir}")

    print(f"Found {len(pred_files)} parquet file(s).")

    needed_cols = ["refl_ids", "qi_mean", "qi_var", "qbg_mean"]

    dfs = []
    for i, p in enumerate(pred_files, start=1):
        if i <= 5 or i % 50 == 0:
            print(f"[{i}/{len(pred_files)}] reading {p.name}")
        dfs.append(pd.read_parquet(p, columns=needed_cols))

    df = pd.concat(dfs, ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.out,
        refl_ids=df["refl_ids"].to_numpy(dtype=np.int64),
        qi_mean=df["qi_mean"].to_numpy(dtype=np.float64),
        qi_var=df["qi_var"].to_numpy(dtype=np.float64),
        qbg_mean=df["qbg_mean"].to_numpy(dtype=np.float64),
    )

    print("Wrote:", args.out)
    print("Rows:", len(df))


if __name__ == "__main__":
    main()