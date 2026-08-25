"""Write the integrator's predictions back into one `.refl` per sweep.

A dataset of several sweeps is trained as one table, but it has to leave as
several: each sweep carries its own crystal model, and DIALS scales each
against that model before merging them. Writing everything into one
reflection file would put reflections measured on one crystal into another
crystal's geometry.

`combine_sweeps.py` kept `refl_ids` per sweep and added `sweep_id` for
exactly this: the pair identifies which table a prediction belongs to and
which row within it.

Usage:
    python scripts/sbgrid/write_predictions.py \
        --pred-dir <predictions>/epoch_0039 --dataset-dir <combined dataset>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Split predictions back per sweep")
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="the combined dataset, which names the sweeps and holds their tables",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def load_predictions(pred_dir: Path) -> pl.DataFrame:
    """Read the prediction parquet whole.

    Not through `get_pred_files`, which selects the four columns a
    single-sweep writeback needs and would silently drop `sweep_id` -- the
    one column this script exists to use.
    """
    single = pred_dir / "pred.parquet"
    files = [single] if single.exists() else sorted(
        pred_dir.glob("preds_epoch_*.parquet")
    )
    if not files:
        raise SystemExit(f"no prediction parquet under {pred_dir}")
    frame = pl.concat([pl.read_parquet(f) for f in files])
    missing = {"refl_ids", "qi_mean", "qi_var", "qbg_mean"} - set(frame.columns)
    if missing:
        raise SystemExit(f"predictions lack {sorted(missing)}")
    return frame


def main():
    args = parse_args()
    out_dir = args.out_dir or args.pred_dir / "per_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = yaml.safe_load((args.dataset_dir / "dataset.yaml").read_text())
    sweeps = spec.get("sweeps")
    if not sweeps:
        raise SystemExit(
            f"{args.dataset_dir}/dataset.yaml lists no sweeps; this dataset "
            "was not built by combine_sweeps.py, so predictions can go back "
            "to its single .refl directly"
        )

    frame = load_predictions(args.pred_dir)
    if "sweep_id" not in frame.columns:
        raise SystemExit(
            "predictions carry no sweep_id; add it to the config's "
            "predict_keys so each prediction knows which table it came from"
        )
    print(f"{len(frame):,} predictions over {len(sweeps)} sweeps")

    from integrator.io.pred_io import write_refl_with_predictions

    written = []
    for i, sweep in enumerate(sweeps):
        subset = frame.filter(pl.col("sweep_id") == i).sort("refl_ids")
        source = args.dataset_dir / f"reflections_{sweep}.refl"
        if not source.exists():
            print(f"  {sweep}: no reflection table at {source.name}, skipping")
            continue
        target = out_dir / f"{sweep}.refl"
        write_refl_with_predictions(
            refl_file=source,
            out_file=target,
            refl_ids=subset["refl_ids"].to_numpy(),
            i_value=subset["qi_mean"].to_numpy(),
            i_variance=subset["qi_var"].to_numpy(),
            bg_mean=subset["qbg_mean"].to_numpy(),
        )
        written.append(target)
        print(f"  {sweep}: {len(subset):,} predictions -> {target.name}")

    if not written:
        raise SystemExit("no reflection tables written")
    print(f"\n{len(written)} table(s) in {out_dir}")
    print("  scale them together, as the reference does:")
    print("    dials.scale " + " ".join(f"{p.name}" for p in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
