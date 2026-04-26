"""Add columns from metadata.pt to prediction parquet files by joining on refl_ids.

Usage:
    python add_d_to_parquet.py \
        --predictions-dir /path/to/predictions/ \
        --data-dir /path/to/pytorch_data/ \
        --keys d intensity.prf.variance
"""

import argparse
from pathlib import Path

import polars as pl
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Path to predictions/ directory containing epoch_*/ subdirs",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to data directory containing metadata.pt",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        required=True,
        help="Metadata keys to add (e.g. d intensity.prf.variance)",
    )
    args = parser.parse_args()

    # Load metadata
    metadata = torch.load(args.data_dir / "metadata.pt", weights_only=False)

    # Build lookup table with refl_ids + requested keys
    lookup_data = {"refl_ids": metadata["refl_ids"].numpy()}
    for key in args.keys:
        if key not in metadata:
            print(f"WARNING: '{key}' not found in metadata.pt, skipping")
            print(f"  Available keys: {list(metadata.keys())}")
            continue
        lookup_data[key] = metadata[key].numpy()

    lookup = pl.DataFrame(lookup_data)
    keys_to_add = [k for k in args.keys if k in metadata]
    print(f"Adding columns: {keys_to_add}")

    # Find all parquet files
    parquets = sorted(args.predictions_dir.glob("**/preds_*.parquet"))
    print(f"Found {len(parquets)} parquet files")

    for pq in parquets:
        df = pl.read_parquet(pq)
        # Drop columns that already exist to avoid join conflicts
        new_keys = [k for k in keys_to_add if k not in df.columns]
        if not new_keys:
            continue
        join_cols = ["refl_ids"] + new_keys
        df = df.join(lookup.select(join_cols), on="refl_ids", how="left")
        df.write_parquet(pq)

    print("Done")


if __name__ == "__main__":
    main()
