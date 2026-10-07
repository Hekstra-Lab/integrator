#!/usr/bin/env python
"""
Write Integrator predictions back into separate MFX DIALS .refl files.

This script is designed for the current MFX workflow where:

    1. Predictions were exported from parquet to NPZ first.
    2. Shoebox metadata contains:
         refl_ids
         image_num
         reflection_id
    3. Original cctbx/DIALS output contains many separate files:
         idx-data_01418_integrated.refl
         idx-data_01418_integrated.expt
         idx-data_02660_integrated.refl
         idx-data_02660_integrated.expt
         ...

Mapping used:

    prediction refl_ids
    -> metadata.npy row
    -> image_num + reflection_id
    -> idx-data_<image_num:05d>_integrated.refl
    -> row reflection_id inside that .refl file

This follows Luis's write_refl_with_predictions() idea, but supports many
separate MFX .refl/.expt files instead of one combined .refl with refl_ids.

Run with dials.python, not regular python:

    dials.python workflows/mfx101555026/scripts/writeback/write_mfx_predictions_to_refl.py ...
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write MFX Integrator qi_mean/qi_var predictions back to "
            "separate DIALS .refl files."
        )
    )

    parser.add_argument(
        "--pred-npz",
        required=True,
        type=Path,
        help=(
            "NPZ prediction file made from parquet. Expected arrays: "
            "refl_ids, qi_mean, qi_var, qbg_mean."
        ),
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="metadata.npy from the shoebox dataset",
    )
    parser.add_argument(
        "--original-refl-dir",
        required=True,
        type=Path,
        help=(
            "Original cctbx output folder containing "
            "idx-data_*_integrated.refl/.expt"
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="New output folder for predicted .refl/.expt files",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional small dry-run limit, e.g. --max-images 5",
    )
    parser.add_argument(
        "--copy-expt",
        action="store_true",
        help="Copy .expt files instead of symlinking them",
    )
    parser.add_argument(
        "--variance-floor",
        type=float,
        default=1e-6,
        help="Minimum allowed variance written to intensity variance columns",
    )

    return parser.parse_args()


def load_prediction_table(pred_npz: Path) -> pd.DataFrame:
    """
    Load prediction arrays from NPZ.

    Expected arrays:
        refl_ids, qi_mean, qi_var, qbg_mean
    """
    if not pred_npz.exists():
        raise FileNotFoundError(f"Missing prediction NPZ: {pred_npz}")

    data = np.load(pred_npz)

    required = ["refl_ids", "qi_mean", "qi_var", "qbg_mean"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{pred_npz} is missing required array(s): {missing}")

    pred_df = pd.DataFrame(
        {
            "refl_ids": data["refl_ids"].astype(np.int64),
            "qi_mean": data["qi_mean"].astype(np.float64),
            "qi_var": data["qi_var"].astype(np.float64),
            "qbg_mean": data["qbg_mean"].astype(np.float64),
        }
    )

    return pred_df


def add_metadata_columns(
    pred_df: pd.DataFrame,
    metadata_path: Path,
) -> pd.DataFrame:
    """
    Join predictions to metadata using refl_ids.

    In the current MFX dataset, refl_ids are row ids into metadata.npy.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    meta = np.load(metadata_path, allow_pickle=True).item()

    required = ["image_num", "reflection_id", "refl_ids"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise KeyError(f"metadata.npy is missing required key(s): {missing}")

    meta_df = pd.DataFrame(
        {
            "refl_ids": np.asarray(meta["refl_ids"], dtype=np.int64),
            "image_num": np.asarray(meta["image_num"], dtype=np.int64),
            "reflection_id": np.asarray(meta["reflection_id"], dtype=np.int64),
        }
    )

    joined = pred_df.merge(meta_df, on="refl_ids", how="inner")

    if len(joined) != len(pred_df):
        print(
            "WARNING: joined "
            f"{len(joined)} rows, but predictions had {len(pred_df)} rows"
        )

    duplicate_count = joined.duplicated(
        subset=["image_num", "reflection_id"]
    ).sum()
    if duplicate_count:
        raise ValueError(
            "Duplicate target rows found after join: "
            f"{duplicate_count} duplicate image_num/reflection_id pairs. "
            "This would overwrite the same reflection row more than once."
        )

    return joined


def link_or_copy_expt(src: Path, dst: Path, copy_expt: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if copy_expt:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def get_or_zeros(rt, column_name: str, n_refl: int) -> np.ndarray:
    """
    Return an existing DIALS reflection column as a NumPy float64 array,
    or a zero array if the column is missing.
    """
    from dxtbx import flumpy

    if column_name in rt:
        return flumpy.to_numpy(rt[column_name]).astype(np.float64)

    return np.zeros(n_refl, dtype=np.float64)


def write_one_image(
    image_num: int,
    rows: pd.DataFrame,
    original_refl_dir: Path,
    out_dir: Path,
    copy_expt: bool,
    variance_floor: float,
) -> tuple[int, int]:
    """
    Open one original .refl, write predictions into selected rows, save new .refl.

    Returns:
        (image_num, number_of_written_rows)
    """
    from dials.array_family import flex
    from dxtbx import flumpy

    refl_name = f"idx-data_{image_num:05d}_integrated.refl"
    expt_name = f"idx-data_{image_num:05d}_integrated.expt"

    src_refl = original_refl_dir / refl_name
    src_expt = original_refl_dir / expt_name

    if not src_refl.exists():
        raise FileNotFoundError(f"Missing source .refl: {src_refl}")
    if not src_expt.exists():
        raise FileNotFoundError(f"Missing source .expt: {src_expt}")

    dst_refl = out_dir / refl_name
    dst_expt = out_dir / expt_name

    rt = flex.reflection_table.from_file(str(src_refl))
    n_refl = len(rt)

    reflection_ids = rows["reflection_id"].to_numpy(dtype=np.int64)
    qi_mean = rows["qi_mean"].to_numpy(dtype=np.float64)
    qi_var = rows["qi_var"].to_numpy(dtype=np.float64)
    qbg_mean = rows["qbg_mean"].to_numpy(dtype=np.float64)

    if np.any(reflection_ids < 0) or np.any(reflection_ids >= n_refl):
        bad = reflection_ids[(reflection_ids < 0) | (reflection_ids >= n_refl)]
        raise IndexError(
            f"{refl_name}: reflection_id out of range. "
            f"n_refl={n_refl}, first bad={bad[:10]}"
        )

    qi_var = np.maximum(qi_var, variance_floor)

    intensity_sum_value = get_or_zeros(rt, "intensity.sum.value", n_refl)
    intensity_sum_variance = get_or_zeros(
        rt, "intensity.sum.variance", n_refl
    )
    intensity_prf_value = get_or_zeros(rt, "intensity.prf.value", n_refl)
    intensity_prf_variance = get_or_zeros(
        rt, "intensity.prf.variance", n_refl
    )
    background_mean = get_or_zeros(rt, "background.mean", n_refl)

    intensity_sum_value[reflection_ids] = qi_mean
    intensity_sum_variance[reflection_ids] = qi_var
    intensity_prf_value[reflection_ids] = qi_mean
    intensity_prf_variance[reflection_ids] = qi_var
    background_mean[reflection_ids] = qbg_mean

    rt["intensity.sum.value"] = flumpy.from_numpy(intensity_sum_value)
    rt["intensity.sum.variance"] = flumpy.from_numpy(intensity_sum_variance)
    rt["intensity.prf.value"] = flumpy.from_numpy(intensity_prf_value)
    rt["intensity.prf.variance"] = flumpy.from_numpy(intensity_prf_variance)
    rt["background.mean"] = flumpy.from_numpy(background_mean)

    out_dir.mkdir(parents=True, exist_ok=True)
    rt.as_file(str(dst_refl))
    link_or_copy_expt(src_expt, dst_expt, copy_expt=copy_expt)

    return image_num, len(reflection_ids)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions from NPZ...")
    pred_df = load_prediction_table(args.pred_npz)
    print(f"Prediction rows: {len(pred_df)}")

    print("Joining predictions to metadata...")
    df = add_metadata_columns(pred_df, args.metadata)
    print(f"Joined rows: {len(df)}")

    image_nums = sorted(df["image_num"].unique().tolist())
    if args.max_images is not None:
        image_nums = image_nums[: args.max_images]

    print(f"Images to write: {len(image_nums)}")
    print(f"Output folder: {args.out_dir}")

    total_rows = 0

    for i, image_num in enumerate(image_nums, start=1):
        rows = df[df["image_num"] == image_num]

        written_image_num, n_rows = write_one_image(
            image_num=int(image_num),
            rows=rows,
            original_refl_dir=args.original_refl_dir,
            out_dir=args.out_dir,
            copy_expt=args.copy_expt,
            variance_floor=args.variance_floor,
        )

        total_rows += n_rows

        if i <= 10 or i % 100 == 0:
            print(
                f"[{i}/{len(image_nums)}] wrote image_num={written_image_num} "
                f"rows={n_rows}"
            )

    print("Done.")
    print(f"Images written: {len(image_nums)}")
    print(f"Prediction rows written: {total_rows}")
    print(f"Output folder: {args.out_dir}")


if __name__ == "__main__":
    main()