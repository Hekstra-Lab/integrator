#!/usr/bin/env python
"""
Production MFX write-back script (run with dials.python).

Reads prediction .npz produced by export_preds_for_dials.py in
integrator-cuda-dev, then writes qi_mean / qi_var / qbg_mean back into
the original MFX .refl files using the optimised write_mfx_refl_with_predictions().

Dependencies (cctbx/psana2 environment only):
  numpy, dials, dxtbx — NO polars, NO pandas, NO pyarrow.

================================================================================
Full two-environment workflow
================================================================================

Step 1 — integrator-cuda-dev  (GPU node)
-----------------------------------------
Run prediction:

  integrator.predict \\
      --run-dir  "$RUN_DIR" \\
      --ckpt     "$RUN_DIR/files/checkpoints/epoch=0024.ckpt" \\
      --batch-size 2048

This writes parquet files to:
  $RUN_DIR/files/predictions/epoch_0024/

Export parquets → .npz (no DIALS needed):

  python workflows/mfx101555026/scripts/writeback/export_preds_for_dials.py \\
      --ckpt-dir "$RUN_DIR/files/predictions/epoch_0024" \\
      --out      "$RUN_DIR/files/predictions/preds_epoch_0024.npz"

Step 2 — cctbx/psana2 + dials.python  (CPU node)
--------------------------------------------------
Activate cctbx environment and run write-back:

  source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh
  export PYTHONPATH=/sdf/home/t/thaoh/s3df_practice/integrator/src:$PYTHONPATH
  cd /sdf/home/t/thaoh/s3df_practice/integrator

  dials.python workflows/mfx101555026/scripts/writeback/run_writeback.py \\
      --npz      "$RUN_DIR/files/predictions/preds_epoch_0024.npz" \\
      --metadata "$DATA_DIR/metadata.npy" \\
      --refl-dir "$REFL_DIR" \\
      --out-dir  "$RUN_DIR/files/predictions/mfx_refl_writeback"

================================================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_text(v) -> str:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, bytes):
        return v.decode()
    return str(v)


def _fmt(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Load .npz predictions (numpy only — no polars/pandas)
# ---------------------------------------------------------------------------


def load_npz(npz_path: Path) -> dict:
    print(f"Loading predictions from {npz_path} ...")
    t0 = time.time()
    data = np.load(npz_path, allow_pickle=False)

    required = ["refl_ids", "qi_mean", "qi_var", "qbg_mean"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"Missing arrays in {npz_path}: {missing}\n"
            "Re-run export_preds_for_dials.py to regenerate the .npz."
        )

    pred_data = {
        "refl_ids": data["refl_ids"].astype(np.int64),
        "qi_mean": data["qi_mean"].astype(np.float64),
        "qi_var": data["qi_var"].astype(np.float64),
        "qbg_mean": data["qbg_mean"].astype(np.float64),
    }

    n = len(pred_data["refl_ids"])
    print(f"  {n:,} prediction rows loaded in {time.time() - t0:.1f}s")
    print(
        f"  qi_mean  : {pred_data['qi_mean'].min():.4f} .. {pred_data['qi_mean'].max():.4f}"
    )
    print(
        f"  qi_var   : {pred_data['qi_var'].min():.6f} .. {pred_data['qi_var'].max():.6f}"
    )
    print(
        f"  qbg_mean : {pred_data['qbg_mean'].min():.4f} .. {pred_data['qbg_mean'].max():.4f}"
    )
    return pred_data


# ---------------------------------------------------------------------------
# Run write-back
# ---------------------------------------------------------------------------


def run_writeback(
    pred_data: dict,
    metadata_path: Path,
    refl_dir: Path,
    out_dir: Path,
    variance_floor: float,
    copy_expt: bool,
) -> dict:
    from integrator.io.refl_io import write_mfx_refl_with_predictions

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting write-back ...")
    print(f"  metadata  : {metadata_path}")
    print(f"  refl-dir  : {refl_dir}")
    print(f"  out-dir   : {out_dir}")
    print(f"  variance_floor: {variance_floor}")

    t0 = time.time()
    result = write_mfx_refl_with_predictions(
        pred_data=pred_data,
        metadata_path=metadata_path,
        original_refl_dir=refl_dir,
        out_dir=out_dir,
        variance_floor=variance_floor,
        copy_expt=copy_expt,
    )
    elapsed = time.time() - t0

    print(f"\nWrite-back complete in {_fmt(elapsed)}")
    print(f"  refl files written : {result['n_refl_files']:,}")
    print(f"  prediction rows    : {result['n_prediction_rows']:,}")
    print(f"  output dir         : {result['out_dir']}")
    return result


# ---------------------------------------------------------------------------
# Quick spot-check on a sample of written .refl files
# ---------------------------------------------------------------------------


def spot_check(
    result: dict,
    refl_dir: Path,
    pred_data: dict,
    meta: dict,
    n_files: int = 3,
    n_rows: int = 5,
) -> None:
    from dials.array_family import flex
    from dxtbx import flumpy

    print(
        f"\nSpot-check: first {n_files} written files, {n_rows} rows each ..."
    )

    meta_refl_ids = np.asarray(meta["refl_ids"], dtype=np.int64)
    meta_rid = np.asarray(meta["reflection_id"], dtype=np.int64)
    meta_sr = np.asarray(meta["source_refl"])

    pred_ids = pred_data["refl_ids"]
    pred_mean = pred_data["qi_mean"].astype(np.float64)
    pred_var = np.clip(pred_data["qi_var"].astype(np.float64), 1e-6, None)

    meta_order = np.argsort(meta_refl_ids)
    meta_sorted = meta_refl_ids[meta_order]

    written = result["written_refl_files"][:n_files]
    all_ok = True

    for out_path in written:
        src_n = _as_text(out_path.name)
        rt = flex.reflection_table.from_file(str(out_path))
        i_sum = flumpy.to_numpy(rt["intensity.sum.value"])
        v_sum = flumpy.to_numpy(rt["intensity.sum.variance"])

        # Find predictions for this file
        file_mask = np.array(
            [_as_text(v) == src_n for v in meta_sr], dtype=bool
        )
        file_rows = np.flatnonzero(file_mask)

        found_idx = np.searchsorted(
            meta_sorted, meta_refl_ids[file_rows[:n_rows]]
        )
        file_ok = True
        for j, fi in enumerate(file_rows[:n_rows]):
            rid = meta_refl_ids[fi]
            row = int(meta_rid[fi])
            p = np.searchsorted(pred_ids, rid)
            if p >= len(pred_ids) or pred_ids[p] != rid:
                continue
            exp_i = float(pred_mean[p])
            exp_v = float(pred_var[p])
            got_i = float(i_sum[row])
            got_v = float(v_sum[row])
            if not (
                np.isclose(got_i, exp_i, rtol=1e-4, atol=1e-6)
                and np.isclose(got_v, exp_v, rtol=1e-4, atol=1e-6)
            ):
                print(
                    f"  MISMATCH {src_n} row {row}: "
                    f"qi_mean exp={exp_i:.4f} got={got_i:.4f}  "
                    f"qi_var exp={exp_v:.6f} got={got_v:.6f}"
                )
                file_ok = False

        print(
            f"  {'OK' if file_ok else 'MISMATCH'}  {src_n}  n_rows={len(rt)}"
        )
        if not file_ok:
            all_ok = False

    if all_ok:
        print("  Spot-check PASSED.")
    else:
        print("  Spot-check found MISMATCHES — review before using output.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--npz",
        required=True,
        type=Path,
        help="preds_epoch_XXXX.npz from export_preds_for_dials.py",
    )
    p.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="metadata.npy from the shoebox dataset (DATA_DIR)",
    )
    p.add_argument(
        "--refl-dir",
        required=True,
        type=Path,
        help="Original .refl/.expt directory (REFL_DIR)",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for written .refl files",
    )
    p.add_argument(
        "--variance-floor",
        type=float,
        default=1e-6,
        help="Minimum allowed variance (default: 1e-6)",
    )
    p.add_argument(
        "--copy-expt",
        action="store_true",
        help="Copy .expt files instead of symlinking them",
    )
    p.add_argument(
        "--skip-spot-check",
        action="store_true",
        help="Skip the post-write spot-check",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("MFX write-back  [dials.python]")
    print(f"  npz       : {args.npz}")
    print(f"  metadata  : {args.metadata}")
    print(f"  refl-dir  : {args.refl_dir}")
    print(f"  out-dir   : {args.out_dir}")
    print("=" * 65)

    t_start = time.time()

    # 1. Load predictions (.npz, numpy only)
    pred_data = load_npz(args.npz)

    # 2. Load metadata (numpy only)
    print(f"\nLoading metadata ...")
    t0 = time.time()
    meta = np.load(args.metadata, allow_pickle=True).item()
    print(
        f"  {len(np.asarray(meta['refl_ids'])):,} rows loaded in {time.time() - t0:.1f}s"
    )

    # 3. Write-back
    result = run_writeback(
        pred_data,
        args.metadata,
        args.refl_dir,
        args.out_dir,
        variance_floor=args.variance_floor,
        copy_expt=args.copy_expt,
    )

    # 4. Optional spot-check
    if not args.skip_spot_check:
        spot_check(result, args.refl_dir, pred_data, meta)

    total = time.time() - t_start
    print(f"\nTotal elapsed: {_fmt(total)}")
    print("Done.")


if __name__ == "__main__":
    main()
