#!/usr/bin/env python
"""
Step 2 of 2 — Write-back and validation smoke test (run with dials.python).

Reads the .npz produced by test_predict_small.py, runs write-back for the
5 test .refl files using the optimised write_mfx_refl_with_predictions(),
and validates 4 correctness checks.

Dependencies (all present in cctbx/psana2 environment):
  numpy, dials, dxtbx — NO polars, NO pandas, NO pyarrow needed.

Usage (cctbx/psana2 environment):

  source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh
  export PYTHONPATH=/sdf/home/t/thaoh/s3df_practice/integrator/src:$PYTHONPATH
  cd /sdf/home/t/thaoh/s3df_practice/integrator

  dials.python workflows/mfx101555026/scripts/writeback/test_writeback_small.py \\
      --npz      "/tmp/test_smoke/preds_small.npz" \\
      --metadata "$DATA_DIR/metadata.npy" \\
      --refl-dir "$REFL_DIR" \\
      --out-dir  "/tmp/test_smoke/writeback"

Validates:
  Check 1  source refl row count == output refl row count
  Check 2  predicted rows written == npz row count
  Check 3  no duplicate (source_refl, reflection_id) in metadata subset
  Check 4  sample qi_mean / qi_var values match npz predictions

Exit 0 = all passed.  Exit 1 = at least one check failed.
"""

from __future__ import annotations

import argparse
import sys
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


# ---------------------------------------------------------------------------
# Load .npz predictions (numpy only — no polars/pandas)
# ---------------------------------------------------------------------------


def load_npz(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    required = ["refl_ids", "qi_mean", "qi_var", "qbg_mean"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing arrays in {npz_path}: {missing}")
    pred_data = {
        "refl_ids": data["refl_ids"].astype(np.int64),
        "qi_mean": data["qi_mean"].astype(np.float64),
        "qi_var": data["qi_var"].astype(np.float64),
        "qbg_mean": data["qbg_mean"].astype(np.float64),
    }
    epoch = int(data["epoch"][0]) if "epoch" in data else -1
    print(
        f"Loaded {len(pred_data['refl_ids']):,} prediction rows from {npz_path.name}"
    )
    print(f"  epoch     : {epoch}")
    print(
        f"  qi_mean   : {pred_data['qi_mean'].min():.4f} .. {pred_data['qi_mean'].max():.4f}"
    )
    print(
        f"  qi_var    : {pred_data['qi_var'].min():.6f} .. {pred_data['qi_var'].max():.6f}"
    )
    return pred_data


# ---------------------------------------------------------------------------
# Run write-back
# ---------------------------------------------------------------------------


def run_writeback(
    pred_data: dict, metadata_path: Path, refl_dir: Path, out_dir: Path
) -> dict:
    # Import here so the module-level import of refl_io (now fixed) works fine
    # without dials being on the path at import time.
    from integrator.io.refl_io import write_mfx_refl_with_predictions

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRunning write-back ...")
    result = write_mfx_refl_with_predictions(
        pred_data=pred_data,
        metadata_path=metadata_path,
        original_refl_dir=refl_dir,
        out_dir=out_dir,
        variance_floor=1e-6,
        copy_expt=False,
    )
    print(
        f"  Written: {result['n_refl_files']} .refl files, "
        f"{result['n_prediction_rows']:,} prediction rows"
    )
    return result


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def check_row_counts(
    refl_dir: Path, out_dir: Path, written_files: list
) -> bool:
    from dials.array_family import flex

    print("\n[Check 1] Source row count == output row count ...")
    passed = True
    for out_path in written_files:
        src_path = refl_dir / out_path.name
        n_src = len(flex.reflection_table.from_file(str(src_path)))
        n_out = len(flex.reflection_table.from_file(str(out_path)))
        ok = n_src == n_out
        print(
            f"  {'PASS' if ok else 'FAIL'}  {out_path.name}  "
            f"src={n_src}  out={n_out}"
        )
        if not ok:
            passed = False
    return passed


def check_predicted_rows(n_prediction_rows: int, n_npz_rows: int) -> bool:
    print("\n[Check 2] Predicted rows written == npz rows ...")
    ok = n_prediction_rows == n_npz_rows
    print(
        f"  {'PASS' if ok else 'FAIL'}  "
        f"written={n_prediction_rows:,}  npz={n_npz_rows:,}"
    )
    return ok


def check_no_duplicates(meta: dict, pred_data: dict) -> bool:
    print(
        "\n[Check 3] No duplicate (source_refl, reflection_id) in subset ..."
    )

    meta_refl_ids = np.asarray(meta["refl_ids"], dtype=np.int64)
    meta_sr = np.asarray(meta["source_refl"])
    meta_rid = np.asarray(meta["reflection_id"], dtype=np.int64)

    # Find metadata rows for prediction refl_ids
    pred_ids = pred_data["refl_ids"]
    meta_order = np.argsort(meta_refl_ids)
    meta_sorted = meta_refl_ids[meta_order]
    found = np.searchsorted(meta_sorted, pred_ids)
    valid = (found < len(meta_sorted)) & (meta_sorted[found] == pred_ids)
    rows = meta_order[found[valid]]

    sr = meta_sr[rows]
    rid = meta_rid[rows].astype(np.int64)

    _, encoded = np.unique(sr, return_inverse=True)
    order = np.lexsort((rid, encoded))
    enc_s = encoded[order]
    rid_s = rid[order]
    dups = np.flatnonzero(
        (enc_s[1:] == enc_s[:-1]) & (rid_s[1:] == rid_s[:-1])
    )

    ok = len(dups) == 0
    if ok:
        print(f"  PASS  no duplicates in {len(sr):,} rows")
    else:
        i = dups[0]
        print(
            f"  FAIL  {len(dups):,} duplicates — first: "
            f"source_refl={_as_text(sr[order[i]])}  "
            f"reflection_id={rid_s[i]}"
        )
    return ok


def check_values(
    refl_dir: Path,
    out_dir: Path,
    pred_data: dict,
    meta: dict,
    n_sample: int = 10,
) -> bool:
    from dials.array_family import flex
    from dxtbx import flumpy

    print(
        "\n[Check 4] Sample qi_mean / qi_var values match npz predictions ..."
    )

    meta_refl_ids = np.asarray(meta["refl_ids"], dtype=np.int64)
    meta_sr = np.asarray(meta["source_refl"])
    meta_rid = np.asarray(meta["reflection_id"], dtype=np.int64)

    pred_ids = pred_data["refl_ids"]
    pred_imean = pred_data["qi_mean"].astype(np.float64)
    pred_ivar = np.clip(pred_data["qi_var"].astype(np.float64), 1e-6, None)

    # Match pred_ids to metadata rows
    meta_order = np.argsort(meta_refl_ids)
    meta_sorted = meta_refl_ids[meta_order]
    found = np.searchsorted(meta_sorted, pred_ids)
    valid = (found < len(meta_sorted)) & (meta_sorted[found] == pred_ids)
    rows = meta_order[found[valid]]

    sr = meta_sr[rows]
    rid = meta_rid[rows].astype(np.int64)
    qi_mean_matched = pred_imean[valid]
    qi_var_matched = pred_ivar[valid]

    rng = np.random.default_rng(42)
    passed = True

    unique_sr, enc = np.unique(sr, return_inverse=True)
    for i, sr_bytes in enumerate(unique_sr):
        sr_name = _as_text(sr_bytes)
        out_path = out_dir / sr_name
        if not out_path.exists():
            print(f"  SKIP  {sr_name}")
            continue

        group = np.flatnonzero(enc == i)
        sample = rng.choice(
            group, size=min(n_sample, len(group)), replace=False
        )

        rt = flex.reflection_table.from_file(str(out_path))
        i_sum = flumpy.to_numpy(rt["intensity.sum.value"])
        v_sum = flumpy.to_numpy(rt["intensity.sum.variance"])

        file_ok = True
        for j in sample:
            row = int(rid[j])
            exp_i = float(qi_mean_matched[j])
            exp_v = float(qi_var_matched[j])
            got_i = float(i_sum[row])
            got_v = float(v_sum[row])
            if not (
                np.isclose(got_i, exp_i, rtol=1e-4, atol=1e-6)
                and np.isclose(got_v, exp_v, rtol=1e-4, atol=1e-6)
            ):
                print(
                    f"    MISMATCH  {sr_name} row {row}: "
                    f"qi_mean exp={exp_i:.6f} got={got_i:.6f}  "
                    f"qi_var  exp={exp_v:.6f} got={got_v:.6f}"
                )
                file_ok = False

        print(f"  {'PASS' if file_ok else 'FAIL'}  {sr_name}")
        if not file_ok:
            passed = False

    return passed


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
        help="preds_small.npz from test_predict_small.py",
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
        help="Output directory for test write-back .refl files",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Write-back validation smoke test  [dials.python]")
    print(f"  npz      : {args.npz}")
    print(f"  metadata : {args.metadata}")
    print(f"  refl-dir : {args.refl_dir}")
    print(f"  out-dir  : {args.out_dir}")
    print("=" * 60)

    # Load .npz predictions (numpy only)
    pred_data = load_npz(args.npz)
    n_npz = len(pred_data["refl_ids"])

    # Load metadata (numpy only)
    print(f"\nLoading metadata ...")
    meta = np.load(args.metadata, allow_pickle=True).item()
    print(f"  {len(np.asarray(meta['refl_ids'])):,} metadata rows")

    # Run write-back
    result = run_writeback(
        pred_data, args.metadata, args.refl_dir, args.out_dir
    )
    written = result["written_refl_files"]

    # Validate
    p1 = check_row_counts(args.refl_dir, args.out_dir, written)
    p2 = check_predicted_rows(result["n_prediction_rows"], n_npz)
    p3 = check_no_duplicates(meta, pred_data)
    p4 = check_values(args.refl_dir, args.out_dir, pred_data, meta)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    checks = {
        "Check 1  source row count == output row count": p1,
        "Check 2  written rows == npz rows": p2,
        "Check 3  no duplicate (source_refl, refl_id)": p3,
        "Check 4  qi_mean / qi_var spot-check": p4,
    }
    all_ok = True
    for label, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED — safe to run full write-back.")
        sys.exit(0)
    else:
        print("ONE OR MORE CHECKS FAILED — investigate before full run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
