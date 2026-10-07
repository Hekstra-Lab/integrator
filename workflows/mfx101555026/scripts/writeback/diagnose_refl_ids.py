#!/usr/bin/env python
"""
Diagnostic: check why prediction refl_ids do not match metadata refl_ids.

Run in integrator-cuda-dev (no DIALS needed):

  python diagnose_refl_ids.py

Paths are hard-coded for run_20260805-011728_9704 (asinh, epoch 24).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BASE = Path(
    "/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx"
)
RUN_DIR = BASE / "runs/run_20260805-011728_9704"
DATA_DIR = BASE / "mfx_shoebox_allruns_269_289_no275_024_rg101"

NPZ_PATH = RUN_DIR / "predictions/preds_epoch_0024.npz"
META_PATH = DATA_DIR / "metadata.npy"


def main():
    # ── Load predictions ──────────────────────────────────────────────────────
    print(f"Loading NPZ: {NPZ_PATH}")
    pred = np.load(NPZ_PATH, allow_pickle=False)
    pred_ids_raw = pred["refl_ids"]

    # ── Load metadata ─────────────────────────────────────────────────────────
    print(f"Loading metadata: {META_PATH}")
    meta = np.load(META_PATH, allow_pickle=True).item()
    meta_ids_raw = np.asarray(meta["refl_ids"])

    # ── Basic info ────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("BASIC INFO")
    print("=" * 60)
    print(f"pred refl_ids dtype  : {pred_ids_raw.dtype}")
    print(f"meta refl_ids dtype  : {meta_ids_raw.dtype}")
    print(f"pred count           : {len(pred_ids_raw):,}")
    print(f"meta count           : {len(meta_ids_raw):,}")
    print(
        f"pred range           : {pred_ids_raw.min()} .. {pred_ids_raw.max()}"
    )
    print(
        f"meta range           : {meta_ids_raw.min()} .. {meta_ids_raw.max()}"
    )

    # ── Float32 precision check ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("FLOAT32 PRECISION CHECK")
    print("=" * 60)
    FLOAT32_EXACT_MAX = 2**23  # 8,388,608
    pred_max = float(pred_ids_raw.max())
    if pred_max > FLOAT32_EXACT_MAX:
        print(
            f"WARNING: pred refl_ids max ({pred_max:.0f}) > float32 exact range "
            f"({FLOAT32_EXACT_MAX:,})"
        )
        print("         Large refl_ids stored as float32 will lose precision.")
    else:
        print(
            f"pred refl_ids max ({pred_max:.0f}) is within float32 exact range — no precision loss."
        )

    # ── Duplicate check ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("DUPLICATE CHECK")
    print("=" * 60)
    pred_int = pred_ids_raw.astype(np.int64)
    meta_int = meta_ids_raw.astype(np.int64)

    n_unique_pred = len(np.unique(pred_int))
    n_unique_meta = len(np.unique(meta_int))
    print(
        f"pred unique count    : {n_unique_pred:,}  (total: {len(pred_int):,})"
    )
    print(
        f"meta unique count    : {n_unique_meta:,}  (total: {len(meta_int):,})"
    )

    if n_unique_pred < len(pred_int):
        n_dups = len(pred_int) - n_unique_pred
        print(
            f"PROBLEM: {n_dups:,} duplicate pred refl_ids — caused by float32 precision loss"
        )
    else:
        print("No duplicates in pred refl_ids.")

    # ── Mismatch check ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("MISMATCH CHECK")
    print("=" * 60)
    meta_sorted = np.sort(meta_int)
    pred_sorted = np.sort(pred_int)

    found = np.searchsorted(meta_sorted, pred_sorted)
    found = np.clip(found, 0, len(meta_sorted) - 1)
    bad_mask = meta_sorted[found] != pred_sorted
    n_bad = int(bad_mask.sum())
    print(f"mismatching pred_ids : {n_bad:,} / {len(pred_sorted):,}")

    if n_bad > 0:
        bad = pred_sorted[bad_mask][:5]
        print(f"first bad pred_ids   : {bad.tolist()}")
        print()
        print("Nearest metadata values for first bad pred_ids:")
        for b in bad:
            i = int(np.searchsorted(meta_sorted, b))
            lo = max(0, i - 2)
            hi = min(len(meta_sorted), i + 3)
            print(
                f"  pred {b:>15,} → nearest meta: {meta_sorted[lo:hi].tolist()}"
            )
    else:
        print("All pred refl_ids found in metadata — no mismatch.")

    # ── Set difference ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SET DIFFERENCE (sample)")
    print("=" * 60)
    pred_set = set(pred_int[:10000].tolist())
    meta_set = set(meta_int[:10000].tolist())
    only_in_pred = sorted(pred_set - meta_set)[:5]
    only_in_meta = sorted(meta_set - pred_set)[:5]
    print(f"In pred but not meta (first 5): {only_in_pred}")
    print(f"In meta but not pred (first 5): {only_in_meta}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if n_unique_pred < len(pred_int):
        print("ROOT CAUSE: float32 precision loss in refl_ids")
        print("  BatchPredWriter stored refl_ids as float32.")
        print("  Large refl_ids (> 8,388,608) lost precision → duplicates.")
        print()
        print("FIX: re-export the NPZ with rounding to nearest integer,")
        print("     OR modify write_mfx_refl_with_predictions to match")
        print("     using source_refl + reflection_id directly from parquets")
        print("     instead of refl_ids → metadata lookup.")
    elif n_bad > 0:
        print(
            "ROOT CAUSE: pred refl_ids are different values from metadata refl_ids."
        )
        print("  This may mean the wrong metadata.npy is being used,")
        print(
            "  or refl_ids were assigned differently in predictions vs metadata."
        )
    else:
        print("No issue found — refl_ids match correctly.")
        print("The error may be intermittent or already resolved.")


if __name__ == "__main__":
    main()
