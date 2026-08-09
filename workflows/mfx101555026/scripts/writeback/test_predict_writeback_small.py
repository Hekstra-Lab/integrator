#!/usr/bin/env python
"""
End-to-end predict + write-back smoke test on first N source .refl files.

Steps:
  1. Load config and model from checkpoint.
  2. Load metadata — identify first N unique source .refl files.
  3. Iterate predict_dataloader; stop as soon as all target refl_ids
     have been seen (first N files are in first chunks — exits quickly).
  4. Write predictions to out_dir/epoch_XXXX/ as parquet.
  5. Run write-back for those N files via write_mfx_refl_with_predictions.
  6. Validate:
       Check 1  source refl row count == output refl row count
       Check 2  predicted rows written == expected
       Check 3  no duplicate (source_refl, reflection_id) in metadata subset
       Check 4  sample qi_mean / qi_var values match parquet predictions

Run with integrator-cuda-dev active (GPU optional — falls back to CPU):

  python test_predict_writeback_small.py \\
      --config   "$CONFIG" \\
      --ckpt     "$RUN_DIR/files/checkpoints/epoch=0024.ckpt" \\
      --metadata "$DATA_DIR/metadata.npy" \\
      --refl-dir "$REFL_DIR" \\
      --out-dir  "/tmp/test_smoke" \\
      --n-images 5

Exit 0 = all checks passed.  Exit 1 = at least one check failed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_text(v) -> str:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, bytes):
        return v.decode()
    return str(v)


def _load_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())


def _load_metadata(metadata_path: Path) -> dict:
    print(f"  Loading metadata from {metadata_path} ...")
    t0 = time.time()
    meta = np.load(metadata_path, allow_pickle=True).item()
    print(
        f"  Done in {time.time() - t0:.1f}s  rows={len(np.asarray(meta['refl_ids'])):,}"
    )
    return meta


# ---------------------------------------------------------------------------
# Step 1 — build model + load checkpoint
# ---------------------------------------------------------------------------


def load_model(config: dict, ckpt_path: Path, device: torch.device):
    from integrator.utils.factory_utils import construct_integrator

    print(f"\nBuilding model from config...")
    integrator = construct_integrator(config)
    print(f"  Loading checkpoint: {ckpt_path.name}")
    state = torch.load(
        ckpt_path.as_posix(), map_location="cpu", weights_only=False
    )
    integrator.load_state_dict(state["state_dict"])
    integrator.eval()
    integrator.to(device)
    print(f"  Model on {device}")
    return integrator


# ---------------------------------------------------------------------------
# Step 2 — find first N source .refl files and their target refl_ids
# ---------------------------------------------------------------------------


def select_target_files(meta: dict, n_images: int):
    source_refl_all = np.asarray(meta["source_refl"])
    refl_ids_all = np.asarray(meta["refl_ids"], dtype=np.int64)
    reflection_id_all = np.asarray(meta["reflection_id"], dtype=np.int64)
    source_expt_all = np.asarray(meta["source_expt"])

    # First N unique source_refl names (alphabetically sorted by np.unique)
    unique_sr, sr_encoded, sr_counts = np.unique(
        source_refl_all, return_inverse=True, return_counts=True
    )
    n_files = min(n_images, len(unique_sr))
    target_names = [_as_text(unique_sr[i]) for i in range(n_files)]
    target_counts = {
        target_names[i]: int(sr_counts[i]) for i in range(n_files)
    }

    # All metadata rows belonging to those N files
    sel_enc = np.arange(n_files)
    sel_mask = np.isin(sr_encoded, sel_enc)
    sel_rows = np.flatnonzero(sel_mask)

    target_refl_ids = refl_ids_all[sel_rows]
    target_source_refl = source_refl_all[sel_rows]
    target_source_expt = source_expt_all[sel_rows]
    target_reflection_id = reflection_id_all[sel_rows]

    print(f"\nSelected {n_files} source .refl files:")
    for name in target_names:
        print(f"  {name}  ({target_counts[name]} rows)")
    print(f"  Total target rows: {len(target_refl_ids):,}")

    return {
        "names": target_names,
        "counts": target_counts,
        "refl_ids": target_refl_ids,
        "source_refl": target_source_refl,
        "source_expt": target_source_expt,
        "reflection_id": target_reflection_id,
    }


# ---------------------------------------------------------------------------
# Step 3 — run inference, stop when all target refl_ids collected
# ---------------------------------------------------------------------------


def run_inference(
    integrator, config: dict, target_info: dict, device: torch.device
) -> dict:
    from integrator.utils.factory_utils import construct_data_loader

    target_ids_set = set(target_info["refl_ids"].tolist())
    total_target = len(target_ids_set)

    print(f"\nBuilding predict dataloader ...")
    data_module = construct_data_loader(config)
    data_module.setup()
    predict_dl = data_module.predict_dataloader()

    predictions: dict[int, dict] = {}  # refl_id -> {key: scalar_value}
    batches_scanned = 0
    t0 = time.time()

    print(f"  Scanning batches for {total_target:,} target rows ...")
    with torch.no_grad():
        for batch in predict_dl:
            counts_b, shoebox_b, mask_b, meta_b = batch
            batch_rids = meta_b["refl_ids"].numpy().astype(np.int64)

            # Check if this batch contains any target rows
            hits = np.isin(batch_rids, list(target_ids_set))
            batches_scanned += 1

            if hits.any():
                # Move batch to device
                counts_b = counts_b.to(device)
                shoebox_b = shoebox_b.to(device)
                mask_b = mask_b.to(device)
                meta_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in meta_b.items()
                }

                pred = integrator.predict_step(
                    (counts_b, shoebox_b, mask_b, meta_dev), 0
                )

                # Store predictions for target rows only
                hit_idx = np.flatnonzero(hits)
                for j in hit_idx:
                    rid = int(batch_rids[j])
                    if rid in target_ids_set:
                        row_pred = {}
                        for k, v in pred.items():
                            if isinstance(v, torch.Tensor):
                                row_pred[k] = v[j].detach().cpu().numpy()
                        predictions[rid] = row_pred
                        target_ids_set.discard(rid)

            if not target_ids_set:
                break

            if batches_scanned % 50 == 0:
                found = total_target - len(target_ids_set)
                elapsed = time.time() - t0
                print(
                    f"  ... {batches_scanned} batches, "
                    f"{found}/{total_target} target rows found "
                    f"({elapsed:.0f}s)"
                )

    elapsed = time.time() - t0
    found = total_target - len(target_ids_set)
    print(
        f"  Done: {batches_scanned} batches scanned, "
        f"{found}/{total_target} target rows found in {elapsed:.1f}s"
    )

    if target_ids_set:
        print(
            f"  WARNING: {len(target_ids_set)} target refl_ids not found in dataloader."
        )

    return predictions


# ---------------------------------------------------------------------------
# Step 4 — write predictions to parquet
# ---------------------------------------------------------------------------


def write_predictions(
    predictions: dict, target_info: dict, pred_dir: Path, epoch: int
) -> Path:
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Build aligned arrays in refl_id order
    rids = target_info["refl_ids"]
    rows = []
    missing = 0

    for rid in rids:
        if int(rid) in predictions:
            rows.append(predictions[int(rid)])
        else:
            missing += 1

    if missing:
        print(
            f"  WARNING: {missing} rows have no prediction (not in dataloader)"
        )

    # Get available rows
    found_rids = [int(r) for r in rids if int(r) in predictions]
    if not found_rids:
        raise RuntimeError("No predictions found for target refl_ids.")

    # Collect scalar prediction columns (skip large arrays like qp_mean)
    needed = ["refl_ids", "qi_mean", "qi_var", "qbg_mean"]
    available = {k for k in predictions[found_rids[0]].keys()}

    col_data: dict = {"refl_ids": np.array(found_rids, dtype=np.int64)}
    for key in ["qi_mean", "qi_var", "qbg_mean"]:
        if key in available:
            col_data[key] = np.array(
                [float(predictions[r][key]) for r in found_rids],
                dtype=np.float32,
            )
        else:
            print(f"  WARNING: {key} not in predictions — filling zeros")
            col_data[key] = np.zeros(len(found_rids), dtype=np.float32)

    col_data["epoch"] = np.full(len(found_rids), epoch, dtype=np.int32)

    df = pl.DataFrame(col_data)
    out_path = (
        pred_dir / f"preds_epoch_{epoch:04d}_rank=0_flush=000000.parquet"
    )
    df.write_parquet(out_path)
    print(f"  Wrote {len(df):,} prediction rows to {out_path.name}")
    return pred_dir


# ---------------------------------------------------------------------------
# Step 5 — run write-back
# ---------------------------------------------------------------------------


def run_writeback(
    pred_dir: Path, metadata_path: Path, refl_dir: Path, out_dir: Path
) -> dict:
    from integrator.io.pred_io import write_mfx_refl_from_preds

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRunning write-back ...")
    result = write_mfx_refl_from_preds(
        ckpt_dir=pred_dir,
        metadata_path=metadata_path,
        original_refl_dir=refl_dir,
        out_dir=out_dir,
        filetype="parquet",
        variance_floor=1e-6,
    )
    print(
        f"  Written: {result['n_refl_files']} .refl files, "
        f"{result['n_prediction_rows']:,} prediction rows"
    )
    return result


# ---------------------------------------------------------------------------
# Step 6 — validation checks
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


def check_predicted_rows(n_prediction_rows: int, pred_dir: Path) -> bool:
    print("\n[Check 2] Predicted rows written == parquet rows ...")
    df = pl.scan_parquet(
        list(pred_dir.glob("preds_epoch_*.parquet"))
    ).collect()
    parquet_rows = len(df)
    ok = parquet_rows == n_prediction_rows
    print(
        f"  {'PASS' if ok else 'FAIL'}  "
        f"parquet={parquet_rows:,}  written={n_prediction_rows:,}"
    )
    return ok


def check_no_duplicates(meta: dict, target_info: dict) -> bool:
    print(
        "\n[Check 3] No duplicate (source_refl, reflection_id) in subset ..."
    )
    sr = target_info["source_refl"]
    rid = target_info["reflection_id"].astype(np.int64)

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
    pred_dir: Path,
    out_dir: Path,
    target_info: dict,
    predictions: dict,
    n_sample: int = 10,
) -> bool:
    from dials.array_family import flex
    from dxtbx import flumpy

    print("\n[Check 4] Sample qi_mean / qi_var values match predictions ...")
    rng = np.random.default_rng(42)
    passed = True

    # Group target rows by source_refl
    sr = target_info["source_refl"]
    rids = target_info["refl_ids"].astype(np.int64)
    rflid = target_info["reflection_id"].astype(np.int64)

    unique_sr, enc = np.unique(sr, return_inverse=True)
    for i, sr_bytes in enumerate(unique_sr):
        sr_name = _as_text(sr_bytes)
        out_path = out_dir / sr_name
        if not out_path.exists():
            print(f"  SKIP  {sr_name} (not written)")
            continue

        group = np.flatnonzero(enc == i)
        # Filter to rows that have predictions
        group_with_pred = [j for j in group if int(rids[j]) in predictions]
        if not group_with_pred:
            print(f"  SKIP  {sr_name} (no predictions)")
            continue

        rt = flex.reflection_table.from_file(str(out_path))
        i_sum = flumpy.to_numpy(rt["intensity.sum.value"])
        v_sum = flumpy.to_numpy(rt["intensity.sum.variance"])

        sample = rng.choice(
            group_with_pred,
            size=min(n_sample, len(group_with_pred)),
            replace=False,
        )
        file_ok = True
        for j in sample:
            rid = int(rids[j])
            row = int(rflid[j])
            exp_i = float(predictions[rid].get("qi_mean", np.nan))
            exp_v = float(predictions[rid].get("qi_var", np.nan))
            # qi_var is clipped to 1e-6
            exp_v = max(exp_v, 1e-6)
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
        "--config",
        required=True,
        type=Path,
        help="Training YAML config (same one used for training)",
    )
    p.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help="Checkpoint file, e.g. epoch=0024.ckpt",
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
        help="Output directory for test predictions and write-back",
    )
    p.add_argument(
        "--n-images",
        type=int,
        default=5,
        help="Number of source .refl files to test (default: 5)",
    )
    p.add_argument(
        "--epoch",
        type=int,
        default=24,
        help="Epoch number label for parquet filenames (default: 24)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    pred_dir = args.out_dir / f"epoch_{args.epoch:04d}"
    wb_dir = args.out_dir / "mfx_refl_writeback"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("Predict + write-back smoke test")
    print(f"  config    : {args.config}")
    print(f"  ckpt      : {args.ckpt}")
    print(f"  metadata  : {args.metadata}")
    print(f"  refl-dir  : {args.refl_dir}")
    print(f"  out-dir   : {args.out_dir}")
    print(f"  n-images  : {args.n_images}")
    print(f"  device    : {device}")
    print("=" * 65)

    config = _load_config(args.config)

    # 1. Model
    model = load_model(config, args.ckpt, device)

    # 2. Target files
    meta = _load_metadata(args.metadata)
    target_info = select_target_files(meta, args.n_images)

    # 3. Inference
    predictions = run_inference(model, config, target_info, device)

    # 4. Write parquet
    write_predictions(predictions, target_info, pred_dir, args.epoch)

    # 5. Write-back
    result = run_writeback(pred_dir, args.metadata, args.refl_dir, wb_dir)

    # 6. Validate
    written = result["written_refl_files"]

    p1 = check_row_counts(args.refl_dir, wb_dir, written)
    p2 = check_predicted_rows(result["n_prediction_rows"], pred_dir)
    p3 = check_no_duplicates(meta, target_info)
    p4 = check_values(pred_dir, wb_dir, target_info, predictions)

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    checks = {
        "Check 1  source row count == output row count": p1,
        "Check 2  predicted rows == parquet rows": p2,
        "Check 3  no duplicate (source_refl, refl_id)": p3,
        "Check 4  qi_mean/qi_var spot-check": p4,
    }
    all_ok = True
    for label, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            all_ok = False

    print("=" * 65)
    if all_ok:
        print("ALL CHECKS PASSED — safe to run full predict + write-back job.")
        sys.exit(0)
    else:
        print("ONE OR MORE CHECKS FAILED — investigate before full run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
