#!/usr/bin/env python
"""
Step 1 of 2 — Prediction smoke test (run in integrator-cuda-dev).

Runs model inference on the first N source .refl files and saves predictions
to a .npz file that can be read by the cctbx/dials.python environment.

The .npz format is used as the interchange file because:
  - integrator-cuda-dev has polars/torch but NOT dials/dxtbx
  - cctbx/psana2 has dials/dxtbx but NOT polars/pandas/pyarrow
  - numpy (.npz) is available in BOTH environments

Usage (integrator-cuda-dev):

  python test_predict_small.py \\
      --config   "$CONFIG" \\
      --ckpt     "$RUN_DIR/files/checkpoints/epoch=0024.ckpt" \\
      --metadata "$DATA_DIR/metadata.npy" \\
      --out-dir  "/tmp/test_smoke" \\
      --n-images 5

Output:
  /tmp/test_smoke/preds_small.npz        <- read by test_writeback_small.py
  /tmp/test_smoke/preds_small_info.txt   <- human-readable summary

The .npz contains:
  refl_ids   int64   (N,)
  qi_mean    float32 (N,)
  qi_var     float32 (N,)
  qbg_mean   float32 (N,)

Activating integrator-cuda-dev:

  export MAMBA_ROOT_PREFIX=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/micromamba_root
  eval "$(/sdf/scratch/.../micromamba/bin/micromamba shell hook --shell bash)"
  micromamba activate integrator-cuda-dev
  export PYTHONNOUSERSITE=1
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
  export PYTHONPATH=/sdf/home/t/thaoh/s3df_practice/integrator/src
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
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


# ---------------------------------------------------------------------------
# Load config and model
# ---------------------------------------------------------------------------


def load_model(config: dict, ckpt_path: Path, device: torch.device):
    from integrator.utils.factory_utils import construct_integrator

    print(f"Building model ...")
    model = construct_integrator(config)
    print(f"Loading checkpoint: {ckpt_path.name}")
    state = torch.load(
        ckpt_path.as_posix(), map_location="cpu", weights_only=False
    )
    model.load_state_dict(state["state_dict"])
    model.eval()
    model.to(device)
    print(f"Model loaded on {device}")
    return model


# ---------------------------------------------------------------------------
# Select first N source .refl files from metadata
# ---------------------------------------------------------------------------


def select_target_files(meta: dict, n_images: int) -> dict:
    source_refl_all = np.asarray(meta["source_refl"])
    refl_ids_all = np.asarray(meta["refl_ids"], dtype=np.int64)

    unique_sr, sr_encoded, sr_counts = np.unique(
        source_refl_all, return_inverse=True, return_counts=True
    )
    n_files = min(n_images, len(unique_sr))

    target_names = [_as_text(unique_sr[i]) for i in range(n_files)]
    target_counts = {
        target_names[i]: int(sr_counts[i]) for i in range(n_files)
    }

    sel_mask = np.isin(sr_encoded, np.arange(n_files))
    sel_rows = np.flatnonzero(sel_mask)
    target_refl_ids = refl_ids_all[sel_rows]

    print(f"\nSelected {n_files} source .refl files:")
    for name in target_names:
        print(f"  {name}  ({target_counts[name]} rows)")
    print(f"  Total target rows: {len(target_refl_ids):,}")

    return {
        "names": target_names,
        "counts": target_counts,
        "refl_ids": target_refl_ids,
    }


# ---------------------------------------------------------------------------
# Run inference — stop when all target refl_ids are collected
# ---------------------------------------------------------------------------


def run_inference(
    model, config: dict, target_info: dict, device: torch.device
) -> dict[int, dict]:
    from integrator.utils.factory_utils import construct_data_loader

    target_set = set(target_info["refl_ids"].tolist())
    total_target = len(target_set)

    print(f"\nBuilding predict dataloader ...")
    dm = construct_data_loader(config)
    dm.setup()
    dl = dm.predict_dataloader()

    predictions: dict[int, dict] = {}
    batches = 0
    t0 = time.time()

    print(
        f"Scanning batches (stops as soon as {total_target:,} rows found) ..."
    )
    with torch.no_grad():
        for batch in dl:
            counts_b, shoebox_b, mask_b, meta_b = batch
            batch_rids = meta_b["refl_ids"].numpy().astype(np.int64)
            hits = np.isin(batch_rids, list(target_set))
            batches += 1

            if hits.any():
                counts_b = counts_b.to(device)
                shoebox_b = shoebox_b.to(device)
                mask_b = mask_b.to(device)
                meta_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in meta_b.items()
                }
                pred = model.predict_step(
                    (counts_b, shoebox_b, mask_b, meta_dev), 0
                )
                for j in np.flatnonzero(hits):
                    rid = int(batch_rids[j])
                    if rid in target_set:
                        predictions[rid] = {
                            k: float(v[j].detach().cpu())
                            for k, v in pred.items()
                            if isinstance(v, torch.Tensor) and v.dim() == 1
                        }
                        target_set.discard(rid)

            if not target_set:
                break

            if batches % 50 == 0:
                found = total_target - len(target_set)
                elapsed = time.time() - t0
                print(
                    f"  {batches} batches | {found}/{total_target} found | {elapsed:.0f}s"
                )

    elapsed = time.time() - t0
    found = total_target - len(target_set)
    print(
        f"Done: {batches} batches scanned, {found}/{total_target} rows found in {elapsed:.1f}s"
    )

    if target_set:
        print(
            f"WARNING: {len(target_set)} target rows not found in dataloader."
        )

    return predictions


# ---------------------------------------------------------------------------
# Save predictions as .npz (numpy interchange format)
# ---------------------------------------------------------------------------


def save_npz(
    predictions: dict, target_info: dict, out_dir: Path, epoch: int
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "preds_small.npz"

    rids = target_info["refl_ids"]
    found_rids = [int(r) for r in rids if int(r) in predictions]
    missing = len(rids) - len(found_rids)

    if missing:
        print(
            f"WARNING: {missing} rows have no prediction — not included in .npz"
        )
    if not found_rids:
        raise RuntimeError("No predictions found for any target refl_id.")

    def _col(key: str) -> np.ndarray:
        vals = [predictions[r].get(key, 0.0) for r in found_rids]
        return np.array(vals, dtype=np.float32)

    np.savez(
        npz_path,
        refl_ids=np.array(found_rids, dtype=np.int64),
        qi_mean=_col("qi_mean"),
        qi_var=_col("qi_var"),
        qbg_mean=_col("qbg_mean"),
        epoch=np.array([epoch], dtype=np.int32),
    )

    print(f"\nSaved {len(found_rids):,} prediction rows → {npz_path}")

    # Human-readable summary
    info_path = out_dir / "preds_small_info.txt"
    lines = [
        f"epoch: {epoch}",
        f"n_rows: {len(found_rids)}",
        f"source_refl_files: {', '.join(target_info['names'])}",
        f"qi_mean range: {_col('qi_mean').min():.4f} .. {_col('qi_mean').max():.4f}",
        f"qi_var  range: {_col('qi_var').min():.6f} .. {_col('qi_var').max():.6f}",
    ]
    info_path.write_text("\n".join(lines) + "\n")
    print(f"Summary       → {info_path}")
    for line in lines:
        print(f"  {line}")

    return npz_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--metadata", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-images", type=int, default=5)
    p.add_argument("--epoch", type=int, default=24)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Prediction smoke test  [integrator-cuda-dev]")
    print(f"  config    : {args.config}")
    print(f"  ckpt      : {args.ckpt}")
    print(f"  metadata  : {args.metadata}")
    print(f"  out-dir   : {args.out_dir}")
    print(f"  n-images  : {args.n_images}")
    print(f"  device    : {device}")
    print("=" * 60)

    config = yaml.safe_load(args.config.read_text())

    # 1. Load model
    model = load_model(config, args.ckpt, device)

    # 2. Load metadata and pick N files
    print(f"\nLoading metadata ...")
    t0 = time.time()
    meta = np.load(args.metadata, allow_pickle=True).item()
    print(
        f"  {len(np.asarray(meta['refl_ids'])):,} rows in {time.time() - t0:.1f}s"
    )

    target_info = select_target_files(meta, args.n_images)

    # 3. Run inference
    predictions = run_inference(model, config, target_info, device)

    # 4. Save .npz
    npz_path = save_npz(predictions, target_info, args.out_dir, args.epoch)

    print("\n" + "=" * 60)
    print("DONE — prediction complete.")
    print(f"Next step: run test_writeback_small.py with dials.python")
    print(f"  dials.python test_writeback_small.py \\")
    print(f"      --npz      {npz_path} \\")
    print(f"      --metadata {args.metadata} \\")
    print(f"      --refl-dir <REFL_DIR> \\")
    print(f"      --out-dir  {args.out_dir}/writeback")
    print("=" * 60)


if __name__ == "__main__":
    main()
