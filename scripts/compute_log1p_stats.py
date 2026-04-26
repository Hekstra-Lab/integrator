"""Streaming log1p (mean, var) over valid voxels of an existing chunks dir.

Saves the result as `log1p_stats.pt` next to chunks/, so the ragged
dataset auto-loads it on the next training run instead of computing
on the fly.

Run:
    uv run python scripts/compute_log1p_stats.py \
        --chunks /n/.../pytorch_data_dials/chunks \
        --max-count 65535
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", required=True, help="path to chunks/ directory")
    ap.add_argument(
        "--max-count", type=float, default=None,
        help="optional clip threshold applied to raw counts before log1p; "
             "match what you'll set in the YAML (e.g. 65535)"
    )
    ap.add_argument(
        "--out", default=None,
        help="output path (default: <chunks>/../log1p_stats.pt)"
    )
    args = ap.parse_args()

    chunks_dir = Path(args.chunks)
    chunk_paths = sorted(chunks_dir.glob("chunk_*.npz"))
    if not chunk_paths:
        raise SystemExit(f"No chunk_*.npz under {chunks_dir}")

    print(f"Streaming {len(chunk_paths)} chunks...")
    sum_v = 0.0
    sumsq_v = 0.0
    n = 0
    for p in chunk_paths:
        with np.load(p) as npz:
            data = npz["data"].astype(np.float64)
            mask = npz["mask"].astype(bool)
        v = data[mask]
        if v.size == 0:
            continue
        np.clip(v, 0, args.max_count if args.max_count is not None else None, out=v)
        v = np.log1p(v)
        sum_v += float(v.sum())
        sumsq_v += float((v * v).sum())
        n += int(v.size)
        print(f"  {p.name}: n_valid={v.size:,}  running n={n:,}")

    if n == 0:
        raise SystemExit("No valid voxels — refusing to write empty stats.")

    mean = sum_v / n
    var = max(sumsq_v / n - mean * mean, 0.0)
    print(f"\nlog1p stats over {n:,} valid voxels:")
    print(f"  mean = {mean:.4f}")
    print(f"  var  = {var:.4f}")
    print(f"  std  = {np.sqrt(var):.4f}")
    print(f"  with max_count clip = {args.max_count}")

    out_path = Path(args.out) if args.out else chunks_dir.parent / "log1p_stats.pt"
    torch.save(torch.tensor([mean, var], dtype=torch.float32), out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
