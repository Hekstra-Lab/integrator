"""Compare raw-pixel distributions across the fixed-pipeline counts.pt and
the ragged chunks_dir. Quantifies what each transform does to outliers.

Run:
    uv run python scripts/compare_pixel_distributions.py \
        --fixed-counts /Users/luis/from_harvard_rc/hewl_9b7c/counts.pt \
        --fixed-masks  /Users/luis/from_harvard_rc/hewl_9b7c/masks.pt \
        --ragged-chunks /n/.../pytorch_data_dials/chunks
"""

import argparse
from pathlib import Path
import numpy as np


def summarize(label, values):
    """Print key stats for a 1-D array of valid pixel values."""
    n = values.size
    print(f"\n=== {label}  (N valid voxels = {n:,}) ===")
    print(f"  min/median/mean/max: "
          f"{values.min():.1f} / {np.median(values):.1f} / {values.mean():.2f} / {values.max():.1f}")
    qs = np.percentile(values, [50, 90, 99, 99.9, 99.99])
    print(f"  p50/p90/p99/p99.9/p99.99: "
          f"{qs[0]:.1f} / {qs[1]:.1f} / {qs[2]:.1f} / {qs[3]:.1f} / {qs[4]:.1f}")
    for t in [256, 1024, 4096, 16384, 65535, 1_000_000]:
        n_above = int((values > t).sum())
        print(f"  > {t:>10d}: {n_above:>12,d}  ({100 * n_above / n:.5f}%)")

    # Transform comparison on the upper tail
    print("  transform of max value:")
    mx = float(values.max())
    print(f"    raw           : {mx:>12.1f}")
    print(f"    sqrt          : {np.sqrt(mx + 0.375):>12.2f}")
    print(f"    Anscombe(2√)  : {2 * np.sqrt(mx + 0.375):>12.2f}")
    print(f"    log1p         : {np.log1p(mx):>12.2f}")
    print(f"    log10(1+x)    : {np.log10(1 + mx):>12.4f}")


def load_fixed(counts_path, masks_path, sample_size=5_000_000):
    import torch
    counts = torch.load(counts_path, weights_only=True)
    masks = torch.load(masks_path, weights_only=True)
    if counts.ndim == 3 and counts.shape[-1] == 1:
        counts = counts.squeeze(-1)
        masks = masks.squeeze(-1)
    flat_c = counts.flatten().float().numpy()
    flat_m = masks.flatten().bool().numpy()
    valid = flat_c[flat_m]
    if valid.size > sample_size:
        rng = np.random.default_rng(0)
        idx = rng.choice(valid.size, size=sample_size, replace=False)
        valid = valid[idx]
    return valid


def load_ragged(chunks_dir, sample_size=5_000_000):
    chunks = sorted(Path(chunks_dir).glob("chunk_*.npz"))
    samples = []
    per_chunk = max(1, sample_size // max(1, len(chunks)))
    rng = np.random.default_rng(0)
    for p in chunks:
        with np.load(p) as npz:
            data = npz["data"].astype(np.float32)
            mask = npz["mask"].astype(bool)
        valid = data[mask]
        if valid.size > per_chunk:
            idx = rng.choice(valid.size, size=per_chunk, replace=False)
            valid = valid[idx]
        samples.append(valid)
    return np.concatenate(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-counts")
    ap.add_argument("--fixed-masks")
    ap.add_argument("--ragged-chunks")
    ap.add_argument("--sample-size", type=int, default=5_000_000)
    args = ap.parse_args()

    if args.fixed_counts and args.fixed_masks:
        v = load_fixed(args.fixed_counts, args.fixed_masks, args.sample_size)
        summarize("FIXED  (hewl_9b7c, working baseline)", v)

    if args.ragged_chunks:
        v = load_ragged(args.ragged_chunks, args.sample_size)
        summarize("RAGGED (dataset 140, training fails)", v)


if __name__ == "__main__":
    main()
