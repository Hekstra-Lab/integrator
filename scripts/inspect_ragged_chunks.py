"""Diagnostic scan over refltorch.mksbox-dials chunk outputs.

Walks every chunk in <chunks_dir>, computes per-reflection statistics, and
reports:
  - distribution of total voxels and valid voxels per reflection
  - foreground / background / overlapped fractions
  - any data-array problems (NaN / inf / negative for uint16 — shouldn't happen)
  - a histogram of "valid voxel count" bins so we can pick a sensible
    `min_valid_pixels` filter (if any)
  - a few sampled "bad" reflections (smallest valid-voxel counts) with full
    breakdowns

Run:
    uv run python scripts/inspect_ragged_chunks.py \
        --chunks /n/hekstra_lab/.../pytorch_data_dials/chunks
"""

import argparse
from pathlib import Path
import numpy as np


def per_refl_breakdown(npz_path: Path):
    """For one chunk, return per-reflection arrays:
        total_vox:  (n,) int   D*H*W
        valid_vox:  (n,) int   training mask True count
        fg_vox:     (n,) int
        bg_vox:     (n,) int
        ov_vox:     (n,) int
        max_data:   (n,) int   max raw count seen
        nan_count:  (n,) int   NaN count in data (should be 0)
    """
    with np.load(npz_path) as npz:
        data = npz["data"]                 # uint16 (or int32) flat
        offsets = npz["offsets"].astype(np.int64)
        shapes = npz["shapes"]
        # mask, foreground, background, overlapped — bool flat
        mask = npz["mask"] if "mask" in npz.files else None
        fg = npz["foreground"] if "foreground" in npz.files else None
        bg = npz["background"] if "background" in npz.files else None
        ov = npz["overlapped"] if "overlapped" in npz.files else None

    n = len(shapes)
    total_vox = (
        shapes[:, 0].astype(np.int64)
        * shapes[:, 1].astype(np.int64)
        * shapes[:, 2].astype(np.int64)
    )

    def _sum_per_refl(arr):
        if arr is None:
            return np.zeros(n, dtype=np.int64)
        return np.add.reduceat(arr.astype(np.int64), offsets[:-1])

    valid_vox = _sum_per_refl(mask)
    fg_vox = _sum_per_refl(fg)
    bg_vox = _sum_per_refl(bg)
    ov_vox = _sum_per_refl(ov)

    # Per-refl max + NaN count of raw data
    data_max = np.zeros(n, dtype=np.int64)
    nan_count = np.zeros(n, dtype=np.int64)
    for i in range(n):
        s, e = int(offsets[i]), int(offsets[i + 1])
        sl = data[s:e]
        data_max[i] = int(sl.max()) if sl.size else 0
        # uint16/int32 doesn't carry NaN, but be safe for float dtypes
        if np.issubdtype(sl.dtype, np.floating):
            nan_count[i] = int(np.isnan(sl).sum())

    return {
        "total_vox": total_vox,
        "valid_vox": valid_vox,
        "fg_vox": fg_vox,
        "bg_vox": bg_vox,
        "ov_vox": ov_vox,
        "data_max": data_max,
        "nan_count": nan_count,
        "shapes": shapes,
        "data_dtype": str(data.dtype),
    }


def histogram_buckets(values, edges):
    """Counts of values falling into [edges[i], edges[i+1])."""
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            counts[i] = int(((values >= edges[i]) & (values <= edges[i + 1])).sum())
        else:
            counts[i] = int(((values >= edges[i]) & (values < edges[i + 1])).sum())
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", required=True, help="Path to chunks/ directory")
    ap.add_argument("--n-bad", type=int, default=10, help="Sample worst-N reflections to print")
    args = ap.parse_args()

    chunks_dir = Path(args.chunks)
    chunk_paths = sorted(chunks_dir.glob("chunk_*.npz"))
    if not chunk_paths:
        raise SystemExit(f"No chunk_*.npz under {chunks_dir}")

    print(f"Scanning {len(chunk_paths)} chunks under {chunks_dir}")
    all_stats = {k: [] for k in (
        "total_vox", "valid_vox", "fg_vox", "bg_vox", "ov_vox", "data_max", "nan_count"
    )}
    chunk_indices = []
    local_indices = []

    for ci, p in enumerate(chunk_paths):
        bd = per_refl_breakdown(p)
        for k in all_stats:
            all_stats[k].append(bd[k])
        n = len(bd["shapes"])
        chunk_indices.append(np.full(n, ci, dtype=np.int64))
        local_indices.append(np.arange(n, dtype=np.int64))
        print(
            f"  chunk {ci:3d}  n={n:6d}  "
            f"data_dtype={bd['data_dtype']}  "
            f"valid: median={int(np.median(bd['valid_vox']))}, "
            f"min={int(bd['valid_vox'].min())}, "
            f"max={int(bd['valid_vox'].max())}"
        )

    s = {k: np.concatenate(all_stats[k]) for k in all_stats}
    chunk_idx = np.concatenate(chunk_indices)
    local_idx = np.concatenate(local_indices)
    n_total = len(s["total_vox"])
    print(f"\nTotal reflections across all chunks: {n_total:,}")

    # ---------- 1) total voxel distribution ----------
    print("\n=== total voxels per reflection (D*H*W) ===")
    qs = np.percentile(s["total_vox"], [0, 25, 50, 75, 90, 95, 99, 100])
    print(f"  min/q25/q50/q75/q90/q95/q99/max = "
          f"{int(qs[0])}/{int(qs[1])}/{int(qs[2])}/{int(qs[3])}/"
          f"{int(qs[4])}/{int(qs[5])}/{int(qs[6])}/{int(qs[7])}")

    # ---------- 2) valid voxel distribution ----------
    print("\n=== valid voxels per reflection (Valid & ~Overlapped) ===")
    qs = np.percentile(s["valid_vox"], [0, 25, 50, 75, 90, 95, 99, 100])
    print(f"  min/q25/q50/q75/q90/q95/q99/max = "
          f"{int(qs[0])}/{int(qs[1])}/{int(qs[2])}/{int(qs[3])}/"
          f"{int(qs[4])}/{int(qs[5])}/{int(qs[6])}/{int(qs[7])}")

    # ---------- 3) histogram by valid count, focused on the low end ----------
    print("\n=== refl count by 'min valid voxels' filter threshold ===")
    print(f"  {'cutoff':>6s}  {'kept':>10s}  {'dropped':>10s}  {'drop %':>7s}")
    for cutoff in [0, 1, 5, 10, 25, 50, 100, 200, 500]:
        kept = int((s["valid_vox"] >= cutoff).sum())
        dropped = n_total - kept
        print(
            f"  {cutoff:>6d}  {kept:>10,d}  {dropped:>10,d}  "
            f"{(dropped/n_total)*100:>6.2f}%"
        )

    # ---------- 4) overlap fraction ----------
    overlap_frac = np.where(
        s["total_vox"] > 0, s["ov_vox"] / s["total_vox"], 0.0
    )
    print("\n=== overlapped-voxel fraction per reflection ===")
    qs = np.percentile(overlap_frac, [50, 75, 90, 95, 99, 100])
    print(f"  q50/q75/q90/q95/q99/max = "
          f"{qs[0]:.3f}/{qs[1]:.3f}/{qs[2]:.3f}/{qs[3]:.3f}/{qs[4]:.3f}/{qs[5]:.3f}")
    fully_overlapped = int(((s["valid_vox"] == 0) & (s["ov_vox"] > 0)).sum())
    print(f"  fully overlapped (0 valid AND >0 overlapped): {fully_overlapped:,}")

    # ---------- 5) data sanity ----------
    print("\n=== raw counts data sanity ===")
    print(f"  total NaN voxels: {int(s['nan_count'].sum())} (expected 0 for integer dtypes)")
    print(f"  data_max per refl: median={int(np.median(s['data_max']))}, "
          f"max={int(s['data_max'].max())}")
    n_zero_max = int((s["data_max"] == 0).sum())
    print(f"  reflections with max=0 (entirely zero pixel data): {n_zero_max:,}")

    # ---------- 6) sample worst reflections ----------
    if args.n_bad > 0:
        print(f"\n=== {args.n_bad} reflections with fewest valid voxels ===")
        order = np.argsort(s["valid_vox"])[: args.n_bad]
        print(f"  {'chunk':>5s}{'local':>7s}{'total':>9s}{'valid':>7s}"
              f"{'fg':>6s}{'bg':>6s}{'ov':>6s}{'data_max':>10s}")
        for k in order:
            print(
                f"  {chunk_idx[k]:>5d}{local_idx[k]:>7d}"
                f"{int(s['total_vox'][k]):>9d}{int(s['valid_vox'][k]):>7d}"
                f"{int(s['fg_vox'][k]):>6d}{int(s['bg_vox'][k]):>6d}"
                f"{int(s['ov_vox'][k]):>6d}{int(s['data_max'][k]):>10d}"
            )

    # ---------- 7) recommended filter ----------
    print("\n=== recommended min_valid_pixels ===")
    for cutoff in (10, 50, 100):
        dropped = int((s["valid_vox"] < cutoff).sum())
        if dropped == 0:
            print(f"  {cutoff:>3d}: drops 0 reflections — safe default")
            return
        if dropped < 0.01 * n_total:
            print(
                f"  {cutoff:>3d}: drops {dropped:,} reflections "
                f"({dropped/n_total*100:.2f}%) — safe trim"
            )
            return
    print("  more than 1% of reflections fall below all the candidates above; "
          "inspect the histogram and pick a value tailored to this dataset.")


if __name__ == "__main__":
    main()
