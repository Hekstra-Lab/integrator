"""Full (non-sampled) scan over chunks: count pixels above each threshold,
across all valid voxels. Use to pick `max_count` properly.

Reports for each threshold:
  - total pixels above
  - fraction of valid voxels above
  - reflections with at least one pixel above (i.e. how many would have a
    voxel masked if max_count were set to that threshold)

Run:
    uv run python scripts/full_pixel_scan.py \
        --chunks /n/.../pytorch_data_dials/chunks
"""

import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", required=True)
    ap.add_argument(
        "--thresholds", nargs="+", type=int,
        default=[1024, 4096, 16384, 32768, 65535, 131072,
                 262144, 524288, 1048576],
        help="pixel-value thresholds to evaluate"
    )
    args = ap.parse_args()

    chunk_paths = sorted(Path(args.chunks).glob("chunk_*.npz"))
    if not chunk_paths:
        raise SystemExit(f"No chunk_*.npz under {args.chunks}")

    n_voxels_total = 0
    n_voxels_valid = 0
    n_voxels_fg = 0  # in DIALS Foreground
    n_voxels_bg = 0  # in DIALS Background
    n_refl_total = 0

    above_counts = {t: 0 for t in args.thresholds}        # voxels above
    above_fg = {t: 0 for t in args.thresholds}            # in foreground
    above_offfg = {t: 0 for t in args.thresholds}         # NOT in foreground
    refl_with_above = {t: 0 for t in args.thresholds}     # reflections affected

    print(f"Scanning {len(chunk_paths)} chunks (this is the FULL pass)...")
    for p in chunk_paths:
        with np.load(p) as npz:
            data = npz["data"]
            mask = npz["mask"].astype(bool)
            fg = npz["foreground"].astype(bool) if "foreground" in npz.files else None
            offsets = npz["offsets"]

        n_voxels_total += data.size
        n_voxels_valid += int(mask.sum())
        if fg is not None:
            n_voxels_fg += int((fg & mask).sum())
            n_voxels_bg += int((~fg & mask).sum())
        n_refl_total += len(offsets) - 1

        # For threshold checks, only count VALID voxels (DIALS mask)
        valid_data = data[mask]
        valid_fg = fg[mask] if fg is not None else None

        for t in args.thresholds:
            above_mask = valid_data > t
            n_above = int(above_mask.sum())
            above_counts[t] += n_above
            if valid_fg is not None:
                above_fg[t] += int((above_mask & valid_fg).sum())
                above_offfg[t] += int((above_mask & ~valid_fg).sum())

        # Per-reflection: any voxel above threshold? Use offsets + reduceat.
        # `data > t` → integer 0/1, sum within each reflection's slice.
        for t in args.thresholds:
            indicator = (data > t).astype(np.int32)
            per_refl = np.add.reduceat(indicator, offsets[:-1])
            refl_with_above[t] += int((per_refl > 0).sum())

        print(
            f"  {p.name}: n_voxels={data.size:,}, "
            f"n_valid={int(mask.sum()):,}"
        )

    print()
    print(f"Totals: {n_voxels_total:,} voxels, "
          f"{n_voxels_valid:,} valid ({100*n_voxels_valid/n_voxels_total:.2f}%)")
    if n_voxels_fg:
        print(f"        {n_voxels_fg:,} in DIALS Foreground, "
              f"{n_voxels_bg:,} in Background")
    print(f"Reflections: {n_refl_total:,}")
    print()

    print(f"{'threshold':>12s}  {'above (valid)':>16s}  {'frac valid':>12s}  "
          f"{'in_fg':>10s}  {'off_fg':>10s}  "
          f"{'refl_with_above':>16s}  {'refl frac':>10s}")
    for t in args.thresholds:
        a = above_counts[t]
        afg = above_fg[t]
        aoff = above_offfg[t]
        r = refl_with_above[t]
        print(
            f"  > {t:>10d}  "
            f"{a:>16,d}  "
            f"{100*a/max(n_voxels_valid,1):>11.5f}%  "
            f"{afg:>10,d}  "
            f"{aoff:>10,d}  "
            f"{r:>16,d}  "
            f"{100*r/max(n_refl_total,1):>9.3f}%"
        )


if __name__ == "__main__":
    main()
