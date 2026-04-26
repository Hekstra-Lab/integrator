"""For every voxel above a threshold (in valid + off-foreground pixels),
compute its absolute detector (x, y, z=frame) coordinate from the chunk
bbox and local position within the shoebox. Then group by (panel, x, y)
to identify single detector pixels that fire repeatedly — true hot
pixels — vs one-off bright detections.

Run:
    uv run python scripts/find_hot_pixels.py \
        --chunks /n/.../pytorch_data_dials/chunks \
        --threshold 131072 \
        --top 25
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--threshold", type=float, default=131072.0)
    ap.add_argument(
        "--require-off-foreground", action="store_true", default=True,
        help="Only consider pixels DIALS marked as NOT Foreground (default)."
    )
    ap.add_argument(
        "--include-foreground", dest="require_off_foreground",
        action="store_false",
        help="Include in-Foreground bright pixels too."
    )
    ap.add_argument("--top", type=int, default=20,
                    help="number of top-recurring detector positions to print")
    args = ap.parse_args()

    chunk_paths = sorted(Path(args.chunks).glob("chunk_*.npz"))
    if not chunk_paths:
        raise SystemExit(f"No chunk_*.npz under {args.chunks}")

    # Per-detector-pixel counter:  (panel, x_det, y_det) -> count
    # We deliberately drop z (frame) from the key so a single hot pixel
    # firing across many frames collapses to one detector address.
    pixel_hits = Counter()
    pixel_max_value = {}            # max raw count seen at that (px, py, panel)
    pixel_z_frames = {}             # set of frames where it fired
    pixel_in_refls = {}             # set of refl_ids it appears in
    total_above = 0
    total_off_fg = 0

    for ci, p in enumerate(chunk_paths):
        with np.load(p) as npz:
            data = npz["data"]
            offsets = npz["offsets"].astype(np.int64)
            shapes = npz["shapes"].astype(np.int64)
            bboxes = npz["bboxes"].astype(np.int64)
            mask = npz["mask"].astype(bool)
            fg = npz["foreground"].astype(bool) if "foreground" in npz.files else None
            refl_ids = npz["refl_ids"].astype(np.int64)
        n_refl = len(shapes)

        # Per-voxel "above threshold AND valid AND optionally off-foreground"
        above = (data > args.threshold) & mask
        if args.require_off_foreground and fg is not None:
            above = above & ~fg
        total_above += int((data > args.threshold).sum())
        if fg is not None:
            total_off_fg += int(((data > args.threshold) & ~fg).sum())

        if not above.any():
            continue

        # For each above-threshold voxel, compute the absolute detector
        # coordinate from the reflection's bbox + the local (d, h, w) within
        # its shoebox. Avoid Python loop over voxels — vectorize.
        voxel_idx = np.flatnonzero(above)                 # (N_above,)
        # Which reflection does each voxel belong to?
        # offsets[i] = start; reflection i covers offsets[i]:offsets[i+1].
        # searchsorted(offsets[1:], voxel_idx, side='right') gives index.
        refl_of = np.searchsorted(offsets[1:], voxel_idx, side="right")
        local_k = voxel_idx - offsets[refl_of]            # offset within shoebox

        # Decompose local_k into (d, h, w) using shapes[refl_of] = (D, H, W)
        D = shapes[refl_of, 0]
        H = shapes[refl_of, 1]
        W = shapes[refl_of, 2]
        # local_k = d * H * W + h * W + w
        d = local_k // (H * W)
        rem = local_k - d * H * W
        h = rem // W
        w = rem - h * W

        # Absolute detector coords from bbox: (x0, x1, y0, y1, z0, z1)
        x_abs = bboxes[refl_of, 0] + w
        y_abs = bboxes[refl_of, 2] + h
        z_abs = bboxes[refl_of, 4] + d
        raw_vals = data[voxel_idx]
        rids = refl_ids[refl_of]

        for x, y, z, v, r in zip(x_abs, y_abs, z_abs, raw_vals, rids):
            key = (int(x), int(y))   # detector pixel (single panel here)
            pixel_hits[key] += 1
            cur = pixel_max_value.get(key, 0)
            if int(v) > cur:
                pixel_max_value[key] = int(v)
            pixel_z_frames.setdefault(key, set()).add(int(z))
            pixel_in_refls.setdefault(key, set()).add(int(r))

        print(f"  chunk {ci}: {int(above.sum())} above-threshold voxels collected")

    print()
    print(f"Total pixels above {args.threshold:.0f}: {total_above:,}  "
          f"(off-foreground: {total_off_fg:,})")
    print(f"Unique detector pixels with at least one hit: {len(pixel_hits):,}")
    print()

    # How many distinct hits per detector pixel?
    hit_counts = np.array(list(pixel_hits.values()))
    if hit_counts.size:
        print(f"Hits per detector pixel:")
        print(f"  min/median/mean/max: "
              f"{hit_counts.min()} / {int(np.median(hit_counts))} / "
              f"{hit_counts.mean():.2f} / {hit_counts.max()}")
        # Histogram of recurrence
        for thresh in (1, 2, 5, 10, 50, 100):
            n = int((hit_counts >= thresh).sum())
            print(f"  detector pixels firing >= {thresh:>3d} times: {n:,}")

    # Top N most-recurring detector pixels
    print(f"\nTop {args.top} most-recurring detector pixels (likely hot pixels):")
    print(f"{'rank':>5s}  {'(x_det, y_det)':>18s}  {'hits':>6s}  "
          f"{'max_val':>10s}  {'unique frames':>14s}  {'unique refls':>14s}")
    for i, (key, cnt) in enumerate(pixel_hits.most_common(args.top), 1):
        x, y = key
        print(
            f"{i:>5d}  ({x:>5d}, {y:>5d})    {cnt:>6d}  "
            f"{pixel_max_value[key]:>10d}  "
            f"{len(pixel_z_frames[key]):>14d}  "
            f"{len(pixel_in_refls[key]):>14d}"
        )

    # If there's a clear hot pixel set (e.g. handful of detector positions
    # accounting for most hits), suggest making a static mask.
    if hit_counts.size:
        n_total = int(hit_counts.sum())
        top_n = int(min(50, hit_counts.size))
        coverage = float(np.sort(hit_counts)[-top_n:].sum()) / n_total
        print(
            f"\nTop {top_n} pixels cover {coverage*100:.1f}% of all above-"
            f"threshold hits."
        )
        if coverage > 0.5:
            print("  → likely a small set of hot pixels driving the artifact "
                  "tail; consider a static detector mask.")


if __name__ == "__main__":
    main()
