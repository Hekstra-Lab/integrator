"""Post-process refltorch masks to mark pixels that are "owned" by a neighbor
reflection rather than the target reflection.

Operates on existing mksbox outputs (counts/masks already extracted). Pixel
data is not modified; only the mask is updated. The intent is a cheap fix for
the overlap-contamination issue — stage 1 of the variable-size shoebox plan.

How ownership works: when two reflections' bboxes share a pixel, the pixel is
"owned" by whichever reflection's predicted centroid is closer. This mirrors
dials/algorithms/shoebox/mask_overlapping.h:108-166.

Run (with DIALS activated):
    dials.python add_overlap_mask.py \
        --refl  /path/to/reflections_.refl \
        --masks /path/to/masks.npy \
        --out   /path/to/masks_overlap.npy \
        --h 33 --w 33 --d 9
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


def _load_masks(path):
    """Load masks from .npy (memmap) or .pt (torch). Returns (numpy_array, fmt)."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p), "npy"
    if p.suffix == ".pt":
        import torch

        return torch.load(p).numpy(), "pt"
    raise ValueError(f"unknown mask extension: {p.suffix}")


def _save_masks(masks_np, path, fmt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(path, masks_np)
    elif fmt == "pt":
        import torch

        torch.save(torch.from_numpy(masks_np), path)
    else:
        raise ValueError(f"unknown format: {fmt}")


def compute_overlap_mask(bboxes, centroids, D, H, W):
    """For each reflection i, return a (D, H, W) bool array that is True
    at pixels within its fixed window that are owned by another reflection.

    bboxes:   (N, 6)  ints, (x0, x1, y0, y1, z0, z1) per reflection
    centroids: (N, 3) floats, (x, y, z) predicted centroid (xyzcal.px)
    """
    N = len(bboxes)
    # max centroid-to-centroid distance at which two bboxes of size (W, H, D)
    # can possibly share a pixel (conservative, overestimates)
    max_r = float(np.sqrt(W * W + H * H + D * D))

    tree = cKDTree(centroids)
    overlap = np.zeros((N, D, H, W), dtype=bool)

    for i in tqdm(range(N), desc="overlap detection"):
        x0, x1, y0, y1, z0, z1 = bboxes[i]
        my_c = centroids[i]

        neighbor_ids = tree.query_ball_point(my_c, r=max_r)
        for nid in neighbor_ids:
            if nid == i:
                continue
            nb = bboxes[nid]
            nb_c = centroids[nid]

            # Intersection in absolute detector coords
            ix0 = max(x0, nb[0])
            ix1 = min(x1, nb[1])
            iy0 = max(y0, nb[2])
            iy1 = min(y1, nb[3])
            iz0 = max(z0, nb[4])
            iz1 = min(z1, nb[5])

            if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
                continue

            # Ownership: neighbor wins pixels closer to its centroid
            zs = np.arange(iz0, iz1)
            ys = np.arange(iy0, iy1)
            xs = np.arange(ix0, ix1)
            zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")

            d_me = (xx - my_c[0]) ** 2 + (yy - my_c[1]) ** 2 + (zz - my_c[2]) ** 2
            d_nb = (xx - nb_c[0]) ** 2 + (yy - nb_c[1]) ** 2 + (zz - nb_c[2]) ** 2
            nb_owned = d_nb < d_me

            # Write into my local (D, H, W) coords
            overlap[
                i,
                iz0 - z0 : iz1 - z0,
                iy0 - y0 : iy1 - y0,
                ix0 - x0 : ix1 - x0,
            ] |= nb_owned

    return overlap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refl", required=True, help="Path to reflections_.refl (has bbox + xyzcal.px)")
    ap.add_argument("--masks", required=True, help="Path to existing masks.npy or masks.pt")
    ap.add_argument("--out", default=None, help="Output path (default: <masks>_overlap.<ext>)")
    ap.add_argument("--h", type=int, required=True, help="Shoebox height")
    ap.add_argument("--w", type=int, required=True, help="Shoebox width")
    ap.add_argument("--d", type=int, required=True, help="Shoebox depth (frames)")
    args = ap.parse_args()

    from dials.array_family import flex

    refl = flex.reflection_table.from_file(args.refl)
    bboxes = np.asarray(refl["bbox"]).reshape(-1, 6).astype(np.int64)
    xyzcal = np.asarray([list(v) for v in refl["xyzcal.px"]], dtype=np.float64)

    N = len(bboxes)
    D, H, W = args.d, args.h, args.w
    V = D * H * W

    print(f"Reflections: {N:,}")
    print(f"Shoebox (D, H, W) = ({D}, {H}, {W}), V = {V:,}")

    # Sanity-check bbox size matches H/W/D
    dx = bboxes[:, 1] - bboxes[:, 0]
    dy = bboxes[:, 3] - bboxes[:, 2]
    dz = bboxes[:, 5] - bboxes[:, 4]
    if not (np.all(dx == W) and np.all(dy == H) and np.all(dz == D)):
        print(
            "WARNING: refl bboxes are not all the requested (W, H, D). "
            f"This file may have DIALS bboxes rather than refltorch's fixed windows. "
            f"dx: min={dx.min()} max={dx.max()} expected={W}; "
            f"dy: min={dy.min()} max={dy.max()} expected={H}; "
            f"dz: min={dz.min()} max={dz.max()} expected={D}."
        )

    # Load masks (N, V)
    masks, fmt = _load_masks(args.masks)
    if masks.ndim != 2 or masks.shape != (N, V):
        raise ValueError(
            f"mask shape {masks.shape} doesn't match ({N}, {V}); "
            "check H/W/D args and the reflection file."
        )

    # Compute overlap (N, D, H, W) and flatten
    overlap = compute_overlap_mask(bboxes, xyzcal, D, H, W)
    overlap_flat = overlap.reshape(N, V)

    # Combine with existing mask — a pixel is valid iff it was valid before AND
    # is not owned by a neighbor. Do not invert the old mask: dead pixels stay dead.
    print("Applying overlap to existing mask...")
    new_masks = masks.astype(bool) & ~overlap_flat

    # Stats
    old_valid = masks.astype(bool).mean(-1)
    new_valid = new_masks.mean(-1)
    overlap_frac = overlap_flat.mean(-1)

    print("\n=== Overlap statistics ===")
    print(f"  Mean overlap per refl:         {overlap_frac.mean() * 100:.2f}%")
    print(f"  Median overlap per refl:       {np.median(overlap_frac) * 100:.2f}%")
    print(f"  Refl with any overlap:         {(overlap_frac > 0).sum():,} / {N:,}  "
          f"({(overlap_frac > 0).mean() * 100:.1f}%)")
    print(f"  Refl with >10% overlap:        {(overlap_frac > 0.10).sum():,}")
    print(f"  Refl with >30% overlap:        {(overlap_frac > 0.30).sum():,}")
    print(f"  Refl with >50% overlap:        {(overlap_frac > 0.50).sum():,}")
    print()
    print(f"  Valid-pixel fraction (before): {old_valid.mean() * 100:.2f}%")
    print(f"  Valid-pixel fraction (after):  {new_valid.mean() * 100:.2f}%")

    # Save
    if args.out is None:
        p = Path(args.masks)
        out_path = p.with_name(f"{p.stem}_overlap{p.suffix}")
    else:
        out_path = Path(args.out)
    _save_masks(new_masks, out_path, fmt)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
