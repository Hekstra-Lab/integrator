"""Convert fixed-window shoeboxes into ragged (spot-sized) chunks.

The fixed pipeline stores every reflection in one padded window
(`counts.npy` / `masks.npy` / `dials_mask.npy`, each `(N, dz*dy*dx)`), which is
mostly background - the source of the weak-high-res anomalous contamination.
This crops each reflection down to its DIALS foreground extent plus a small
guard ring and writes the `chunk_*.npz` layout that `RaggedShoeboxDataModule`
reads, carrying the foreground bit through for `profile_support: foreground`.

The crop stays in fixed-window voxel coordinates (the foreground bit is already
aligned to `counts.npy` row for row), so no detector-frame arithmetic is needed.

Several sweeps can be fused into one ragged dataset: pass more than one
`--data-dir`. Each sweep's reflections get a global `refl_id` offset, chunk
numbering continues across sweeps, and `metadata.npy` / `group_labels_*.pt` are
concatenated in sweep order, so a Friedel pair's observations from every sweep
live in one dataset with aligned metadata.

Run:
    uv run python scripts/ragged/fixed_to_ragged.py \
        --data-dir  <sweep1 dir> <sweep2 dir> \
        --out-dir   <ragged dataset dir to create> \
        --dz 5 --dy 37 --dx 33            # fixed window (or read from dataset.yaml)
"""

import argparse
from pathlib import Path

import numpy as np

# DIALS MaskCode bits as written by make_shoeboxes (dials_mask.npy, uint8).
VALID = 1
FOREGROUND = 4
OVERLAPPED = 32


def _window_shape(args, first_dir: Path) -> tuple[int, int, int]:
    """Fixed window (dz, dy, dx) from CLI or the first sweep's dataset.yaml."""
    if args.dz and args.dy and args.dx:
        return int(args.dz), int(args.dy), int(args.dx)
    spec_path = first_dir / "dataset.yaml"
    if spec_path.exists():
        import yaml

        geom = yaml.safe_load(spec_path.read_text()).get("geometry", {})
        d, h, w = geom.get("d"), geom.get("h"), geom.get("w")
        if None not in (d, h, w):
            return int(d), int(h), int(w)
    raise SystemExit(
        "window shape unknown: pass --dz/--dy/--dx or provide dataset.yaml "
        "with geometry.d/h/w"
    )


def _fg_bbox(fg_3d: np.ndarray):
    """Inclusive-exclusive (z0,z1,y0,y1,x0,x1) of the foreground voxels, or None."""
    zs, ys, xs = np.nonzero(fg_3d)
    if zs.size == 0:
        return None
    return (
        int(zs.min()), int(zs.max()) + 1,
        int(ys.min()), int(ys.max()) + 1,
        int(xs.min()), int(xs.max()) + 1,
    )


def _expand(box, guard, shape):
    """Grow a bbox by per-axis guard, clipped to the window shape."""
    dz, dy, dx = shape
    gz, gy, gx = guard
    z0, z1, y0, y1, x0, x1 = box
    return (
        max(0, z0 - gz), min(dz, z1 + gz),
        max(0, y0 - gy), min(dy, y1 + gy),
        max(0, x0 - gx), min(dx, x1 + gx),
    )


def _centered_box(size, shape):
    """A crop of `size` (dz,dy,dx) centered in the window, clipped to it."""
    out = []
    for s, full in zip(size, shape, strict=True):
        s = min(s, full)
        lo = max(0, (full - s) // 2)
        out.extend((lo, lo + s))
    return tuple(out)


def _touches_edge(box, shape) -> bool:
    """True if the (unguarded) foreground bbox reaches a window boundary,
    meaning the fixed window was too small to contain the whole spot."""
    dz, dy, dx = shape
    z0, z1, y0, y1, x0, x1 = box
    return z0 == 0 or z1 == dz or y0 == 0 or y1 == dy or x0 == 0 or x1 == dx


def _scan_sweep(dmask, n, shape, guard):
    """Pass 1: crop boxes from the foreground bit + has_fg / edge flags."""
    dz, dy, dx = shape
    crop_boxes = np.zeros((n, 6), dtype=np.int32)  # (z0,z1,y0,y1,x0,x1)
    has_fg = np.zeros(n, dtype=bool)
    edge = np.zeros(n, dtype=bool)
    for i in range(n):
        fg = ((np.asarray(dmask[i]) & FOREGROUND) != 0).reshape(dz, dy, dx)
        box = _fg_bbox(fg)
        if box is None:
            continue
        has_fg[i] = True
        edge[i] = _touches_edge(box, shape)
        crop_boxes[i] = _expand(box, guard, shape)
    return crop_boxes, has_fg, edge


def _print_stats(crop_boxes, has_fg, edge, med, K):
    n = len(crop_boxes)
    n_fg = int(has_fg.sum())
    vols = (
        (crop_boxes[:, 1] - crop_boxes[:, 0])
        * (crop_boxes[:, 3] - crop_boxes[:, 2])
        * (crop_boxes[:, 5] - crop_boxes[:, 4])
    )
    print(
        f"  foreground: {n_fg:,} ({100*n_fg/n:.1f}%)  "
        f"no-fg (centered {med}): {n-n_fg:,}  "
        f"edge-touching (truncated spot): {int(edge.sum()):,}"
    )
    for label, a in (
        ("depth ", crop_boxes[:, 1] - crop_boxes[:, 0]),
        ("height", crop_boxes[:, 3] - crop_boxes[:, 2]),
        ("width ", crop_boxes[:, 5] - crop_boxes[:, 4]),
        ("voxels", vols),
    ):
        q = np.percentile(a, [0, 50, 90, 99, 100]).astype(int)
        print(f"  {label}: min/med/p90/p99/max = "
              f"{q[0]}/{q[1]}/{q[2]}/{q[3]}/{q[4]}")
    print(f"  mean voxels/refl = {vols.mean():.0f} "
          f"({100*vols.mean()/K:.1f}% of fixed window)")


def _write_sweep(
    counts, masks, dmask, crop_boxes, n, shape, chunks_dir,
    id_offset, chunk_start, chunk_size,
):
    """Pass 2: crop counts/mask/foreground and write chunk_*.npz for one sweep."""
    dz, dy, dx = shape
    counts_dtype = counts.dtype
    chunk_no = chunk_start
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        data_parts, mask_parts, fg_parts = [], [], []
        offsets = [0]
        shapes = np.zeros((end - start, 3), dtype=np.int32)
        bboxes = np.zeros((end - start, 6), dtype=np.int32)
        refl_ids = np.arange(id_offset + start, id_offset + end, dtype=np.int64)

        for j, i in enumerate(range(start, end)):
            z0, z1, y0, y1, x0, x1 = crop_boxes[i]
            c = np.asarray(counts[i]).reshape(dz, dy, dx)[z0:z1, y0:y1, x0:x1]
            code = np.asarray(dmask[i]).reshape(dz, dy, dx)[z0:z1, y0:y1, x0:x1]
            validity = (
                np.asarray(masks[i]).reshape(dz, dy, dx)[z0:z1, y0:y1, x0:x1]
            ).astype(bool)
            # DIALS never sets the Valid(1) bit in dials_mask, so validity comes
            # from masks.npy (dead pixels / gaps); drop only overlapped neighbours.
            m = validity & ((code & OVERLAPPED) == 0)
            fg = (code & FOREGROUND) != 0

            data_parts.append(c.reshape(-1).astype(counts_dtype, copy=False))
            mask_parts.append(m.reshape(-1))
            fg_parts.append(fg.reshape(-1))
            D, H, W = z1 - z0, y1 - y0, x1 - x0
            shapes[j] = (D, H, W)
            bboxes[j] = (x0, x1, y0, y1, z0, z1)  # DIALS order, window frame
            offsets.append(offsets[-1] + D * H * W)

        chunk_path = chunks_dir / f"chunk_{chunk_no:03d}.npz"
        np.savez(
            chunk_path,
            data=np.concatenate(data_parts),
            mask=np.concatenate(mask_parts),
            foreground=np.concatenate(fg_parts),
            offsets=np.asarray(offsets, dtype=np.int64),
            shapes=shapes,
            bboxes=bboxes,
            refl_ids=refl_ids,
        )
        chunk_no += 1
        print(f"  wrote {chunk_path.name}: reflections {start:,}..{end:,}")
    return chunk_no


def _merge_metadata(sweep_dirs, counts_per_sweep, out_dir, fname):
    """Concatenate each sweep's metadata dict in sweep order (aligns global refl_id)."""
    import torch

    from integrator.io import load_metadata, save_data

    per_sweep = []
    for d in sweep_dirs:
        p = d / fname
        if not p.exists():
            raise SystemExit(f"{p} missing; every sweep needs {fname}")
        per_sweep.append(load_metadata(p))
    keys = set(per_sweep[0])
    for m in per_sweep[1:]:
        keys &= set(m)
    merged = {}
    for k in sorted(keys):
        vals = [torch.as_tensor(m[k]) for m in per_sweep]
        merged[k] = torch.cat(vals, dim=0)
        if merged[k].shape[0] != sum(counts_per_sweep):
            raise SystemExit(
                f"metadata '{k}' has {merged[k].shape[0]} rows, expected "
                f"{sum(counts_per_sweep)}"
            )
    save_data(merged, out_dir / fname)
    print(f"  merged {fname}: {sum(counts_per_sweep):,} rows, keys={sorted(keys)}")


def _merge_group_labels(sweep_dirs, out_dir):
    """Concatenate matching group_labels_*.pt across sweeps."""
    import torch

    names = set.intersection(
        *[{p.name for p in d.glob("group_labels_*.pt")} for d in sweep_dirs]
    ) if sweep_dirs else set()
    for name in sorted(names):
        parts = [torch.load(d / name, weights_only=True) for d in sweep_dirs]
        torch.save(torch.cat(parts, dim=0), out_dir / name)
        print(f"  merged {name}: {sum(len(p) for p in parts):,} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir", required=True, nargs="+",
        help="one or more fixed shoebox dirs (sweeps) to fuse",
    )
    ap.add_argument("--out-dir", required=True, help="ragged dataset dir to create")
    ap.add_argument("--counts-fname", default="counts.npy")
    ap.add_argument("--masks-fname", default="masks.npy")
    ap.add_argument("--dials-mask-fname", default="dials_mask.npy")
    ap.add_argument("--metadata-fname", default="metadata.npy")
    ap.add_argument("--dz", type=int, default=None)
    ap.add_argument("--dy", type=int, default=None)
    ap.add_argument("--dx", type=int, default=None)
    ap.add_argument("--guard-z", type=int, default=1, help="depth guard ring")
    ap.add_argument("--guard-y", type=int, default=2, help="height guard ring")
    ap.add_argument("--guard-x", type=int, default=2, help="width guard ring")
    ap.add_argument(
        "--fixed-size", type=int, nargs=3, default=None, metavar=("DZ", "DY", "DX"),
        help="CONTROL: use one constant centered box (dz dy dx) for ALL reflections "
        "instead of the per-reflection foreground crop (ablates ragged vs right-sized).",
    )
    ap.add_argument("--chunk-size", type=int, default=50000,
                    help="reflections per output chunk_*.npz")
    ap.add_argument("--limit", type=int, default=None,
                    help="convert only the first N reflections PER SWEEP (debugging)")
    args = ap.parse_args()

    sweep_dirs = [Path(d) for d in args.data_dir]
    out_dir = Path(args.out_dir)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    dz, dy, dx = _window_shape(args, sweep_dirs[0])
    K = dz * dy * dx
    guard = (args.guard_z, args.guard_y, args.guard_x)
    print(f"fixed window {dz}x{dy}x{dx}={K}, guard {guard}, {len(sweep_dirs)} sweep(s)")

    # ---- pass 1 over every sweep: crop boxes + a global no-fg median ----
    scans = []  # (dir, n, crop_boxes, has_fg, edge)
    for d in sweep_dirs:
        counts = np.load(d / args.counts_fname, mmap_mode="r")
        dmask = np.load(d / args.dials_mask_fname, mmap_mode="r")
        if counts.shape[1] != K or dmask.shape[1] != K:
            raise SystemExit(
                f"{d}: {counts.shape[1]}/{dmask.shape[1]} voxels/refl but "
                f"window={K}; check --dz/--dy/--dx"
            )
        n = min(counts.shape[0], args.limit) if args.limit else counts.shape[0]
        crop_boxes, has_fg, edge = _scan_sweep(dmask, n, (dz, dy, dx), guard)
        scans.append((d, n, crop_boxes, has_fg, edge))
        print(f"scanned {d.name}: {n:,} reflections")

    all_fg = np.concatenate([s[2][s[3]] for s in scans])
    if all_fg.size == 0:
        raise SystemExit("no reflection has a foreground voxel; check dials_mask")
    med = (
        int(np.median(all_fg[:, 1] - all_fg[:, 0])),
        int(np.median(all_fg[:, 3] - all_fg[:, 2])),
        int(np.median(all_fg[:, 5] - all_fg[:, 4])),
    )
    for _, _, crop_boxes, has_fg, _ in scans:
        if (~has_fg).any():
            crop_boxes[~has_fg] = _centered_box(med, (dz, dy, dx))

    if args.fixed_size is not None:
        # CONTROL: replace every crop with one constant centered box (shoeboxes
        # are centroid-centered, so window-centered == spot-centered).
        fixed = _centered_box(tuple(args.fixed_size), (dz, dy, dx))
        for _, _, crop_boxes, _, _ in scans:
            crop_boxes[:] = fixed
        print(f"--fixed-size {tuple(args.fixed_size)} -> constant box {fixed} for all reflections")

    print("\ncombined crop stats:")
    _print_stats(
        np.concatenate([s[2] for s in scans]),
        np.concatenate([s[3] for s in scans]),
        np.concatenate([s[4] for s in scans]),
        med, K,
    )

    # ---- pass 2: write chunks with global refl_id offsets ----
    print()
    id_offset = 0
    chunk_no = 0
    counts_per_sweep = []
    for d, n, crop_boxes, _, _ in scans:
        counts = np.load(d / args.counts_fname, mmap_mode="r")
        masks = np.load(d / args.masks_fname, mmap_mode="r")
        dmask = np.load(d / args.dials_mask_fname, mmap_mode="r")
        print(f"writing {d.name} (refl_id {id_offset:,}..{id_offset+n:,}):")
        chunk_no = _write_sweep(
            counts, masks, dmask, crop_boxes, n, (dz, dy, dx),
            chunks_dir, id_offset, chunk_no, args.chunk_size,
        )
        id_offset += n
        counts_per_sweep.append(n)

    # ---- merge metadata + resolution-bin labels for global refl_id alignment ----
    print()
    _merge_metadata(sweep_dirs, counts_per_sweep, out_dir, args.metadata_fname)
    _merge_group_labels(sweep_dirs, out_dir)

    print(f"\ndone: {id_offset:,} reflections -> {chunk_no} chunk(s) in {chunks_dir}")


if __name__ == "__main__":
    main()
