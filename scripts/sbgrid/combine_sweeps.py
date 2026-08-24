"""Combine per-sweep shoebox datasets into the one directory training reads.

`make_shoeboxes` writes a dataset per sweep, and `RotationDataModule` takes a
single `data_dir`. Training on one sweep of three would use a third of the
data, and the sweeps are three passes over one crystal, so they belong
together.

Two details the concatenation has to get right:

`refl_ids` restart at zero in every sweep, so they collide. A `sweep_id`
column is added and the original ids are kept unchanged, which keeps each
reflection traceable back to the `.refl` file it came from -- renumbering
would make the prediction writeback ambiguous.

The count statistics are recomputed over the combined data rather than
averaged across sweeps. Averaging means and variances of different-sized
groups is not the mean and variance of their union.

Usage:
    python scripts/sbgrid/combine_sweeps.py --shoebox-dir <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Combine per-sweep shoeboxes")
    p.add_argument("--shoebox-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--chunk",
        type=int,
        default=20000,
        help="reflections copied at a time, to bound memory",
    )
    return p.parse_args()


def sweep_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir() if (d / "dataset.yaml").is_file())


def anscombe_stats(counts: np.memmap, masks: np.memmap, chunk: int) -> tuple:
    """Streaming mean and variance of the raw and Anscombe-transformed counts.

    Computed in one pass over the combined array: the transform is applied to
    masked pixels only, which is what the data module does when it
    standardizes.
    """
    total = raw_sum = raw_sq = ans_sum = ans_sq = 0.0
    for start in range(0, counts.shape[0], chunk):
        c = np.asarray(counts[start : start + chunk], dtype=np.float64)
        m = np.asarray(masks[start : start + chunk], dtype=bool)
        values = c[m]
        if values.size == 0:
            continue
        transformed = 2.0 * np.sqrt(np.clip(values, 0, None) + 0.375)
        total += values.size
        raw_sum += values.sum()
        raw_sq += (values**2).sum()
        ans_sum += transformed.sum()
        ans_sq += (transformed**2).sum()
    raw_mean = raw_sum / total
    ans_mean = ans_sum / total
    return (
        [raw_mean, raw_sq / total - raw_mean**2],
        [ans_mean, ans_sq / total - ans_mean**2],
    )


def main():
    args = parse_args()
    sweeps = sweep_dirs(args.shoebox_dir)
    if not sweeps:
        raise SystemExit(f"no per-sweep datasets under {args.shoebox_dir}")

    specs = [yaml.safe_load((d / "dataset.yaml").read_text()) for d in sweeps]
    geometry = specs[0]["geometry"]
    for d, spec in zip(sweeps, specs, strict=True):
        if spec["geometry"] != geometry:
            raise SystemExit(
                f"{d.name} has geometry {spec['geometry']}, expected {geometry}; "
                "the sweeps must be cut with one window to be combined"
            )

    counts = [np.load(d / "counts.npy", mmap_mode="r") for d in sweeps]
    masks = [np.load(d / "masks.npy", mmap_mode="r") for d in sweeps]
    n_total = sum(c.shape[0] for c in counts)
    n_pixels = counts[0].shape[1]
    print(f"{len(sweeps)} sweeps, {n_total:,} reflections of {n_pixels} pixels")
    for d, c in zip(sweeps, counts, strict=True):
        print(f"  {d.name}: {c.shape[0]:,}")

    args.out.mkdir(parents=True, exist_ok=True)
    out_counts = np.lib.format.open_memmap(
        args.out / "counts.npy",
        mode="w+",
        dtype=counts[0].dtype,
        shape=(n_total, n_pixels),
    )
    out_masks = np.lib.format.open_memmap(
        args.out / "masks.npy",
        mode="w+",
        dtype=masks[0].dtype,
        shape=(n_total, n_pixels),
    )

    offset = 0
    for i, (c, m) in enumerate(zip(counts, masks, strict=True)):
        for start in range(0, c.shape[0], args.chunk):
            stop = min(start + args.chunk, c.shape[0])
            out_counts[offset + start : offset + stop] = c[start:stop]
            out_masks[offset + start : offset + stop] = m[start:stop]
        offset += c.shape[0]
        print(f"  copied sweep {i}: {offset:,} / {n_total:,}")
    out_counts.flush()
    out_masks.flush()

    from integrator.io import load_metadata, save_data

    combined: dict[str, np.ndarray] = {}
    sweep_id = []
    for i, d in enumerate(sweeps):
        meta = load_metadata(d / "metadata.npy")
        n = len(next(iter(meta.values())))
        sweep_id.append(np.full(n, i, dtype=np.int32))
        for key, value in meta.items():
            combined.setdefault(key, []).append(np.asarray(value))
    stacked = {k: np.concatenate(v) for k, v in combined.items()}
    # which sweep each reflection came from, so a prediction can be written
    # back to the right .refl file
    stacked["sweep_id"] = np.concatenate(sweep_id)
    save_data(stacked, args.out / "metadata.npy")
    print(f"  metadata: {len(stacked)} columns, {len(stacked['sweep_id']):,} rows")

    raw, ans = anscombe_stats(out_counts, out_masks, args.chunk)
    spec = {
        "geometry": geometry,
        "n_reflections": int(n_total),
        "polychromatic": bool(specs[0].get("polychromatic", False)),
        "anscombe": True,
        "files": {
            "counts": "counts.npy",
            "masks": "masks.npy",
            "reference": "metadata.npy",
        },
        "stats": {"raw": raw, "anscombe": ans},
        "sweeps": [d.name for d in sweeps],
    }
    if "crystal" in specs[0]:
        spec["crystal"] = specs[0]["crystal"]
    (args.out / "dataset.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

    # the per-sweep reflection tables stay where they are; sweep_id maps back
    for d in sweeps:
        refl = d / "reflections_.refl"
        if refl.exists():
            shutil.copy2(refl, args.out / f"reflections_{d.name}.refl")

    print(f"\nwrote {args.out}")
    print(f"  raw stats      mean {raw[0]:.3f}  var {raw[1]:.1f}")
    print(f"  anscombe stats mean {ans[0]:.3f}  var {ans[1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
