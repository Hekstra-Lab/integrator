"""Generate a tiny synthetic ragged dataset for local pipeline tests.

Produces the same files refltorch.mksbox-dials would, minus the .refl file
(since that requires DIALS to write). Layout:

    <out_dir>/
      chunks/chunk_000.npz, chunk_001.npz   — ragged shoebox data
      metadata.pt                             — per-reflection scalars
      stats.pt                                — raw counts (mean, var)
      anscombe_stats.pt                       — anscombe (mean, var)
      group_labels_<N>.pt                     — quantile-binned resolution
      bin_edges_<N>.pt
      identifiers.yaml                        — empty placeholder
      manifest.yaml

Synthetic reflection content:
- Each shoebox is filled with a Gaussian peak at center + uniform background,
  with realistic Poisson noise.
- Variable shape per reflection: dz in [3, 11], h/w in [11, 31].
- mask = (background <= count) AND not in DIALS-overlap region (random).
- d (resolution) drawn uniformly in [1.5, 6.0] Å.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml


def make_one_reflection(rng, dz_range=(3, 11), h_range=(11, 31), w_range=(11, 31)):
    D = int(rng.integers(*dz_range))
    H = int(rng.integers(*h_range))
    W = int(rng.integers(*w_range))

    bg_rate = float(rng.uniform(0.5, 5.0))
    peak_amp = float(rng.uniform(50.0, 1000.0))
    peak_sigma_z = max(D / 6.0, 0.8)
    peak_sigma_xy = float(rng.uniform(2.0, 4.0))

    zc, yc, xc = D / 2.0, H / 2.0, W / 2.0
    z, y, x = np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing="ij"
    )
    gauss = peak_amp * np.exp(
        -0.5
        * (
            ((z - zc) / peak_sigma_z) ** 2
            + ((y - yc) / peak_sigma_xy) ** 2
            + ((x - xc) / peak_sigma_xy) ** 2
        )
    )
    rate = gauss + bg_rate
    counts = rng.poisson(rate).astype(np.uint16)

    # ~95% mask validity per reflection: random tiny dead/overlap dropouts
    mask = rng.random(counts.shape) > 0.05

    # Foreground bit: rough ellipse around the peak (just for completeness)
    fg = (gauss > 0.2 * peak_amp)

    return D, H, W, counts, mask, fg


def make_chunk(rng, n_refl: int, refl_id_start: int):
    """Build a single chunk's payload for np.savez."""
    data_chunks = []
    mask_chunks = []
    fg_chunks = []
    bg_chunks = []
    ov_chunks = []
    shapes = np.empty((n_refl, 3), dtype=np.int32)
    bboxes = np.empty((n_refl, 6), dtype=np.int32)
    refl_ids = np.arange(refl_id_start, refl_id_start + n_refl, dtype=np.int64)

    for i in range(n_refl):
        D, H, W, counts, mask, fg = make_one_reflection(rng)
        shapes[i] = (D, H, W)
        # synthetic detector bbox; not used by the dataloader's loss path
        bboxes[i] = (0, W, 0, H, 0, D)

        data_chunks.append(counts.ravel())
        mask_chunks.append(mask.ravel())
        fg_chunks.append(fg.ravel())
        bg_chunks.append((~fg & mask).ravel())
        ov_chunks.append(np.zeros_like(mask).ravel())

    offsets = np.cumsum([0] + [d.size for d in data_chunks]).astype(np.int64)

    return {
        "data": np.concatenate(data_chunks),
        "mask": np.concatenate(mask_chunks),
        "foreground": np.concatenate(fg_chunks),
        "background": np.concatenate(bg_chunks),
        "overlapped": np.concatenate(ov_chunks),
        "offsets": offsets,
        "shapes": shapes,
        "bboxes": bboxes,
        "refl_ids": refl_ids,
    }


def make_metadata(n_total: int, rng, refl_ids: np.ndarray) -> dict:
    """Synthetic per-reflection scalars matching what refl_as_pt would write."""
    d = rng.uniform(1.5, 6.0, size=n_total).astype(np.float32)
    md = {
        "d": torch.from_numpy(d),
        "refl_ids": torch.from_numpy(refl_ids.astype(np.float32)),
        "panel": torch.zeros(n_total, dtype=torch.float32),
        "flags": torch.zeros(n_total, dtype=torch.float32),
        "intensity.prf.value": torch.from_numpy(
            rng.uniform(0, 1000, size=n_total).astype(np.float32)
        ),
        "intensity.prf.variance": torch.from_numpy(
            rng.uniform(10, 100, size=n_total).astype(np.float32)
        ),
        "intensity.sum.value": torch.from_numpy(
            rng.uniform(0, 1000, size=n_total).astype(np.float32)
        ),
        "intensity.sum.variance": torch.from_numpy(
            rng.uniform(10, 100, size=n_total).astype(np.float32)
        ),
        "background.mean": torch.from_numpy(
            rng.uniform(0, 5, size=n_total).astype(np.float32)
        ),
        "xyzcal.px.0": torch.from_numpy(
            rng.uniform(0, 2400, size=n_total).astype(np.float32)
        ),
        "xyzcal.px.1": torch.from_numpy(
            rng.uniform(0, 2400, size=n_total).astype(np.float32)
        ),
        "xyzcal.px.2": torch.from_numpy(
            rng.uniform(0, 360, size=n_total).astype(np.float32)
        ),
    }
    return md


def make_stats(chunk_files: list[Path]):
    """Streaming raw + anscombe (mean, var) over valid voxels."""
    sum_c = sumsq_c = sum_a = sumsq_a = 0.0
    n = 0
    for f in chunk_files:
        with np.load(f) as npz:
            data = npz["data"].astype(np.float64)
            mask = npz["mask"]
        v = data[mask]
        a = 2.0 * np.sqrt(v + 0.375)
        sum_c += float(v.sum())
        sumsq_c += float((v * v).sum())
        sum_a += float(a.sum())
        sumsq_a += float((a * a).sum())
        n += int(v.size)
    mean_c = sum_c / n
    var_c = sumsq_c / n - mean_c * mean_c
    mean_a = sum_a / n
    var_a = sumsq_a / n - mean_a * mean_a
    return (
        torch.tensor([mean_c, var_c], dtype=torch.float32),
        torch.tensor([mean_a, var_a], dtype=torch.float32),
    )


def bin_by_resolution(d: torch.Tensor, n_bins: int):
    qs = torch.linspace(0, 1, n_bins + 1)
    edges = torch.quantile(d.float(), qs)
    labels = torch.searchsorted(edges[1:-1], d).long()
    return labels, edges


def make_toy_dataset(
    out_dir: Path,
    n_chunks: int = 2,
    refl_per_chunk: int = 64,
    n_bins: int = 5,
    seed: int = 0,
) -> Path:
    out_dir = Path(out_dir)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Chunks
    chunk_paths = []
    for ci in range(n_chunks):
        payload = make_chunk(
            rng, n_refl=refl_per_chunk, refl_id_start=ci * refl_per_chunk
        )
        p = chunks_dir / f"chunk_{ci:03d}.npz"
        np.savez(p, **payload)
        chunk_paths.append(p)

    # Metadata
    n_total = n_chunks * refl_per_chunk
    refl_ids = np.arange(n_total, dtype=np.int64)
    md = make_metadata(n_total, rng, refl_ids)
    torch.save(md, out_dir / "metadata.pt")

    # Stats
    raw, ans = make_stats(chunk_paths)
    torch.save(raw, out_dir / "stats.pt")
    torch.save(ans, out_dir / "anscombe_stats.pt")

    # Group labels
    labels, edges = bin_by_resolution(md["d"], n_bins)
    torch.save(labels, out_dir / f"group_labels_{n_bins}.pt")
    torch.save(edges, out_dir / f"bin_edges_{n_bins}.pt")

    # Identifiers + manifest (placeholder content)
    (out_dir / "identifiers.yaml").write_text(yaml.safe_dump({}))
    (out_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "total_reflections": n_total,
                "n_chunks": n_chunks,
                "synthetic": True,
            }
        )
    )

    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-chunks", type=int, default=2)
    ap.add_argument("--refl-per-chunk", type=int, default=64)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    p = make_toy_dataset(
        Path(args.out_dir),
        n_chunks=args.n_chunks,
        refl_per_chunk=args.refl_per_chunk,
        n_bins=args.n_bins,
        seed=args.seed,
    )
    print(f"Wrote toy dataset to {p}")


if __name__ == "__main__":
    main()
