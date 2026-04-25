"""PyTorch Dataset + collate for ragged shoebox data produced by refltorch.mksbox-dials.

Each chunk .npz contains concatenated pixel data for a subset of reflections,
with offsets/shapes arrays that let us slice out individual ragged shoeboxes.
This module provides:

- RaggedShoeboxDataset: random-access over all reflections across chunks.
- pad_collate_ragged: batches variable-size shoeboxes with explicit masks.
- VolumeBucketSampler: group reflections with similar voxel-counts into batches
  so padding within a batch stays tight.
- RaggedShoeboxDataModule: Lightning wrapper for train/val/test splits.

Disk format expected (as written by refltorch.mksbox-dials):
    <chunks_dir>/chunk_000.npz, chunk_001.npz, ...
    <chunks_dir>/../manifest.yaml
    <chunks_dir>/../reflections.refl
"""

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger(__name__)


# Default arrays exposed per reflection. Add "foreground"/"background"/"overlapped"
# if you want to drive a background-aware loss.
DEFAULT_KEYS = ("data", "mask")


class RaggedShoeboxDataset(Dataset):
    """Indexes every reflection across all chunk_*.npz files in a directory.

    Args:
        chunks_dir: directory of chunk_*.npz files (from refltorch.mksbox-dials).
        metadata_path: path to metadata.pt (dict of per-reflection scalars
            including 'd'). If None, no metadata is attached to items.
        stats_path: path to stats.pt ([mean, var] of raw counts) or
            anscombe_stats.pt. If None, auto-resolves based on `anscombe`.
        group_labels_path: path to group_labels_{N}.pt. If None, looks for any
            group_labels_*.pt in the parent of chunks_dir. If not found, all
            reflections get group_label=0.
        anscombe: if True, standardize via the Anscombe transform; otherwise
            plain z-score of raw counts. Either way, `counts` (raw) is also
            returned for the Poisson NLL.
        keys: which per-voxel arrays to expose per item. 'mask' is always
            included as a training validity mask (Valid & ~Overlapped from
            DIALS). Others (foreground/background/overlapped) can be added.
        eager: if True, load all chunk arrays into RAM at construction time.
        extra_metadata_keys: additional scalar columns from metadata.pt to
            attach per reflection (e.g. 'intensity.prf.value' for debugging).
    """

    def __init__(
        self,
        chunks_dir,
        metadata_path=None,
        stats_path=None,
        group_labels_path=None,
        anscombe: bool = True,
        keys=DEFAULT_KEYS,
        eager: bool = True,
        extra_metadata_keys: tuple[str, ...] = (),
        min_valid_pixels: int = 10,
        max_count: float | None = None,
        transform: str = "anscombe",
        protect_foreground: bool = True,
    ):
        # If max_count is set with foreground protection, we need the
        # foreground bit loaded — promote it into keys.
        if max_count is not None and protect_foreground:
            keys = tuple(set(tuple(keys)) | {"foreground"})

        self.keys = tuple(dict.fromkeys(("data",) + tuple(keys)))  # dedupe, data first
        self.eager = eager
        self.anscombe = bool(anscombe)
        self.min_valid_pixels = int(min_valid_pixels)
        # If set, raw counts above this threshold are treated as artifacts.
        # If `protect_foreground=True`, only pixels above this threshold AND
        # outside DIALS' Foreground bit get masked out (the reasoning being
        # that bright in-Foreground pixels are genuine Bragg signal we
        # don't want to discard). All above-threshold pixels are clipped
        # in `counts` regardless, so the Poisson NLL never sees the extreme
        # value and grad through it stays well-behaved.
        self.max_count = float(max_count) if max_count is not None else None
        self.protect_foreground = bool(protect_foreground)
        # When max_count is set, we precompute the artifact mask once per
        # chunk and cache it next to the chunk file. The cache filename
        # encodes the threshold + protect_foreground flag so changing
        # parameters invalidates the cache automatically.
        self._artifact_caches = []  # list of bool arrays parallel to chunks
        # Encoder-input transform applied to (mask-clipped) raw counts:
        #   "anscombe"  : 2*sqrt(x + 0.375) — Poisson variance-stabilizing,
        #                 paired with the saved anscombe_stats.pt for
        #                 standardization. Default; matches fixed pipeline.
        #   "log1p"     : log(1 + x) — much tighter dynamic range. Skips
        #                 the dataset-level standardization (encoder
        #                 GroupNorm normalizes per-batch). Better for
        #                 datasets with very bright outliers.
        #   "none"      : raw counts (encoder GroupNorm does all the work).
        if transform not in ("anscombe", "log1p", "none"):
            raise ValueError(f"unknown transform: {transform!r}")
        self.transform = transform
        # `anscombe=True` retained for backward compat — overrides transform
        # only when explicitly set False (legacy "raw" path).
        if transform == "anscombe" and not self.anscombe:
            self.transform = "none"

        chunks_dir = Path(chunks_dir)
        parent_dir = chunks_dir.parent

        # ---------- per-reflection metadata (d, miller_index, etc.) ----------
        if metadata_path is None:
            candidate = parent_dir / "metadata.pt"
            metadata_path = candidate if candidate.exists() else None
        self.metadata: dict = (
            torch.load(metadata_path, weights_only=True) if metadata_path else {}
        )
        self._extra_keys = tuple(extra_metadata_keys)

        # ---------- standardization stats ----------
        # Resolve the saved stats file path corresponding to the chosen
        # transform. If absent, we'll compute streaming stats from chunks
        # below once they're loaded.
        #
        # log1p is intentionally OFF this path: scvi-tools'
        # `log_variational=True` recipe (the literature standard for
        # skewed-count VAEs spanning 0 → 10⁵+ counts) feeds raw `log1p(x)`
        # straight to the encoder and lets GroupNorm/LayerNorm handle the
        # per-batch normalization. Z-scoring against a global (mean, std)
        # of the heavy-tailed log1p distribution dominates the std with
        # the bright Bragg-peak tail; bulk voxels get squashed near zero
        # and the encoder learns from a near-degenerate input. Empirically
        # this was a contributor to the qbg NaN around step 70 on dataset
        # 140 — the fixed pipeline doesn't have this issue because every
        # shoebox has the same shape, so the global std is well-defined.
        self._stat_mean = 0.0
        self._stat_std = 1.0
        self._stats_resolved_path = None
        if stats_path is None and transform != "log1p":
            default_names = {
                "anscombe": "anscombe_stats.pt",
                "none":     "stats.pt",
            }
            # Honor the legacy `anscombe=False` shorthand for "none"
            t_for_path = (
                transform if transform in default_names
                else ("anscombe" if anscombe else "none")
            )
            candidate = parent_dir / default_names[t_for_path]
            stats_path = candidate if candidate.exists() else None
        if stats_path is not None:
            s = torch.load(stats_path, weights_only=True)
            self._stat_mean = float(s[0])
            self._stat_std = float(torch.sqrt(torch.clamp(s[1], min=1e-12)))
            self._stats_resolved_path = str(stats_path)

        # ---------- group_labels (resolution bins) ----------
        if group_labels_path is None:
            matches = sorted(parent_dir.glob("group_labels_*.pt"))
            group_labels_path = matches[0] if matches else None
        else:
            # Resolve a bare filename relative to parent_dir, matching how
            # metadata_path/stats_path are resolved above. Lets the YAML say
            # `group_labels_path: group_labels_30.pt` and Just Work.
            gp = Path(group_labels_path)
            if not gp.is_absolute():
                group_labels_path = parent_dir / gp
        if group_labels_path is not None:
            self.group_labels = torch.load(group_labels_path, weights_only=True).long()
        else:
            self.group_labels = None

        # ---------- chunks ----------
        chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk_*.npz in {chunks_dir}")

        self._chunks = []           # per-chunk dict of numpy arrays (or NpzFile if lazy)
        self._global_index = []     # (chunk_idx, local_idx) for each global idx
        self._volumes = []          # total voxels per reflection, for bucket sampler

        for ci, f in enumerate(chunk_files):
            with np.load(f) as npz:
                required = ("offsets", "shapes", "bboxes", "refl_ids")
                for r in required:
                    if r not in npz.files:
                        raise KeyError(f"{f}: missing '{r}'")
                if eager:
                    wanted = set(self.keys) | set(required)
                    arrays = {k: npz[k] for k in wanted if k in npz.files}
                else:
                    arrays = np.load(f, allow_pickle=False)

            self._chunks.append(arrays)

            shapes = arrays["shapes"]
            offsets = arrays["offsets"]
            n = len(shapes)

            # ----- valid-pixel filter (mirrors fixed-pipeline) -----
            # Drop reflections whose final training mask has < min_valid_pixels
            # True voxels. This removes fully-overlapped reflections + things
            # the encoder/loss can't usefully fit.
            if "mask" in arrays:
                m_flat = arrays["mask"]  # (total_voxels,) bool
                # per-reflection valid count via offsets
                # numpy add.reduceat on bool->int gives sums between offsets
                valid_per_refl = np.add.reduceat(
                    m_flat.astype(np.int64), offsets[:-1]
                )
            else:
                # No mask available — keep everything
                valid_per_refl = np.full(n, self.min_valid_pixels, dtype=np.int64)

            keep_mask = valid_per_refl >= self.min_valid_pixels
            n_drop = int((~keep_mask).sum())
            if n_drop > 0:
                logger.info(
                    "  chunk %d: dropping %d / %d reflections with < %d valid pixels",
                    ci, n_drop, n, self.min_valid_pixels,
                )

            for li in range(n):
                if keep_mask[li]:
                    self._global_index.append((ci, li))

            vols = (shapes[:, 0].astype(np.int64)
                    * shapes[:, 1].astype(np.int64)
                    * shapes[:, 2].astype(np.int64))
            self._volumes.append(vols[keep_mask])

        self._volumes = np.concatenate(self._volumes)

        logger.info(
            "RaggedShoeboxDataset: %d reflections kept across %d chunks "
            "(eager=%s, transform=%s, min_valid_pixels=%d)",
            len(self._global_index), len(self._chunks), eager, self.transform,
            self.min_valid_pixels,
        )

        # Build / load per-chunk artifact masks (raw > max_count outside
        # foreground). Cached to disk under chunks_dir as
        # `<chunk_name>.artifact_<thresh>_pf<0|1>.npy`. With caches present
        # the per-getitem path skips the comparison entirely — useful for
        # large datasets where __getitem__ throughput matters.
        if self.max_count is not None:
            self._artifact_caches = [
                self._load_or_build_artifact_cache(chunk_files[ci], ci)
                for ci in range(len(self._chunks))
            ]
        else:
            self._artifact_caches = [None] * len(self._chunks)

        # If we didn't find a stats file matching the chosen transform,
        # compute streaming (mean, var) over valid voxels on the fly, then
        # cache them to disk so subsequent runs skip this work.
        # Skip for log1p — the literature recipe is to feed raw log1p(x)
        # into the encoder without standardization (see comment above).
        if (
            self._stats_resolved_path is None
            and self.transform not in ("none", "log1p")
        ):
            cache_name = f"{self.transform}_stats.pt"
            cache_path = parent_dir / cache_name
            mean, var = self._compute_streaming_stats(self.transform)
            self._stat_mean = float(mean)
            self._stat_std = float(np.sqrt(max(var, 1e-12)))
            logger.info(
                "Computed on-the-fly %s stats: mean=%.3f, std=%.3f  "
                "(%d valid voxels)",
                self.transform, self._stat_mean, self._stat_std,
                self._stats_n_valid,
            )
            try:
                torch.save(
                    torch.tensor([float(mean), float(var)], dtype=torch.float32),
                    cache_path,
                )
                logger.info("Cached %s stats to %s", self.transform, cache_path)
                self._stats_resolved_path = str(cache_path)
            except OSError as exc:
                # Read-only mount or similar — proceed without caching.
                logger.warning("Could not cache stats to %s: %s", cache_path, exc)

    def _artifact_cache_path(self, chunk_path: Path) -> Path:
        """Filename encodes the threshold and protect_foreground flag, so
        changing either invalidates the cache without manual cleanup."""
        thresh = int(self.max_count) if self.max_count is not None else 0
        pf = 1 if self.protect_foreground else 0
        return chunk_path.with_name(
            f"{chunk_path.stem}.artifact_{thresh}_pf{pf}.npy"
        )

    def _load_or_build_artifact_cache(
        self, chunk_path: Path, chunk_idx: int
    ) -> "np.ndarray | None":
        """Return a per-voxel bool array marking artifact pixels (above
        max_count and, if protect_foreground=True, outside DIALS Foreground).

        Loaded from a sibling `.artifact_<thresh>_pf<0|1>.npy` cache when
        present; otherwise computed once and saved for next time.
        """
        cache_path = self._artifact_cache_path(chunk_path)
        if cache_path.exists():
            try:
                arr = np.load(cache_path, mmap_mode="r")
                # Sanity: cache size matches data size in this chunk.
                expected_n = self._chunks[chunk_idx]["data"].shape[0]
                if arr.shape == (expected_n,):
                    logger.info("Loaded artifact cache: %s", cache_path.name)
                    return np.asarray(arr)  # materialize for fast indexing
                logger.warning(
                    "Stale artifact cache at %s (size mismatch); rebuilding.",
                    cache_path,
                )
            except Exception as exc:
                logger.warning("Failed to load %s: %s; rebuilding.", cache_path, exc)

        # Build it
        c = self._chunks[chunk_idx]
        data = c["data"]
        above = data > self.max_count
        if self.protect_foreground and "foreground" in c:
            artifact = above & ~c["foreground"].astype(bool)
        else:
            artifact = above

        n_artifact = int(artifact.sum())
        logger.info(
            "Built artifact cache for chunk %d: %d / %d voxels above %.0f%s",
            chunk_idx, n_artifact, artifact.size, self.max_count,
            " (protected by foreground)" if self.protect_foreground else "",
        )
        try:
            np.save(cache_path, artifact)
        except OSError as exc:
            logger.warning("Could not save artifact cache to %s: %s", cache_path, exc)
        return artifact

    def _compute_streaming_stats(self, transform: str) -> tuple[float, float]:
        """One streaming pass over chunks to compute (mean, var) of the
        chosen transform applied to valid voxels of *kept* reflections
        (those that survive min_valid_pixels filter and max_count clipping).

        Vectorized — no Python per-reflection loop. Roughly 100x faster
        than the naive version on 1M-reflection datasets."""
        sum_v = 0.0
        sumsq_v = 0.0
        n_total = 0

        for c in self._chunks:
            data = c["data"]
            offsets = c["offsets"]
            n_refl = len(offsets) - 1
            mask = c["mask"] if "mask" in c else None

            # Per-reflection valid-pixel count, then the keep mask
            if mask is not None:
                valid_per_refl = np.add.reduceat(
                    mask.astype(np.int64), offsets[:-1]
                )
            else:
                valid_per_refl = np.full(
                    n_refl, self.min_valid_pixels, dtype=np.int64
                )
            keep_refl = valid_per_refl >= self.min_valid_pixels

            # Project keep_refl to a per-voxel array via offsets diff
            refl_size = np.diff(offsets).astype(np.int64)
            voxel_in_kept = np.repeat(keep_refl, refl_size)

            # Combine: voxel is "use" iff DIALS-valid AND in kept reflection
            if mask is not None:
                use = voxel_in_kept & mask.astype(bool)
            else:
                use = voxel_in_kept

            d = data[use].astype(np.float64)
            if d.size == 0:
                continue
            if self.max_count is not None:
                np.clip(d, 0, self.max_count, out=d)
            else:
                np.clip(d, 0, None, out=d)

            if transform == "anscombe":
                v = 2.0 * np.sqrt(d + 0.375)
            elif transform == "log1p":
                v = np.log1p(d)
            else:
                v = d

            sum_v += float(v.sum())
            sumsq_v += float((v * v).sum())
            n_total += int(v.size)

        self._stats_n_valid = n_total
        if n_total == 0:
            return 0.0, 1.0
        mean = sum_v / n_total
        var = sumsq_v / n_total - mean * mean
        return mean, max(var, 0.0)

    def __len__(self):
        return len(self._global_index)

    @property
    def volumes(self) -> np.ndarray:
        """Voxel count per reflection, aligned with dataset index order."""
        return self._volumes

    def __getitem__(self, idx):
        ci, li = self._global_index[idx]
        c = self._chunks[ci]

        start = int(c["offsets"][li])
        end = int(c["offsets"][li + 1])
        D, H, W = (int(x) for x in c["shapes"][li])

        # ---------- raw counts (for Poisson NLL) ----------
        raw_np = c["data"][start:end].reshape(D, H, W).astype(np.float32, copy=False)
        raw = torch.from_numpy(np.ascontiguousarray(raw_np))

        # ---------- mask and other per-voxel bool arrays ----------
        item = {"counts": raw}
        for k in self.keys:
            if k == "data":
                continue
            arr = c[k][start:end].reshape(D, H, W)
            item[k] = torch.from_numpy(np.ascontiguousarray(arr))

        # ---------- saturation / artifact mask ----------
        # Neutralize pixels whose raw count exceeds max_count. With
        # protect_foreground=True we keep DIALS-Foreground brights (real
        # Bragg signal) and only mask off-Foreground brights (hot pixels,
        # ice rings, neighbouring-spot bleed). The decision is precomputed
        # per chunk in `_artifact_caches[ci]` (a flat bool array over the
        # chunk's voxel buffer), so the per-getitem cost is just a slice.
        if self.max_count is not None and "mask" in item:
            cache = self._artifact_caches[ci]
            if cache is not None:
                artifact = torch.from_numpy(
                    np.ascontiguousarray(cache[start:end].reshape(D, H, W))
                )
            else:
                # Fallback: compute live (used if cache failed to build)
                above = raw > self.max_count
                if self.protect_foreground and "foreground" in item:
                    artifact = above & ~item["foreground"]
                else:
                    artifact = above
            item["mask"] = item["mask"] & ~artifact
            # Clip raw counts so the Poisson NLL gradient stays bounded
            # even on the genuine bright pixels we kept in the mask.
            raw = raw.clamp(max=self.max_count)
            item["counts"] = raw

        # ---------- standardized data (for encoder input) ----------
        # Apply transform, then standardize against the global (mean, std)
        # of the transformed valid voxels. Stats come from disk if a file
        # matching the transform existed, otherwise computed once at init.
        if self.transform == "anscombe":
            pre = 2.0 * (raw.clamp(min=0) + 0.375).sqrt()
        elif self.transform == "log1p":
            pre = torch.log1p(raw.clamp(min=0))
        else:  # "none"
            pre = raw
        std = (pre - self._stat_mean) / self._stat_std

        # Zero out masked voxels in the standardized view so padded + dead
        # pixels don't propagate spurious activations through conv biases.
        if "mask" in item:
            std = std * item["mask"].float()
        item["standardized_data"] = std

        # ---------- geometry ----------
        item["shape"] = (D, H, W)
        item["bbox"] = torch.from_numpy(np.ascontiguousarray(c["bboxes"][li]))
        refl_id = int(c["refl_ids"][li])
        item["refl_id"] = refl_id

        # ---------- per-reflection scalar metadata ----------
        if "d" in self.metadata:
            item["d"] = float(self.metadata["d"][refl_id])
        for k in self._extra_keys:
            if k in self.metadata:
                item[k] = float(self.metadata[k][refl_id])
        if self.group_labels is not None:
            item["group_label"] = int(self.group_labels[refl_id])
        else:
            item["group_label"] = 0

        return item


_SCALAR_METADATA_KEYS = ("d", "group_label")
_NON_VOXEL_KEYS = {"shape", "bbox", "refl_id"} | set(_SCALAR_METADATA_KEYS)


def pad_collate_ragged(batch, pad_values: Optional[dict] = None):
    """Pad variable-size shoeboxes to (Dmax, Hmax, Wmax) within the batch.

    Returns a dict with:
        counts:             (B, Dmax, Hmax, Wmax)   float — raw pixel data (Poisson target)
        standardized_data:  (B, Dmax, Hmax, Wmax)   float — anscombe+z-scored (encoder input)
        mask:               (B, Dmax, Hmax, Wmax)   bool — DIALS mask ∧ real-region
        shapes:             (B, 3)                   int32 — (D, H, W) per refl
        bboxes:             (B, 6)                   int32 — DIALS bbox
        refl_ids:           (B,)                     int64
        metadata: {
            'd':            (B,)                     float  — per-refl resolution
            'group_label':  (B,)                     int64  — resolution bin index
            ... (any extra_metadata_keys in the dataset)
        }
    """
    pad_values = pad_values or {}
    B = len(batch)
    Ds, Hs, Ws = zip(*(b["shape"] for b in batch))
    Dmax, Hmax, Wmax = max(Ds), max(Hs), max(Ws)

    # Detect per-voxel keys by filtering out geometry + scalar metadata +
    # anything else the dataset attaches as a scalar.
    per_voxel_keys = []
    scalar_keys = []
    for k, v in batch[0].items():
        if k in _NON_VOXEL_KEYS:
            continue
        if torch.is_tensor(v) and v.ndim == 3:
            per_voxel_keys.append(k)
        else:
            scalar_keys.append(k)

    out = {}
    for k in per_voxel_keys:
        ref = batch[0][k]
        pad_v = pad_values.get(k, 0)
        padded = torch.full(
            (B, Dmax, Hmax, Wmax), pad_v,
            dtype=ref.dtype,
            device=ref.device,
        )
        for i, b in enumerate(batch):
            D, H, W = b["shape"]
            padded[i, :D, :H, :W] = b[k]
        out[k] = padded

    # Intersect the per-voxel mask with the real-region mask so padded voxels
    # are always invalid.
    if "mask" in out:
        valid_region = torch.zeros(B, Dmax, Hmax, Wmax, dtype=torch.bool)
        for i, (D, H, W) in enumerate(zip(Ds, Hs, Ws)):
            valid_region[i, :D, :H, :W] = True
        out["mask"] = out["mask"].bool() & valid_region

    out["shapes"] = torch.tensor([list(b["shape"]) for b in batch], dtype=torch.int32)
    out["bboxes"] = torch.stack([b["bbox"] for b in batch]).to(torch.int32)
    out["refl_ids"] = torch.tensor([b["refl_id"] for b in batch], dtype=torch.int64)

    # ---------- scalar per-refl metadata dict ----------
    metadata = {}
    if "d" in batch[0]:
        metadata["d"] = torch.tensor(
            [b["d"] for b in batch], dtype=torch.float32
        )
    if "group_label" in batch[0]:
        metadata["group_label"] = torch.tensor(
            [b["group_label"] for b in batch], dtype=torch.int64
        )
    # Any extras surfaced by the dataset that aren't geometry/voxel/d/group_label
    for k in scalar_keys:
        if k in ("d", "group_label"):
            continue
        vals = [b[k] for b in batch]
        if all(isinstance(v, (int, float)) for v in vals):
            metadata[k] = torch.tensor(vals, dtype=torch.float32)
    if metadata:
        out["metadata"] = metadata

    return out


class VolumeBucketSampler(Sampler):
    """Yields batch indices grouped by similar voxel count.

    Motivation: with random shuffling, a batch can mix a small reflection
    (~500 voxels) with a big one (~12000 voxels). The collate pads everyone
    up to the big one's shape, wasting 24× the memory on the small reflection.
    Grouping by volume keeps padding tight while still giving epoch-scale
    shuffling by permuting bucket order and within-bucket order each epoch.

    Args:
        volumes: (N,) voxel count per reflection (from dataset.volumes).
        batch_size
        bucket_mult: each bucket holds batch_size * bucket_mult reflections.
                     Larger -> more within-bucket shuffling, more padding waste.
                     Smaller -> tighter padding, less shuffling.
        shuffle: permute bucket order and within-bucket order each epoch.
        drop_last: drop trailing partial batch in each bucket.
    """

    def __init__(
        self,
        volumes: np.ndarray,
        batch_size: int,
        bucket_mult: int = 50,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.batch_size = batch_size
        self.bucket_mult = bucket_mult
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._order_by_volume = np.argsort(volumes, kind="stable")
        self._n = len(volumes)

    def __iter__(self):
        bucket = self.batch_size * self.bucket_mult
        buckets = [self._order_by_volume[i:i + bucket]
                   for i in range(0, self._n, bucket)]
        if self.shuffle:
            random.shuffle(buckets)
        for b in buckets:
            if self.shuffle:
                b = b.copy()
                np.random.shuffle(b)
            end = len(b) - (len(b) % self.batch_size) if self.drop_last else len(b)
            for i in range(0, end, self.batch_size):
                batch = b[i:i + self.batch_size]
                if len(batch) == 0:
                    continue
                yield batch.tolist()

    def __len__(self):
        if self.drop_last:
            return self._n // self.batch_size
        return (self._n + self.batch_size - 1) // self.batch_size


class RaggedShoeboxDataModule:
    """Lightweight DataModule — not pytorch_lightning-typed, but same shape.

    Handles train/val/test split on the global reflection index (not chunk-local,
    so splits stay random across the dataset) and wires bucket sampling.
    """

    def __init__(
        self,
        data_dir=None,
        chunks_dir=None,
        batch_size: int = 64,
        val_frac: float = 0.05,
        test_frac: float = 0.05,
        num_workers: int = 2,
        bucket_mult: int = 50,
        keys=DEFAULT_KEYS,
        seed: int = 0,
        eager: bool = True,
        anscombe: bool = True,
        metadata_path=None,
        stats_path=None,
        group_labels_path=None,
        extra_metadata_keys: tuple[str, ...] = (),
        min_valid_pixels: int = 10,
        max_count: float | None = None,
        transform: str = "anscombe",
        protect_foreground: bool = True,
        **_unused_kwargs,
    ):
        # Resolve chunks_dir from data_dir if not given. data_dir mirrors the
        # fixed-pipeline convention so prepare_priors and other utilities can
        # find their inputs (group_labels.pt, stats, etc.) under one root.
        if chunks_dir is None:
            if data_dir is None:
                raise ValueError(
                    "RaggedShoeboxDataModule needs either `data_dir` or "
                    "`chunks_dir`. data_dir is preferred — chunks/ is "
                    "expected to live directly under it."
                )
            chunks_dir = Path(data_dir) / "chunks"
        elif data_dir is None:
            data_dir = Path(chunks_dir).parent

        self.data_dir = Path(data_dir)
        self.chunks_dir = Path(chunks_dir)
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.num_workers = num_workers
        self.bucket_mult = bucket_mult
        self.keys = keys
        self.seed = seed
        self.eager = eager
        self.anscombe = anscombe
        self.metadata_path = metadata_path
        self.stats_path = stats_path
        self.group_labels_path = group_labels_path
        self.extra_metadata_keys = extra_metadata_keys
        self.min_valid_pixels = min_valid_pixels
        self.max_count = max_count
        self.transform = transform
        self.protect_foreground = protect_foreground

    def setup(self, stage=None):
        self.dataset = RaggedShoeboxDataset(
            self.chunks_dir,
            metadata_path=self.metadata_path,
            stats_path=self.stats_path,
            group_labels_path=self.group_labels_path,
            anscombe=self.anscombe,
            keys=self.keys,
            eager=self.eager,
            extra_metadata_keys=self.extra_metadata_keys,
            min_valid_pixels=self.min_valid_pixels,
            max_count=self.max_count,
            transform=self.transform,
            protect_foreground=self.protect_foreground,
        )
        n = len(self.dataset)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_val = int(n * self.val_frac)
        n_test = int(n * self.test_frac)
        self.test_idx = perm[:n_test]
        self.val_idx = perm[n_test:n_test + n_val]
        self.train_idx = perm[n_test + n_val:]

        vols = self.dataset.volumes
        self._train_vols = vols[self.train_idx]
        self._val_vols = vols[self.val_idx]
        self._test_vols = vols[self.test_idx]

    def _make_loader(self, indices, volumes, shuffle):
        sampler = VolumeBucketSampler(
            volumes=volumes,
            batch_size=self.batch_size,
            bucket_mult=self.bucket_mult,
            shuffle=shuffle,
        )
        # Sampler yields *positions into the subset*; we need to map back
        # to dataset indices. Use a Subset-style proxy.
        subset = _IndexedSubset(self.dataset, indices)
        return DataLoader(
            subset,
            batch_sampler=sampler,
            collate_fn=pad_collate_ragged,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_idx, self._train_vols, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_idx, self._val_vols, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_idx, self._test_vols, shuffle=False)


class _IndexedSubset(Dataset):
    """Like torch.utils.data.Subset but materializes as a clean Dataset that
    the BatchSampler (which yields subset-local indices) can address directly."""

    def __init__(self, dataset: Dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[int(self.indices[i])]
