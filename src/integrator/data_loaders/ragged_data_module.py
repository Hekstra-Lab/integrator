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
    ):
        self.keys = tuple(dict.fromkeys(("data",) + tuple(keys)))  # dedupe, data first
        self.eager = eager
        self.anscombe = anscombe
        self.min_valid_pixels = int(min_valid_pixels)

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
        if stats_path is None:
            default_name = "anscombe_stats.pt" if anscombe else "stats.pt"
            candidate = parent_dir / default_name
            stats_path = candidate if candidate.exists() else None
        if stats_path is not None:
            s = torch.load(stats_path, weights_only=True)
            self._stat_mean = float(s[0])
            self._stat_std = float(torch.sqrt(torch.clamp(s[1], min=1e-12)))
        else:
            # Not standardizing — encoder will see raw float counts.
            self._stat_mean = 0.0
            self._stat_std = 1.0
            logger.warning(
                "No stats file found; standardized_data will equal raw data. "
                "Consider running mksbox-dials to produce stats.pt / anscombe_stats.pt."
            )

        # ---------- group_labels (resolution bins) ----------
        if group_labels_path is None:
            matches = sorted(parent_dir.glob("group_labels_*.pt"))
            group_labels_path = matches[0] if matches else None
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
            "(eager=%s, anscombe=%s, min_valid_pixels=%d)",
            len(self._global_index), len(self._chunks), eager, anscombe,
            self.min_valid_pixels,
        )

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

        # ---------- standardized data (for encoder input) ----------
        if self.anscombe:
            pre = 2.0 * (raw.clamp(min=0) + 0.375).sqrt()
        else:
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
