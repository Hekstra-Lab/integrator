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
        keys: which per-voxel arrays to expose per item. "data" and "mask" are
              the defaults; others in the chunk file (foreground/background/
              overlapped) can be added if needed.
        eager: if True, load all chunk arrays into RAM at construction time.
               Fastest __getitem__, uses more memory. If False, keep NpzFile
               handles and slice on demand (slower but bounded RAM).
        float_data: if True, cast data to float32 in __getitem__. If False,
                    keep the native dtype (e.g. uint16) and cast later.
    """

    def __init__(
        self,
        chunks_dir,
        keys=DEFAULT_KEYS,
        eager: bool = True,
        float_data: bool = True,
    ):
        self.keys = tuple(dict.fromkeys(("data",) + tuple(keys)))  # dedupe, data first
        self.eager = eager
        self.float_data = float_data

        chunk_files = sorted(Path(chunks_dir).glob("chunk_*.npz"))
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
                    # Load everything we'll need into RAM
                    wanted = set(self.keys) | set(required)
                    arrays = {k: npz[k] for k in wanted if k in npz.files}
                else:
                    # Re-open lazily — npz must be held open for slicing
                    arrays = np.load(f, allow_pickle=False)  # NpzFile

            self._chunks.append(arrays)

            shapes = arrays["shapes"] if eager else arrays["shapes"]
            offsets = arrays["offsets"] if eager else arrays["offsets"]
            n = len(shapes)
            self._global_index.extend((ci, li) for li in range(n))
            # Per-refl voxel count = shape product (equivalent to offsets diff)
            vols = (shapes[:, 0].astype(np.int64)
                    * shapes[:, 1].astype(np.int64)
                    * shapes[:, 2].astype(np.int64))
            self._volumes.append(vols)

        self._volumes = np.concatenate(self._volumes)

        logger.info(
            "RaggedShoeboxDataset: %d reflections across %d chunks (eager=%s)",
            len(self._global_index), len(self._chunks), eager,
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

        item = {}
        data_np = c["data"][start:end].reshape(D, H, W)
        if self.float_data:
            data_np = data_np.astype(np.float32, copy=False)
        item["data"] = torch.from_numpy(np.ascontiguousarray(data_np))

        for k in self.keys:
            if k == "data":
                continue
            arr = c[k][start:end].reshape(D, H, W)
            item[k] = torch.from_numpy(np.ascontiguousarray(arr))

        item["shape"] = (D, H, W)
        item["bbox"] = torch.from_numpy(np.ascontiguousarray(c["bboxes"][li]))
        item["refl_id"] = int(c["refl_ids"][li])
        return item


def pad_collate_ragged(batch, pad_values: Optional[dict] = None):
    """Pad variable-size shoeboxes to (Dmax, Hmax, Wmax) within the batch.

    Returns a dict with:
        data:     (B, Dmax, Hmax, Wmax)     float or int, depending on dataset
        mask:     (B, Dmax, Hmax, Wmax)     bool — original mask AND "is-real-pixel"
        shapes:   (B, 3)                     int32 — original (D, H, W) per refl
        bboxes:   (B, 6)                     int32 — DIALS bbox
        refl_ids: (B,)                       int64
        (plus any extra per-voxel keys from the dataset)

    The returned `mask` is the intersection of the per-voxel mask (from DIALS)
    and the "is this a real (not padded) voxel" mask. Use it directly in the
    loss.
    """
    pad_values = pad_values or {}
    B = len(batch)
    Ds, Hs, Ws = zip(*(b["shape"] for b in batch))
    Dmax, Hmax, Wmax = max(Ds), max(Hs), max(Ws)

    per_voxel_keys = [k for k in batch[0].keys()
                      if k not in ("shape", "bbox", "refl_id")]

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

    # Ensure `mask` exists and also marks padded voxels as invalid.
    if "mask" in out:
        valid_region = torch.zeros(B, Dmax, Hmax, Wmax, dtype=torch.bool)
        for i, (D, H, W) in enumerate(zip(Ds, Hs, Ws)):
            valid_region[i, :D, :H, :W] = True
        out["mask"] = out["mask"].bool() & valid_region

    out["shapes"] = torch.tensor([list(b["shape"]) for b in batch], dtype=torch.int32)
    out["bboxes"] = torch.stack([b["bbox"] for b in batch]).to(torch.int32)
    out["refl_ids"] = torch.tensor([b["refl_id"] for b in batch], dtype=torch.int64)
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
        chunks_dir,
        batch_size: int = 64,
        val_frac: float = 0.05,
        test_frac: float = 0.05,
        num_workers: int = 2,
        bucket_mult: int = 50,
        keys=DEFAULT_KEYS,
        seed: int = 0,
        eager: bool = True,
    ):
        self.chunks_dir = chunks_dir
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.num_workers = num_workers
        self.bucket_mult = bucket_mult
        self.keys = keys
        self.seed = seed
        self.eager = eager

    def setup(self, stage=None):
        self.dataset = RaggedShoeboxDataset(
            self.chunks_dir, keys=self.keys, eager=self.eager
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
