import logging
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# LAZY-READING EDIT: read .npy files with mmap_mode='r' so the full arrays
# stay on disk instead of being copied into CPU RAM at setup time.
def _load_shoebox_array(path, weights_only=True):
    """Load counts/masks from .npy lazily, or fall back to eager .pt loading."""
    p = Path(path)
    npy = p.with_suffix(".npy")
    if npy.exists():
        return np.load(npy, mmap_mode="r")

    logger.warning(
        "Lazy loading is only available for .npy files; loading %s eagerly.", p
    )
    try:
        return torch.load(p, weights_only=weights_only)
    except TypeError:
        return torch.load(p)


def _squeeze_last_axis(array):
    """Remove a trailing singleton axis without materializing the whole array."""
    if array.ndim > 0 and array.shape[-1] == 1:
        return array[..., 0]
    return array


def _metadata_value_at(value, idx):
    """Read one metadata value while preserving tensors when possible."""
    return value[idx]


def _metadata_numpy(value):
    """Return a NumPy view/copy suitable for indexing and boolean tests."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _compute_valid_indices(
    masks,
    reference,
    min_valid_pixels,
    resolution_cutoff,
    chunk_size=250_000,
):
    """Build valid row indices in chunks without loading all masks into RAM."""
    n_rows = len(masks)
    variance_key = None
    if "intensity.prf.variance" in reference:
        variance_key = "intensity.prf.variance"
    elif "intensity.sum.variance" in reference:
        variance_key = "intensity.sum.variance"

    variance = (
        _metadata_numpy(reference[variance_key]) if variance_key is not None else None
    )
    d_values = (
        _metadata_numpy(reference["d"])
        if resolution_cutoff is not None
        else None
    )

    valid_chunks = []
    n_dead = 0
    n_bad_variance = 0
    n_cut = 0

    # LAZY-READING EDIT: inspect only a manageable slice of masks at a time.
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        mask_chunk = np.asarray(masks[start:stop])
        keep = mask_chunk.sum(axis=-1) >= min_valid_pixels
        n_dead += int((~keep).sum())

        if variance is not None:
            variance_ok = variance[start:stop] >= 0
            n_bad_variance += int((~variance_ok).sum())
            keep &= variance_ok

        if d_values is not None:
            resolution_ok = d_values[start:stop] < resolution_cutoff
            n_cut += int((~resolution_ok).sum())
            keep &= resolution_ok

        local = np.flatnonzero(keep).astype(np.int64, copy=False)
        if local.size:
            valid_chunks.append(local + start)

    if n_dead:
        logger.info(
            "Removed %d reflections with < %d valid pixels",
            n_dead,
            min_valid_pixels,
        )
    if variance_key is not None and n_bad_variance:
        logger.info(
            "Removed %d reflections with %s < 0",
            n_bad_variance,
            variance_key,
        )
    if resolution_cutoff is not None and n_cut:
        logger.info(
            "Removed %d reflections with d >= %.2f",
            n_cut,
            resolution_cutoff,
        )
    if variance_key is None:
        logger.info("No intensity variance key found; skipping variance filtering")

    if not valid_chunks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(valid_chunks)


# Default columns from rs.io.read_dials_stills
DEFAULT_DS_COLS = [
    "zeta",
    "xyzobs.px.variance.0",
    "xyzobs.px.variance.1",
    "xyzobs.px.variance.2",
    "xyzobs.px.value.0",
    "xyzobs.px.value.1",
    "xyzobs.px.value.2",
    "xyzobs.mm.variance.0",
    "xyzobs.mm.variance.1",
    "xyzobs.mm.variance.2",
    "xyzobs.mm.value.0",
    "xyzobs.mm.value.1",
    "xyzobs.mm.value.2",
    "xyzcal.mm.0",
    "xyzcal.mm.1",
    "xyzcal.mm.2",
    "refl_ids",
    "qe",
    "profile.correlation",
    "partiality",
    "partial_id",
    "panel",
    "num_pixels.valid",
    "num_pixels.foreground",
    "num_pixels.background_used",
    "num_pixels.background",
    "lp",
    "intensity.prf.variance",
    "intensity.prf.value",
    "imageset_id",
    "flags",
    "entering",
    "d",
    "bbox.0",
    "bbox.1",
    "bbox.2",
    "bbox.3",
    "bbox.4",
    "bbox.5",
    "background.sum.variance",
    "background.sum.value",
    "background.mean",
    "s1.0",
    "s1.1",
    "s1.2",
    "xyzcal.px.0",
    "xyzcal.px.1",
    "xyzcal.px.2",
    "intensity.sum.variance",
    "intensity.sum.value",
    "H",
    "K",
    "L",
    "is_test",
    "is_coset",
    "group_label",
    "profile_group_label",
    "image_num",
    "image_id",
    "n_images",
]


class IntegratorDataset(Dataset):
    def __init__(
        self,
        counts,
        masks,
        reference,
        valid_indices,
        stats,
        transform,
        column_names: list = DEFAULT_DS_COLS,
    ):
        self.counts = counts
        self.masks = masks
        self.reference = reference
        self.valid_indices = np.asarray(valid_indices, dtype=np.int64)
        self.stats = torch.as_tensor(stats, dtype=torch.float32)
        self.transform = transform
        self.column_names = column_names

    def __len__(self):
        return len(self.valid_indices)

    def _transform_counts(self, counts, masks):
        """Apply the configured transform to one reflection only."""
        stats = self.stats

        if counts.dim() == 1:
            if self.transform == "anscombe":
                transformed = 2 * (counts.clamp(min=0) + 0.375).sqrt()
                return ((transformed - stats[0]) / stats[1].sqrt()) * masks

            if self.transform == "log1p":
                return torch.log1p(counts.clamp(min=0)) * masks

            if self.transform == "asinh":
                scale = stats[1].sqrt().clamp(min=1e-8)
                return torch.asinh(counts / scale) * masks

            if self.transform == "log_softplus":
                return torch.log(torch.nn.functional.softplus(counts) + 1e-8) * masks

            if self.transform == "sqrt_squareplus":
                b = 4.0
                squareplus = 0.5 * (counts + torch.sqrt(counts * counts + b))
                return torch.sqrt(squareplus + 1e-8) * masks

            return ((counts * masks) - stats[0]) / stats[1].sqrt()

        # Preserve the original behavior for arrays with extra feature channels.
        standardized = ((counts[..., -1] * masks) - stats[0]) / stats[1].sqrt()
        counts = counts.clone()
        if counts.dim() >= 2 and counts.size(-1) >= 3:
            for channel in range(3):
                channel_max = counts[..., channel].max().clamp(min=1e-8)
                counts[..., channel] = 2 * (counts[..., channel] / channel_max) - 1
        return standardized

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        source_idx = int(self.valid_indices[idx])

        # LAZY-READING EDIT: copy only this one row from the disk-backed arrays.
        # The DataLoader later combines individual rows into a batch.
        counts_np = np.array(self.counts[source_idx], copy=True)
        masks_np = np.array(self.masks[source_idx], copy=True)

        if counts_np.dtype == np.uint16:
            counts_np = counts_np.astype(np.int32, copy=False)

        counts = torch.from_numpy(counts_np).to(torch.float32)
        masks = torch.from_numpy(masks_np).bool()
        standardized_counts = self._transform_counts(counts, masks)

        meta = {
            key: _metadata_value_at(self.reference[key], source_idx)
            for key in self.column_names
            if key in self.reference
        }

        return counts, standardized_counts, masks, meta


class IndexedDataset(Dataset):
    """Memory-efficient subset that stores NumPy indices instead of Python lists."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[int(self.indices[idx])]


class RotationDataModule(pl.LightningDataModule):
    """LightningDataModule for rotation-geometry shoebox data."""

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 10,
        validation_split: float = 0.2,
        num_workers: int = 3,
        include_test: bool = False,
        subset_size: int | None = None,
        resolution_cutoff: float | None = None,
        min_valid_pixels: int = 10,
        shoebox_file_names: dict | None = None,
        transform: str | None = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.include_test = include_test
        self.subset_size = subset_size
        self.num_workers = num_workers
        self.resolution_cutoff = resolution_cutoff
        self.min_valid_pixels = min_valid_pixels
        self.full_dataset = None
        self.n_images = None

        if shoebox_file_names is None:
            shoebox_file_names = {
                "counts": "counts.npy",
                "masks": "masks.npy",
                "reference": "metadata.npy",
            }
        self.shoebox_file_names = shoebox_file_names

        transform = transform or "standardization"
        if transform not in (
            "anscombe",
            "log1p",
            "standardization",
            "asinh",
            "log_softplus",
            "sqrt_squareplus",
        ):
            raise ValueError(
                f"transform must be 'anscombe', 'log1p', 'standardization', "
                f"'asinh', 'log_softplus', or 'sqrt_squareplus'; "
                f"got {transform!r}"
            )
        self.transform = transform

    def setup(self, stage=None):
        # LAZY-READING EDIT: these are NumPy memory maps, not full RAM copies.
        counts = _squeeze_last_axis(
            _load_shoebox_array(
                os.path.join(self.data_dir, self.shoebox_file_names["counts"])
            )
        )
        masks = _squeeze_last_axis(
            _load_shoebox_array(
                os.path.join(self.data_dir, self.shoebox_file_names["masks"])
            )
        )

        from integrator.io import load_metadata, read_dataset_spec

        spec = read_dataset_spec(self.data_dir)
        if spec is None:
            raise FileNotFoundError(
                f"dataset.yaml not found in {self.data_dir}; "
                "regenerate the dataset with mksbox"
            )

        stats_key = "anscombe" if self.transform == "anscombe" else "raw"
        stats = torch.tensor(spec["stats"][stats_key], dtype=torch.float32)

        reference = load_metadata(
            os.path.join(self.data_dir, self.shoebox_file_names["reference"])
        )

        if "image_id" in reference:
            reference["image_id"] = torch.as_tensor(
                reference["image_id"], dtype=torch.long
            )
            self.n_images = int(reference["image_id"].max().item()) + 1
        else:
            self.n_images = None

        # LAZY-READING EDIT: keep the original arrays untouched and retain only
        # integer row indices for reflections that pass the filters.
        valid_indices = _compute_valid_indices(
            masks=masks,
            reference=reference,
            min_valid_pixels=self.min_valid_pixels,
            resolution_cutoff=self.resolution_cutoff,
        )

        self.full_dataset = IntegratorDataset(
            counts=counts,
            masks=masks,
            reference=reference,
            valid_indices=valid_indices,
            stats=stats,
            transform=self.transform,
        )

        all_indices = np.arange(len(self.full_dataset), dtype=np.int64)

        if self.subset_size is not None and self.subset_size < len(all_indices):
            all_indices = np.random.default_rng().choice(
                all_indices,
                size=self.subset_size,
                replace=False,
            )

        is_test = reference.get("is_test")
        if is_test is not None:
            source_indices = valid_indices[all_indices]
            is_test_values = _metadata_numpy(is_test)[source_indices].astype(bool)
        else:
            is_test_values = None

        if is_test_values is not None and is_test_values.any():
            test_idx = all_indices[is_test_values]
            train_val_idx = all_indices[~is_test_values]
        else:
            test_idx = np.empty(0, dtype=np.int64)
            train_val_idx = all_indices

        self.test_dataset = IndexedDataset(self.full_dataset, test_idx)

        # This permutation is only an int64 index array, not a copy of counts.
        perm = np.random.default_rng().permutation(len(train_val_idx))
        val_size = int(len(train_val_idx) * self.validation_split)
        val_idx = train_val_idx[perm[:val_size]]
        train_idx = train_val_idx[perm[val_size:]]

        self.val_dataset = IndexedDataset(self.full_dataset, val_idx)
        self.train_dataset = IndexedDataset(self.full_dataset, train_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        if self.include_test:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
        return None

    def predict_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
