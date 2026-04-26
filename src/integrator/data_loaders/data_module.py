import logging
import os
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)
import pytorch_lightning as pl
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    TensorDataset,
    random_split,
)


def _load_shoebox_array(path, weights_only=True):
    """Load counts/masks from either new .npy (refltorch mksbox >= memmap era)
    or legacy .pt (torch.save). If a sibling .npy exists next to the requested
    .pt path, prefer it. Returns torch.Tensor either way.

    For very large .npy datasets that don't fit in RAM, swap the materialization
    line below for a lazy Dataset that slices the memmap in __getitem__.
    """
    p = Path(path)
    npy = p.with_suffix(".npy")
    if npy.exists():
        arr = np.load(npy)
        if arr.dtype == np.uint16:
            arr = arr.astype(np.int32)
        return torch.from_numpy(arr)
    try:
        return torch.load(p, weights_only=weights_only)
    except TypeError:
        return torch.load(p)


SIMULATED_COLS = [
    "shoebox_median",
    "shoebox_var",
    "shoebox_mean",
    "shoebox_min",
    "shoebox_max",
    "intensity",
    "background",
    "refl_ids",
    "is_test",
    "group_label",
    "profile_group_label",
]

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
    "group_label",
    "profile_group_label",
]


class IntegratorDataset(Dataset):
    def __init__(
        self,
        counts,
        standardized_counts,
        masks,
        reference,
        column_names: list = DEFAULT_DS_COLS,
    ):
        self.counts = counts
        self.standardized_counts = standardized_counts
        self.masks = masks
        self.reference = reference
        self.column_names = column_names

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        counts = self.counts[idx]
        standardized_counts = self.standardized_counts[idx]
        masks = self.masks[idx]

        meta = {
            k: self.reference[k][idx]
            for k in self.column_names
            if k in self.reference
        }

        return counts, standardized_counts, masks, meta


# Filter to remove reflections with variance = -1
def _remove_flagged_variance(
    counts: torch.Tensor,
    masks: torch.Tensor,
    metadata: dict,
    filter_key: str = "intensity.prf.variance",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    bad = metadata[filter_key] < 0
    n_bad = bad.sum().item()
    if n_bad > 0:
        logger.info("Removed %d reflections with %s < 0", n_bad, filter_key)
    counts = counts[~bad]
    masks = masks[~bad]
    metadata = {k: v[~bad] for k, v in metadata.items()}
    return counts, masks, metadata


class ShoeboxDataModule2D(pl.LightningDataModule):
    """

    Attributes:
        data_dir:
        batch_size:
        val_split:
        test_split:
        include_test:
        subset_size:
        single_sample_index:
        num_workers:
        cutoff:
        full_dataset:

        shoebox_file_names:
        H:
        W:
        Z:
        standardized_counts:
        get_dxyz:
        anscombe: Boolean indicating whether to use Anscome transformation
        full_dataset:
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 100,
        val_split: float | None = 0.2,
        test_split: float | None = 0.1,
        num_workers: int = 3,
        include_test: bool = False,
        subset_size: int | None = None,
        single_sample_index: None = None,
        cutoff=None,
        persistent_workers: bool = False,
        shoebox_file_names={
            "counts": "counts.pt",
            # "metadata": "metadata.pt",
            "masks": "masks.pt",
            "stats": "stats.pt",
            "reference": "reference.pt",
            "x_coords": None,
            "y_coords": None,
            "standardized_counts": None,
        },
        refl_file=None,
        D=1,
        H=21,
        W=21,
        get_dxyz=False,
        anscombe=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.include_test = include_test
        self.subset_size = subset_size
        self.single_sample_index = single_sample_index
        self.num_workers = num_workers
        self.cutoff = cutoff
        self.full_dataset = None  # Will store the full dataset
        self.shoebox_file_names = shoebox_file_names
        self.H = H
        self.W = W
        self.standardized_counts = shoebox_file_names["standardized_counts"]
        self.x_coords = shoebox_file_names["x_coords"]
        self.y_coords = shoebox_file_names["y_coords"]
        self.get_dxyz = get_dxyz
        self.anscombe = anscombe

    def setup(self, stage=None):
        counts = _load_shoebox_array(
            os.path.join(self.data_dir, self.shoebox_file_names["counts"]),
        )
        masks = _load_shoebox_array(
            os.path.join(self.data_dir, self.shoebox_file_names["masks"]),
        )
        stats = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["stats"]),
        )
        reference = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["reference"]),
        )

        self.dataset_mean = stats[0]
        self.dataset_var = stats[1]
        all_dead = masks.sum(-1) < 10

        # filter out samples with all dead pixels
        counts = counts[~all_dead]
        masks = masks[~all_dead]
        reference = reference[~all_dead]

        # dataset
        processed_counts = counts.clone()
        processed_counts[~masks.bool()] = self.dataset_mean.round()

        if self.anscombe:
            ans = 2 * torch.sqrt(processed_counts + (3.0 / 8.0))
            standardized_counts = (ans - stats[0]) / stats[1].sqrt()
        else:
            standardized_counts = ((processed_counts) - stats[0]) / stats[
                1
            ].sqrt()

        if self.x_coords is not None:
            x = torch.load(os.path.join(self.data_dir, self.x_coords))[
                ~all_dead
            ]
            y = torch.load(os.path.join(self.data_dir, self.y_coords))[
                ~all_dead
            ]

            standardized_counts = torch.stack(
                (standardized_counts, x, y)
            ).permute(1, 0, 2)

        self.full_dataset = TensorDataset(
            processed_counts, standardized_counts, masks, reference
        )
        # If single_sample_index is specified, use only that sample
        if self.single_sample_index is not None:
            self.full_dataset = Subset(
                self.full_dataset, [self.single_sample_index]
            )

        # Optionally, create a subset of the dataset
        if self.subset_size is not None and self.subset_size < len(
            self.full_dataset
        ):
            indices = torch.randperm(len(self.full_dataset))[
                : self.subset_size
            ]
            self.full_dataset = Subset(self.full_dataset, indices=indices)

        # Calculate lengths for train/val/test splits
        total_size: int = len(self.full_dataset)
        val_size = int(total_size * self.val_split)
        if self.include_test:
            test_size = int(total_size * self.test_split)
            train_size = total_size - val_size - test_size
        else:
            test_size = 0
            train_size = total_size - val_size

        # Split the dataset
        if self.include_test:
            self.train_dataset, self.val_dataset, self.test_dataset = (
                random_split(
                    self.full_dataset, [train_size, val_size, test_size]
                )
            )
        else:
            self.train_dataset, self.val_dataset = random_split(
                self.full_dataset, [train_size, val_size]
            )
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self):
        if self.include_test:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
            )
        else:
            return None

    def predict_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )


class ShoeboxDataModule(pl.LightningDataModule):
    """

    Attributes:
        data_dir:
        batch_size:
        val_split:
        test_split:
        include_test:
        subset_size:
        single_sample_index:
        num_workers:
        cutoff:
        min_valid_pixels:
        full_dataset:

        shoebox_file_names:
        H:
        W:
        D:
        standardized_counts:
        get_dxyz:
        anscombe: Boolean indicating whether to use Anscombe transformation
        full_dataset:
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 10,
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 3,
        include_test: bool = False,
        subset_size: int | None = None,
        single_sample_index=None,
        cutoff: float | None = None,
        min_valid_pixels: int = 10,
        persistent_workers: bool = True,
        shoebox_file_names={
            "counts": "counts.pt",
            "masks": "masks.pt",
            "stats": "stats.pt",
            "reference": "reference.pt",
            "standardized_counts": None,
        },
        refl_file: Path | None = None,
        H: int = 21,
        W: int = 21,
        D: int = 3,
        get_dxyz: bool = False,
        anscombe: bool = False,
        transform: str | None = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.include_test = include_test
        self.subset_size = subset_size
        self.single_sample_index = single_sample_index
        self.num_workers = num_workers
        self.cutoff = cutoff
        self.min_valid_pixels = min_valid_pixels
        self.full_dataset = None  # Will store the full dataset
        self.shoebox_file_names = shoebox_file_names
        self.H = H
        self.W = W
        self.D = D
        self.standardized_counts = shoebox_file_names["standardized_counts"]
        self.get_dxyz = get_dxyz
        self.anscombe = anscombe
        # Resolve encoder-input transform. Explicit `transform` wins; absent
        # that we honor the legacy `anscombe` flag for backward compat.
        # log1p deliberately skips global z-scoring — matches scvi-tools'
        # `log_variational=True` recipe (encoder GroupNorm handles
        # per-batch normalization). On heavy-tailed counts the global std
        # would otherwise be dominated by the bright Bragg peaks and
        # squash bulk voxels near zero.
        if transform is None:
            self.transform = "anscombe" if anscombe else "none"
        elif transform not in ("anscombe", "log1p", "none"):
            raise ValueError(
                f"transform must be 'anscombe', 'log1p', or 'none'; "
                f"got {transform!r}"
            )
        else:
            self.transform = transform

    def setup(self, stage=None):
        counts = _load_shoebox_array(
            os.path.join(self.data_dir, self.shoebox_file_names["counts"])
        ).squeeze(-1)
        masks = _load_shoebox_array(
            os.path.join(self.data_dir, self.shoebox_file_names["masks"])
        ).squeeze(-1)
        stats = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["stats"])
        )
        reference = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["reference"])
        )

        # Filter out reflections with too few valid pixels
        all_dead = masks.sum(-1) < self.min_valid_pixels
        n_dead = all_dead.sum().item()
        if n_dead > 0:
            logger.info(
                "Removed %d reflections with < %d valid pixels",
                n_dead,
                self.min_valid_pixels,
            )
        counts = counts[~all_dead]
        masks = masks[~all_dead]
        reference = {k: v[~all_dead] for k, v in reference.items()}

        counts, masks, reference = _remove_flagged_variance(
            counts, masks, reference
        )

        # Apply resolution cutoff before standardization
        if self.cutoff is not None:
            selection = reference["d"] < self.cutoff
            n_cut = (~selection).sum().item()
            if n_cut > 0:
                logger.info(
                    "Removed %d reflections with d >= %.2f",
                    n_cut,
                    self.cutoff,
                )
            counts = counts[selection]
            masks = masks[selection]
            reference = {k: v[selection] for k, v in reference.items()}

        # Standardize counts
        if counts.dim() == 2:
            if self.transform == "anscombe":
                anscombe_transformed = 2 * (counts.clamp(min=0) + 0.375).sqrt()
                standardized_counts = (
                    (anscombe_transformed - stats[0]) / stats[1].sqrt()
                ) * masks
            elif self.transform == "log1p":
                # No standardization — encoder GroupNorm handles per-batch
                # normalization (scvi-tools `log_variational=True` recipe).
                standardized_counts = torch.log1p(counts.clamp(min=0)) * masks
            else:
                standardized_counts = ((counts * masks) - stats[0]) / stats[
                    1
                ].sqrt()
        else:
            standardized_counts = (
                (counts[..., -1] * masks) - stats[0]
            ) / stats[1].sqrt()
            # Normalize first three channels of counts
            if counts.dim() >= 3 and counts.size(-1) >= 3:
                counts[:, :, 0] = (
                    2 * (counts[:, :, 0] / (counts[:, :, 0].max() + 1e-8)) - 1
                )
                counts[:, :, 1] = (
                    2 * (counts[:, :, 1] / (counts[:, :, 1].max() + 1e-8)) - 1
                )
                counts[:, :, 2] = (
                    2 * (counts[:, :, 2] / (counts[:, :, 2].max() + 1e-8)) - 1
                )

        self.full_dataset = IntegratorDataset(
            counts,
            standardized_counts,
            masks,
            reference,
        )

        # indicators for reflections flagged for the test set
        is_test = reference["is_test"]

        # indices
        all_indices = torch.arange(len(self.full_dataset))

        # Optionally, create a subset of the dataset
        if self.subset_size is not None and self.subset_size < len(
            self.full_dataset
        ):
            all_indices = all_indices[
                torch.randperm(len(all_indices))[: self.subset_size]
            ]

        # Split the indices using is_test
        test_mask = is_test[all_indices]
        test_idx = all_indices[test_mask]
        train_val_idx = all_indices[~test_mask]

        # test dataset
        self.test_dataset = Subset(self.full_dataset, test_idx.tolist())

        perm = torch.randperm(len(train_val_idx))

        val_size = int(len(train_val_idx) * self.val_split)

        val_idx = train_val_idx[perm[:val_size]]
        train_idx = train_val_idx[perm[val_size:]]

        # Train and validation test sets
        self.val_dataset = Subset(self.full_dataset, val_idx.tolist())
        self.train_dataset = Subset(self.full_dataset, train_idx.tolist())

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.include_test:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            return None

    def predict_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class SimulatedShoeboxLoader(pl.LightningDataModule):
    """

    Attributes:
        data_dir:
        batch_size:
        val_split:
        test_split:
        include_test:
        subset_size:
        single_sample_index:
        num_workers:
        cutoff:
        min_valid_pixels:
        full_dataset:

        shoebox_file_names:
        H:
        W:
        D:
        standardized_counts:
        get_dxyz:
        anscombe: Boolean indicating whether to use Anscombe transformation
        full_dataset:
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 10,
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 3,
        include_test: bool = False,
        subset_size: int | None = None,
        single_sample_index=None,
        cutoff: float | None = None,
        min_valid_pixels: int = 10,
        persistent_workers: bool = True,
        shoebox_file_names={
            "counts": "counts.pt",
            "masks": "masks.pt",
            "stats": "stats.pt",
            "reference": "reference.pt",
            "standardized_counts": None,
        },
        refl_file: Path | None = None,
        H: int = 21,
        W: int = 21,
        D: int = 3,
        get_dxyz: bool = False,
        anscombe: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.include_test = include_test
        self.subset_size = subset_size
        self.single_sample_index = single_sample_index
        self.num_workers = num_workers
        self.cutoff = cutoff
        self.min_valid_pixels = min_valid_pixels
        self.full_dataset = None  # Will store the full dataset
        self.shoebox_file_names = shoebox_file_names
        self.H = H
        self.W = W
        self.D = D
        self.standardized_counts = shoebox_file_names["standardized_counts"]
        self.get_dxyz = get_dxyz
        self.anscombe = anscombe

    def setup(self, stage=None):
        counts = _load_shoebox_array(
            os.path.join(self.data_dir, self.shoebox_file_names["counts"]),
            weights_only=False,
        ).squeeze(-1)
        masks = _load_shoebox_array(
            os.path.join(self.data_dir, self.shoebox_file_names["masks"]),
            weights_only=False,
        ).squeeze(-1)
        stats = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["stats"]),
            weights_only=False,
        )
        reference = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["reference"]),
            weights_only=False,
        )

        # Alias refl_id -> refl_ids for compatibility with callbacks
        if "refl_id" in reference and "refl_ids" not in reference:
            reference["refl_ids"] = reference["refl_id"]

        # Alias radial_bin -> group_label for hierarchical model
        if "radial_bin" in reference and "group_label" not in reference:
            reference["group_label"] = reference["radial_bin"]

        if self.anscombe:
            anscombe_transformed = 2 * (counts.clamp(min=0) + 0.375).sqrt()
            standardized_counts = (
                (anscombe_transformed - stats[0]) / stats[1].sqrt()
            ) * masks
        else:
            standardized_counts = ((counts * masks) - stats[0]) / stats[
                1
            ].sqrt()

        self.full_dataset = IntegratorDataset(
            counts,
            standardized_counts,
            masks,
            reference,
            column_names=SIMULATED_COLS,
        )

        # indicators for reflections flagged for the test set
        is_test = reference["is_test"]

        # indices
        all_indices = torch.arange(len(self.full_dataset))

        # Optionally, create a subset of the dataset
        if self.subset_size is not None and self.subset_size < len(
            self.full_dataset
        ):
            all_indices = all_indices[
                torch.randperm(len(all_indices))[: self.subset_size]
            ]

        # Split the indices using is_test
        test_mask = is_test[all_indices]
        test_idx = all_indices[test_mask]
        train_val_idx = all_indices[~test_mask]

        # test dataset
        self.test_dataset = Subset(self.full_dataset, test_idx.tolist())

        perm = torch.randperm(len(train_val_idx))

        val_size = int(len(train_val_idx) * self.val_split)

        val_idx = train_val_idx[perm[:val_size]]
        train_idx = train_val_idx[perm[val_size:]]

        # Train and validation test sets
        self.val_dataset = Subset(self.full_dataset, val_idx.tolist())
        self.train_dataset = Subset(self.full_dataset, train_idx.tolist())

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.include_test:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            return None

    def predict_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
