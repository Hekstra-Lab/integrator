import logging
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)

logger = logging.getLogger(__name__)


def _load_shoebox_array(path, weights_only=True):
    """Load counts/masks from either new .npy  or .pt. Returns torch.Tensor."""
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
    # Merge / scaling columns (selected only if present in the reference file).
    "miller_idx_friedelized",
    "miller_idx_unfriedelized",
    "centric",
    "friedel_plus",
    "absorption_sh",
    "s_sq",
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
    if filter_key not in metadata:
        return counts, masks, metadata
    bad = metadata[filter_key] < 0
    n_bad = bad.sum().item()
    if n_bad > 0:
        logger.info("Removed %d reflections with %s < 0", n_bad, filter_key)
    counts = counts[~bad]
    masks = masks[~bad]
    metadata = {k: v[~bad] for k, v in metadata.items()}
    return counts, masks, metadata


class RotationDataModule(pl.LightningDataModule):
    """LightningDataModule for rotation-geometry shoebox data.

    Attributes:
        transform: Count transform `anscombe`, `log1p`, or `standardization`.
    """

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
        group_by_asu_id: bool = False,
        group_by_key: str = "miller_idx_friedelized",
        max_obs_per_hkl: int | None = None,
        ice_ring_ranges: list | None = None,
        split_by_miller_idx: bool = False,
        test_split: float = 0.0,
        split_seed: int = 42,
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
        # Grouped (by asu_id / Miller index) batching for the merging models.
        # Defaults keep the standard per-observation behavior.
        self.group_by_asu_id = group_by_asu_id
        self.group_by_key = group_by_key
        self.max_obs_per_hkl = max_obs_per_hkl
        self.ice_ring_ranges = ice_ring_ranges
        # Hold out whole reflections (by group_by_key) for val/test instead of
        # random observations -- the crystallographic free-set convention.
        self.split_by_miller_idx = split_by_miller_idx
        self.test_split = test_split
        self.split_seed = split_seed
        self.full_dataset = None
        if shoebox_file_names is None:
            shoebox_file_names = {
                "counts": "counts.npy",
                "masks": "masks.npy",
                "reference": "metadata.npy",
            }
        self.shoebox_file_names = shoebox_file_names
        transform = transform or "standardization"
        if transform not in ("anscombe", "log1p", "standardization"):
            raise ValueError(
                f"transform must be 'anscombe', 'log1p', or 'standardization'; "
                f"got {transform!r}"
            )
        self.transform = transform

    def setup(self, stage=None):
        counts = _load_shoebox_array(
            os.path.join(self.data_dir, self.shoebox_file_names["counts"])
        ).squeeze(-1)
        masks = _load_shoebox_array(
            os.path.join(self.data_dir, self.shoebox_file_names["masks"])
        ).squeeze(-1)
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
        if self.resolution_cutoff is not None:
            selection = reference["d"] < self.resolution_cutoff
            n_cut = (~selection).sum().item()
            if n_cut > 0:
                logger.info(
                    "Removed %d reflections with d >= %.2f",
                    n_cut,
                    self.resolution_cutoff,
                )
            counts = counts[selection]
            masks = masks[selection]
            reference = {k: v[selection] for k, v in reference.items()}

        # Drop contaminated resolution bands (e.g. ice rings) at load time.
        if self.ice_ring_ranges:
            keep = torch.ones(len(reference["d"]), dtype=torch.bool)
            for lo, hi in self.ice_ring_ranges:
                keep &= ~((reference["d"] >= lo) & (reference["d"] <= hi))
            n_ice = int((~keep).sum())
            if n_ice > 0:
                logger.info(
                    "Removed %d reflections in ice-ring ranges %s",
                    n_ice,
                    self.ice_ring_ranges,
                )
            counts = counts[keep]
            masks = masks[keep]
            reference = {k: v[keep] for k, v in reference.items()}

        if counts.dim() == 2:
            if self.transform == "anscombe":
                anscombe_transformed = 2 * (counts.clamp(min=0) + 0.375).sqrt()
                standardized_counts = (
                    (anscombe_transformed - stats[0]) / stats[1].sqrt()
                ) * masks
            elif self.transform == "log1p":
                standardized_counts = torch.log1p(counts.clamp(min=0)) * masks
            else:
                standardized_counts = ((counts * masks) - stats[0]) / stats[
                    1
                ].sqrt()
        else:
            standardized_counts = (
                (counts[..., -1] * masks) - stats[0]
            ) / stats[1].sqrt()
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

        all_indices = torch.arange(len(self.full_dataset))
        if self.subset_size is not None and self.subset_size < len(
            self.full_dataset
        ):
            all_indices = all_indices[
                torch.randperm(len(all_indices))[: self.subset_size]
            ]

        if self.split_by_miller_idx:
            # Free-set split: hold out whole reflections (by group_by_key) so a
            # reflection's observations never straddle train/val/test. Seeded and
            # reproducible; overrides any per-observation is_test for consistency.
            if self.group_by_key not in reference:
                raise KeyError(
                    f"split_by_miller_idx requires '{self.group_by_key}' in the "
                    "metadata reference (re-run make_shoeboxes --anomalous)."
                )
            groups = reference[self.group_by_key].long()
            uniq, inverse = torch.unique(groups, return_inverse=True)
            gen = torch.Generator().manual_seed(self.split_seed)
            perm = torch.randperm(len(uniq), generator=gen)
            n_test = int(len(uniq) * self.test_split)
            n_val = int(len(uniq) * self.validation_split)
            label = torch.zeros(len(uniq), dtype=torch.long)  # 0 = train
            label[perm[:n_test]] = 2  # test
            label[perm[n_test : n_test + n_val]] = 1  # val
            obs_label = label[inverse]  # per-observation
            reference["is_test"] = obs_label == 2  # keep is_test consistent
            logger.info(
                "split_by_miller_idx: %d reflections -> train/val/test = "
                "%d/%d/%d",
                len(uniq),
                int((label == 0).sum()),
                n_val,
                n_test,
            )
            sel = obs_label[all_indices]
            test_idx = all_indices[sel == 2]
            val_idx = all_indices[sel == 1]
            train_idx = all_indices[sel == 0]
        else:
            is_test = reference.get("is_test")
            if is_test is not None and is_test.any():
                test_mask = is_test[all_indices].bool()
                test_idx = all_indices[test_mask]
                train_val_idx = all_indices[~test_mask]
            else:
                test_idx = torch.tensor([], dtype=torch.long)
                train_val_idx = all_indices
            perm = torch.randperm(len(train_val_idx))
            val_size = int(len(train_val_idx) * self.validation_split)
            val_idx = train_val_idx[perm[:val_size]]
            train_idx = train_val_idx[perm[val_size:]]

        self.test_dataset = Subset(self.full_dataset, test_idx.tolist())
        self.val_dataset = Subset(self.full_dataset, val_idx.tolist())
        self.train_dataset = Subset(self.full_dataset, train_idx.tolist())

        # Keep the split index tensors + the grouping key for grouped sampling.
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self._miller_idx = None
        if self.group_by_asu_id:
            if self.group_by_key not in reference:
                raise KeyError(
                    f"group_by_asu_id requires '{self.group_by_key}' in the "
                    "metadata reference; add it offline (e.g. an asu_id / "
                    "nonanom_id column) and point the loader at that file."
                )
            self._miller_idx = reference[self.group_by_key].long()

    def train_dataloader(self):
        if self.group_by_asu_id:
            from integrator.data_loaders.grouped_sampler import (
                GroupedAsuIdBatchSampler,
            )

            sampler = GroupedAsuIdBatchSampler(
                miller_idx=self._miller_idx,
                indices=self.train_idx,
                batch_size=self.batch_size,
                shuffle=True,
                max_obs_per_hkl=self.max_obs_per_hkl,
            )
            return DataLoader(
                self.full_dataset,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.group_by_asu_id:
            from integrator.data_loaders.grouped_sampler import (
                GroupedAsuIdSampler,
            )

            sampler = GroupedAsuIdSampler(
                miller_idx=self._miller_idx,
                indices=self.val_idx,
                shuffle=False,
            )
            return DataLoader(
                self.full_dataset,
                sampler=sampler,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
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

    def predict_dataloader(self, grouped: bool | None = None):
        # Default to grouped batching when the module was configured for it, so
        # `finalize_merge` sees each HKL complete in one batch.
        if grouped is None:
            grouped = self.group_by_asu_id
        if grouped:
            if self._miller_idx is None:
                raise RuntimeError(
                    "grouped predict requires group_by_asu_id=True so the asu "
                    "ids are loaded."
                )
            from integrator.data_loaders.grouped_sampler import (
                GroupedAsuIdBatchSampler,
            )

            sampler = GroupedAsuIdBatchSampler(
                miller_idx=self._miller_idx,
                indices=None,
                batch_size=self.batch_size,
                shuffle=False,
                max_obs_per_hkl=None,
            )
            return DataLoader(
                self.full_dataset,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
