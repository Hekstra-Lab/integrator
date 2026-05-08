import logging
import os

import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from integrator.data_loaders.data_module import (
    IntegratorDataset,
    _load_shoebox_array,
)

logger = logging.getLogger(__name__)

POLY_DS_COLS = [
    "refl_ids",
    "is_test",
    "d",
    "wavelength",
    "image_num",
    "group_label",
    "H",
    "K",
    "L",
    "intensity.sum.value",
    "intensity.sum.variance",
    "background.sum.value",
    "background.sum.variance",
    "s1.0",
    "s1.1",
    "s1.2",
    "xyzcal.px.0",
    "xyzcal.px.1",
    "xyzcal.px.2",
    "panel",
    "flags",
]


class PolychromaticDataModule(pl.LightningDataModule):
    """DataModule for polychromatic (Laue) stills shoeboxes."""

    def __init__(
        self,
        data_dir,
        batch_size: int = 256,
        val_split: float = 0.2,
        test_split: float = 0.0,
        num_workers: int = 8,
        include_test: bool = True,
        subset_size: int | None = None,
        cutoff: float | None = None,
        min_valid_pixels: int = 10,
        shoebox_file_names: dict | None = None,
        D: int = 1,
        H: int = 25,
        W: int = 25,
        anscombe: bool = True,
        transform: str | None = None,
    ):
        super().__init__()
        self.data_dir = str(data_dir)
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.include_test = include_test
        self.subset_size = subset_size
        self.num_workers = num_workers
        self.cutoff = cutoff
        self.min_valid_pixels = min_valid_pixels
        self.D = D
        self.H = H
        self.W = W

        self.shoebox_file_names = shoebox_file_names or {
            "counts": "counts.npy",
            "masks": "masks.npy",
            "stats": "anscombe_stats.pt",
            "reference": "metadata.pt",
            "standardized_counts": None,
        }

        if transform is None:
            self.transform = "anscombe" if anscombe else "none"
        elif transform not in ("anscombe", "log1p", "none"):
            raise ValueError(
                f"transform must be 'anscombe', 'log1p', or 'none'; "
                f"got {transform!r}"
            )
        else:
            self.transform = transform

        self.full_dataset = None
        self.beam_center_px = None

    def setup(self, stage=None):
        counts = _load_shoebox_array(
            os.path.join(self.data_dir, self.shoebox_file_names["counts"])
        ).squeeze(-1)
        masks = _load_shoebox_array(
            os.path.join(self.data_dir, self.shoebox_file_names["masks"])
        ).squeeze(-1)
        stats = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["stats"]),
        )
        reference = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["reference"]),
        )

        crystal_path = os.path.join(self.data_dir, "crystal.yaml")
        if os.path.exists(crystal_path):
            with open(crystal_path) as f:
                crystal_meta = yaml.safe_load(f)
            self.beam_center_px = crystal_meta.get("beam_center_px")
            if self.beam_center_px is not None:
                logger.info(
                    "Beam center (px): %.1f, %.1f",
                    self.beam_center_px[0],
                    self.beam_center_px[1],
                )

        # Filter reflections with too few valid pixels
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

        # Resolution cutoff
        if self.cutoff is not None and "d" in reference:
            selection = reference["d"] < self.cutoff
            n_cut = (~selection).sum().item()
            if n_cut > 0:
                logger.info(
                    "Removed %d reflections with d >= %.2f", n_cut, self.cutoff
                )
            counts = counts[selection]
            masks = masks[selection]
            reference = {k: v[selection] for k, v in reference.items()}

        # Standardize counts (2D path — counts is always (N, H*W) for stills)
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

        self.full_dataset = IntegratorDataset(
            counts,
            standardized_counts,
            masks,
            reference,
            column_names=POLY_DS_COLS,
        )

        # Split using is_test if available
        is_test = reference.get("is_test")
        all_indices = torch.arange(len(self.full_dataset))

        if self.subset_size is not None and self.subset_size < len(
            self.full_dataset
        ):
            all_indices = all_indices[
                torch.randperm(len(all_indices))[: self.subset_size]
            ]

        if is_test is not None and is_test.any():
            test_mask = is_test[all_indices].bool()
            test_idx = all_indices[test_mask]
            train_val_idx = all_indices[~test_mask]
        else:
            test_idx = torch.tensor([], dtype=torch.long)
            train_val_idx = all_indices

        self.test_dataset = Subset(self.full_dataset, test_idx.tolist())

        perm = torch.randperm(len(train_val_idx))
        val_size = int(len(train_val_idx) * self.val_split)
        val_idx = train_val_idx[perm[:val_size]]
        train_idx = train_val_idx[perm[val_size:]]

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
        return None

    def predict_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
