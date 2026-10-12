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
from integrator.data_loaders.grouped_sampler import (
    GroupedAsuIdBatchSampler,
    GroupedAsuIdSampler,
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
    # how many observations share this reflection's latent, so the
    # Wilson prior can be tempered by 1/N instead of applied per observation
    "multiplicity",
    # window origin on the panel (fast, slow), so a physics profile can be
    # rendered on the shoebox's own pixels
    "bbox.0",
    "bbox.2",
    # orders of the same direction in band; >1 means the spot is a harmonic sum
    "n_harmonics",
    # Merging/scaling identity + anomalous metadata. Each is included in the
    # batch only when present in the reference file (IntegratorDataset filters by
    # `if k in reference`), so older poly metadata without them is unaffected.
    #   asu_id       - canonical reflection id (scripts/prepare_asu_ids.py),
    #                  required for grouped merging.
    #   nonanom_id   - Friedel-pooled id; centric / friedel_plus for the
    #                  anomalous-preserving terms (scripts/add_friedel_metadata.py).
    #   absorption_sh- per-obs crystal-frame real-SH basis for the MLP scale's
    #                  absorption inputs (a stills variant of
    #                  scripts/extract_crystal_frame_sh.py).
    "asu_id",
    "nonanom_id",
    "centric",
    "friedel_plus",
    "absorption_sh",
]


class PolychromaticDataModule(pl.LightningDataModule):
    """DataModule for polychromatic (Laue) stills shoeboxes."""

    def __init__(
        self,
        data_dir,
        batch_size: int = 256,
        validation_split: float = 0.2,
        num_workers: int = 8,
        include_test: bool = True,
        subset_size: int | None = None,
        resolution_cutoff: float | None = None,
        min_valid_pixels: int = 10,
        shoebox_file_names: dict | None = None,
        transform: str | None = None,
        mask_fill: str = "legacy",
        group_by_asu_id: bool = False,
        group_by_key: str = "asu_id",
        max_obs_per_hkl: int | None = None,
        extra_columns: list[str] | None = None,
        ice_ring_ranges: list | None = None,
        # Half-open [min, max) on `image_num`, so the bound reads like a frame
        # count: max_image_num 360 keeps frames 0..359.
        min_image_num: int | None = None,
        max_image_num: int | None = None,
    ):
        super().__init__()
        self.min_image_num = min_image_num
        self.max_image_num = max_image_num
        self.data_dir = str(data_dir)
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.include_test = include_test
        self.subset_size = subset_size
        self.num_workers = num_workers
        self.resolution_cutoff = resolution_cutoff
        self.min_valid_pixels = min_valid_pixels
        # Grouped batching for merging/scaling: pack complete HKL groups per
        # batch so the per-HKL sufficient-statistic sum is complete in one batch.
        self.group_by_asu_id = group_by_asu_id
        self.group_by_key = group_by_key
        self.max_obs_per_hkl = max_obs_per_hkl
        # Extra metadata keys to expose in the batch beyond POLY_DS_COLS (e.g. a
        # custom grouping id, or scale inputs). Included only when present in the
        # reference file, so listing an absent key is harmless.
        self.column_names = POLY_DS_COLS + list(extra_columns or [])
        # Drop resolution bands (e.g. ice rings) at load time.
        self.ice_ring_ranges = ice_ring_ranges

        self.shoebox_file_names = shoebox_file_names or {
            "counts": "counts.npy",
            "masks": "masks.npy",
            "reference": "metadata.npy",
        }

        transform = transform or "standardization"
        if transform not in ("anscombe", "log1p", "standardization"):
            raise ValueError(
                f"transform must be 'anscombe', 'log1p', or 'standardization'; "
                f"got {transform!r}"
            )
        if mask_fill not in ("legacy", "zero", "local_median"):
            raise ValueError(
                "mask_fill must be 'legacy', 'zero', or 'local_median'; "
                f"got {mask_fill!r}"
            )
        self.transform = transform
        self.mask_fill = mask_fill

        self.full_dataset = None
        self.beam_center_px = None

    def setup(self, stage=None):
        counts = _load_shoebox_array(
            os.path.join(self.data_dir, self.shoebox_file_names["counts"])
        ).squeeze(-1)
        masks = _load_shoebox_array(
            os.path.join(self.data_dir, self.shoebox_file_names["masks"])
        ).squeeze(-1)
        from integrator.io import load_data, read_dataset_spec

        spec = read_dataset_spec(self.data_dir)
        if spec is None:
            raise FileNotFoundError(
                f"dataset.yaml not found in {self.data_dir}; "
                "regenerate the dataset with mksbox"
            )
        stats_key = "anscombe" if self.transform == "anscombe" else "raw"
        stats = torch.tensor(spec["stats"][stats_key], dtype=torch.float32)
        reference = load_data(
            os.path.join(self.data_dir, self.shoebox_file_names["reference"])
        )

        crystal_meta = spec.get("crystal") or {}
        self.beam_center_px = crystal_meta.get("beam_center_px")
        if self.beam_center_px is not None:
            logger.info(
                "Beam center (px): %.1f, %.1f",
                self.beam_center_px[0],
                self.beam_center_px[1],
            )
        if self.group_by_asu_id and self.group_by_key not in reference:
            raise KeyError(
                f"group_by_key={self.group_by_key!r} is not in the reference "
                f"metadata. Available keys include {sorted(reference)[:20]}... "
                "Run scripts/prepare_asu_ids.py to add 'asu_id' (and "
                "scripts/add_friedel_metadata.py for 'nonanom_id')."
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

        # Image range, so the batch cut is a config knob rather than something
        # baked into a dataset at shoebox-build time. `mksbox --max-images`
        # does the same thing permanently; doing it here lets one dataset
        # covering every frame serve a 360-frame run and a 999-frame run, and
        # lets a model be scored against a baseline MTZ cut to the same
        # batches without regenerating shoeboxes.
        if self.min_image_num is not None or self.max_image_num is not None:
            if "image_num" not in reference:
                raise KeyError(
                    "min_image_num/max_image_num need an 'image_num' column in "
                    "the reference metadata; this dataset has "
                    f"{sorted(reference)[:12]}..."
                )
            img = reference["image_num"]
            selection = torch.ones_like(img, dtype=torch.bool)
            if self.min_image_num is not None:
                selection &= img >= self.min_image_num
            if self.max_image_num is not None:
                selection &= img < self.max_image_num
            logger.info(
                "Image range [%s, %s): kept %d reflections, removed %d",
                "-inf" if self.min_image_num is None else self.min_image_num,
                "inf" if self.max_image_num is None else self.max_image_num,
                int(selection.sum().item()),
                int((~selection).sum().item()),
            )
            if not bool(selection.any()):
                raise ValueError(
                    "the image range selected no reflections; image_num spans "
                    f"{float(img.min()):.0f}..{float(img.max()):.0f}"
                )
            counts = counts[selection]
            masks = masks[selection]
            reference = {k: v[selection] for k, v in reference.items()}

        if self.resolution_cutoff is not None and "d" in reference:
            selection = reference["d"] < self.resolution_cutoff
            n_cut = (~selection).sum().item()
            if n_cut > 0:
                logger.info(
                    "Removed %d reflections with d >= %.2f", n_cut, self.resolution_cutoff
                )
            counts = counts[selection]
            masks = masks[selection]
            reference = {k: v[selection] for k, v in reference.items()}

        # Standardize counts
        if self.mask_fill != "legacy":
            # One semantics for every transform: transform, standardize, then fill the
            # masked pixels. Under `legacy` the `* masks` lands on different sides of
            # the centering, so a masked pixel means a different thing per transform.
            if self.transform == "anscombe":
                t = 2 * (counts.clamp(min=0) + 0.375).sqrt()
                t = (t - stats[0]) / stats[1].sqrt()
            elif self.transform == "log1p":
                t = torch.log1p(counts.clamp(min=0))
            else:
                t = (counts - stats[0]) / stats[1].sqrt()
            valid = masks.bool()
            if self.mask_fill == "zero":
                standardized_counts = t * masks
            else:
                big = torch.finfo(t.dtype).max
                fill = torch.where(valid, t, torch.full_like(t, big)).median(
                    dim=-1, keepdim=True
                ).values
                fill = torch.where(
                    valid.any(dim=-1, keepdim=True), fill, torch.zeros_like(fill)
                )
                standardized_counts = torch.where(valid, t, fill.expand_as(t))
        elif self.transform == "anscombe":
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
            column_names=self.column_names,
        )

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
        val_size = int(len(train_val_idx) * self.validation_split)
        val_idx = train_val_idx[perm[:val_size]]
        train_idx = train_val_idx[perm[val_size:]]

        self.val_dataset = Subset(self.full_dataset, val_idx.tolist())
        self.train_dataset = Subset(self.full_dataset, train_idx.tolist())

    def train_dataloader(self):
        if self.group_by_asu_id:
            sampler = GroupedAsuIdBatchSampler(
                asu_ids=self.full_dataset.reference[self.group_by_key],
                indices=torch.tensor(
                    self.train_dataset.indices, dtype=torch.long
                ),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
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
            # Single-index GroupedAsuIdSampler (not BatchSampler) for val, so
            # Lightning sees dataloader.sampler / .batch_size; see RotationData-
            # Module.val_dataloader for why batch_sampler= breaks the val cadence.
            sampler = GroupedAsuIdSampler(
                asu_ids=self.full_dataset.reference[self.group_by_key],
                indices=torch.tensor(
                    self.val_dataset.indices, dtype=torch.long
                ),
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
        return None

    def predict_dataloader(self, grouped: bool = False):
        # `grouped=True` packs complete HKL groups per batch (needs
        # group_by_asu_id), which the merging integrators' finalize_merge
        # requires. max_obs_per_hkl is forced off so the merge sees every
        # observation.
        if grouped and self.group_by_asu_id:
            sampler = GroupedAsuIdBatchSampler(
                asu_ids=self.full_dataset.reference[self.group_by_key],
                indices=torch.arange(len(self.full_dataset), dtype=torch.long),
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
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
