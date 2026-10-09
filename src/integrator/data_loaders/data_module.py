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

from integrator.data_loaders.grouped_sampler import (
    GroupedAsuIdBatchSampler,
    GroupedAsuIdSampler,
)

logger = logging.getLogger(__name__)


def _load_shoebox_array(path, weights_only=True):
    """Load counts/masks from either new .npy (integrator.mksbox memmap era)
    or .pt (torch.save). Returns torch.Tensor.
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
    # which sweep a reflection came from, for datasets combined across
    # several crystals: without it a prediction cannot be written back to
    # the reflection table it belongs to
    "sweep_id",
    # centric reflections follow Gamma(1/2), acentric Gamma(1); the Wilson
    # prior needs to know which is which
    "centric",
    # how many observations share this reflection's latent, so the
    # Wilson prior can be tempered by 1/N instead of applied per observation
    "multiplicity",
    "asu_id",
    # Friedel-pair metadata for the anomalous coupling terms, precomputed by
    # scripts/add_friedel_metadata.py. Absent unless the reference was augmented.
    #   nonanom_id   - Friedel-POOLED reflection id (mates share an id).
    #   centric      - bool, centric reflection (I(+)==I(-) by symmetry).
    #   friedel_plus - bool, the I(+) member of the pair (ISYM odd).
    "nonanom_id",
    "centric",
    "friedel_plus",
    # Per-reflection crystal-frame spherical-harmonic absorption basis (B, n_sh),
    # precomputed by scripts/extract_crystal_frame_sh.py and consumed by
    # PhysicalScale. Absent unless the reference file was augmented.
    "absorption_sh",
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


class RotationDataModule(pl.LightningDataModule):
    """LightningDataModule for rotation-geometry shoebox data.

    Attributes:
        transform: Count transform `anscombe`, `log1p`, `asinh`, or `standardization`.
            `asinh` is the one defined for negative counts, so it is the transform for
            real-valued detector output; `asinh_scale` sets its linear->log knee.
        mask_fill: What a masked pixel becomes in the encoder input. The transforms
            disagree under `legacy`, because `* masks` is applied on different sides
            of the centering: `anscombe` yields 0 in standardized space, which reads
            as an average-brightness pixel; `standardization` yields `-mean/sd`, a
            deep hole; `log1p` and `asinh` yield a true zero count. `zero` makes
            every transform give 0 in standardized space. `local_median` fills with
            each shoebox's own median of valid pixels, so a masked region looks like
            that shoebox's background instead of inventing an edge the convolutions
            respond to. Masked pixels are excluded from the likelihood either way, so
            this only changes what the encoder sees. It matters in proportion to how
            much is masked: negligible at 0.2% (PDZ2), not at 35% (6VYN).
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
        d_min: float | None = None,
        d_max: float | None = None,
        min_valid_pixels: int = 10,
        shoebox_file_names: dict | None = None,
        transform: str | None = None,
        asinh_scale: float = 1.0,
        mask_fill: str = "legacy",
        group_by_asu_id: bool = False,
        group_by_key: str = "asu_id",
        max_obs_per_hkl: int | None = None,
        ice_ring_ranges: list | None = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.include_test = include_test
        self.subset_size = subset_size
        self.num_workers = num_workers
        self.resolution_cutoff = resolution_cutoff
        self.d_min = d_min
        self.d_max = d_max
        self.min_valid_pixels = min_valid_pixels
        self.full_dataset = None
        if shoebox_file_names is None:
            shoebox_file_names = {
                "counts": "counts.npy",
                "masks": "masks.npy",
                "reference": "metadata.npy",
            }
        self.shoebox_file_names = shoebox_file_names
        transform = transform or "standardization"
        if transform not in ("anscombe", "log1p", "asinh", "standardization"):
            raise ValueError(
                "transform must be 'anscombe', 'log1p', 'asinh', or "
                f"'standardization'; got {transform!r}"
            )
        if asinh_scale <= 0:
            raise ValueError(f"asinh_scale must be > 0; got {asinh_scale}")
        if mask_fill not in ("legacy", "zero", "local_median"):
            raise ValueError(
                "mask_fill must be 'legacy', 'zero', or 'local_median'; "
                f"got {mask_fill!r}"
            )
        self.transform = transform
        self.asinh_scale = asinh_scale
        self.mask_fill = mask_fill
        self.group_by_asu_id = group_by_asu_id
        self.group_by_key = group_by_key
        self.max_obs_per_hkl = max_obs_per_hkl
        self.ice_ring_ranges = ice_ring_ranges

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

        if self.group_by_asu_id and self.group_by_key not in reference:
            raise KeyError(
                f"group_by_key={self.group_by_key!r} is not in the reference "
                f"metadata. Available keys include {sorted(reference)[:20]}... "
                "For 'nonanom_id' run scripts/add_friedel_metadata.py and point "
                "the loader's reference at the augmented file."
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

        # Resolution limits, before standardization so the statistics describe
        # the data actually trained on.
        #
        # d_min/d_max keep d_min <= d <= d_max, the crystallographic convention
        # and the one dials_port/stats.py already uses: d_min is the
        # high-resolution limit. Prefer them.
        #
        # resolution_cutoff predates these and keeps d < cutoff, which discards
        # the LOW-resolution data -- the opposite of what its name suggests and
        # of what several config comments claim. It is left alone rather than
        # redefined, because silently reversing a parameter would change the
        # meaning of any config that already sets it.
        if self.d_min is not None or self.d_max is not None:
            selection = torch.ones_like(reference["d"], dtype=torch.bool)
            if self.d_min is not None:
                selection &= reference["d"] >= self.d_min
            if self.d_max is not None:
                selection &= reference["d"] <= self.d_max
            n_cut = (~selection).sum().item()
            if n_cut > 0:
                logger.info(
                    "Removed %d reflections outside d in [%s, %s]",
                    n_cut,
                    "-inf" if self.d_min is None else f"{self.d_min:.2f}",
                    "inf" if self.d_max is None else f"{self.d_max:.2f}",
                )
            counts = counts[selection]
            masks = masks[selection]
            reference = {k: v[selection] for k, v in reference.items()}

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

        if counts.dim() == 2 and self.mask_fill != "legacy":
            # One semantics for every transform: transform, standardize, then fill the
            # masked pixels. Under `legacy` the `* masks` below lands on different
            # sides of the centering, so a masked pixel means three different things
            # depending on the transform.
            if self.transform == "anscombe":
                t = 2 * (counts.clamp(min=0) + 0.375).sqrt()
                t = (t - stats[0]) / stats[1].sqrt()
            elif self.transform == "log1p":
                t = torch.log1p(counts.clamp(min=0))
            elif self.transform == "asinh":
                t = torch.asinh(counts / self.asinh_scale)
            else:
                t = (counts - stats[0]) / stats[1].sqrt()
            valid = masks.bool()
            if self.mask_fill == "zero":
                standardized_counts = t * masks
            else:
                # each shoebox's own median over its valid pixels
                big = torch.finfo(t.dtype).max
                fill = torch.where(valid, t, torch.full_like(t, big)).median(
                    dim=-1, keepdim=True
                ).values
                fill = torch.where(
                    valid.any(dim=-1, keepdim=True), fill, torch.zeros_like(fill)
                )
                standardized_counts = torch.where(valid, t, fill.expand_as(t))
        elif counts.dim() == 2:
            if self.transform == "anscombe":
                anscombe_transformed = 2 * (counts.clamp(min=0) + 0.375).sqrt()
                standardized_counts = (
                    (anscombe_transformed - stats[0]) / stats[1].sqrt()
                ) * masks
            elif self.transform == "log1p":
                standardized_counts = torch.log1p(counts.clamp(min=0)) * masks
            elif self.transform == "asinh":
                # inverse hyperbolic sine, asinh(c/s) = log(c/s + sqrt((c/s)^2 + 1)):
                # defined on the whole real line (anscombe/log1p need c >= 0 and clamp
                # away the negatives), ~linear for |c| < s and ~log in the tails. Handles
                # the negative background pixels of real-valued detector output and
                # compresses bright peaks so a 180-photon spot does not dominate the
                # encoder input. `s = asinh_scale` sets the linear->log knee, in photons.
                standardized_counts = torch.asinh(counts / self.asinh_scale) * masks
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
            # Use the single-index GroupedAsuIdSampler (not BatchSampler) for
            # val. Lightning's `Trainer.reset_val_dataloader` introspects
            # dataloader.batch_size and dataloader.sampler — both are None
            # when `batch_sampler=` is used, which silently breaks the val
            # cadence after the sanity check. Using sampler= + batch_size=
            # gives Lightning the standard surface area it expects. Batches
            # are chunks of `batch_size` from the HKL-sorted index sequence,
            # so they still contain mostly-complete HKL groups (~1 split per
            # batch boundary).
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
        else:
            return None

    def predict_dataloader(self, grouped: bool = False):
        # `grouped=True` packs complete HKL groups per batch (requires
        # group_by_asu_id), which the merging integrators' finalize_merge needs
        # when an HKL must be processed as a set (e.g. merge_attention).
        # max_obs_per_hkl is forced off so the merge sees every observation.
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


class SimulatedShoeboxLoader(pl.LightningDataModule):
    """

    Attributes:
        data_dir:
        batch_size:
        validation_split:
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
        validation_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 3,
        include_test: bool = False,
        subset_size: int | None = None,
        single_sample_index=None,
        cutoff: float | None = None,
        min_valid_pixels: int = 10,
        shoebox_file_names: dict | None = None,
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
        self.validation_split = validation_split
        self.test_split = test_split
        self.include_test = include_test
        self.subset_size = subset_size
        self.single_sample_index = single_sample_index
        self.num_workers = num_workers
        self.cutoff = cutoff
        self.min_valid_pixels = min_valid_pixels
        self.full_dataset = None  # Will store the full dataset
        if shoebox_file_names is None:
            shoebox_file_names = {
                "counts": "counts.pt",
                "masks": "masks.pt",
                "stats": "stats.pt",
                "reference": "reference.pt",
                "standardized_counts": None,
            }
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
        val_size = int(len(train_val_idx) * self.validation_split)
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
