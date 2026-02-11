import os
from pathlib import Path

import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    TensorDataset,
    random_split,
)

from integrator.data_loaders import BaseDataModule

SIMULATED_COLS = [
    "shoebox_median",
    "shoebox_var",
    "shoebox_mean",
    "shoebox_min",
    "shoebox_max",
    "intensity",
    "background",
    "refl_id",
    "refl_ids",
    "is_test",
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

        meta = {k: self.reference[k][idx] for k in self.column_names}

        return counts, standardized_counts, masks, meta


# Filter to remove reflections with variance = -1
def _remove_flagged_variance(
    counts: torch.Tensor,
    masks: torch.Tensor,
    metadata: dict,
    filter_key: str = "intensity.prf.variance",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    filter_ = metadata[filter_key] == -1

    counts = counts[~filter_]
    masks = masks[~filter_]
    metadata = {k: v[~filter_] for k, v in metadata.items()}

    return counts, masks, metadata


class ShoeboxDataModule2D(BaseDataModule):
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
        use_metadata:
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
        use_metadata=None,
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
        self.use_metadata = use_metadata
        self.shoebox_file_names = shoebox_file_names
        self.H = H
        self.W = W
        self.standardized_counts = shoebox_file_names["standardized_counts"]
        self.x_coords = shoebox_file_names["x_coords"]
        self.y_coords = shoebox_file_names["y_coords"]
        self.get_dxyz = get_dxyz
        self.anscombe = anscombe

    def setup(self):
        counts = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["counts"]),
        )
        masks = torch.load(
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
            standardized_counts = (ans - stats[1]) / stats[1].sqrt()
        else:
            standardized_counts = ((processed_counts) - stats[0]) / stats[1].sqrt()

        if self.x_coords is not None:
            x = torch.load(os.path.join(self.data_dir, self.x_coords))[~all_dead]
            y = torch.load(os.path.join(self.data_dir, self.y_coords))[~all_dead]

            standardized_counts = torch.stack((standardized_counts, x, y)).permute(
                1, 0, 2
            )

        self.full_dataset = TensorDataset(
            processed_counts, standardized_counts, masks, reference
        )
        # If single_sample_index is specified, use only that sample
        if self.single_sample_index is not None:
            self.full_dataset = Subset(self.full_dataset, [self.single_sample_index])

        # Optionally, create a subset of the dataset
        if self.subset_size is not None and self.subset_size < len(self.full_dataset):
            indices = torch.randperm(len(self.full_dataset))[: self.subset_size]
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
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.full_dataset, [train_size, val_size, test_size]
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


class ShoeboxDataModule(BaseDataModule):
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
        use_metadata:
        shoebox_file_names:
        H:
        W:
        D:
        standardized_counts:
        get_dxyz:
        anscombe: Boolean indicating whether to use Anscome transformation
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
        use_metadata: bool | None = None,
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
        self.full_dataset = None  # Will store the full dataset
        self.use_metadata = use_metadata
        self.shoebox_file_names = shoebox_file_names
        self.H = H
        self.W = W
        self.D = D
        self.standardized_counts = shoebox_file_names["standardized_counts"]
        self.get_dxyz = get_dxyz
        self.anscombe = False

    def setup(self, stage=None):
        counts = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["counts"])
        ).squeeze(-1)
        masks = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["masks"])
        ).squeeze(-1)
        stats = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["stats"])
        )
        reference = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["reference"])
        )

        # Filter out all refls with less than 10 valid pixels
        all_dead = masks.sum(-1) < 10

        # filter out samples with all dead pixels
        counts = counts[~all_dead]
        masks = masks[~all_dead]
        reference = {k: v[~all_dead] for k, v in reference.items()}

        counts, masks, reference = _remove_flagged_variance(counts, masks, reference)

        # Apply cutoff before standardization to ensure we only process needed data
        if self.cutoff is not None:
            # Make sure we're checking the first column of reference against cutoff
            # Ensure reference has the right shape before filtering

            if reference.dim() > 1:
                selection = reference[:, 13] < self.cutoff
            else:
                selection = reference < self.cutoff

            # Apply selection filter to all tensors
            counts = counts[selection]
            masks = masks[selection]
            reference = reference[selection]

        else:
            if counts.dim() == 2:
                if self.anscombe:
                    anscombe_transformed = 2 * (counts + 0.375).sqrt()
                    standardized_counts = (
                        (anscombe_transformed - stats[1]) / stats[1].sqrt()
                    ) * masks
                else:
                    standardized_counts = ((counts * masks) - stats[0]) / stats[
                        1
                    ].sqrt()
            else:
                standardized_counts = (counts[..., -1] * masks) - stats[0] / stats[
                    1
                ].sqrt()
                # Normalize first three channels of counts
                # Only attempt this if counts has enough dimensions
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
        if self.subset_size is not None and self.subset_size < len(self.full_dataset):
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


class SimulatedShoeboxLoader(BaseDataModule):
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
        use_metadata:
        shoebox_file_names:
        H:
        W:
        D:
        standardized_counts:
        get_dxyz:
        anscombe: Boolean indicating whether to use Anscome transformation
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
        use_metadata: bool | None = None,
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
        self.full_dataset = None  # Will store the full dataset
        self.use_metadata = use_metadata
        self.shoebox_file_names = shoebox_file_names
        self.H = H
        self.W = W
        self.D = D
        self.standardized_counts = shoebox_file_names["standardized_counts"]
        self.get_dxyz = get_dxyz
        self.anscombe = False

    def setup(self, stage=None):
        counts = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["counts"]),
            weights_only=False,
        ).squeeze(-1)
        masks = torch.load(
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

        if self.anscombe:
            anscombe_transformed = 2 * (counts + 0.375).sqrt()
            standardized_counts = (
                (anscombe_transformed - stats[1]) / stats[1].sqrt()
            ) * masks
        else:
            standardized_counts = ((counts * masks) - stats[0]) / stats[1].sqrt()

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
        if self.subset_size is not None and self.subset_size < len(self.full_dataset):
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


# %%
if __name__ == "__main__":
    data_dir = Path(
        "/Users/luis/master/notebooks/integrator_notes/code/simulating_shoeboxes/"
    )

    loader = SimulatedShoeboxLoader(
        data_dir=data_dir,
        shoebox_file_names={
            "counts": "counts.pt",
            "masks": "masks.pt",
            "stats": "stats_anscombe.pt",
            "reference": "reference.pt",
            "standardized_counts": None,
        },
        include_test=True,
        num_workers=0,
    )

    loader.setup()

    next(iter(loader.train_dataloader()))

    from integrator.utils import load_config

    yaml = "/Users/luis/master/notebooks/integrator_notes/code/simulating_shoeboxes/simulated_data_config.yaml"

    import os

    import torch

    from integrator.utils import (
        construct_data_loader,
    )

    cfg = load_config(yaml)
    data_loader = construct_data_loader(cfg)
