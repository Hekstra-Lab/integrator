import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset
from integrator.data_loaders import BaseDataModule


class ShoeboxDataModule(BaseDataModule):
    """
    Attributes:
        data_dir: Path to the directory containing the data
        batch_size: Batch size for the data loaders
        val_split:
        test_split:
        include_test:
        subset_size:
        single_sample_index:
        num_workers:
        cutoff:
        full_dataset:
        H:
        W:
        Z:
        full_dataset:
    """

    def __init__(
        self,
        data_dir,
        batch_size=100,
        val_split=0.2,
        test_split=0.1,
        num_workers=3,
        include_test=False,
        subset_size=None,
        single_sample_index=None,
        cutoff=None,
        use_metadata=None,
        persistent_workers=True,
        shoebox_file_names={
            "counts": "counts.pt",
            # "metadata": "metadata.pt",
            "masks": "masks.pt",
            "stats": "stats.pt",
            "reference": "reference.pt",
            "standardized_counts": None,
        },
        refl_file=None,
        H=21,
        W=21,
        Z=3,
        get_dxyz=False,
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
        self.Z = Z
        self.standardized_counts = shoebox_file_names["standardized_counts"]
        self.get_dxyz = get_dxyz

    def setup(self, stage=None):
        counts = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["counts"])
        )
        #        metadata = torch.load(
        #            os.path.join(self.data_dir, self.shoebox_file_names["metadata"])
        #        ).type(torch.float32)
        #
        masks = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["masks"])
        )
        stats = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["stats"])
        )
        reference = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["reference"])
        )

        all_dead = masks.sum(-1) < 10

        # print("all_dead", all_dead.sum())

        # filter out samples with all dead pixels
        counts = counts[~all_dead]
        masks = masks[~all_dead]
        reference = reference[~all_dead]

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
        #            if self.use_metadata is not None:
        #                metadata = metadata[selection]
        #
        # Standardize counts after filtering
        if self.standardized_counts is not None:
            standardized_counts = torch.load(
                os.path.join(
                    self.data_dir, self.shoebox_file_names["standardized_counts"]
                )
            )
            if self.cutoff is not None:
                standardized_counts = standardized_counts[selection]
        else:
            if counts.dim() == 2:
                #standardized_counts = (counts * masks) - stats[0] / stats[1].sqrt()
                ans = 2*torch.sqrt(counts + (3.0/8.0))
                standardized_counts = ((ans - stats[1])/stats[1].sqrt()) * masks

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

        # Create the full dataset based on whether metadata is present
        #        if self.use_metadata is not None:
        #            self.full_dataset = TensorDataset(
        #                counts,
        #                standardized_counts,
        #                metadata,
        #                masks,
        #                reference,
        #            )
        #        else:
        #            self.full_dataset = TensorDataset(
        #                counts, standardized_counts, masks, reference
        #            )
        self.full_dataset = TensorDataset(counts, standardized_counts, masks, reference)
        # If single_sample_index is specified, use only that sample
        if self.single_sample_index is not None:
            self.full_dataset = Subset(self.full_dataset, [self.single_sample_index])

        # Optionally, create a subset of the dataset
        if self.subset_size is not None and self.subset_size < len(self.full_dataset):
            indices = torch.randperm(len(self.full_dataset))[: self.subset_size]
            self.full_dataset = Subset(self.full_dataset, indices)

        # Calculate lengths for train/val/test splits
        total_size = len(self.full_dataset)
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
        # Use the full dataset for prediction
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# %%
class ShoeboxDataModule2(BaseDataModule):
    """
    Attributes:
        data_dir: Path to the directory containing the data
        batch_size: Batch size for the data loaders
        val_split:
        test_split:
        include_test:
        subset_size:
        single_sample_index:
        num_workers:
        cutoff:
        full_dataset:
        H:
        W:
        Z:
        full_dataset:
    """

    def __init__(
        self,
        data_dir,
        batch_size=100,
        val_split=0.2,
        test_split=0.1,
        num_workers=3,
        include_test=False,
        subset_size=None,
        single_sample_index=None,
        cutoff=None,
        use_metadata=None,
        persistent_workers=True,
        shoebox_file_names={
            "counts": "counts.pt",
            "masks": "masks.pt",
            "reference": "reference.pt",
            "standardized_counts": None,
        },
        refl_file=None,
        H=21,
        W=21,
        Z=3,
        get_dxyz=False,
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
        self.Z = Z
        self.standardized_counts = shoebox_file_names["standardized_counts"]
        self.get_dxyz = get_dxyz

    def setup(self, stage=None):
        counts = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["counts"])
        )

        masks = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["masks"])
        )
        reference = torch.load(
            os.path.join(self.data_dir, self.shoebox_file_names["reference"])
        )

        self.full_dataset = TensorDataset(counts, masks, reference)

        # Optionally, create a subset of the dataset
        if self.subset_size is not None and self.subset_size < len(self.full_dataset):
            indices = torch.randperm(len(self.full_dataset))[: self.subset_size]
            self.full_dataset = Subset(self.full_dataset, indices)

        # Calculate lengths for train/val/test splits
        total_size = len(self.full_dataset)
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
            pin_memory=True,
            # prefetch_factor=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            # prefetch_factor=2,
        )

    def test_dataloader(self):
        if self.include_test:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                # prefetch_factor=2,
            )
        else:
            return None

    def predict_dataloader(self):
        # Use the full dataset for prediction
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# %%
