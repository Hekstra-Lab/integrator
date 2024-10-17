import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset


class ShoeboxDataModule(pl.LightningDataModule):
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
        batch_size=32,
        val_split=0.2,
        test_split=0.1,
        num_workers=4,
        include_test=False,
        subset_size=None,
        single_sample_index=None,
        cutoff=None,
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

    def setup(self, stage=None):
        # Load the tensors
        shoeboxes = torch.load(os.path.join(self.data_dir, "samples.pt"))
        metadata = torch.load(os.path.join(self.data_dir, "metadata.pt"))
        dead_pixel_mask = torch.load(os.path.join(self.data_dir, "masks.pt"))

        if self.cutoff is not None:
            selection = metadata[:, 0] < self.cutoff
            shoeboxes = shoeboxes[selection]
            metadata = metadata[selection]
            dead_pixel_mask = dead_pixel_mask[selection]

        self.H = torch.unique(shoeboxes[..., 0], dim=1).size(-1)
        self.W = torch.unique(shoeboxes[..., 1], dim=1).size(-1)
        self.Z = torch.unique(shoeboxes[..., 2], dim=1).size(-1)

        # Create the full dataset
        self.full_dataset = TensorDataset(shoeboxes, metadata, dead_pixel_mask)

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
