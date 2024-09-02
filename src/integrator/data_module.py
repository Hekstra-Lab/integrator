import torch
import pytorch_lightning
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset


class ShoeboxDataModule(pytorch_lightning.LightningDataModule):
    def __init__(
        self,
        shoebox_data,
        metadata,
        dead_pixel_mask,
        batch_size=32,
        val_split=0.2,
        test_split=0.1,
        num_workers=4,
        include_test=False,
        subset_size=None,
        single_sample_index=None,  # Add parameter for single sample index
    ):
        super().__init__()
        self.shoebox_data = shoebox_data
        self.metadata = metadata
        self.dead_pixel_mask = dead_pixel_mask
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.include_test = include_test
        self.subset_size = subset_size
        self.single_sample_index = single_sample_index  # Store single sample index
        self.num_workers = (num_workers,)

    def setup(self, stage=None):
        # Load the tensors
        shoeboxes = torch.load(self.shoebox_data)
        metadata = torch.load(self.metadata)
        dead_pixel_mask = torch.load(self.dead_pixel_mask)

        self.H = torch.unique(shoeboxes[..., 0], dim=1).size(-1)
        self.W = torch.unique(shoeboxes[..., 1], dim=1).size(-1)
        self.Z = torch.unique(shoeboxes[..., 2], dim=1).size(-1)

        # Create the full dataset
        full_dataset = TensorDataset(shoeboxes, metadata, dead_pixel_mask)

        # If single_sample_index is specified, use only that sample
        if self.single_sample_index is not None:
            full_dataset = Subset(full_dataset, [self.single_sample_index])

        # Optionally, create a subset of the dataset
        if self.subset_size is not None and self.subset_size < len(full_dataset):
            indices = torch.randperm(len(full_dataset))[: self.subset_size]
            full_dataset = Subset(full_dataset, indices)

        # Calculate lengths for train/val/test splits
        total_size = len(full_dataset)
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
                full_dataset, [train_size, val_size, test_size]
            )
        else:
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=3,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=3,
            drop_last=True,
        )

    def test_dataloader(self):
        if self.include_test:
            return DataLoader(self.test_dataset, batch_size=self.batch_size)
        else:
            return None
