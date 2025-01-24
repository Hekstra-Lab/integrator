import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset
from abc import ABC, abstractmethod


class BaseDataModule(pl.LightningDataModule, ABC):
    @abstractmethod
    def setup(self, *args, **kwargs):
        pass

    # @abstractmethod
    # def train_dataloader(self, *input):
    # pass

    # @abstractmethod
    # def val_dataloader(self):
    # pass

    # @abstractmethod
    # def test_dataloader(self):
    # pass

    # @abstractmethod
    # def predict_dataloader(self):
    # pass
