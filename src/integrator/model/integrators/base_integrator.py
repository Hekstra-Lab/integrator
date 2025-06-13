from abc import ABC, abstractmethod

import polars as plr
import pytorch_lightning as pl
import torch


class BaseIntegrator(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()

        # lists to track avg traning metrics
        self.train_loss = []
        self.train_kl = []
        self.train_nll = []

        # lists to track avg validation metrics
        self.val_loss = []

        # dataframes to keep track of val/train epoch metrics
        self.schema = [
            ("epoch", int),
            ("avg_loss", float),
            ("avg_kl", float),
            ("avg_nll", float),
        ]
        self.train_df = plr.DataFrame(schema=self.schema)

    @abstractmethod
    def forward(self, *args, **kwargs):
        "Forward method to be implemented by the subclass integrator"
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):
        # calculate epoch averages
        avg_train_loss = sum(self.train_loss) / len(self.train_loss)
        avg_kl = sum(self.train_kl) / len(self.train_kl)

        # log averages to weights & biases
        self.log("train_loss", avg_loss)

        # create epoch dataframe
        epoch_df = plr.Dataframe(
            {
                "epoch": self.current_epoch,
                "avg_loss": avg_train_loss,
            }
        )

        # udpate training dataframe
        self.train_df = plr.concat([self.train_df, epoch_df])
        # clear all lists
        self.train_loss = []
        self.val_loss = []
