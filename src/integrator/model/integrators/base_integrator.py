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
        self.val_kl = []
        self.val_nll = []

        # dataframes to keep track of val/train epoch metrics
        self.schema = [
            ("epoch", int),
            ("avg_loss", float),
            ("avg_kl", float),
            ("avg_nll", float),
        ]
        self.train_df = plr.DataFrame(schema=self.schema)
        self.val_df = plr.DataFrame(schema=self.schema)

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
        avg_nll = sum(self.train_nll) / len(self.train_nll)

        # log averages to weights & biases
        self.log("train_loss", avg_train_loss)
        self.log("avg_kl", avg_kl)
        self.log("avg_nll", avg_nll)

        # create epoch dataframe
        epoch_df = plr.DataFrame(
            {
                "epoch": self.current_epoch,
                "avg_loss": avg_train_loss,
                "avg_kl": avg_kl,
                "avg_nll": avg_nll,
            }
        )

        # udpate training dataframe
        self.train_df = plr.concat([self.train_df, epoch_df])
        # clear all lists
        self.train_loss = []
        self.train_kl = []
        self.train_nll = []

    def on_validation_epoch_end(self):
        avg_val_loss = sum(self.val_loss) / len(self.val_loss)
        avg_kl = sum(self.val_kl) / len(self.val_kl)
        avg_nll = sum(self.val_nll) / len(self.val_nll)

        self.log("validation_loss", avg_val_loss)
        self.log("validation_avg_kl", avg_kl)
        self.log("validation_avg_nll", avg_nll)

        epoch_df = plr.DataFrame(
            {
                "epoch": self.current_epoch,
                "avg_loss": avg_val_loss,
                "avg_kl": avg_kl,
                "avg_nll": avg_nll,
            }
        )
        self.val_df = plr.concat([self.val_df, epoch_df])

        self.val_loss = []
        self.avg_kl = []
        self.val_nll = []
