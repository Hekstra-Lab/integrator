from abc import ABC, abstractmethod
import torch
import pytorch_lightning as pl


class BaseIntegrator(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self.train_loss = []

    @abstractmethod
    def forward(self, *args, **kwargs):
        "Forward method to be implemented by the subclass integrator"
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):
        avg_loss = sum(self.train_loss) / len(self.train_loss)
        self.log("train_loss", avg_loss)
        self.train_loss = []
