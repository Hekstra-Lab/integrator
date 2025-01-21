from abc import ABC, abstractmethod
import torch
import pytorch_lightning as pl


class BaseIntegrator(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        "Forward method to be implemented by the subclass integrator"
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
