from abc import ABC, abstractmethod
import torch


class BaseLoss(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, *input):
        return self.loss(*input)
