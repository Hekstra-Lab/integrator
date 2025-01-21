from abc import ABC, abstractmethod
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
