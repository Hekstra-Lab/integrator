from abc import ABC, abstractmethod
import torch.nn as nn


class BaseProfile(nn.Module, ABC):
    @abstractmethod
    def forward(self, representation):
        pass


class DirichletProfile(BaseProfile):
    def __init__(self):
        super().__init__()
