import torch
from abc import ABC, abstractmethod
from integrator.layers import Linear, Constraint


class BaseDistribution(torch.nn.Module, ABC):
    def __init__(
        self,
        q,
    ):
        super().__init__()
        self.q = q

    @abstractmethod
    def distribution(self, params):
        pass

    @abstractmethod
    def forward(self, representation):
        pass
