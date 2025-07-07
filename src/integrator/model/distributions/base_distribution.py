from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseDistribution(torch.nn.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        q: torch.distributions.Distribution | Any,
    ):
        super().__init__()
        self.q = q

    @abstractmethod
    def distribution(self, params):
        pass

    @abstractmethod
    def forward(self, representation):
        pass
