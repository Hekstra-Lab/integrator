import torch
from abc import ABC, abstractmethod


class BaseDecoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        *args,
        **kwargs,
    ):
        pass
