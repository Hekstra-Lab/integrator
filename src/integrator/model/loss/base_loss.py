from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution


class BaseLoss(nn.Module, ABC):
    def __init__(self, mc_samples: int = 100):
        super().__init__()
        self.mc_samples = mc_samples

    @abstractmethod
    def forward(
        self,
        rate: Tensor,
        counts: Tensor,
        q_p: Distribution,
        q_i: Distribution,
        q_bg: Distribution,
        masks: Tensor,
    ) -> dict: ...

    def compute_kl(
        self,
        q_dist: Distribution,
        pdist: Distribution,
    ) -> Tensor:
        """Compute KL divergence between distributions, with fallback sampling if needed."""
        try:
            return torch.distributions.kl.kl_divergence(q_dist, pdist)
        except NotImplementedError:
            samples = q_dist.rsample([self.mc_samples])
            log_q = q_dist.log_prob(samples)
            log_p = pdist.log_prob(samples)
            return (log_q - log_p).mean(dim=0)
