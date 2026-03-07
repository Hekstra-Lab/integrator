from dataclasses import dataclass
from typing import Literal

from integrator.configs.priors import PriorConfig


@dataclass
class LossArgs:
    mc_samples: int
    eps: float
    pprf_cfg: PriorConfig | None
    pbg_cfg: PriorConfig | None
    pi_cfg: PriorConfig | None

    def __post_init__(self):
        if self.mc_samples < 0:
            raise ValueError(
                "The number of Monte Carlo samples must be greater than 0"
            )
        if self.eps < 0:
            raise ValueError("The epsilon offset value must be greater than 0")


@dataclass
class HierarchicalLossArgs(LossArgs):
    """LossArgs extended with empirical-Bayes hyperparameters."""

    hierarchical_intensity: bool = True
    hierarchical_background: bool = False
    hyperprior_scale: float = 2.0


@dataclass
class ConditionalLossArgs(LossArgs):
    """LossArgs extended with input-dependent (conditional) prior parameters."""

    conditional_intensity: bool = True
    conditional_background: bool = False
    hidden_dim: int = 16
    hyperprior_scale: float = 2.0


@dataclass
class LossConfig:
    name: Literal["default", "hierarchical", "conditional"]
    args: LossArgs
