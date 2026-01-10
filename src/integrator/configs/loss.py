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
class LossConfig:
    name: Literal["default"]
    args: LossArgs
