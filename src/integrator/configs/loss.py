from dataclasses import dataclass
from typing import Literal

from integrator.configs.priors import PriorConfig


@dataclass
class LossArgs:
    """Constructor arguments for the ELBO loss.

    Attributes:
        mc_samples: Monte Carlo samples used to estimate the expected NLL; must be positive.
        eps: Numerical floor added for stability; must be positive.
        pprf_cfg: Prior over the profile, or `None` to use the loss default.
        pbg_cfg: Prior over the background, or `None` to use the loss default.
        pi_cfg: Prior over the intensity, or `None` to use the loss default.
    """

    mc_samples: int = 100
    eps: float = 1.0e-6
    pprf_cfg: PriorConfig | None = None
    pbg_cfg: PriorConfig | None = None
    pi_cfg: PriorConfig | None = None

    def __post_init__(self):
        if self.mc_samples <= 0:
            raise ValueError(
                "The number of Monte Carlo samples must be greater than 0"
            )
        if self.eps <= 0:
            raise ValueError("The epsilon offset value must be greater than 0")


@dataclass
class LossConfig:
    """Registry selection for the loss: a `name` plus its typed `args`.

    Attributes:
        name: Loss variant, `monochromatic_wilson` or `polychromatic_wilson`.
        args: Loss constructor arguments.
    """

    name: Literal["monochromatic_wilson", "polychromatic_wilson"]
    args: LossArgs
