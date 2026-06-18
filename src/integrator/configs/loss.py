from dataclasses import dataclass
from typing import Literal

from integrator.configs.priors import PriorConfig


@dataclass
class LossArgs:
    """The structured priors of the ELBO loss.

    Attributes:
        pprf_cfg: Prior over the profile, or `None` to use the loss default.
        pbg_cfg: Prior over the background, or `None` to use the loss default.
        pi_cfg: Prior over the intensity, or `None` to use the loss default.
    """

    pprf_cfg: PriorConfig | None = None
    pbg_cfg: PriorConfig | None = None
    pi_cfg: PriorConfig | None = None


@dataclass
class LossConfig:
    """Registry selection for the loss: a `name` plus its typed `args`.

    Attributes:
        name: Loss variant, `monochromatic_wilson` or `polychromatic_wilson`.
        args: Loss constructor arguments.
    """

    name: Literal["monochromatic_wilson", "polychromatic_wilson"]
    args: LossArgs
