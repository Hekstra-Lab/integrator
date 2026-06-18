from dataclasses import dataclass

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
