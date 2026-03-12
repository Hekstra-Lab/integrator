from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

P = TypeVar("P")


@dataclass
class DirichletParams:
    concentration: float | str
    shape: tuple[int, int, int]

    def __post_init__(self):
        if isinstance(self.concentration, str):
            path = Path(self.concentration).expanduser()

            if not path.exists():
                raise ValueError(f"Concentration path does not exist: {path}")

            if not path.is_file():
                raise ValueError(f"Concentration path is not a file: {path}")

        else:
            if self.concentration <= 0:
                raise ValueError(
                    "Dirichlet concentration scalar must be positive"
                )

        if len(self.shape) != 3:
            raise ValueError(
                f"""
                The input shape must have three elements ([depth, height, width]),
                but only {len(self.shape)} were passed.
                """
            )


@dataclass
class ExponentialParams:
    rate: float

    def __post_init__(self):
        if self.rate < 0:
            raise ValueError(
                f"Exponential rate parameter must be non-negative, got {self.rate}"
            )


@dataclass
class GammaParams:
    concentration: float
    rate: float

    def __post_init__(self):
        if self.concentration < 0:
            raise ValueError(
                f"""
                Gamma concentration parameter must be non-negative,
                got {self.concentration}
                """
            )
        if self.rate < 0:
            raise ValueError(
                f"Gamma rate parameter must be non-negative, got {self.rate}"
            )


@dataclass
class HalfCauchyParams:
    scale: float


@dataclass
class LogNormalParams:
    loc: float
    scale: float


@dataclass
class GaussianProfilePriorParams:
    """Prior params for the low-rank logistic-normal profile surrogate.

    The prior on h is N(0, sigma_prior² I).  sigma_prior is stored in
    ``profile_basis.pt`` — no params need to be configured here.
    This dataclass exists only so that ``pprf_cfg`` can carry a ``weight``.
    Leave ``params: {}`` in the YAML.
    """


@dataclass
class PriorConfig[P]:
    name: Literal[
        "dirichlet",
        "exponential",
        "gamma",
        "half_cauchy",
        "log_normal",
        "gaussian_profile",
    ]
    params: P
    weight: float
