from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

P = TypeVar("P")


@dataclass
class DirichletParams:
    """Parameters of a Dirichlet profile prior.

    Attributes:
        concentration: Scalar concentration, or a path to a precomputed concentration tensor.
        shape: Profile shape `(depth, height, width)`.
        quantile: Optional quantile in `(0, 1)` used to set the concentration from data.
        conc_factor: Scale applied to the data-derived concentration.
    """

    concentration: float | str
    shape: tuple[int, int, int]
    quantile: float | None = None
    conc_factor: float = 40.0

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

        if self.quantile is not None and not 0.0 < self.quantile < 1.0:
            raise ValueError(
                f"quantile must be in (0, 1), got {self.quantile}"
            )


@dataclass
class GammaParams:
    """Parameters of a Gamma prior.

    Attributes:
        concentration: Gamma shape parameter; must be non-negative.
        rate: Gamma rate parameter; must be non-negative.
    """

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
class PriorConfig[P]:
    """A weighted prior selection, generic over its parameter payload `P`.

    Attributes:
        name: Prior family, `dirichlet`, `exponential`, or `gamma`.
        params: Family-specific parameters; a `DirichletParams` or `GammaParams` depending on `name`.
        weight: Scalar weight applied to this prior's KL term.
    """

    name: Literal[
        "dirichlet",
        "exponential",
        "gamma",
    ]
    params: P
    weight: float
