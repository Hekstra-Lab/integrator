from dataclasses import dataclass
from typing import Literal


@dataclass
class IntegratorCfg:
    """Architecture and inference hyperparameters for an integrator `LightningModule`.

    Attributes:
        data_dim: Shoebox dimensionality, either `2d` or `3d`.
        d: Shoebox depth in pixels (number of z-slices); ignored for `2d` data but still validated positive.
        h: Shoebox height in pixels; must be positive.
        w: Shoebox width in pixels; must be positive.
        encoder_out: Width of the encoder embedding consumed by the surrogates.
        mc_samples: Monte Carlo samples drawn from each surrogate per forward pass; must be `>= 1`.
        predict_keys: Output columns to emit at predict time, or `default` for the built-in set.
    """

    data_dim: Literal["2d", "3d"]
    d: int
    h: int
    w: int
    encoder_out: int = 64
    mc_samples: int = 4
    predict_keys: Literal["default"] | list[str] = "default"

    def __post_init__(self):
        if self.data_dim not in ("2d", "3d"):
            raise ValueError(
                f"data_dim must be '2d' or '3d', got {self.data_dim!r}"
            )

        for name in ("d", "h", "w"):
            v = getattr(self, name)
            if v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")

        if self.mc_samples < 1:
            raise ValueError(f"mc_samples must be >= 1, got {self.mc_samples}")


@dataclass
class IntegratorConfig:
    """Registry selection for the integrator: a `name` plus its typed `args`.

    Attributes:
        name: Registry key naming the integrator class to construct.
        args: Integrator hyperparameters as an `IntegratorCfg`.
    """

    name: str
    args: IntegratorCfg

    def __post_init__(self):
        from integrator.registry import REGISTRY

        valid = REGISTRY["integrator"].keys()
        if self.name not in valid:
            raise ValueError(
                f"Unknown integrator '{self.name}'. "
                f"Available integrators: {sorted(valid)}"
            )
