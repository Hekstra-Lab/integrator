from dataclasses import dataclass
from typing import Literal


@dataclass
class IntegratorCfg:
    """Hyperparameters for an integrator `LightningModule`.

    Attributes:
        data_dim: Shoebox dimensionality, either `2d` or `3d`.
        d: Shoebox depth in pixels (number of z-slices); ignored for `2d` data but still validated positive.
        h: Shoebox height in pixels; must be positive.
        w: Shoebox width in pixels; must be positive.
        lr: Peak optimizer learning rate; must be positive.
        encoder_out: Width of the encoder embedding consumed by the surrogates.
        weight_decay: Adam weight decay applied to all parameters; must be non-negative.
        decoder_weight_decay: Separate weight decay for the `qp` decoder weight; `None` reuses `weight_decay`.
        qp_smoothness_weight: Penalty weight on profile-basis spatial smoothness; `None` disables it.
        qp_orthogonality_weight: Penalty weight on profile-basis column orthogonality; `None` disables it.
        lr_schedule: Schedule name `cosine_warmup` or `step_linear_warmup`, or `None` for a constant rate.
        warmup_epochs: Number of linear-warmup epochs used by `cosine_warmup`.
        warmup_steps: Number of linear-warmup steps used by `step_linear_warmup`.
        lr_min: Floor learning rate for the schedule; must be non-negative and `<= lr`.
        mc_samples: Monte Carlo samples drawn from each surrogate per forward pass; must be `>= 1`.
        predict_keys: Output columns to emit at predict time, or `default` for the built-in set.
    """

    data_dim: Literal["2d", "3d"]
    d: int
    h: int
    w: int
    lr: float = 0.001
    encoder_out: int = 64
    weight_decay: float = 0.0
    decoder_weight_decay: float | None = None
    qp_smoothness_weight: float | None = None
    qp_orthogonality_weight: float | None = None
    lr_schedule: Literal["cosine_warmup", "step_linear_warmup"] | None = None
    warmup_epochs: int = 5
    warmup_steps: int = 0
    lr_min: float = 1.0e-5
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

        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")

        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )

        if (
            self.decoder_weight_decay is not None
            and self.decoder_weight_decay < 0
        ):
            raise ValueError(
                "decoder_weight_decay must be non-negative, got "
                f"{self.decoder_weight_decay}"
            )

        for name in (
            "qp_smoothness_weight",
            "qp_orthogonality_weight",
        ):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{name} must be non-negative, got {v}")

        if self.warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be non-negative, got {self.warmup_epochs}"
            )
        if self.lr_min < 0:
            raise ValueError(f"lr_min must be non-negative, got {self.lr_min}")
        if self.lr_min > self.lr:
            raise ValueError(
                f"lr_min ({self.lr_min}) must be <= lr ({self.lr})"
            )

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
