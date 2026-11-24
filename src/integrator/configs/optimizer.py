from dataclasses import dataclass
from typing import Literal


@dataclass
class OptimizerConfig:
    """Optimizer and learning-rate-schedule settings for an integrator.

    Attributes:
        lr: Peak optimizer learning rate; must be positive.
        weight_decay: Adam weight decay applied to all parameters; must be non-negative.
        decoder_weight_decay: Separate weight decay for the `qp` decoder weight; `None` reuses `weight_decay`.
        lr_schedule: Schedule name `cosine_warmup` or `step_linear_warmup`, or `None` for a constant rate.
        warmup_epochs: Number of linear-warmup epochs used by `cosine_warmup`.
        warmup_steps: Number of linear-warmup steps used by `step_linear_warmup`.
        lr_min: Floor learning rate for the schedule; must be non-negative and `<= lr`.
        loss_lr: Learning rate for every `loss.*` parameter (the Wilson B and G(lambda)
            coefficients, a learned background map), in a group of its own with no weight
            decay; a schedule scales it by the same factor as `lr`. `None` (default) keeps
            them in the groups and at the rate they have always had. Under Adam a parameter
            whose gradient keeps one sign moves about `lr` per step whatever the gradient's
            size, so at lr 1e-3 the Wilson B rose a constant 0.18/epoch for 60 epochs and
            never converged.
    """

    lr: float = 0.001
    weight_decay: float = 0.0
    decoder_weight_decay: float | None = None
    lr_schedule: Literal["cosine_warmup", "step_linear_warmup"] | None = None
    warmup_epochs: int = 5
    warmup_steps: int = 0
    lr_min: float = 1.0e-5
    loss_lr: float | None = None

    def __post_init__(self):
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

        if self.warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be non-negative, got {self.warmup_epochs}"
            )
        if self.loss_lr is not None and self.loss_lr <= 0:
            raise ValueError(f"loss_lr must be positive, got {self.loss_lr}")
        if self.lr_min < 0:
            raise ValueError(f"lr_min must be non-negative, got {self.lr_min}")
        if self.lr_min > self.lr:
            raise ValueError(
                f"lr_min ({self.lr_min}) must be <= lr ({self.lr})"
            )
