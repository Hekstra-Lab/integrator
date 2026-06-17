from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainerConfig:
    """PyTorch Lightning `Trainer` settings.

    Attributes:
        max_epochs: Maximum number of training epochs; must be non-negative.
        accelerator: Lightning accelerator, `cpu`, `gpu`, or `auto`.
        devices: Number of devices to train on.
        logger: Whether Lightning's built-in logging is enabled.
        precision: Numerical precision, `16` or `32`.
        check_val_every_n_epoch: Run validation every N epochs.
        log_every_n_steps: Log scalar metrics every N steps.
        enable_checkpointing: Whether Lightning writes checkpoints.
        deterministic: Whether to force deterministic algorithms.
        gradient_clip_val: Gradient-clipping threshold, or `None` to disable.
        gradient_clip_algorithm: Clipping mode, `norm` or `value`; `None` to disable.
    """

    max_epochs: int = 100
    accelerator: Literal["cpu", "auto", "gpu"] = "auto"
    devices: int = 1
    logger: bool = True
    precision: Literal["16", "32"] = "32"
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 50
    enable_checkpointing: bool = True
    deterministic: bool = False
    gradient_clip_val: float | None = None
    gradient_clip_algorithm: Literal["norm", "value"] | None = None

    def __post_init__(self):
        if self.max_epochs < 0:
            raise ValueError(
                f"""
                    Epochs must be an integer value greater than 0:
                    epochs = {self.max_epochs}
                    """
            )
