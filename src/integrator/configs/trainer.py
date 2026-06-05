from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainerConfig:
    max_epochs: int
    accelerator: Literal["cpu", "auto", "gpu"]
    devices: int
    logger: bool
    precision: Literal["16", "32"]
    check_val_every_n_epoch: int
    log_every_n_steps: int
    enable_checkpointing: bool
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


@dataclass
class CheckpointConfig:
    """The `checkpoint:` YAML section — pytorch_lightning ModelCheckpoint.

    Built into the ModelCheckpoint callback in cli/train.py (the Trainer itself
    only has the on/off switch `enable_checkpointing`; save frequency/retention
    are callback settings, which is why they live here, not in `trainer:`).

    `None` for save_top_k / monitor / mode means "derive from the early_stop
    config when present, else the legacy default" (see cli/train.py).
    """

    every_n_epochs: int = 1
    save_top_k: int | None = None
    monitor: str | None = None
    mode: str | None = None
    save_last: bool | str | None = "link"

    def __post_init__(self):
        if self.every_n_epochs < 1:
            raise ValueError(
                "checkpoint.every_n_epochs must be >= 1, got "
                f"{self.every_n_epochs}"
            )
        if self.mode is not None and self.mode not in ("min", "max"):
            raise ValueError(
                f"checkpoint.mode must be 'min' or 'max', got {self.mode!r}"
            )
        if self.save_last not in (True, False, None, "link"):
            raise ValueError(
                "checkpoint.save_last must be true, false, null, or 'link', "
                f"got {self.save_last!r}"
            )
