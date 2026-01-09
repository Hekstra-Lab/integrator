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
    deterministic: bool
    enable_checkpointing: bool

    def __post_init__(self):
        if self.max_epochs < 0:
            raise ValueError(
                f"""
                    Epochs must be an integer value greater than 0:
                    epochs = {self.max_epochs}
                    """
            )
