"""Exponential moving average of the trainable parameters, with buffers following the live model."""

from __future__ import annotations

from typing import Any

import torch
from pytorch_lightning.callbacks import WeightAveraging


class ParameterEMA(WeightAveraging):
    """EMA of the floating-point parameters, theta_avg <- decay theta_avg + (1 - decay) theta, after every step.

    Lightning's `EMAWeightAveraging` averages buffers too and swaps them into
    the model at validation and at checkpoint time. Buffers here are not
    weights: the EB loss's prior (`eb_g`, `eb_h`, the boolean `eb_ready`) is set
    in closed form at epoch end, and its epoch accumulators must not be mixed
    across steps. So buffers are copied from the live model at every update and
    again before each swap and checkpoint, and only parameters are averaged.

    The averaged parameters are what validation sees and what every checkpoint
    holds in `state_dict` (the live ones go to `current_model_state`), so an
    export from a checkpoint uses the average with no extra step. Updates start
    at `start_epoch`.

    Why: at a constant learning rate the narrowest profile component's weight
    wandered with sd ~0.01 from epoch to epoch and the final epoch landed
    anywhere in that range; averaging epochs 39-59 by hand cut the replicate sd
    of the faint-intensity bias from 0.026 to 0.004.
    """

    def __init__(self, decay: float = 0.999, start_epoch: int = 0):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema decay must be in (0, 1), got {decay}")
        self.decay, self.start_epoch = float(decay), int(start_epoch)
        self._buffer_ptrs: set[int] = set()
        super().__init__(multi_avg_fn=self._ema, use_buffers=True)

    def should_update(self, step_idx: int | None = None, epoch_idx: int | None = None) -> bool:
        return step_idx is not None and self._epoch >= self.start_epoch

    def setup(self, trainer, pl_module, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        self._epoch = 0
        if self._average_model is not None:
            self._buffer_ptrs = {b.data_ptr() for b in self._average_model.module.buffers()}

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch = trainer.current_epoch

    def _ema(self, averaged: list[torch.Tensor], current: list[torch.Tensor], n_averaged) -> None:
        for a, c in zip(averaged, current):
            if a.data_ptr() in self._buffer_ptrs or not a.is_floating_point():
                a.copy_(c)
            else:
                a.mul_(self.decay).add_(c, alpha=1.0 - self.decay)

    def _sync_buffers(self, pl_module) -> None:
        if self._average_model is None:
            return
        with torch.no_grad():
            for a, c in zip(self._average_model.module.buffers(), pl_module.buffers()):
                a.copy_(c)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._sync_buffers(pl_module)
        super().on_validation_epoch_start(trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict[str, Any]) -> None:
        self._sync_buffers(pl_module)
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def on_train_end(self, trainer, pl_module) -> None:
        self._sync_buffers(pl_module)
        super().on_train_end(trainer, pl_module)
