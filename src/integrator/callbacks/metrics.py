"""Lightweight per-step / per-epoch metric recorders (CSV or parquet).

Both recorders hold no GPU tensors across steps and flush to disk once per
epoch, so they add negligible overhead to training.
"""

from pathlib import Path

import polars as pl
import torch
from pytorch_lightning.callbacks import Callback


class LossTraceRecorder(Callback):
    """Record per-step loss components to CSV/parquet without slowing training.

    Accumulates scalar loss values in plain Python lists (no GPU tensors held),
    then flushes to disk once per epoch.

    Columns: step, loss, nll, kl, kl_prf, kl_i, kl_bg
    """

    _KEYS = ("loss", "nll", "kl", "kl_prf", "kl_i", "kl_bg")

    def __init__(
        self,
        out_dir: str | Path,
        use_parquet: bool = True,
    ):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.use_parquet = use_parquet
        self._rows: list[dict[str, float]] = []

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        lc = (
            outputs.get("loss_components")
            if isinstance(outputs, dict)
            else None
        )
        if lc is None:
            return
        row = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
        }
        for k in self._KEYS:
            v = lc.get(k)
            row[k] = float(v) if v is not None else float("nan")
        self._rows.append(row)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        lc = (
            outputs.get("loss_components")
            if isinstance(outputs, dict)
            else None
        )
        if lc is None:
            return
        row = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
        }
        for k in self._KEYS:
            v = lc.get(k)
            row[k] = float(v) if v is not None else float("nan")
        # tag so train/val are distinguishable if someone merges files
        row["split"] = "val"
        self._rows.append(row)

    def on_train_epoch_end(self, trainer, pl_module):
        self._flush(trainer, split="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._flush(trainer, split="val")

    def _flush(self, trainer, split: str):
        if not self._rows:
            return

        epoch = trainer.current_epoch
        suffix = "parquet" if self.use_parquet else "csv"
        fname = self.out_dir / f"loss_trace_{split}_epoch_{epoch:04d}.{suffix}"

        df = pl.DataFrame(self._rows)
        if self.use_parquet:
            df.write_parquet(fname)
        else:
            df.write_csv(fname)

        self._rows.clear()


class EpochMetricRecorder(Callback):
    def __init__(
        self,
        out_dir: str | Path,
        keys: list[str],
        split: str = "train",  # "train" or "val"
        every_n_epochs: int = 1,
        max_rows_per_epoch: int | None = None,
        use_parquet: bool = True,
    ):
        super().__init__()
        self.out_dir = out_dir
        self.keys = keys
        self.split = split
        self.every_n_epochs = every_n_epochs
        self.max_rows_per_epoch = max_rows_per_epoch
        self.use_parquet = use_parquet

        self.buffers: dict[str, list[torch.Tensor]] = {}
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    def _append(self, key, tensor):
        self.buffers.setdefault(key, []).append(tensor.detach())

    def _collect(self, outputs):
        out = outputs["forward_out"]
        for key in self.keys:
            if key in out:
                self._append(key, out[key])

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        if self.split == "train":
            self._collect(outputs)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        if self.split == "val":
            self._collect(outputs)

    def _flush(self, trainer):
        epoch = trainer.current_epoch

        if epoch % self.every_n_epochs != 0:
            self.buffers.clear()
            return

        if not self.buffers:
            return

        data = {}

        for key, chunks in self.buffers.items():
            x = torch.cat(chunks)

            if (
                self.max_rows_per_epoch
                and x.shape[0] > self.max_rows_per_epoch
            ):
                idx = torch.randperm(x.shape[0])[: self.max_rows_per_epoch]
                x = x[idx]

            data[key] = x.cpu().numpy()

        df = pl.DataFrame(data).select(
            pl.lit(epoch).alias("epoch"), pl.all()
        )

        suffix = "parquet" if self.use_parquet else "csv"
        fname = f"{self.out_dir}/{self.split}_epoch_{epoch:04d}.{suffix}"

        if self.use_parquet:
            df.write_parquet(fname)
        else:
            df.write_csv(fname)

        print(f"[Recorder] wrote {fname}")

        self.buffers.clear()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.split == "train":
            self._flush(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.split == "val":
            self._flush(trainer)
