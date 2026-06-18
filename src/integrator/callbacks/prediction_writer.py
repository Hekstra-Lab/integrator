import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter

logger = logging.getLogger(__name__)


def assign_labels(
    dataset,
    save_dir: str,
):
    train_id_df = pl.DataFrame(schema=[("train_ids", int)])
    val_id_df = pl.DataFrame(schema=[("val_ids", int)])

    with torch.no_grad():
        for batch in dataset.train_dataloader():
            _, _, _, reference = batch
            refl_key = "refl_ids" if "refl_ids" in reference else "refl_id"
            train_ids = pl.DataFrame(
                {"train_ids": (reference[refl_key].int()).tolist()}
            )
            train_id_df = pl.concat([train_id_df, train_ids])

        for batch in dataset.val_dataloader():
            _, _, _, reference = batch
            refl_key = "refl_ids" if "refl_ids" in reference else "refl_id"
            val_ids = pl.DataFrame(
                {"val_ids": (reference[refl_key].int()).tolist()}
            )
            val_id_df = pl.concat([val_id_df, val_ids])

    train_id_df.write_csv(save_dir + "/train_labels.csv")
    logger.debug("train labels saved to %s/train_labels.csv", save_dir)
    val_id_df.write_csv(save_dir + "/val_labels.csv")
    logger.debug("val labels saved to %s/val_labels.csv", save_dir)


class BatchPredWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: Path,
        write_interval="batch",
        dtype=np.float32,
        epoch: int | None = None,
        filename_prefix: str = "preds",
        flush_every: int = 10,
        partition: bool = False,
    ):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.dtype = dtype
        self.epoch = epoch
        self.filename_prefix = filename_prefix
        self.flush_every = flush_every
        self.partition = partition

        self._buffer: list[pl.DataFrame] = []
        self._flush_idx = 0

    def _flush(self, trainer):
        if not self._buffer:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        df = pl.concat(self._buffer, rechunk=True)
        self._buffer.clear()

        rank = trainer.global_rank
        fname = (
            f"{self.filename_prefix}"
            f"_epoch_{self.epoch:04d}"
            f"_rank={rank}"
            f"_flush={self._flush_idx:06d}.parquet"
        )

        df.write_parquet(self.output_dir / fname)
        self._flush_idx += 1

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        batch_cpu = {}

        for k, v in prediction.items():
            if isinstance(v, torch.Tensor):
                batch_cpu[k] = (
                    v.detach().cpu().numpy().astype(self.dtype, copy=False)
                )

            elif isinstance(v, dict):
                for k_, v_ in v.items():
                    key = f"{k}.{k_}"
                    batch_cpu[key] = (
                        v_.detach()
                        .cpu()
                        .numpy()
                        .astype(self.dtype, copy=False)
                    )

            elif isinstance(v, list):
                batch_cpu[k] = np.asarray(v)

        n_rows = next(iter(batch_cpu.values())).shape[0]

        if self.epoch is not None:
            batch_cpu["epoch"] = np.full((n_rows,), self.epoch, dtype=np.int32)

        self._buffer.append(pl.DataFrame(batch_cpu))

        # only shard when partitioning; otherwise accumulate for one file
        if self.partition and len(self._buffer) >= self.flush_every:
            self._flush(trainer)

        del prediction
        torch.cuda.empty_cache()

    def on_predict_epoch_end(self, trainer, pl_module):
        if self.partition:
            self._flush(trainer)
        elif self._buffer:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            df = pl.concat(self._buffer, rechunk=True)
            self._buffer.clear()
            df.write_parquet(self.output_dir / "pred.parquet")
