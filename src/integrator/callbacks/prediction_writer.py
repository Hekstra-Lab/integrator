from pathlib import Path

import numpy as np
import polars as pl
import polars as plr
import torch
from pytorch_lightning.callbacks import BasePredictionWriter


def assign_labels(
    dataset,
    save_dir: str,
):
    train_id_df = plr.DataFrame(schema=[("train_ids", int)])
    val_id_df = plr.DataFrame(schema=[("val_ids", int)])

    with torch.no_grad():
        for batch in dataset.train_dataloader():
            _, _, _, reference = batch
            train_ids = plr.DataFrame(
                {"train_ids": (reference["refl_ids"].int()).tolist()}
            )
            train_id_df = plr.concat([train_id_df, train_ids])

        for batch in dataset.val_dataloader():
            _, _, _, reference = batch
            val_ids = plr.DataFrame(
                {"val_ids": (reference["refl_ids"].int()).tolist()}
            )
            val_id_df = plr.concat([val_id_df, val_ids])

    train_id_df.write_csv(save_dir + "/train_labels.csv")
    print(f"train labels saved to {save_dir + '/train_labels.csv'}")
    val_id_df.write_csv(save_dir + "/val_labels.csv")
    print(f"val labels saved to {save_dir + '/val_labels.csv'}")


class EpochPredWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: Path,
        write_interval,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices,
    ):
        merged_predictions = {}

        for batch_prediction in predictions:
            batch_cpu = dict()
            for k, v in batch_prediction.items():
                if isinstance(v, torch.Tensor):
                    batch_cpu[k] = v.cpu().numpy()
                elif isinstance(v, list):
                    batch_cpu[k] = v

            for key, value in batch_cpu.items():
                if key not in merged_predictions:
                    merged_predictions[key] = []
                merged_predictions[key].append(value)

        # Save the merged predictions as a single .pt file
        preds_fname = self.output_dir / "preds.pt"
        torch.save(merged_predictions, preds_fname)


## Parquet writer
# class BatchPredWriter(BasePredictionWriter):
#     def __init__(
#         self,
#         output_dir: Path,
#         write_interval="batch",
#         dtype=np.float32,
#         epoch: int | None = None,
#         filename_prefix: str = "preds",
#     ):
#         super().__init__(write_interval)
#         self.output_dir = Path(output_dir)
#         self.dtype = dtype
#         self.epoch = epoch
#         self.filename_prefix = filename_prefix
#
#     def write_on_batch_end(
#         self,
#         trainer,
#         pl_module,
#         prediction,
#         batch_indices,
#         batch,
#         batch_idx,
#         dataloader_idx,
#     ):
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#
#         batch_cpu = {}
#
#         # ---- flatten prediction dict (unchanged logic) ----
#         for k, v in prediction.items():
#             if isinstance(v, torch.Tensor):
#                 batch_cpu[k] = (
#                     v.detach().cpu().numpy().astype(self.dtype, copy=False)
#                 )
#
#             elif isinstance(v, dict):
#                 for k_, v_ in v.items():
#                     key = f"{k}.{k_}"
#                     batch_cpu[key] = (
#                         v_.detach()
#                         .cpu()
#                         .numpy()
#                         .astype(self.dtype, copy=False)
#                     )
#
#             elif isinstance(v, list):
#                 batch_cpu[k] = np.asarray(v)
#
#         n_rows = next(iter(batch_cpu.values())).shape[0]
#
#         if self.epoch is not None:
#             batch_cpu["epoch"] = np.full((n_rows,), self.epoch, dtype=np.int32)
#
#         df = pl.DataFrame(batch_cpu)
#
#         rank = trainer.global_rank
#         fname = (
#             f"{self.filename_prefix}"
#             f"_epoch={self.epoch:04d}"
#             f"_rank={rank}"
#             f"_batch={batch_idx:06d}.parquet"
#         )
#
#         path = self.output_dir / fname
#         df.write_parquet(path)
#
#         del prediction
#         torch.cuda.empty_cache()


# TODO: Add argument to select writer based of pred.py --save-preds-as argument
# We should have a writer for .h5, .parquet, and .pt files


## Parquet writer
class BatchPredWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: Path,
        write_interval="batch",
        dtype=np.float32,
        epoch: int | None = None,
        filename_prefix: str = "preds",
        flush_every: int = 10,
    ):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.dtype = dtype
        self.epoch = epoch
        self.filename_prefix = filename_prefix
        self.flush_every = flush_every

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

        if len(self._buffer) >= self.flush_every:
            self._flush(trainer)

        del prediction
        torch.cuda.empty_cache()

    def on_predict_epoch_end(self, trainer, pl_module):
        self._flush(trainer)
