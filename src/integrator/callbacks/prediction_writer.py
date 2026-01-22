from pathlib import Path

import h5py
import numpy as np
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


class BatchPredWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: Path,
        write_interval="batch",
        dtype=np.float32,
        epoch: int | None = None,
        filename: str | None = None,
    ):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.dtype = dtype
        self.epoch = epoch
        self.filename = filename or (
            f"preds_epoch_{epoch:04d}.h5" if epoch is not None else "preds.h5"
        )
        self._h5 = None

    def _open_file(self):
        if self._h5 is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._h5 = h5py.File(self.output_dir / self.filename, "w")

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
        self._open_file()

        # Flatten prediction structure
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

        if self.epoch is not None:
            batch_cpu["epoch"] = np.full(
                (next(iter(batch_cpu.values())).shape[0],),
                self.epoch,
                dtype=np.int32,
            )

        # Append to datasets
        for k, arr in batch_cpu.items():
            if k not in self._h5:
                self._h5.create_dataset(
                    k,
                    shape=(0, *arr.shape[1:]),
                    maxshape=(None, *arr.shape[1:]),
                    chunks=(arr.shape[0], *arr.shape[1:]),
                    dtype=arr.dtype,
                )

            dset = self._h5[k]
            n = dset.shape[0]
            dset.resize(n + arr.shape[0], axis=0)
            dset[n : n + arr.shape[0]] = arr

        del prediction
        torch.cuda.empty_cache()

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices,
    ):
        # Important: close file cleanly
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
