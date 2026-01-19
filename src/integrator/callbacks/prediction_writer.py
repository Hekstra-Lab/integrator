from pathlib import Path

import numpy as np
import polars as plr
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.loggers import WandbLogger


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


# class PredWriter(BasePredictionWriter):
#     def __init__(
#         self,
#         output_dir: Path,
#         write_interval,
#     ):
#         super().__init__(write_interval)
#         self.output_dir = output_dir
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
#         # Getting log direcotory
#         logger = trainer.logger
#         if isinstance(logger, WandbLogger):
#             self.output_dir = logger.experiment.dir
#             Path(self.output_dir).mkdir(parents=True, exist_ok=True)
#         else:
#             self.output_dir = trainer.default_root_dir
#             Path(self.output_dir).mkdir(parents=True, exist_ok=True)
#
#         # Move predictions to CPU and save
#         prediction_cpu = {k: v.cpu().numpy() for k, v in prediction.items()}
#         torch.save(
#             prediction_cpu,
#             os.path.join(self.output_dir, f"batch_{batch_idx}.pt"),
#         )
#         del prediction
#
#     def write_on_epoch_end(
#         self,
#         trainer,
#         pl_module,
#         predictions,
#         batch_indices,
#     ):
#         merged_predictions = {}
#
#         for batch_prediction in predictions:
#             batch_cpu = dict()
#             for k, v in batch_prediction.items():
#                 if isinstance(v, torch.Tensor):
#                     batch_cpu[k] = v.cpu().numpy()
#                 elif isinstance(v, list):
#                     batch_cpu[k] = v
#
#             for key, value in batch_cpu.items():
#                 if key not in merged_predictions:
#                     merged_predictions[key] = []
#                 merged_predictions[key].append(value)
#
#         # Save the merged predictions as a single .pt file
#         preds_fname = self.output_dir / "preds.pt"
#         torch.save(merged_predictions, preds_fname)


class PredWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: Path,
        write_interval="batch",
        dtype=np.float32,
    ):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.dtype = dtype

    def _resolve_output_dir(self, trainer):
        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            out = Path(logger.experiment.dir)
        else:
            out = Path(trainer.default_root_dir)

        out.mkdir(parents=True, exist_ok=True)
        return out

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
        outdir = self._resolve_output_dir(trainer)

        for k, v in prediction.items():
            if isinstance(v, torch.Tensor):
                arr = v.detach().cpu().numpy().astype(self.dtype, copy=False)
                np.save(outdir / f"{k}_batch_{batch_idx:06d}.npy", arr)
            elif isinstance(v, list):
                # save lists separately if needed
                np.save(
                    outdir / f"{k}_batch_{batch_idx:06d}.npy",
                    np.array(v, dtype=object),
                )

        # Clean
        del prediction
        torch.cuda.empty_cache()

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices,
    ):
        return
