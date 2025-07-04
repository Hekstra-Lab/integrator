import os
from pathlib import Path

import polars as plr
import torch
from pytorch_lightning.callbacks import BasePredictionWriter, Callback


def assign_labels(dataset, save_dir: str):
    train_id_df = plr.DataFrame(schema=[("train_ids", int)])
    val_id_df = plr.DataFrame(schema=[("val_ids", int)])

    with torch.no_grad():
        for batch in dataset.train_dataloader():
            _, _, _, reference = batch
            train_ids = plr.DataFrame({"train_ids": (reference[:, -1].int()).tolist()})
            train_id_df = plr.concat([train_id_df, train_ids])

        for batch in dataset.val_dataloader():
            _, _, _, reference = batch
            val_ids = plr.DataFrame({"val_ids": (reference[:, -1].int()).tolist()})
            val_id_df = plr.concat([val_id_df, val_ids])

    train_id_df.write_csv(save_dir + "/train_labels.csv")
    print(f"train labels saved to {save_dir + '/train_labels.csv'}")
    val_id_df.write_csv(save_dir + "/val_labels.csv")
    print(f"val labels saved to {save_dir + '/val_labels.csv'}")


class IntensityPlotter(Callback):
    # def on_validation_epoch_end(self,trainer, pl_module):
    def __init__(self):
        super().__init__()
        self.batch_predictions = []
        self.val_predictions = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            predictions = pl_module(batch)
            self.batch_predictions.append(predictions)


class PredWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

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
        if self.output_dir is None:
            # Default to logger directory
            self.output_dir = os.path.join(
                # trainer.logger.log_dir, "predictions", "last"
                trainer.logger.experiment.dir,
                "predictions",
                "last",
            )
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Move predictions to CPU and save
        prediction_cpu = {
            k: v.cpu().numpy() for k, v in prediction.items()
        }  # Ensure CPU transfer
        torch.save(
            prediction_cpu,
            os.path.join(self.output_dir, f"batch_{batch_idx}.pt"),
        )
        del prediction

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        if self.output_dir is None:
            # Default to logger directory
            # self.output_dir = trainer.logger.log_dir
            self.output_dir = trainer.logger.experiment.dir
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize a merged dictionary where each key accumulates all values in a list
        merged_predictions = {}

        for batch_prediction in predictions:
            batch_cpu = {k: v.cpu().numpy() for k, v in batch_prediction.items()}

            for key, value in batch_cpu.items():
                if key not in merged_predictions:
                    merged_predictions[key] = []  # Initialize list for this key
                merged_predictions[key].append(value)  # Append batch values to list

        # Save the merged predictions as a single .pt file
        torch.save(merged_predictions, os.path.join(self.output_dir, "preds.pt"))
