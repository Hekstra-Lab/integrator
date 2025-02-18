import torch
import os
from pytorch_lightning.callbacks import BasePredictionWriter
from pathlib import Path


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
                trainer.logger.log_dir, "predictions", "last"
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
            self.output_dir = trainer.logger.log_dir
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


#    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
#        if self.output_dir is None:
#            # Default to logger directory
#            self.output_dir = trainer.logger.log_dir
#            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
#
#
#        # Move all predictions to CPU and save
#        predictions_cpu = [
#            {k: v.cpu().numpy() for k, v in batch_prediction.items()}
#            for batch_prediction in predictions
#        ]
#        torch.save(predictions_cpu, os.path.join(self.output_dir, "preds.pt"))


# class PredWriter(BasePredictionWriter):
# def __init__(self, output_dir, write_interval):
# super().__init__(write_interval)
# self.output_dir = output_dir

# def write_on_batch_end(
# self,
# trainer,
# pl_module,
# prediction,
# batch_indices,
# batch,
# batch_idx,
# dataloader_idx,
# ):
# if self.output_dir is None:
# # get logger directory
# # defaults to storing the last epoch predictions
# self.output_dir = trainer.logger.log_dir + "/predictions/" + "/last/"
# # create directory to store predictions
# Path(self.output_dir).mkdir(parents=True, exist_ok=True)
# # save predictions
# torch.save(
# prediction,
# os.path.join(self.output_dir, f"batch_{batch_idx}.pt"),
# )

# def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
# if self.output_dir is None:
# # get logger directory
# self.output_dir = trainer.logger.log_dir
# # create directory to store predictions
# Path(self.output_dir).mkdir(parents=True, exist_ok=True)
# # save predictions
# torch.save(predictions, os.path.join(self.output_dir, "preds.pt"))

