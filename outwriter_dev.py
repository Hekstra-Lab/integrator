import os

import polars as plr
import torch

# This is the development script for writing outputs
# The OutWriter object should take as input the path where the output.pt files live
# The output dir may contain either a batch of output.pt files (one per batch), or a single output.pt file from entire epoch.

# Lets first open one of the predict() outputs

output_dir = "./logs/"


os.listdir(output_dir)


class OutWriter:
    def __init__(self, path):
        self.path = path


outputs = torch.load(output_dir + "batch_1.pt")
out_dict = dict(
    {
        "qI_mean": outputs["qI_mean"].numpy(),
        "qI_variance": outputs["qI_variance"].numpy(),
        "weighted_sum_mean": outputs["weighted_sum_mean"].numpy(),
        "thresholded_mean": outputs["thresholded_mean"].numpy(),
        "thresholded_var": outputs["thresholded_var"].numpy(),
        "refl_ids": outputs["refl_ids"].numpy(),
    }
)

plr.DataFrame(out_dict)
