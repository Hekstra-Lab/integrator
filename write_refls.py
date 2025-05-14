from integrator.callbacks import PredWriter
import yaml
import json
from pytorch_lightning.callbacks import Callback
import os
import glob
from integrator.utils import (
    load_config,
    create_integrator,
    create_integrator_from_checkpoint,
    create_data_loader,
    create_trainer,
    parse_args,
    override_config,
    clean_from_memory,
    predict_from_checkpoints,
    reflection_file_writer,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import torch
import subprocess

# from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from integrator.callbacks import (
    IntensityPlotter,
    MVNPlotter,
    UNetPlotter,
    IntegratedPlotter,
)
import argparse


if __name__ == "__main__":
    # load data
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--path",
        type=str,
    )

    args = argparser.parse_args()

    path = Path(args.path)
    pred_path = path.as_posix() + "/files/predictions"
    pred_dirs = glob.glob(pred_path + "/epoch*")
    prediction_files = glob.glob(pred_path + "/epoch*/*.pt")

    print(prediction_files)
    print(pred_dirs)

    reflection_file_writer(
        pred_dirs,
        prediction_files,
        "/n/holylabs/LABS/hekstra_lab/Users/laldama/integratorv2/integrator/data/pass1/reflections_.refl"
    )


