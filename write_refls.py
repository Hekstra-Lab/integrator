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

config= '/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/lightning_logs/wandb/run-20250421_121835-mxp871sk/files/config_copy.yaml'

config = load_config(config)

pred_dirs = ['/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/lightning_logs/wandb/run-20250421_121835-mxp871sk/files/predictions/epoch_5/']

prediction_files = glob.glob(pred_dirs[0] + "*.pt")

print(prediction_files)

reflection_file_writer(
    pred_dirs,
    prediction_files,
    "/n/holylabs/LABS/hekstra_lab/Users/laldama/integratorv2/integrator/data/pass1/reflections_.refl"
)


