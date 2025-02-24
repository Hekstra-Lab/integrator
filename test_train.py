from integrator.callbacks import PredWriter
from dials.array_family import flex
import numpy as np
import torch
import glob
from integrator.utils import (
    load_config,
    create_integrator,
    create_integrator_from_checkpoint,
    create_data_loader,
    create_trainer,
    clean_from_memory,
    predict_from_checkpoints,
    reflection_file_writer,
)
from integrator.callbacks import IntensityPlotter
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import RichProgressBar
import torchvision.transforms.functional as F

# %%
config = "./src/integrator/configs/config_3d_cnn.yaml"
config = load_config(config)
data = create_data_loader(config)

# %%
# TODO: Incorporate this code block into the logging function.
# This code block gets the resolution vector for each reflection id
# Use this to plot data as a function of resolution

# getting refl ids
refl_tbl = flex.reflection_table.from_file(config["output"]["refl_file"])

refl_ids = []
for batch in data.train_dataloader():
    refl_ids.extend(batch[1][:, 4].int().tolist())

sel = np.asarray([False] * len(refl_tbl))
for id in refl_ids:
    sel[id] = True

refl_tbl_subset = refl_tbl.select(flex.bool(sel))

# %%
logger = WandbLogger(
    project="integrator",
    name="test-3d_encoder-local-3",
    save_dir="lightning_logs",
)

logdir = logger.experiment.dir

integrator = create_integrator(config)

# Create callbacks
## create prediction callback
pred_writer = PredWriter(
    output_dir=None,
    write_interval=config["trainer"]["params"]["callbacks"]["pred_writer"][
        "write_interval"
    ],
)

plotter = IntensityPlotter(num_profiles=10)

## create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=logger.experiment.dir + "/checkpoints",  # when using wandb logger
    filename="{epoch}-{val_loss:.2f}",
    every_n_epochs=1,
    save_top_k=-1,
    save_last="link",
)

# Create trainer
trainer = create_trainer(
    config,
    data,
    callbacks=[
        pred_writer,
        checkpoint_callback,
        plotter,
        RichProgressBar(),
    ],
    logger=logger,
)

# Fit the model
trainer.fit(
    integrator,
    train_dataloaders=data.train_dataloader(),
    val_dataloaders=data.val_dataloader(),
)

# %%
# Create trainer from checkpoints
pred_integrator = create_integrator_from_checkpoint(
    config,
    logdir + "/checkpoints/last.ckpt",
)

# Predict using the latest weights
trainer.predict(
    pred_integrator,
    return_predictions=False,
    dataloaders=data.predict_dataloader(),
    ckpt_path=logdir + "/checkpoints/last.ckpt",
)

version_dir = logdir
path = version_dir + "/checkpoints/epoch*.ckpt"

# override to stop new version dirs from being created
config["trainer"]["params"]["logger"] = False

# clean from memory
clean_from_memory(trainer, pred_writer, pred_writer, checkpoint_callback)

# %%
# predict from checkpoints
predict_from_checkpoints(config, trainer, pred_integrator, data, version_dir, path)

prediction_path = version_dir + "/predictions/"
prediction_directories = glob.glob(prediction_path + "epoch*")
prediction_files = glob.glob(prediction_path + "epoch*/*.pt")

# write reflections
reflection_file_writer(
    prediction_directories,
    prediction_files,
    config["output"]["refl_file"],
)

# %%
# save config as yaml to version_dir
import yaml

config_path = version_dir + "/config.yaml"
with open(config_path, "w") as file:
    yaml.dump(config, file)
