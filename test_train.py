import glob

import numpy as np
from dials.array_family import flex
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from integrator.callbacks import IntensityPlotter, PredWriter
from integrator.utils import (
    clean_from_memory,
    create_data_loader,
    create_integrator,
    create_integrator_from_checkpoint,
    create_trainer,
    load_config,
    predict_from_checkpoints,
    reflection_file_writer,
)

# %%
config = "./src/integrator/configs/dev_config.yaml"
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

encoder_name = config["components"]["encoder"]["name"]
I_pairing_name = config["components"]["loss"]["params"]["I_pairing"]
bg_pairing_name = config["components"]["loss"]["params"]["bg_pairing"]
p_pairing_name = config["components"]["loss"]["params"]["p_pairing"]


logger = WandbLogger(
    project="integrator",
    name="Encoder_"
    + encoder_name
    + "_I_"
    + I_pairing_name
    + "_Bg_"
    + bg_pairing_name
    + "_P_"
    + p_pairing_name,
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
