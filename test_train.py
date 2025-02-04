from integrator.callbacks import PredWriter
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
from pytorch_lightning.callbacks import ModelCheckpoint

config = "./src/integrator/configs/config.yaml"
config = load_config(config)

data = create_data_loader(config)

integrator = create_integrator(config)

# Create callbacks

## create prediction callback
pred_writer = PredWriter(
    output_dir=None,
    write_interval=config["trainer"]["params"]["callbacks"]["pred_writer"][
        "write_interval"
    ],
)

## create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filename="{epoch}-{val_loss:.2f}",
    every_n_epochs=2,
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
    ],
)

# Fit the model
trainer.fit(
    integrator,
    train_dataloaders=data.train_dataloader(),
    val_dataloaders=data.val_dataloader(),
)

# Create trainer from checkpoints
pred_integrator = create_integrator_from_checkpoint(
    config,
    trainer.logger.log_dir + "/checkpoints/last.ckpt",
)

# Predict using the latest weights
trainer.predict(
    pred_integrator,
    return_predictions=False,
    dataloaders=data.predict_dataloader(),
)

version_dir = trainer.logger.log_dir
path = version_dir + "/checkpoints/epoch*.ckpt"

# override to stop new version dirs from being created
config["trainer"]["params"]["logger"] = False

# clean from memory
clean_from_memory(trainer, pred_writer, pred_writer, checkpoint_callback)

# predict from checkpoints
predict_from_checkpoints(config, data, version_dir, path)

prediction_path = version_dir + "/predictions/"
prediction_directories = glob.glob(prediction_path + "epoch*")
prediction_files = glob.glob(prediction_path + "epoch*/*.pt")

# write reflections
reflection_file_writer(prediction_directories, prediction_files)

# save config as yaml to version_dir
import yaml

config_path = version_dir + "/config.yaml"
with open(config_path, "w") as file:
    yaml.dump(config, file)
