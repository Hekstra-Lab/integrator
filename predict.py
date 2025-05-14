import torch
import glob
import re
from pathlib import Path
import gc
from integrator.callbacks import PredWriter
from integrator.utils import (
    create_integrator_from_checkpoint,
    load_config,
    create_data_loader,
    create_trainer,
    create_integrator
)

torch.set_float32_matmul_precision("medium")

# Create configuration file

config = load_config(
'/n/hekstra_lab/people/aldama/lightning_logs/wandb/run-20250501_165045-l8e9r0ox'
)

# update batch_size
config["data_loader"]["params"]["batch_size"] = 1000
 
# create DataLoader
data = create_data_loader(config)

# path to weights
checkpoint_path = '/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/lightning_logs/wandb/run-20250421_121835-mxp871sk/files/checkpoints/epoch=8-val_loss=0.00.ckpt'

checkpoint = torch.load(
        checkpoint_path,
        weights_only=False
        )

# create integrator
integrator = create_integrator(config)

# load checkpoint weights
integrator.load_state_dict(checkpoint['state_dict'])

# create prediction writer
pred_writer = PredWriter(
    output_dir='./pred_10k/',
    write_interval="epoch"
)

# create trainer object
trainer = create_trainer(
    config,
    data,
    callbacks=[
        pred_writer,
    ],
)

# prediction step
trainer.predict(
    integrator,
    return_predictions=False,
    dataloaders=data.predict_dataloader(),
)

