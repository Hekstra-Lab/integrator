# Updated 2024-07-25
# This uses the new DataLoader

import os
import glob
import torch
import numpy as np
from dials.array_family import flex
from integrator.models import (
    Encoder,
    PoissonLikelihoodV2,
    DistributionBuilder,
)
from integrator.layers import Standardize
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import polars as pl
from integrator.layers import Standardize
from integrator.models import IntegratorModel
from integrator.io import ShoeboxDataModule


# %%
# Hyperparameters
depth = 10
dmodel = 64
feature_dim = 7
dropout = 0.5
beta = 1.0
mc_samples = 100
max_size = 1024
eps = 1e-5
batch_size = 50
learning_rate = 0.001
epochs = 2

# Load training data

# Device to train on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

shoebox_file = "./out/shoebox_tensor.pt"
metadata_file = "./out/metadata_tensor.pt"
is_flat_file = "./out/is_flat_tensor.pt"
dead_pixel_mask_file = "./out/padded_dead_pixel_mask_tensor.pt"

# Initialize the DataModule
data_module = ShoeboxDataModule(
    shoebox_data=shoebox_file,
    metadata=metadata_file,
    is_flat=is_flat_file,
    dead_pixel_mask=dead_pixel_mask_file,
    batch_size=32,
    val_split=0.2,
    test_split=0.1,
    include_test=False,
    subset_size=100,
)

# Setup data module
data_module.setup()

# Length (number of samples) of train loader
train_loader_len = len(data_module.train_dataloader())


# Intensity variational distribution
intensity_dist = torch.distributions.gamma.Gamma

# Background variational distribution
background_dist = torch.distributions.gamma.Gamma

# Intensity prior distribution
prior_I = torch.distributions.exponential.Exponential(rate=torch.tensor(1.0))

# Concentration for background prior distribution
concentration = torch.tensor([1.0], device=device)

# Rate for background prior distribution
rate = torch.tensor([1.0], device=device)

# Background prior distribution
prior_bg = torch.distributions.gamma.Gamma(concentration, rate)

# Standardization module
standardization = Standardize(max_counts=train_loader_len)

# Encoder module
encoder = Encoder(depth, dmodel, feature_dim, dropout=dropout)

# Distribution builder module
distribution_builder = DistributionBuilder(
    dmodel, intensity_dist, background_dist, eps, beta
)

# Loss likelihood module
poisson_loss = PoissonLikelihoodV2(
    beta=beta,
    eps=eps,
    prior_I=prior_I,
    prior_bg=prior_bg,
    p_I_scale=0.0001,
    p_bg_scale=0.0001,
)

# Number of steps to train for
steps = 1000 * train_loader_len

# Integration model
model = IntegratorModel(
    encoder,
    distribution_builder,
    poisson_loss,
    standardization,
    total_steps=steps,
    n_cycle=4,
    lr=learning_rate,
    anneal=False,
    max_epochs=epochs,
)

# Logging
logger = TensorBoardLogger(save_dir="./integrator_logs", name="integrator_model")

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",
    dirpath="./integrator_logs/checkpoints/",
    filename="integrator-{epoch:02d}-{train_loss:.2f}",
    save_top_k=3,
    mode="min",
)

# Progress bar
progress_bar = TQDMProgressBar(refresh_rate=1)


# Training module
trainer = Trainer(
    max_epochs=epochs,
    accelerator="cpu",  # Use "cpu" for CPU training
    devices="auto",
    num_nodes=1,
    precision="32",  # Use 32-bit precision for CPU
    accumulate_grad_batches=1,
    check_val_every_n_epoch=1,
    callbacks=[checkpoint_callback, progress_bar],
    logger=logger,
    log_every_n_steps=10,
)

# Train the model
trainer.fit(model, data_module)

# %%
# Code to store outputs

refl_ids_train = np.array(model.training_preds["refl_id"], dtype=np.int32)
tbl_ids_train = np.array(model.training_preds["tbl_id"], dtype=np.int32)
refl_ids_val = np.array(model.validation_preds["refl_id"], dtype=np.int32)
tbl_ids_val = np.array(model.validation_preds["tbl_id"], dtype=np.int32)

# Training predictions
train_res_df = pl.DataFrame(
    {
        "tbl_id": tbl_ids_train,
        "refl_id": refl_ids_train,
        "q_I_mean": model.training_preds["q_I_mean"],
        "q_I_stddev": model.training_preds["q_I_stddev"],
    }
)

# Validation predictions
val_res_df = pl.DataFrame(
    {
        "tbl_id": tbl_ids_val,
        "refl_id": refl_ids_val,
        "q_I_mean": model.validation_preds["q_I_mean"],
        "q_I_stddev": model.validation_preds["q_I_stddev"],
    }
)

# Concatenate train_res_df and val_res_df
res_df = pl.concat([train_res_df, val_res_df])

tbl_ids = res_df["tbl_id"].unique().to_list()

# Directory to shoeboxes
refl_dir = "/Users/luis/integrator/rotation_data_examples/data/"

# shoebox filenames
filenames = glob.glob(os.path.join(refl_dir, "shoebox*"))

refl_tables = [flex.reflection_table.from_file(filename) for filename in filenames]

# Iterate over reflection id
for tbl_id in tbl_ids:
    sel = np.asarray([False] * len(refl_tables[tbl_id]))

    filtered_df = res_df.filter(res_df["tbl_id"] == tbl_id)

    # Reflection ids
    reflection_ids = filtered_df["refl_id"].to_list()

    # Intensity predictions
    intensity_preds = filtered_df["q_I_mean"].to_list()
    intensity_stddev = filtered_df["q_I_stddev"].to_list()

    for i, id in enumerate(reflection_ids):
        sel[id] = True

    refl_temp_tbl = refl_tables[tbl_id].select(flex.bool(sel))

    refl_temp_tbl["intensity.sum.value"] = flex.double(intensity_preds)

    refl_temp_tbl["intensity.sum.variance"] = flex.double(intensity_stddev)

    # save the updated reflection table
    refl_temp_tbl.as_file(f"integrator_preds_{tbl_id}_test.refl")