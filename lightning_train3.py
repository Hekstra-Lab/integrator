# Updated 2024-07-30
# This uses the uniform sized shoeboxes

import os
import glob
import torch
import numpy as np
from dials.array_family import flex
from rs_distributions import distributions as rsd
from integrator.models import (
    Encoder,
    PoissonLikelihoodV2,
    Builder,
)
from integrator.layers import Standardize
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import polars as pl
from integrator.layers import Standardize
from integrator.io import ShoeboxDataModule
import pickle

# %%
# Hyperparameters
depth = 10
dmodel = 32
feature_dim = 7
dropout = 0.5
beta = 1.0
mc_samples = 100
max_size = 1024
eps = 1e-5
batch_size = 50
learning_rate = 0.001
epochs = 500

# %%
# Load training data

# Device to train on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

shoebox_file = "./samples.pt"
metadata_file = "./metadata.pt"
dead_pixel_mask_file = "./masks.pt"
is_flat_file = "./out/is_flat_tensor.pt"

size = torch.load(metadata_file).size(0)
is_flat = torch.ones(size, dtype=torch.bool).unsqueeze(1)

sboxes = torch.load(shoebox_file)

# Initialize the DataModule
data_module = ShoeboxDataModule(
    shoebox_data=shoebox_file,
    metadata=metadata_file,
    is_flat=is_flat,
    dead_pixel_mask=dead_pixel_mask_file,
    batch_size=3,
    val_split=0.2,
    test_split=0.1,
    include_test=False,
    subset_size=10,
    single_sample_index=None,
)

# Setup data module
data_module.setup()

# %%
# Length (number of samples) of train loader
train_loader_len = len(data_module.train_dataloader())

# Intensity variational distribution
intensity_dist = torch.distributions.gamma.Gamma

# Background variational distribution
background_dist = torch.distributions.gamma.Gamma

# Intensity prior distribution
prior_I = torch.distributions.exponential.Exponential(rate=torch.tensor(0.05))

# Concentration for background prior distribution
concentration = torch.tensor([1.0], device=device)

# Rate for background prior distribution
rate = torch.tensor([1.0], device=device)

# Background prior distribution
# prior_bg = torch.distributions.gamma.Gamma(concentration, rate)
# prior_bg = torch.distributions.exponential.Exponential(rate=torch.tensor(100.0))
# prior_bg = torch.distributions.log_normal.LogNormal(0, 0.1)
prior_bg = rsd.FoldedNormal(0, 0.1)
# prior_bg = torch.distributions.uniform.Uniform(0.0, 3.0)

# Standardization module
standardization = Standardize(max_counts=train_loader_len)

# Encoder module
encoder = Encoder(depth, dmodel, feature_dim, dropout=dropout)

# Distribution builder module
distribution_builder = Builder(
    dmodel, intensity_dist, background_dist, eps, beta, num_components=8
)

# Loss likelihood module
poisson_loss = PoissonLikelihoodV2(
    beta=beta,
    eps=eps,
    prior_I=prior_I,
    prior_bg=prior_bg,
    p_I_scale=0.0001,
    p_bg_scale=0.001,
)

# Number of steps to train for
steps = 1000 * train_loader_len

# Integration model
    encoder,
    distribution_builder,
    poisson_loss,
    standardization,
    total_steps=steps,
    n_cycle=4,
    lr=learning_rate,
    anneal=False,
    max_epochs=epochs,
    penalty_scale=0.0,
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

model.training_preds["q_I_mean"]
model.training_preds["DIALS_I_sum_val"]
model.training_preds["DIALS_I_prf_val"]
model.training_preds["q_bg_mean"]

# %%
import matplotlib.pyplot as plt

shoebox = next(iter(data_module.train_dataloader()))[0]
mask = next(iter(data_module.train_dataloader()))[-1]
tbl = flex.reflection_table.from_file(
    "/Users/luis/integrator_mvn/integrator/reflections_.refl"
)

filter = flex.reflection_table.from_file(
    "/Users/luis/integrator_mvn/integrator/reflections_.refl"
)["refl_ids"].as_numpy_array() == int(
    next(iter(data_module.train_dataloader()))[1][..., -1]
)


xzycal = tbl.select(flex.bool(filter))["xyzcal.px"].as_numpy_array()

xyzobs = tbl.select(flex.bool(filter))["xyzobs.px.value"].as_numpy_array()


xcal = xzycal[0][0] - shoebox[0][..., :3][:, 0].min()
ycal = xzycal[0][1] - shoebox[0][..., :3][:, 1].min()
xobs = xyzobs[0][0] - shoebox[0][..., :3][:, 0].min()
yobs = xyzobs[0][1] - shoebox[0][..., :3][:, 1].min()


plt.imshow(next(iter(data_module.train_dataloader()))[0][..., -1].reshape(3, 21, 21)[1])

plt.scatter(xcal.item(), ycal.item(), color="red")
plt.scatter(xobs.item(), yobs.item(), color="blue")

plt.show()

# %%
plt.imshow(mask[0].reshape(3, 21, 21)[0])
plt.show()

plt.imshow(shoebox[0][..., 5].reshape(3, 21, 21)[-1])
plt.show()


# %%
# Code to store outputs
refl_ids_train = np.array(model.training_preds["refl_id"], dtype=np.int32)
tbl_ids_train = np.array(model.training_preds["tbl_id"], dtype=np.int32)
refl_ids_val = np.array(model.validation_preds["refl_id"], dtype=np.int32)


# Training predictions
train_res_df = pl.DataFrame(
    {
        "refl_id": refl_ids_train,
        "q_I_mean": model.training_preds["q_I_mean"],
        "q_I_stddev": model.training_preds["q_I_stddev"],
    }
)


import matplotlib.pyplot as plt

model.training_preds["q_I_mean"]
model.training_preds["DIALS_I_sum_val"]

plt.plot(
    model.training_preds["q_I_mean"],
    model.training_preds["DIALS_I_prf_val"],
    "o",
    color="black",
    alpha=0.2,
)
plt.yscale("log")
plt.xscale("log")
plt.show()

# Validation predictions
val_res_df = pl.DataFrame(
    {
        "refl_id": refl_ids_val,
        "q_I_mean": model.validation_preds["q_I_mean"],
        "q_I_stddev": model.validation_preds["q_I_stddev"],
    }
)

with open("train_res.pkl", "wb") as f:
    pickle.dump(model.training_preds, f)

with open("val_res.pkl", "wb") as f:
    pickle.dump(model.validation_preds, f)

# Concatenate train_res_df and val_res_df
res_df = pl.concat([train_res_df, val_res_df])

# refl file
# refl_dir = "/Users/luis/integrator/rotation_data_examples/data/"

refl_filename = "/Users/luis/integrator_mvn/integrator/reflections_.refl"
refl_table = flex.reflection_table.from_file(refl_filename)


sel = np.asarray([False] * len(refl_table))
reflection_ids = res_df["refl_id"].to_list()
intensity_preds = res_df["q_I_mean"].to_list()
intensity_stddev = res_df["q_I_stddev"].to_list()

for i, id in enumerate(reflection_ids):
    sel[id] = True

refl_temp_tbl = refl_table.select(flex.bool(sel))

refl_temp_tbl["intensity.sum.value"] = flex.double(intensity_preds)

refl_temp_tbl["intensity.sum.variance"] = flex.double(intensity_stddev)

# save the updated reflection table
refl_temp_tbl.as_file(f"integrator_preds_test.refl")
