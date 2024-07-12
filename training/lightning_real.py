import torch
from dials.array_family import flex
import pickle
import os
import polars as plrs
import pandas as pd
import numpy as np
from integrator.io import RotationData, RotationDataModule
from torch.distributions.transforms import ExpTransform
from integrator.models import (
    Encoder,
    PoissonLikelihoodV2,
    DistributionBuilder,
    Integrator,
)
from torch.utils.data import DataLoader
import torch.nn as nn
from rs_distributions import distributions as rsd
from integrator.layers import Standardize
import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import math
from integrator.models import MLP
import polars as pl
from rs_distributions import distributions as rsd
from rs_distributions.transforms import FillScaleTriL
from torch.utils.data import DataLoader
from tqdm import tqdm
from integrator.layers import Linear, ResidualLayer, Standardize
from integrator.models import MLP
from integrator.models.encoder import MeanPool
import torch.distributions.constraints as constraints
from integrator.models import IntegratorModel

torch.set_float32_matmul_precision('high')

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"


# Hyperparameters
depth = 32
dmodel = 64
feature_dim = 7
dropout = 0.5
beta = 1.0
mc_samples = 100
max_size = 1024
eps = 1e-5
batch_size = 5
learning_rate = 0.001
epochs = 1000
subset_ratio = .001

# Directory with .refl files
shoebox_dir = '/n/holylabs/LABS/hekstra_lab/Users/laldama/integrator_/rotation_data_examples/data_temp/temp'
# Set device to cpu or gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataLoader and Dataset setup

rotation_data = RotationData(shoebox_dir=shoebox_dir)

data_module = RotationDataModule(
    shoebox_dir=shoebox_dir, batch_size=batch_size, subset_ratio=subset_ratio
)
data_module.setup()

train_loader_len = len(data_module.train_dataloader())

# Variational distributions
intensity_dist = torch.distributions.gamma.Gamma
background_dist = torch.distributions.gamma.Gamma
prior_I = torch.distributions.exponential.Exponential(rate=torch.tensor(1.0))
concentration = torch.tensor([1.0],device=device)
rate = torch.tensor([1.0],device=device)
prior_bg = torch.distributions.gamma.Gamma(concentration, rate)

# Instantiate standardization, encoder, distribution builder, and likelihood
standardization = Standardize(max_counts=train_loader_len)
encoder = Encoder(depth, dmodel, feature_dim, dropout=None)
distribution_builder = DistributionBuilder(
    dmodel, intensity_dist, background_dist, eps, beta
)
poisson_loss = PoissonLikelihoodV2(
    beta=beta,
    eps=eps,
    prior_I=prior_I,
    prior_bg=prior_bg,
    p_I_scale=.001,
    p_bg_scale=.001,
)


total_steps = 1000 * train_loader_len
print(train_loader_len)
print(total_steps)

model = IntegratorModel(
    encoder, distribution_builder, poisson_loss, standardization, total_steps = total_steps,n_cycle=4 
)

logger = TensorBoardLogger("tb_logs", name="integrator_model")

checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",
    dirpath="checkpoints/",
    filename="integrator-{epoch:02d}-{train_loss:.2f}",
    save_top_k=3,
    mode="min",
)
progress_bar = TQDMProgressBar(refresh_rate=1)

trainer = Trainer(
    max_epochs=epochs,
    accelerator="auto",  # Use "cpu" for CPU training
    devices='auto',
    num_nodes=1,
    precision='16-mixed',  # Use 32-bit precision for CPU
    accumulate_grad_batches=2,
    check_val_every_n_epoch = 100,
    callbacks=[checkpoint_callback, progress_bar],
    logger=logger,
    log_every_n_steps=10,
)

trainer.fit(model, data_module)

# %%
# intensity prediction array
intensity_preds = np.array(model.training_preds["q_I_mean"])

# Reflection id array
refl_ids = np.array(model.training_preds["refl_id"])

# Table ids
tbl_ids = np.unique(np.array(model.training_preds["tbl_id"]))


# DataFrame to store predictions
res_df = pl.DataFrame(
    {
        "tbl_id": model.training_preds["tbl_id"],
        "refl_id": model.training_preds["refl_id"],
        "q_I_mean": model.training_preds["q_I_mean"],
        "q_I_stddev": model.training_preds["q_I_stddev"],
    }
)

#for tbl_id in tbl_ids:
#    sel = np.asarray([False] * len(rotation_data.refl_tables[tbl_id]))
#    filtered_df = res_df.filter(res_df["tbl_id"] == tbl_id)
#    reflection_ids = filtered_df["refl_id"].to_list()
#    intensity_preds = filtered_df["q_I_mean"].to_list()
#    intensity_stddev = filtered_df["q_I_stddev"].to_list()

#    for i in reflection_ids:
#        sel[i] = True

#    refl_temp_tbl = data_module.full_dataset.refl_tables[tbl_id].select(flex.bool(sel))
#    refl_temp_tbl["intensity.sum.value"] = flex.double(intensity_preds)
#    refl_temp_tbl["intensity.sum.variance"] = flex.double(intensity_stddev)

    # save the updated reflection table
#    refl_temp_tbl.as_file(f"integrator_preds_{tbl_id}.refl")


# %%
# Save weights
torch.save(model.state_dict(), "integrator_weights.pth")

# Function to recursively move tensors to CPU
def move_to_cpu(data):
    if isinstance(data, dict):
        return {key: move_to_cpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_cpu(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_cpu(item) for item in data)
    elif torch.is_tensor(data):
        return data.cpu()
    else:
        return data

results = {
    "train_preds": model.training_preds,
    "test_preds": model.validation_preds,
    "train_avg_loss": model.train_avg_loss,
    "test_avg_loss": model.validation_avg_loss,
}
results_cpu = move_to_cpu(results)

with open("results.pkl", "wb") as f:
    pickle.dump(results_cpu, f)

