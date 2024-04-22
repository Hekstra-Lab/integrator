import torch
import polars as pl
import pandas as pd
import numpy as np
from integrator.io import RotationData
from tqdm import trange
from tqdm import tqdm
from integrator.models import (
    Encoder,
    PoissonLikelihoodV2,
    DistributionBuilder,
    IntegratorV3,
)
from torch.utils.data import DataLoader
import torch.nn as nn
from rs_distributions import distributions as rsd
from integrator.layers import Standardize

# %%
# Hyperparameters
depth = 10
dmodel = 64
feature_dim = 7
dropout = 0.5
beta = 1.0
mc_samples = 100
max_size = 1024
eps = 1e-12
batch_size = 100
learning_rate = 0.001
epochs = 10

# Variational distributions
intensity_dist = rsd.FoldedNormal
background_dist = rsd.FoldedNormal
# intensity_dist = torch.distributions.log_normal.LogNormal
# background_dist = torch.distributions.log_normal.LogNormal

# Prior distributions
prior_I = torch.distributions.log_normal.LogNormal(
    loc=torch.tensor(7.0, requires_grad=False),
    scale=torch.tensor(1.4, requires_grad=False),
)
p_I_scale = 0.1
prior_bg = torch.distributions.normal.Normal(
    loc=torch.tensor(10, requires_grad=False),
    scale=torch.tensor(1, requires_grad=False),
)
p_bg_scale = 0.1

# %%
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading data
shoebox_dir = "/Users/luis/integrator/rotation_data_examples/data/"
# shoebox_dir = "/n/holylabs/LABS/hekstra_lab/Users/laldama/integrator_/rotation_data_examples/data_temp/"

rotation_data = RotationData(shoebox_dir=shoebox_dir)
train_, test_ = torch.utils.data.random_split(rotation_data, [0.001, 0.999])

# train and test loaders
train_loader = DataLoader(train_, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_, batch_size=batch_size, shuffle=False)

# Layers
encoder = Encoder(depth, dmodel, feature_dim, dropout=None)
standardization = Standardize(max_counts=len(train_loader))
distribution_builder = DistributionBuilder(
    dmodel, intensity_dist, background_dist, eps, beta
)
poisson_loss = PoissonLikelihoodV2(
    beta=beta,
    eps=eps,
    prior_I=prior_I,
    prior_bg=prior_bg,
    p_I_scale=p_I_scale,
    p_bg_scale=p_bg_scale,
)
integrator = IntegratorV3(standardization, encoder, distribution_builder, poisson_loss)
integrator = integrator.to(device)

# optimizer
opt = torch.optim.Adam(integrator.parameters(), lr=learning_rate)

# %%
trace = []
grad_norms = []
steps = len(train_loader)
bar = trange(steps)
n_batches = 100

# anomaly detection
torch.autograd.set_detect_anomaly(True)

# DataFrame to store evaluation metrics
eval_metrics = pl.DataFrame(
    {
        "epoch": pl.Series([], dtype=pl.Int64),
        "average_loss": pl.Series([], dtype=pl.Float64),
        "test_loss": pl.Series([], dtype=pl.Float32),
        "min_intensity": pl.Series([], dtype=pl.Float32),
        "max_intensity": pl.Series([], dtype=pl.Float32),
        "mean_intensity": pl.Series([], dtype=pl.Float32),
        "min_sigma": pl.Series([], dtype=pl.Float32),
        "max_sigma": pl.Series([], dtype=pl.Float32),
        "mean_sigma": pl.Series([], dtype=pl.Float32),
        # "min_background": pl.Series([], dtype=pl.Float32),
        # "max_background": pl.Series([], dtype=pl.Float32),
        # "mean_background": pl.Series([], dtype=pl.Float32),
        "min_pij": pl.Series([], dtype=pl.Float32),
        "max_pij": pl.Series([], dtype=pl.Float32),
        "mean_pij": pl.Series([], dtype=pl.Float32),
        "corr_prf_train": pl.Series([], dtype=pl.Float64),
        "corr_sum_train": pl.Series([], dtype=pl.Float64),
        "corr_prf_test": pl.Series([], dtype=pl.Float64),
        "corr_sum_test": pl.Series([], dtype=pl.Float64),
    }
)

# %%
# Training loop
num_epochs = epochs
num_steps = len(train_loader)

with tqdm(total=num_epochs * num_steps, desc="Training") as pbar:
    for epoch in range(num_epochs):
        I_train, SigI_train, I_test, SigI_test = [], [], [], []
        I_dials_prf_train, I_dials_prf_test, I_dials_sum_train, I_dials_sum_test = (
            [],
            [],
            [],
            [],
        )
        batch_loss = []
        i = 0

        integrator.train()
        # Get batch of data
        for step, (sbox, dead_pixel_mask, DIALS_I) in enumerate(train_loader):
            sbox = sbox.to(device)
            dead_pixel_mask = dead_pixel_mask.to(device)

            # Forward and Backward pass
            opt.zero_grad()
            loss = integrator(sbox, dead_pixel_mask, mc_samples=mc_samples)
            loss.backward()
            opt.step()

            # Store metrics
            trace.append(loss.item())
            batch_loss.append(loss.item())
            grad_norm = torch.nn.utils.clip_grad_norm(
                integrator.parameters(), max_norm=max_size
            )
            grad_norms.append(grad_norm)

            # Update progress bar
            pbar.set_postfix(
                {
                    "Epoch": epoch + 1,
                    "Step": step + 1,
                    "Loss": loss.item(),
                    "Grad Norm": grad_norm,
                }
            )
            pbar.update(1)

        # Evaluation Loop
        integrator.eval()
        val_loss = []

        # set loader to test mode

        num_batches = n_batches
        pij_ = []
        bg = []
        counts = []

        I_train, SigI_train = [], []
        I_dials_prf_train = []
        I_dials_sum_train = []

        # Evaluation loop for train set
        with torch.no_grad():
            for i, (sbox, dead_pixel_mask, DIALS_I) in enumerate(train_loader):
                if i >= num_batches:
                    break
                sbox = sbox.to(device)
                dead_pixel_mask = dead_pixel_mask.to(device)

                # forward pass
                output = integrator.get_intensity_sigma_batch(sbox, dead_pixel_mask)

                I_dials_prf_train.append(DIALS_I[0].detach().cpu())
                I_dials_sum_train.append(DIALS_I[2].detach().cpu())
                I_train.append(output[0].cpu())
                SigI_train.append(output[1].cpu())
                pij_.append(output[2].cpu())
                counts.append(output[3].cpu())
                val_loss.append(output[4].cpu())

        # Compute evaluation metrics for train set
        I_dials_prf_ = np.concatenate(I_dials_prf_train)
        I_dials_sum_ = np.concatenate(I_dials_sum_train)
        I_pred = np.concatenate(I_train).flatten()
        corr_prf_train = np.corrcoef(I_dials_prf_, I_pred)[0][-1]
        corr_sum_train = np.corrcoef(I_dials_sum_, I_pred)[0][-1]

        # to store test metrics
        I_test, SigI_test = [], []
        I_dials_prf_test = []
        I_dials_sum_test = []

        # Evaluation loop for test set
        with torch.no_grad():
            for i, (sbox, dead_pixel_mask, DIALS_I) in enumerate(test_loader):
                if i >= num_batches:
                    break
                sbox = sbox.to(device)
                dead_pixel_mask = dead_pixel_mask.to(device)

                # forward pass
                output = integrator.get_intensity_sigma_batch(sbox, dead_pixel_mask)

                I_dials_prf_test.append(DIALS_I[0].detach().cpu())
                I_dials_sum_test.append(DIALS_I[2].detach().cpu())
                I_test.append(output[0].cpu())
                SigI_test.append(output[1].cpu())
                pij_.append(output[2].cpu())
                counts.append(output[3].cpu())
                val_loss.append(output[4].cpu())

        # Compute evaluation metrics for test set
        I_dials_prf_ = np.concatenate(I_dials_prf_test)
        I_dials_sum_ = np.concatenate(I_dials_sum_test)
        I_pred = np.concatenate(I_test).flatten()
        corr_prf_test = np.corrcoef(I_dials_prf_, I_pred)[0][-1]
        corr_sum_test = np.corrcoef(I_dials_sum_, I_pred)[0][-1]

        # I_pred_arr_test = np.vstack((I_pred_arr, I_pred))
        # SigI_pred_arr_test = np.vstack((SigI_pred_arr, np.array(SigI_test).ravel()))

        eval_metrics = eval_metrics.vstack(
            pl.DataFrame(
                {
                    "epoch": [epoch],
                    "average_loss": [np.mean([batch_loss])],
                    "test_loss": [np.mean([val_loss])],
                    "min_intensity": [np.concatenate(I_test).min()],
                    "max_intensity": [np.concatenate(I_test).max()],
                    "mean_intensity": [np.concatenate(I_test).mean()],
                    "min_sigma": [np.concatenate(SigI_test).min()],
                    "max_sigma": [np.concatenate(SigI_test).max()],
                    "mean_sigma": [np.concatenate(SigI_test).mean()],
                    "min_pij": [np.concatenate(pij_).min()],
                    "max_pij": [np.concatenate(pij_).max()],
                    "mean_pij": [np.concatenate(pij_).mean()],
                    "corr_prf_train": [corr_prf_train],
                    "corr_sum_train": [corr_sum_train],
                    "corr_prf_test": [corr_prf_test],
                    "corr_sum_test": [corr_sum_test],
                }
            )
        )


eval_metrics.write_csv(f"evaluation_metrics_.csv")
np.savetxt(f"trace.csv", trace, fmt="%s")
# np.savetxt(f"IPred_train.csv", I_pred_arr_train, delimiter=",")
# np.savetxt(f"SigIPred_train.csv", SigI_pred_arr_train, delimiter=",")
# np.savetxt(f"IPred_test.csv", I_pred_arr_test, delimiter=",")
# np.savetxt(f"SigIPred_test.csv", SigI_pred_arr_test, delimiter=",")
torch.save(integrator.state_dict(), f"weights.pth")


# %%
