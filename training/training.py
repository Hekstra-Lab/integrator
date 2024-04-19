import torch
import polars as pl
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

# Prior distributions
prior_I = torch.distributions.log_normal.LogNormal(
    loc=torch.tensor(7.0, requires_grad=False),
    scale=torch.tensor(1.4, requires_grad=False),
)
p_I_scale = 0.1
# prior_bg = torch.distributions.normal.Normal(
#    loc=torch.tensor(10, requires_grad=False),
#    scale=torch.tensor(2, requires_grad=False),
# )
# p_bg_scale = 0.1

# %%
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading data
# shoebox_dir = "/Users/luis/integrator/rotation_data_examples/data/"
shoebox_dir = "/n/holylabs/LABS/hekstra_lab/Users/laldama/integrator_/rotation_data_examples/data_temp/"
rotation_data = RotationData(shoebox_dir=shoebox_dir, val_split=None)

# Set data loader to training mode
rotation_data.set_mode("train")

# training loader
train_loader = DataLoader(rotation_data, batch_size=batch_size, shuffle=True)

# Training Loop
encoder = Encoder(depth, dmodel, feature_dim, dropout=None)
standardization = Standardize(max_counts=len(train_loader))
distribution_builder = DistributionBuilder(
    dmodel, intensity_dist, background_dist, eps, beta
)
poisson_loss = PoissonLikelihoodV2(
    beta=beta,
    eps=eps,
    prior_I=None,
    prior_bg=None,
)
integrator = IntegratorV3(standardization, encoder, distribution_builder, poisson_loss)
integrator = integrator.to(device)

trace = []
grad_norms = []
steps = len(train_loader)
bar = trange(steps)
n_batches = 100

# Optimizer
opt = torch.optim.Adam(integrator.parameters(), lr=learning_rate)

torch.autograd.set_detect_anomaly(True)

# set loader to training mode
rotation_data.set_mode("train")

# DIALS Intensity from profile model
I_dials_prf = rotation_data.test_df["intensity.prf.value"][0 : n_batches * batch_size]

# DIALS Intensity from summation model
I_dials_sum = rotation_data.test_df["intensity.prf.value"][0 : n_batches * batch_size]

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
        "corr_prf": pl.Series([], dtype=pl.Float64),
        "corr_sum": pl.Series([], dtype=pl.Float64),
    }
)

# Array to store predicted Intensities
I_pred_arr = np.empty((0, batch_size * n_batches))
# Array to store predicted SigI
SigI_pred_arr = np.empty((0, batch_size * n_batches))

# %%
# Training loop
num_epochs = epochs
num_steps = len(train_loader)

with tqdm(total=num_epochs * num_steps, desc="Training") as pbar:
    for epoch in range(num_epochs):
        batch_loss = []
        i = 0

        # Get batch of data
        for step, (sbox, masks, dead_pixel_mask) in enumerate(train_loader):
            sbox = sbox.to(device)
            masks = masks.to(device)
            dead_pixel_mask = dead_pixel_mask.to(device)

            # Forward pass
            opt.zero_grad()
            loss = integrator(sbox, masks, dead_pixel_mask, mc_samples=mc_samples)

            # Backward pass
            loss.backward()
            opt.step()

            # Record metrics
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
        rotation_data.set_mode("test")

        num_batches = n_batches
        I, SigI = [], []
        pij_ = []
        bg = []
        counts = []

        # Evaluation loop
        with torch.no_grad():
            for i, (sbox, masks, dead_pixel_mask) in enumerate(train_loader):
                if i >= num_batches:
                    break
                sbox = ims.to(device)
                masks = masks.to(device)
                dead_pixel_mask = dead_pixel_mask.to(device)
                sbox_ = standardization(sbox, masks)

                # forward pass
                output = integrator.get_intensity_sigma_batch(sbox_, dead_pixel_mask)

                I.append(output[0].cpu())
                SigI.append(output[1].cpu())
                pij_.append(output[2].cpu())
                counts.append(output[3].cpu())
                val_loss.append(output[4].cpu())

        rotation_data.set_mode("train")

        # Compute evaluation metrics
        I_pred = np.ravel([I])
        corr_prf = np.corrcoef(I_dials_prf, I_pred)[0][-1]
        corr_sum = np.corrcoef(I_dials_sum, I_pred)[0][-1]

        I_pred_arr = np.vstack((I_pred_arr, I_pred))
        SigI_pred_arr = np.vstack((SigI_pred_arr, np.array(SigI).ravel()))

        eval_metrics = eval_metrics.vstack(
            pl.DataFrame(
                {
                    "epoch": [epoch],
                    "average_loss": [np.mean([batch_loss])],
                    "test_loss": [np.mean([val_loss])],
                    "min_intensity": [np.array(I).ravel().min()],
                    "max_intensity": [np.array(I).ravel().max()],
                    "mean_intensity": [np.array(I).ravel().mean()],
                    "min_sigma": [np.array(SigI).ravel().min()],
                    "max_sigma": [np.array(SigI).ravel().max()],
                    "mean_sigma": [np.array(SigI).ravel().mean()],
                    "min_pij": [np.array(pij_).ravel().min()],
                    "max_pij": [np.array(pij_).ravel().max()],
                    "mean_pij": [np.array(pij_).ravel().mean()],
                    "corr_prf": [corr_prf],
                    "corr_sum": [corr_sum],
                }
            )
        )

eval_metrics.write_csv(f"evaluation_metrics_.csv")
np.savetxt(f"trace.csv", trace, fmt="%s")
np.savetxt(f"IPred.csv", I_pred_arr, delimiter=",")
np.savetxt(f"SigIPred.csv", SigI_pred_arr, delimiter=",")
torch.save(integrator.state_dict(), f"weights.pth")
