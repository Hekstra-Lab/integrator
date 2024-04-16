import torch
import polars as pl
import math
import numpy as np
from integrator.io import RotationData, Standardize
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
prior_mean = 7
prior_std = 1.4

batch_size = 10
learning_rate = 0.001
epochs = 2

bg_penalty_scaling = [0, 1]
kl_bern_scale = [0, 1]
kl_lognorm_scale = [0]

# %%
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading data
shoebox_dir = "/Users/luis/integrator/rotation_data_examples/data/"
# shoebox_dir = "/n/holylabs/LABS/hekstra_lab/Users/laldama/integrator_/rotation_data_examples/data_temp/temp"
rotation_data = RotationData(shoebox_dir=shoebox_dir, val_split=None)

# Set data loader to training mode
rotation_data.set_mode("train")

# training loader
train_loader = DataLoader(rotation_data, batch_size=batch_size, shuffle=True)


# %%
# Weight initializer
def weight_initializer(weight):
    fan_avg = 0.5 * (weight.shape[-1] + weight.shape[-2])
    std = math.sqrt(1.0 / fan_avg / 10.0)
    a = -2.0 * std
    b = 2.0 * std
    torch.nn.init.trunc_normal_(weight, 0.0, std, a, b)
    return weight


def initialize_weights(model):
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                weight_initializer(p)


# Training Loop

encoder = Encoder(depth, dmodel, feature_dim, dropout=None)
standardization = Standardize()
distribution_builder = DistributionBuilder(dmodel, eps, beta)
poisson_loss = PoissonLikelihoodV2(beta, eps)
integrator = IntegratorV3(encoder, distribution_builder, poisson_loss)
integrator = integrator.to(device)

trace = []
grad_norms = []
steps = len(train_loader)
bar = trange(steps)
n_batches = 10

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
# Array to store predicted intensities and sigI
I_pred_arr = np.empty((0, batch_size * n_batches))
SigI_pred_arr = np.empty((0, batch_size * n_batches))


# Training loop
for epoch in range(epochs):
    bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    batch_loss = []
    i = 0

    # Get batch data
    for ims, masks in bar:
        if i != 2:
            ims, masks = next(iter(train_loader))
            ims = ims.to(device)
            masks = masks.to(device)
            ims_ = standardization(ims, masks)

            # Forward pass

            # reset gradients of model parameters
            opt.zero_grad()

            # calculate loss
            loss = integrator(ims_, masks, mc_samples=mc_samples)

            # Backward pass
            loss.backward()

            # update model parameters
            opt.step()

            # Record metrics
            trace.append(loss.item())
            batch_loss.append(loss.item())
            grad_norm = (
                sum(
                    p.grad.norm().item() ** 2
                    for p in integrator.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            # grad_norms.append(grad_norm)
            grad_norms.append(
                torch.nn.utils.clip_grad_norm(
                    integrator.parameters(), max_norm=max_size
                )
            )

            bar.set_description(f"Step {(steps+1)}/{steps},Loss:{loss.item():.4f}")
            i += 1
        else:
            break

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
        for i, (ims, masks) in enumerate(train_loader):
            if i >= num_batches:
                break
            ims = ims.to(device)
            ims_ = standardization(ims, masks)
            masks = masks.to(device)

            # forward pass
            output = integrator.get_intensity_sigma_batch(ims_, masks)

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
                # "min_background": [np.array(bg).ravel().min()],
                # "max_background": [np.array(bg).ravel().max()],
                # "mean_background": [np.array(bg).ravel().mean()],
                "min_pij": [np.array(pij_).ravel().min()],
                "max_pij": [np.array(pij_).ravel().max()],
                "mean_pij": [np.array(pij_).ravel().mean()],
                "corr_prf": [corr_prf],
                "corr_sum": [corr_sum],
            }
        )
    )

eval_metrics.write_csv(f"metrics.csv")
np.savetxt(f"trace_.csv", trace, fmt="%s")
np.savetxt(f"I_pred.csv", I_pred_arr, delimiter=",")  # save network predicted intensity
np.savetxt(f"SigI_pred.csv", I_pred_arr, delimiter=",")  # save network predicted SigI
torch.save(integrator.state_dict(), f"weights_run.pth")  # save final network weights


# %%
# Training loop
num_epochs = epochs
num_steps = len(train_loader)

with tqdm(total=num_epochs * num_steps, desc="Training") as pbar:
    for epoch in range(num_epochs):
        batch_loss = []
        i = 0

        # Get batch data
        for step, (ims, masks) in enumerate(train_loader):
            ims = ims.to(device)
            masks = masks.to(device)
            ims_ = standardization(ims, masks)

            # Forward pass
            opt.zero_grad()
            loss = integrator(ims_, masks, mc_samples=mc_samples)

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
            for i, (ims, masks) in enumerate(train_loader):
                if i >= num_batches:
                    break
                ims = ims.to(device)
                ims_ = standardization(ims, masks)
                masks = masks.to(device)

                # forward pass
                output = integrator.get_intensity_sigma_batch(ims_, masks)

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
