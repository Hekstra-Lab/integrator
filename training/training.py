import torch
import os
import polars as pl
import pandas as pd
import numpy as np
from integrator.io import RotationData
from tqdm import trange
from torch.distributions.transforms import ExpTransform
from tqdm import tqdm
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
batch_size = 100
learning_rate = 0.001

# Load training data
loaded_data_ = torch.load("shoebox_data.pt")
data_dir = "./training/shoebox_data"

# Get data as list
data_list = []
bg_counts = []
true_rate = []
true_I = []
true_bg = []
true_rate = []
I_counts = []
true_covariances = []
true_weighted_intensity = []

for shoebox_file in os.listdir(data_dir):
    file_path = os.path.join(data_dir, shoebox_file)
    if file_path != "./training/shoebox_data/.DS_Store":
        loaded_data_ = torch.load(file_path)
        true_I.extend(torch.unbind(loaded_data_["true_intensities"]))
        true_bg.extend(torch.unbind(loaded_data_["poisson_rate"]))
        true_rate.extend(torch.unbind(loaded_data_["rates"], dim=0))
        bg_counts.extend(torch.unbind(loaded_data_["bg_counts"]))
        I_counts.extend(torch.unbind(loaded_data_["I_counts"]))
        true_covariances.extend(torch.unbind(loaded_data_["covariances"], dim=0))
        true_weighted_intensity.extend(torch.unbind(loaded_data_["weighted_intensies"]))
        data_ = loaded_data_["dataset"]
        data_ = data_.to(torch.float32)
        data_list.extend(torch.unbind(data_, dim=0))

# number of voxels in each shoebox
num_voxels = [x.size(0) for x in data_list]

# list of shoebox shapes
shapes = [
    (len(x[:, 0].unique()), len(x[:, 1].unique()), len(x[:, 2].unique()))
    for x in data_list
]

# DataFrame to hold data and other relevant information
df = pl.DataFrame(
    {
        "X": data_list,
        "num_voxels": num_voxels,
        "shape": shapes,
        "true_I": true_I,
        "true_bg": true_bg,
        "true_rate": true_rate,
        "bg_counts": bg_counts,
        "I_counts": I_counts,
        "true_cov": true_covariances,
        "true_weighted_intensity": true_weighted_intensity,
    }
)

# Largest number of voxels in a shoebox
max_voxels = np.unique(num_voxels).max()


# %%
class SimulatedData(torch.utils.data.Dataset):
    def __init__(self, df, max_voxels):
        self.df = df
        self.max_voxels = max_voxels
        self.data = df["X"]
        self.true_I = df["true_I"]
        self.true_bg = df["true_bg"]
        self.true_rate = df["true_rate"]
        self.bg_counts = df["bg_counts"]
        self.I_counts = df["I_counts"]
        self.true_cov = df["true_cov"]
        self.true_weighted_intensity = df["true_weighted_intensity"]
        self.shapes = df["shape"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        shoebox = self.data[idx]
        shape = self.shapes[idx]
        num_voxels = shape[0] * shape[1] * shape[2]
        pad_size = self.max_voxels - num_voxels
        padded_shoebox = torch.nn.functional.pad(
            shoebox, (0, 0, 0, max(pad_size, 0)), "constant", 0
        )
        I_counts = torch.nn.functional.pad(
            self.I_counts[idx], (0, max(pad_size, 0)), "constant", 0
        )
        weighted_intensity = torch.nn.functional.pad(
            self.true_weighted_intensity[idx], (0, max(pad_size, 0)), "constant", 0
        )
        bg_counts = torch.nn.functional.pad(
            self.bg_counts[idx], (0, max(pad_size, 0)), "constant", 0
        )

        pad_mask = torch.nn.functional.pad(
            torch.ones_like(shoebox[:, -1], dtype=torch.bool),
            (0, max(pad_size, 0)),
            "constant",
            False,
        )
        if shape[-1] == 1:
            is_flat = torch.tensor(True)
            cov = torch.zeros(3, 3)
            cov[:2, :2] = self.true_cov[idx]
        else:
            is_flat = torch.tensor(False)
            cov = self.true_cov[idx]

        return (
            padded_shoebox,
            pad_mask,
            self.true_I[idx],
            cov,
            self.true_bg[idx],
            is_flat,
            bg_counts,
            I_counts,
            weighted_intensity,
            torch.tensor(shape),
        )


simulated_data = SimulatedData(df.sample(fraction=1, shuffle=True), max_voxels)

# train loader
subset_ratio = 0.001
subset_size = int(len(simulated_data) * subset_ratio)
subset_indices = list(range(subset_size))
subset_data = torch.utils.data.Subset(simulated_data, subset_indices)
train_subset, test_subset = torch.utils.data.random_split(subset_data, [0.8, 0.2])
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
# train_loader = DataLoader(simulated_data, batch_size=1)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Variational distributions
intensity_dist = rsd.FoldedNormal

# intensity_dist = torch.distributions.log_normal.LogNormal
background_dist = rsd.FoldedNormal

prior_I = torch.distributions.log_normal.LogNormal(
    loc=torch.tensor(7.0, requires_grad=False),
    scale=torch.tensor(1.5, requires_grad=False),
)

p_I_scale = 1

prior_bg = torch.distributions.gamma.Gamma(torch.tensor([1.0]), torch.tensor([1]))
p_bg_scale = 1

prior_profile = torch.distributions.multivariate_normal.MultivariateNormal(
    torch.zeros(2), true_covariances[0][:2][:, :2]
)

# %%
epochs = 2000

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_steps = len(train_loader)

grad_norms = []
q_I_list = []
q_bg_list = []
train_profile = []
train_avg_loss = []
train_traces = []
train_rate = []
train_L = []

test_avg_loss = []
test_profile = []
test_rate = []
test_traces = []
test_L = []

# evaluate every number of epochs
evaluate_every = 2

# for i in range(num_runs):
standardization = Standardize(max_counts=len(train_loader))
encoder = Encoder(depth, dmodel, feature_dim, dropout=None)

distribution_builder = DistributionBuilder(
    dmodel, intensity_dist, background_dist, eps, beta
)

poisson_loss = PoissonLikelihoodV2(
    beta=beta,
    eps=eps,
    prior_I=prior_I,
    prior_bg=prior_bg,
    prior_profile=None,
    p_I_scale=0.0001,
    p_bg_scale=0.0001,
    p_profile_scale=0.1,
)

integrator = Integrator(
    standardize=standardization,
    encoder=encoder,
    distribution_builder=distribution_builder,
    likelihood=poisson_loss,
)

integrator = integrator.to(device)
opt = torch.optim.Adam(integrator.parameters(), lr=learning_rate)
trace = []

q_I_mean_train_list = []
q_I_stddev_train_list = []
q_bg_mean_train_list = []
q_bg_stddev_train_list = []
true_L_train_list = []

train_preds = {
    "q_I_mean": [],
    "q_I_stddev": [],
    "q_bg_mean": [],
    "q_bg_stddev": [],
    "L_pred": [],
    # "rate_pred": [],
    "profile_pred": [],
    "true_I": [],
    "true_L": [],
    "true_bg": [],
    "bg_counts": [],
    "I_counts": [],
    "weighted_I": [],
}

test_preds = {
    "q_I_mean_list": [],
    "q_I_stddev_list": [],
    "q_bg_mean_list": [],
    "q_bg_stddev_list": [],
    "L_pred_list": [],
    "rate_pred_list": [],
    "profile_pred_list": [],
    "true_I": [],
    "true_L": [],
    "bg_counts": [],
    "I_counts": [],
    "weighted_I": [],
}

with tqdm(total=epochs * num_steps, desc="training") as pbar:
    for epoch in range(epochs):
        # Train
        integrator.train()
        for step, (
            sbox,
            mask,
            true_I,
            true_L,
            true_bg,
            is_flat,
            bg_counts,
            I_counts,
            weighted_I,
            shape,
        ) in enumerate(train_loader):
            sbox = sbox.to(device)
            mask = mask.to(device)

            opt.zero_grad()
            loss, rate, q_I, profile, q_bg, counts, L = integrator(sbox, mask, is_flat)
            loss.backward()
            opt.step()
            trace.append(loss.item())

            grad_norm = torch.nn.utils.clip_grad_norm_(
                integrator.parameters(), max_norm=max_size
            )

            grad_norms.append(grad_norm)

            # Update progress bar
            if epoch == epochs - 1:
                train_preds["q_I_mean"].extend(q_I.mean.ravel().tolist())
                train_preds["q_I_stddev"].extend(q_I.stddev.ravel().tolist())
                train_preds["q_bg_mean"].extend(q_bg.mean.ravel().tolist())
                train_preds["q_bg_stddev"].extend(q_bg.stddev.ravel().tolist())
                train_preds["L_pred"].extend(L.cpu())
                train_preds["profile_pred"].extend(profile.cpu())
                # train_preds["rate_pred"].extend(rate.cpu())
                train_preds["true_I"].extend(true_I.ravel().tolist())
                train_preds["true_L"].extend(true_L.cpu())
                train_preds["true_bg"].extend(true_bg.cpu())
                train_preds["bg_counts"].extend(bg_counts.cpu())
                train_preds["I_counts"].extend(I_counts.cpu())
                train_preds["weighted_I"].extend(weighted_I.cpu())

            pbar.set_postfix(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "loss": loss.item(),
                    "grad norm": grad_norm,
                }
            )

            pbar.update(1)

        # store metrics/outputs
        train_avg_loss.append(torch.mean(torch.tensor(trace)))
        train_rate.append(rate)
        train_profile.append(profile)
        train_L.append(L[0])

        # # %%
        # # Evaluate
        # if (epoch + 1) % evaluate_every == 0 or epoch == epochs - 1:
        # integrator.eval()
        # test_loss = []

        # with torch.no_grad():
        # for i, (shoebox, mask, true_I, true_L,true_bg,bg_counts,I_counts,weighted_I) in enumerate(test_loader):
        # shoebox = shoebox.to(device)
        # mask = mask.to(device)

        # # Forward pass
        # eval_loss, rate, q_I, profile, q_bg, counts, L = integrator(
        # shoebox, mask,
        # )
        # test_loss.append(eval_loss.item())

        # if epoch == epochs - 1:
        # test_preds["q_I_mean_list"].extend(q_I.mean.ravel().tolist())
        # test_preds["q_I_stddev_list"].extend(
        # q_I.stddev.ravel().tolist()
        # )
        # test_preds["q_bg_mean_list"].extend(q_bg.mean.ravel().tolist())
        # test_preds["q_bg_stddev_list"].extend(
        # q_bg.stddev.ravel().tolist()
        # )
        # test_preds["L_pred_list"].extend(L)
        # test_preds["rate_pred_list"].extend(rate)
        # test_preds["true_I"].extend(true_I.ravel().tolist())
        # test_preds["true_L"].extend(true_L)
        # test_preds["profile_pred_list"].extend(profile)

        # test_avg_loss.append(torch.mean(torch.tensor(eval_loss)))


results = {
    "train_preds": train_preds,
    "test_preds": test_preds,
    "train_avg_loss": train_avg_loss,
    "test_avg_loss": test_avg_loss,
}
