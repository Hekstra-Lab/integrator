import torch
import math
import csv
import polars as pl
import numpy as np
from integrator.io import RotationData
from tqdm import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
from integrator.layers import Transformer
from integrator.models import (
    MLP,
    MLPImage,
    MLPOut1,
    MLPPij,
    MLPPij2,
    LogNormDistribution,
    ProfilePredictor,
    ReflectionTransformerEncoder,
    IntegratorTransformer,
    IntensityBgPredictor,
    RotationReflectionEncoder,
    RotationPixelEncoder,
    PoissonLikelihood,
)
from torch.utils.data import DataLoader
import itertools


# %%
# Hyperparameters
steps = 10000
batch_size = 10
max_size = 1024
beta = 1.0
eps = 1e-12
depth = 10
learning_rate = 0.0001
dmodel = 64
feature_dim = 7
mc_samples = 100
epochs = 1
prior_mean = 7
prior_std = 1.4
num_epochs = 2
dropout = 0.5

bg_penalty_scaling = [0, 1]
kl_bern_scale = [0, 1]
kl_lognorm_scale = [1]

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading data
shoebox_dir = "/Users/luis/integrator/rotation_data_examples/data/"
# shoebox_dir = "/n/holylabs/LABS/hekstra_lab/Users/laldama"
rotation_data = RotationData(shoebox_dir=shoebox_dir, val_split=None)

# %%
train_loader = DataLoader(rotation_data, batch_size=batch_size, shuffle=True)
im, mask = next(iter(train_loader))

# %%

# Encoders and predictors
# reflection encoder
pixel_encoder = RotationPixelEncoder(
    depth=depth, dmodel=dmodel, d_in_refl=6, dropout=None
)
profile_ = ProfilePredictor(dmodel, depth)
likelihood = PoissonLikelihood(
    beta=beta, eps=eps, prior_mean=prior_mean, prior_std=prior_std
)
bglognorm = LogNormDistribution(dmodel, eps, beta)
pixel_transformer = Transformer(
    d_model=64, d_hid=2000, nhead=8, batch_first=True, dropout=dropout, nlayers=6
)

refl_transformer = ReflectionTransformerEncoder(
    depth=depth, dmodel=dmodel, feature_dim=feature_dim, dropout=dropout
)
intensity_bacground = IntensityBgPredictor(depth, dmodel, dropout=dropout)

# %%

# Initiating transformers
integrator = IntegratorTransformer(
    refl_transformer,
    intensity_bacground,
    pixel_encoder,
    pixel_transformer,
    profile_,
    bglognorm,
    likelihood,
)

integrator(
    im, mask, emp_bg=8, kl_lognorm_scale=1, bg_penalty_scaling=None, kl_bern_scale=1
)


# Trunacated Normal
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


# %%
def train_and_eval(
    kl_lognorm_scale, n_batches, bg_penalty_scaling=None, kl_bern_scale=None
):
    # Encoders

    integrator = IntegratorTransformer(
        refl_transformer,
        intensity_bacground,
        pixel_encoder,
        pixel_transformer,
        profile_,
        bglognorm,
        likelihood,
    )

    initialize_weights(integrator)
    integrator = integrator.to(device)

    # reset_model_weights(integrator)
    trace = []
    grad_norms = []
    bar = trange(steps)
    opt = torch.optim.Adam(integrator.parameters(), lr=learning_rate)  # optimizer
    torch.autograd.set_detect_anomaly(True)

    integrator.train()
    for epoch in range(num_epochs):
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        i = 0
        # Get batch data
        for ims, masks in bar:
            # ims, masks = next(iter(train_loader))
            ims = ims.to(device)
            masks = masks.to(device)

            # Forward pass
            opt.zero_grad()
            loss = integrator(
                ims,
                masks,
                emp_bg,
                kl_lognorm_scale,
                bg_penalty_scaling=None,
                kl_bern_scale=None,
            )

            # Backward pass
            loss.backward()
            opt.step()

            # Record metrics
            trace.append(loss.item())
            grad_norm = (
                sum(
                    p.grad.norm().item() ** 2
                    for p in integrator.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            grad_norms.append(grad_norm)
            # grad_norms.append(
            # torch.nn.utils.clip_grad_norm(integrator.parameters(), max_norm=max_size)
            # )

            bar.set_description(f"Step {(steps+1)}/{steps},Loss:{loss.item():.4f}")
            i += 1

    final_loss = trace[-1]
    torch.save(integrator.state_dict(), f"weights_gamma{gamma}.pth")

    # Evaluation Loop
    integrator.eval()
    eval_metrics = {}
    # val_loss = []

    rotation_data.set_mode = "test"
    eval_loader = DataLoader(rotation_data, batch_size=batch_size, shuffle=False)

    num_batches = n_batches
    I, SigI = [], []
    pij_ = []
    bg = []
    counts = []

    with torch.no_grad():
        for i, (ims, masks) in enumerate(eval_loader):
            if i >= num_batches:
                break
            ims = ims.to(device)
            masks = masks.to(device)

            # forward pass
            output = integrator.get_intensity_sigma_batch(ims, masks)

            I.append(output[0].cpu())
            SigI.append(output[1].cpu())
            bg.append(output[2].cpu())
            pij_.append(output[3].cpu())
            counts.append(output[4].cpu())
            # val_loss.append(output[5].cpu())
    rotation_data.set_mode = "train"
    integrator.train()

    # I = np.concatenate(I, axis=0)
    # SigI = np.concatenate(SigI, axis=0)
    # bg = np.concatenate(bg, axis=0)

    eval_metrics["min_intensity"] = np.array(I).ravel().min()
    eval_metrics["max_intensity"] = np.array(I).ravel().max()
    eval_metrics["mean_intensity"] = np.array(I).ravel().mean()
    eval_metrics["min_sigma"] = np.array(SigI).ravel().min()
    eval_metrics["max_sigma"] = np.array(SigI).ravel().max()
    eval_metrics["mean_sigma"] = np.array(SigI).ravel().mean()
    eval_metrics["min_background"] = np.array(bg).ravel().min()
    eval_metrics["max_background"] = np.array(bg).ravel().max()
    eval_metrics["mean_background"] = np.array(bg).ravel().mean()
    eval_metrics["min_pij"] = np.array(pij_).ravel().min()
    eval_metrics["max_pij"] = np.array(pij_).ravel().max()
    eval_metrics["mean_pij"] = np.array(pij_).ravel().mean()

    return final_loss, eval_metrics, I, trace


# %%
# training loop

emp_bg = rotation_data.train_df["background.mean"].mean()

results = []
n_batches = 10
steps = len(train_loader)
for gamma in kl_lognorm_scale:
    try:
        final_loss, eval_metrics, I_, trace_ = train_and_eval(
            gamma, n_batches, bg_penalty_scaling=None, kl_bern_scale=None
        )

        # Plotting loss as function of step
        plt.clf()
        # plt.plot(np.linspace(0, steps * num_epochs, steps * num_epochs), trace_)
        plt.plot(np.linspace(0, 2, 2), trace_)
        plt.ylabel("loss")
        plt.xlabel("step")
        plt.savefig(f"loss_gamma_{gamma}.png", dpi=300)

        # Scatter plot of network-intensity vs DIALS summmation model
        plt.clf()
        plt.scatter(
            np.array(I_).ravel(),
            rotation_data.test_df["intensity.sum.value"][0 : n_batches * batch_size],
            alpha=0.15,
        )
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(alpha=0.5)
        plt.savefig(
            f"NN_vs_sum_model_gamma_{gamma}.png",
            dpi=300,
        )

        # Network vs DIALS profile model
        plt.clf()
        plt.scatter(
            np.array(I_).ravel(),
            rotation_data.test_df["intensity.prf.value"][0 : n_batches * batch_size],
            alpha=0.15,
        )
        plt.grid(alpha=0.5)
        plt.yscale("log")
        plt.xscale("log")
        plt.savefig(
            f"NN_vs_prf_model_gamma_{gamma}.png",
            dpi=300,
        )

        # append reults
        results.append(
            {
                # "bg_penalty_scaling": beta,
                # "kl_bern_scale": alpha,
                "kl_lognorm_scale": gamma,
                "final_loss": final_loss,
                "min_intensity": eval_metrics["min_intensity"],
                "max_intensity": eval_metrics["max_intensity"],
                "mean_intensity": eval_metrics["mean_intensity"],
                "min_sigma": eval_metrics["min_sigma"],
                "max_sigma": eval_metrics["max_sigma"],
                "mean_sigma": eval_metrics["mean_sigma"],
                "min_background": eval_metrics["min_background"],
                "max_background": eval_metrics["max_background"],
                "mean_background": eval_metrics["mean_background"],
                "min_pij": eval_metrics["min_pij"],
                "max_pij": eval_metrics["max_pij"],
                "mean_pij": eval_metrics["mean_pij"],
            }
        )

    except Exception as e:
        print(f"Failed with kl_lognorm_scale: {gamma}, Error:{e}")
# %%
with open("hyperparameter_results.csv", "w", newline="") as csvfile:
    fieldnames = [
        # "bg_penalty_scaling",
        # "kl_bern_scale",
        "kl_lognorm_scale",
        "final_loss",
        "min_intensity",
        "max_intensity",
        "mean_intensity",
        "min_sigma",
        "max_sigma",
        "mean_sigma",
        "min_background",
        "max_background",
        "mean_background",
        "min_pij",
        "max_pij",
        "mean_pij",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()  # Write the header row
    for result in results:
        writer.writerow(result)
