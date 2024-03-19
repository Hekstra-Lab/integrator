import torch
import math
import polars as pl
import numpy as np
from integrator.io import RotationData
from tqdm import trange
from tqdm import tqdm
from integrator.models import (
    LogNormDistribution,
    ProfilePredictor,
    IntegratorV2,
    IntensityBgPredictor,
    RotationReflectionEncoder,
    RotationPixelEncoder,
    PoissonLikelihood,
)
from torch.utils.data import DataLoader

# Hyperparameters
steps = 10000
batch_size = 10
max_size = 1024
beta = 1.0
eps = 1e-12
depth = 10
learning_rate = 0.001
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
kl_lognorm_scale = [0]

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading data
shoebox_dir = "/Users/luis/integrator/rotation_data_examples/data/"
# shoebox_dir = "/n/holylabs/LABS/hekstra_lab/Users/laldama/integrator_/rotation_data_examples/data_temp/temp"
rotation_data = RotationData(shoebox_dir=shoebox_dir, val_split=None)


# loads a shoebox and corresponding mask
rotation_data.set_mode("train")
train_loader = DataLoader(rotation_data, batch_size=batch_size, shuffle=True)

# Encoders
refl_encoder = RotationReflectionEncoder(depth, dmodel, feature_dim, dropout=dropout)
intensity_bacground = IntensityBgPredictor(depth, dmodel, dropout=dropout)
pixel_encoder = RotationPixelEncoder(
    depth=depth, dmodel=dmodel, d_in_refl=6, dropout=dropout
)
profile_ = ProfilePredictor(dmodel, depth)
likelihood = PoissonLikelihood(
    beta=beta, eps=eps, prior_mean=prior_mean, prior_std=prior_std
)
bglognorm = LogNormDistribution(dmodel, eps, beta)

# integration model
integrator = IntegratorV2(
    refl_encoder, intensity_bacground, pixel_encoder, profile_, bglognorm, likelihood
)
integrator = integrator.to(device)

# train set empirical background
emp_bg = rotation_data.train_df["background.mean"].mean()

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

    # integration model
    integrator = IntegratorV2(
        refl_encoder,
        intensity_bacground,
        pixel_encoder,
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

    # DataLoader for evaluation
    eval_loader = DataLoader(rotation_data, batch_size=batch_size, shuffle=False)

    # DIALS Intensity from profile model
    I_dials_prf = rotation_data.test_df["intensity.prf.value"][
        0 : n_batches * batch_size
    ]

    # DIALS Intensity from summation model
    I_dials_sum = rotation_data.test_df["intensity.prf.value"][
        0 : n_batches * batch_size
    ]

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
            "min_background": pl.Series([], dtype=pl.Float32),
            "max_background": pl.Series([], dtype=pl.Float32),
            "mean_background": pl.Series([], dtype=pl.Float32),
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

    for epoch in range(num_epochs):
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = []
        i = 0
        # Get batch data
        for ims, masks in bar:
            if i != 2:
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

        rotation_data.set_mode("train")

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
                output = integrator.get_intensity_sigma_batch(
                    ims, masks, emp_bg=emp_bg, kl_lognorm_scale=kl_lognorm_scale
                )

                I.append(output[0].cpu())
                SigI.append(output[1].cpu())
                bg.append(output[2].cpu())
                pij_.append(output[3].cpu())
                counts.append(output[4].cpu())
                val_loss.append(output[5].cpu())

        rotation_data.set_mode("train")
        # Correlation

        # Correlation Calculation
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
                    "min_background": [np.array(bg).ravel().min()],
                    "max_background": [np.array(bg).ravel().max()],
                    "mean_background": [np.array(bg).ravel().mean()],
                    "min_pij": [np.array(pij_).ravel().min()],
                    "max_pij": [np.array(pij_).ravel().max()],
                    "mean_pij": [np.array(pij_).ravel().mean()],
                    "corr_prf": [corr_prf],
                    "corr_sum": [corr_sum],
                }
            )
        )

    # save weights after final epoch

    return trace, eval_metrics, I_pred_arr, SigI_pred_arr


# %%
# training loop

n_batches = 10
I_list = []
steps = len(train_loader)

for i, gamma in enumerate(kl_lognorm_scale):
    try:
        (trace_, metrics, I_pred_arr, Sig_I_pred_arr) = train_and_eval(
            gamma, n_batches, bg_penalty_scaling=None, kl_bern_scale=None
        )

        metrics.write_csv(f"bern{gamma}_run{i}.csv")  # Save evaluation metrics
        np.savetxt(f"trace_run_{i}.csv", trace_, fmt="%s")  # save loss trace
        np.savetxt(
            f"I_pred_run_{i}.csv", I_pred_arr, delimiter=","
        )  # save network predicted intensity
        np.savetxt(
            f"SigI_pred_run_{i}.csv", I_pred_arr, delimiter=","
        )  # save network predicted SigI
        torch.save(
            integrator.state_dict(), f"weights_gamma{gamma}.pth"
        )  # save final network weights

    except Exception as e:
        print(f"Failed with kl_lognorm_scale: {gamma}, Error:{e}")

# %%
