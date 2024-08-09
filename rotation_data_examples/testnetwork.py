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
import torch.nn as nn

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
profile_ = ProfilePredictor(dmodel, depth, max_pixel=rotation_data.max_voxels)
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

        rotation_data.set_mode("test")

        num_batches = n_batches
        I, SigI = [], []
        pij_ = []
        bg = []
        counts = []

        with torch.no_grad():
            for i, (ims, masks) in enumerate(train_loader):
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

next(iter(train_loader))[0].min()
print(refl_encoder)
print(pixel_encoder)
print(intensity_bacground)
print(profile_)


# %%
mvnn = Builder(dmodel, eps, beta)

mvnn(out1, out2, ims[..., 3:6])

dxy = ims[..., 3:6]


# 10 images/masks


im, mask = next(iter(train_loader))
counts = im[..., -1]


refl_rep = refl_encoder(im, mask)

refl_rep.shape

param_rep = intensity_bacground(refl_rep)

pix_rep = pixel_encoder(im[:, :, 0:-1])

profile = profile_(refl_rep, pix_rep, mask=mask)


profile.sum(axis=1)


bg, q = bglognorm(param_rep)


likelihood(counts, profile, bg, q, emp_bg, kl_lognorm_scale)
z = q.rsample([100])


z.shape
profile.shape

print(profile_)

(z.squeeze(0) * profile + bg).shape


z * profile.squeeze(-2)

z[0] * profile.squeeze(-2)


profile.squeeze(-2).sum


pool = MeanPool()

integrator(im, mask, emp_bg, kl_lognorm_scale)

pool(out, mask).shape

print(profile_)

# %%


# Embed shoeboxes
class MeanPool(torch.nn.Module):
    def __init__(self, dim=-2):
        super().__init__()
        # self.register_buffer(
        # "dim",
        # torch.tensor(),
        # )

    def forward(self, data, mask=None):
        out = data.sum(1, keepdim=True)
        if mask is None:
            denom = data.shape[-1]
        else:
            denom = mask.sum(-1, keepdim=True)
        out = out / denom.unsqueeze(-1)

        return out


print(profile_)


# %%
class RunningMoments:
    def __init__(self, axis=-2):
        self.n = 0  # number samples
        self.s = 0.0  # variance
        self.mean = 0.0  # mean
        self.axis = axis

    def update(self, x):
        k = len(x)  # number of samples
        self.n += k  # update number samples
        diff = x - self.mean  # subtract mean from samples
        self.mean = self.mean + np.sum(diff / self.n, axis=self.axis, keepdims=True)
        diff *= x - self.mean  # update squared difference
        self.s = self.s + diff.sum(axis=self.axis, keepdims=True)  # update variance

    @property
    def var(self):
        if self.n <= 1:
            return None
        return self.s / self.n

    @property
    def std(self):
        return np.sqrt(self.var)


# %%
class Standardize(nn.Module):
    def __init__(
        self, center=True, feature_dim=7, max_counts=float("inf"), epsilon=1e-6
    ):
        super().__init__()
        self.epsilon = epsilon
        self.center = center
        self.max_counts = max_counts
        self.register_buffer("mean", torch.zeros((1, 1, feature_dim)))
        self.register_buffer("m2", torch.zeros((1, 1, feature_dim)))
        self.register_buffer("count", torch.tensor(0.0))

    @property
    def var(self):
        m2 = torch.clamp(self.m2, min=self.epsilon)
        return m2 / self.count.clamp(min=1)

    @property
    def std(self):
        return torch.sqrt(self.var)

    def update(self, im, mask=None):
        if mask is None:
            k = len(im)
        else:
            k = mask.sum()  # count num of pixels in batch
        self.count += k

        if mask is None:
            diff = im - self.mean
        else:
            diff = (im - self.mean) * mask.unsqueeze(-1)

        self.mean += torch.sum(diff / self.count, dim=(1, 0))

        if mask is None:
            diff *= im - self.mean
        else:
            diff *= (im - self.mean) * mask.unsqueeze(-1)
        self.m2 += torch.sum(diff, dim=(1, 0))

    def standardize(self, im, mask=None):
        if mask is None:
            if self.center:
                return (im - self.mean) / self.std
        else:
            if self.center:
                return ((im - self.mean) * mask.unsqueeze(-1)) / self.std
        return im / self.std

    def forward(self, im, mask, training=True):
        if self.count > self.max_counts:
            training = False
        if training:
            self.update(im, mask)
        return self.standardize(im, mask)


# %%
l = 10_000
d = 5
x = np.arange(l)[..., None] * np.ones((l, d))
x = x.astype("float32")
n = RunningMoments()
s = Standardize(feature_dim=d)

for batch in np.split(x, 10):
    x_tens = torch.tensor(batch, dtype=torch.float32)
    n.update(batch)
    s(x_tens, mask=None)
    assert np.allclose(n.mean, s.mean)
    assert np.allclose(n.var, s.var)
    assert np.allclose(n.std, s.std)

# %%
torch.tensor(x)

# %%

standardize = Standardize()
standardize.update(im, mask)

standardize(im, mask=mask).min()

standardize.mean

k = mask.sum()
mean = torch.sum(im / torch.sum(mask), dim=(1, 0))
mean

diff = (im * (im - mean)) * mask.unsqueeze(-1)
diff.shape


torch.sum(im, dim=(1, 0)) / torch.sum(mask)


# %%


epoch = 0
bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

s = Standardize()

for i, (im, mask) in enumerate(bar):
    if i < 1000:
        s(im, mask)
        print(s.var)
    else:
        break

from integrator.layers import Linear


# %%
class DynamicLinear(torch.nn.Module):
    def __init__(self, input_dim, output_dim_2D, output_dim_3D):
        super().__init__()
        self.output_dim_2D = output_dim_2D
        self.output_dim_3D = output_dim_3D

        self.linear1 = Linear(input_dim, self.output_dim_2D)
        self.linear2 = Linear(input_dim, self.output_dim_3D)

    def forward(self, x, d_flag):
        out1 = self.linear1(x)
        out2 = self.linear2(x)
        return out1, out2
