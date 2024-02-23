import torch
import numpy as np
from integrator.io import RotationData
from tqdm import trange
import matplotlib.pyplot as plt
from integrator.models import (
    LogNormDistribution,
    ProfilePredictor,
    IntegratorV2,
    RotationReflectionEncoder,
    IntensityBgPredictor,
    RotationPixelEncoder,
    PoissonLikelihood,
)
from torch.utils.data import DataLoader

# %%


# %%
# Hyperparams
steps = 20
batch_size = 100
max_size = 1024
beta = 1.0
eps = 1e-12
depth = 10
learning_rate = 1e-3
dmodel = 64
feature_dim = 7
mc_samples = 100

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")

# %%

# Specify directory to `shoebox.refl` files
# shoebox_dir = "/Users/luis/dials_out/816_sbgrid_HEWL/pass1/"
shoebox_dir = "./"
# Get Dataset
rotation_data = RotationData(shoebox_dir=shoebox_dir)

train_loader = DataLoader(rotation_data, batch_size=batch_size, shuffle=True)


# Encoders
refl_encoder = RotationReflectionEncoder(depth, dmodel, feature_dim)
intensity_bacground = IntensityBgPredictor(depth, dmodel)
pixel_encoder = RotationPixelEncoder(depth=depth, dmodel=dmodel, d_in_refl=6)
profile_ = ProfilePredictor(dmodel, depth)
likelihood = PoissonLikelihood(beta=beta, eps=eps)
bglognorm = LogNormDistribution(dmodel, eps, beta)

# %%

# ims, masks = next(iter(train_loader))

# # Encoding reflections
# encoded_refls = refl_encoder(ims, mask=masks)
# # Encoding pixel
# encoded_pixels = pixel_encoder(ims[:, :, 0:-1])
# # Output reflection pathway
# refl_out = intensity_bacground(encoded_refls)

# # output pixel pathway
# pixel_out = profile_(encoded_refls, encoded_pixels)

integrator = IntegratorV2(
    refl_encoder, intensity_bacground, pixel_encoder, profile_, bglognorm, likelihood
)

integrator = integrator.to(device)


# integrator = integrator.cuda()


# %%
trace = []
grad_norms = []
bar = trange(steps)
I_values = []
SigI_values = []

# %%
opt = torch.optim.Adam(integrator.parameters(), lr=learning_rate)


# %%
# dir(train_loader)

# torch.isnan(next(iter(train_loader))[0][..., -1]).sum()
# torch.isinf(next(iter(train_loader))[0][..., -1]).sum()


# %%

torch.autograd.set_detect_anomaly(True)
for step in trange(steps):
    # Get batch data
    ims, masks = next(iter(train_loader))
    ims = ims.to(device)
    masks = masks.to(device)

    # Forward pass
    opt.zero_grad()
    loss = integrator(ims, masks)

    # Backward pass

    loss.backward()
    opt.step()

    # Record metrics
    trace.append(loss.item())
    grad_norms.append(
        torch.nn.utils.clip_grad_norm(integrator.parameters(), max_norm=max_size)
    )

    bar.set_description(f"Step {(step+1)}/{steps},Loss:{loss.item():.4f}")

# %%
data_loader = DataLoader(rotation_data, batch_size=1000, shuffle=False)
all_I = []
all_SigI = []
i = 0

# %%

ims, masks = next(iter(data_loader))
# %%
xyz = ims[..., 0:3]
dxyz = ims[..., 3:6]
counts = torch.clamp(ims[..., -1], min=0)


# %%
def get_intensity_sigma(shoebox):
    I, SigI = [], []
    xyz = shoebox[..., 0:3]
    dxyz = shoebox[..., 3:6]
    counts = torch.clamp(shoebox[..., -1], min=0)
    device = next(integrator.parameters()).device

    reflrep = reflencoder(shoebox)
    paramrep = paramencoder(reflrep)
    pixelrep = pixel_encoder(shoebox[:, :, 0:-1])
    pijrep = pijencoder(reflrep, pixelrep)

    I.append(i.detach().cpu().numpy())
    SigI.append(s.detach().cpu().numpy())

    for batch in zip(
        torch.split(xyz, batch_size, dim=0),
        torch.split(dxyz, batch_size, dim=0),
        torch.split(counts, batch_size, dim=0),
    ):
        device = next(parameters()).device
        i, s = get_intensity_sigma_batch(*(i.to(device=device) for i in batch))
        I.append(i.detach().cpu().numpy())
        SigI.append(s.detach().cpu().numpy())
    I, SigI = np.concatenate(I), np.concatenate(SigI)
    return I, SigI


def get_intensity_sigma_batch(shoebox):
    # norm_factor = get_per_spot_normalization(shoebox)
    # shoebox[:, :, -1] = shoebox[:, :, -1] / norm_factor.unsqueeze(-1)
    reflrep = reflencoder(shoebox)
    paramrep = paramencoder(reflrep)
    pixelrep = pixelencoder(shoebox[:, :, 0:-1])
    pijrep = pijencoder(reflrep, pixelrep)

    q = bglognorm.distribution(paramrep)
    I, SigI = q.mean, q.stddev

    # I, SigI = I * norm_factor, SigI * norm_factor
    return I, SigI


# %%
for padded_data, mask in data_loader:
    i += 1
    print(f"{i}")

# %%

# %%


# %%
plt.plot(grad_norms)
plt.title("Gradient Norms During Training")
plt.xlabel("Training Steps")
plt.ylabel("Gradient Norm")
plt.grid(True)
plt.savefig("grad_norms")


if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")
