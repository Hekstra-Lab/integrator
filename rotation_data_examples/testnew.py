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
steps = 10_000
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

# Specify directory to `shoebox.refl` files
# shoebox_dir = "/Users/luis/dials_out/816_sbgrid_HEWL/pass1/"
shoebox_dir = "data/"
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


# integrator = integrator.cuda()


trace = []
grad_norms = []
bar = trange(steps)
I_values = []
SigI_values = []

# %%
opt = torch.optim.Adam(integrator.parameters(), lr=learning_rate)


# dir(train_loader)

# torch.isnan(next(iter(train_loader))[0][..., -1]).sum()
# torch.isinf(next(iter(train_loader))[0][..., -1]).sum()


# %%

torch.autograd.set_detect_anomaly(True)
for step in trange(steps):
    # Get batch data
    ims, masks = next(iter(train_loader))
    # ims = ims.cuda()
    # masks = masks.cuda()

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
plt.plot(grad_norms)
plt.title("Gradient Norms During Training")
plt.xlabel("Training Steps")
plt.ylabel("Gradient Norm")
plt.grid(True)
plt.savefig("grad_norms")
