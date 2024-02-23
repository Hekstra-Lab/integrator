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
# Hyperparameters
steps = 100
batch_size = 100
max_size = 1024
beta = 1.0
eps = 1e-12
depth = 10
learning_rate = 0.0001
dmodel = 64
feature_dim = 7
mc_samples = 100

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# Loading data
shoebox_dir = "/Users/luis/integrator/rotation_data_examples/data/"
rotation_data = RotationData(shoebox_dir=shoebox_dir)

# %%
# loads a shoebox and corresponding mask
train_loader = DataLoader(rotation_data, batch_size=batch_size, shuffle=True)


# Encoders
refl_encoder = RotationReflectionEncoder(depth, dmodel, feature_dim)
intensity_bacground = IntensityBgPredictor(depth, dmodel)
pixel_encoder = RotationPixelEncoder(depth=depth, dmodel=dmodel, d_in_refl=6)
profile_ = ProfilePredictor(dmodel, depth)
likelihood = PoissonLikelihood(beta=beta, eps=eps)
bglognorm = LogNormDistribution(dmodel, eps, beta)

# integration model

integrator = IntegratorV2(
    refl_encoder, intensity_bacground, pixel_encoder, profile_, bglognorm, likelihood
)

# %%
# training loop
trace = []
grad_norms = []
bar = trange(steps)
I_values = []
SigI_values = []
opt = torch.optim.Adam(integrator.parameters(), lr=learning_rate)  # optimizer

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

    bar.set_description(f"Step {(step+1)}/{steps},Loss:{loss.item():.4f}")

# %%
# Evaluation Loop
eval_loader = DataLoader(rotation_data, batch_size=batch_size, shuffle=False)

# Evaluation
# model_weights = integrator.state_dict()
# torch.save(model_weights, "integrator_model_weights.pth")
integrator.load_state_dict(torch.load("integrator_model_weights.pth"))
integrator.to(device)
# set model to evaluation mode
integrator.eval()

num_batches = 100
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

        I.append(output[0])
        SigI.append(output[1])
        bg.append(output[2])
        pij_.append(output[3])
        counts.append(output[4])
# %%
# Stats
# Mean intensity across dataset
print(f"Dataset mean intensity: {np.array(I).ravel().mean()}")
# np.array(I).ravel().shape

# Mean background across dataset
print(f"Dataset mean background {np.array(bg).ravel().mean()}")

# Max background
print(f"Max background: {np.array(bg).ravel().max()}")
print(f"Min background: np.array(bg).ravel().min()")

# Min background
rotation_data.refl_tables.columns

# %%

# Plots

# Plotting loss as function of step
plt.plot(np.linspace(0, steps, steps - 10), trace[10::])
plt.ylabel("loss")
plt.xlabel("step")
plt.show()

# Scatter plot of network-intensity vs DIALS summmation model
plt.scatter(
    np.array(I).ravel(),
    rotation_data.refl_tables["intensity.sum.value"][0 : num_batches * batch_size],
)
plt.savefig("intensity_vs_summation_model", dpi=300)

# Scatter plot of network-intensity vs DIALS profile model
plt.scatter(
    np.array(bg).ravel(), rotation_data.refl_tables["intensity.sum.value"][0:1000]
)
plt.yscale("log")
plt.xscale("log")
plt.show()
plt.savefig("intensity_vs_profile.png", dpi=300)
