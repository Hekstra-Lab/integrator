from IPython import embed
from tqdm import trange
from pylab import *
import torch
from scipy.spatial import cKDTree
import pandas as pd

from integrator.io import ImageData
from integrator.models import Integrator, MLPEncoder, PoissonLikelihood, Elliptical


image_file = "e080_001.mccd"
prediction_file = "e080_001.mccd.ii"

steps = 10
batch_size = 100
max_size = 1024

depth = 10
dmodel = 64
eps = 1e-12
learning_rate = 1e-4
beta = 10.0
mc_samples = 100

img_data = ImageData([image_file], [prediction_file], max_size=max_size)
ds = img_data.get_data_set(0)
xy_idx, xy, dxy, counts, mask = img_data[0]


encoder = MLPEncoder(depth, dmodel)
likelihood = PoissonLikelihood(beta=beta, eps=eps)

integrator = Integrator(encoder, profile, likelihood)

integrator = integrator.cuda()

# Need to do a batch to initialize all the weights
idx = torch.multinomial(
    torch.ones(len(mask)) / len(mask), num_samples=batch_size, replacement=False
)
loss = integrator(xy[idx].cuda(), dxy[idx].cuda(), counts[idx].cuda(), mask[idx].cuda())

opt = torch.optim.Adam(integrator.parameters(), lr=learning_rate)
trace = []
grad_norms = []
bar = trange(steps)
for i in bar:
    opt.zero_grad()
    idx = torch.multinomial(
        torch.ones(len(mask)) / len(mask), num_samples=batch_size, replacement=False
    )
    batch = (xy[idx].cuda(), dxy[idx].cuda(), counts[idx].cuda(), mask[idx].cuda())
    loss = integrator(*batch, mc_samples=mc_samples)
    loss.backward()
    trace.append(float(loss.detach().cpu().numpy()))
    gnorm = integrator.grad_norm().detach().cpu().numpy()
    grad_norms.append(gnorm)
    bar.set_postfix({"loss": f"{trace[-1]:0.2e}", "grad": f"{gnorm:0.2e}"})
    opt.step()

    # Compute I and SigI and add to their lists
    I, SigI = integrator.get_intensity_sigma(xy, dxy, counts, mask, batch_size)
    I_values.append(I.mean())
    SigI_values.append(SigI.mean())

I, SigI = integrator.get_intensity_sigma(xy, dxy, counts, mask, batch_size)
plt.plot(ds.I, I, "k.")
plt.xlabel("Intensity (Precognition)")
plt.ylabel("Intensity (Neural net)")
plt.loglog()
plt.grid(which="both", axis="both", linestyle="--")
plt.show()

# Function to get intensity
I, SigI = integrator.get_intensity_sigma(xy, dxy, counts, mask, batch_size)
plt.plot(ds.I, I, "k.")
plt.xlabel("Intensity (Precognition)")
plt.ylabel("Intensity (Neural net)")
plt.loglog()
plt.grid(which="both", axis="both", linestyle="--")
plt.savefig("plot.png")

# Plot the loss
plt.figure()
plt.plot(trace, "k-")
plt.title("Loss over steps")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.savefig("loss_plot.png")

# Plot the gradient norm
plt.figure()
plt.plot(grad_norms, "k-")
plt.title("Gradient norm over steps")
plt.xlabel("Step")
plt.ylabel("Gradient norm")
plt.savefig("grad_norm_plot.png")

# Plot the average intensity
plt.figure()
plt.plot(I_values, "k-")
plt.title("Average intensity over steps")
plt.xlabel("Step")
plt.ylabel("Average intensity")
plt.savefig("I_plot.png")

# Plot the average SigI
plt.figure()
plt.plot(SigI_values, "k-")
plt.title("Average SigI over steps")
plt.xlabel("Step")
plt.ylabel("Average SigI")
plt.savefig("SigI_plot.png")


embed(colors="linux")
