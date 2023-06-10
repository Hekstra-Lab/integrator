from IPython import embed
from tqdm import trange
from pylab import *
import torch
from scipy.spatial import cKDTree
import pandas as pd

from integrator.io import ImageData
from integrator.models import Integrator,MLPEncoder,PoissonLikelihood,EllipticalProfile


image_file = "e080_001.mccd"
prediction_file = "e080_001.mccd.ii"

steps = 10_000
batch_size=100
max_size = 1024

depth = 10
dmodel = 64
eps = 1e-12
learning_rate=1e-4
beta = 10.
mc_samples = 100

img_data = ImageData([image_file], [prediction_file], max_size=max_size)
ds = img_data.get_data_set(0)
xy_idx, xy, dxy, counts, mask = img_data[0]


profile = EllipticalProfile(dmodel, eps=eps, beta=beta)
encoder = MLPEncoder(depth, dmodel)
likelihood = PoissonLikelihood(beta=beta, eps=eps)

integrator = Integrator(encoder, profile, likelihood)

integrator = integrator.cuda()

#Need to do a batch to initialize all the weights
idx = torch.multinomial(torch.ones(len(mask))/len(mask), num_samples=batch_size, replacement=False)
loss = integrator(xy[idx].cuda(),dxy[idx].cuda(),counts[idx].cuda(),mask[idx].cuda())

opt = torch.optim.Adam(integrator.parameters(), lr=learning_rate)
trace = []
grad_norms = []
bar = trange(steps)
for i in bar:
    opt.zero_grad()
    idx = torch.multinomial(torch.ones(len(mask))/len(mask), num_samples=batch_size, replacement=False)
    batch = (xy[idx].cuda(),dxy[idx].cuda(),counts[idx].cuda(),mask[idx].cuda())
    loss = integrator(*batch, mc_samples=mc_samples)
    loss.backward()
    trace.append(float(loss.detach().cpu().numpy()))
    gnorm = integrator.grad_norm().detach().cpu().numpy()
    grad_norms.append(gnorm)
    bar.set_postfix({'loss' : f'{trace[-1]:0.2e}', 'grad': f'{gnorm:0.2e}'})
    opt.step()

I,SigI = integrator.get_intensity_sigma(xy, dxy, counts, mask, batch_size)
plt.plot(ds.I, I, 'k.')
plt.xlabel("Intensity (Precognition)")
plt.ylabel("Intensity (Neural net)")
plt.loglog()
plt.grid(which='both', axis='both', linestyle='--')
plt.show()

embed(colors='linux')
