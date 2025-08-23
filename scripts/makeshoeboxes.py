# to get to the starting point:
#
# dials.import ~/data/i04_bag_training/*gz
# dials.find_spots imported.expt
# dials.index imported.expt strong.refl
# dials.refine indexed.*
# dials.predict refined.expt
# then
#
# dials.python make_shoeboxes.py predicted.refl imported.expt x=2 ny=2 nz=2

import gc

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dials.array_family import flex
from dials.util import Sorry
from dials.util.options import ArgumentParser, flatten_experiments, flatten_reflections
from libtbx.phil import parse

# %%

phil_scope = parse(
    """
  nx = 1
    .type = int(value_min=1)
    .help = "+/- x around centroid"

  ny = 1
    .type = int(value_min=1)
    .help = "+/- y around centroid"

  nz = 1
    .type = int(value_min=1)
    .help = "+/- z around centroid"

  output {
    reflections = 'shoeboxes.refl'
      .type = str
      .help = "The integrated output filename"
  }
"""
)

# Create the parser
parser = ArgumentParser(
    phil=phil_scope,
    read_experiments=True,
    read_reflections=True,
)

# arguments and options
params, options = parser.parse_args(
    ["integrated.refl", "integrated.expt", "nx=10", "ny=10", "nz=1"]
)


# reflections
reflections = flatten_reflections(params.input.reflections)

# experiments
experiments = flatten_experiments(params.input.experiments)

#
diff_phil = parser.diff_phil.as_str()

if not any([experiments, reflections]):
    parser.print_help()
    exit(0)
elif len(experiments) > 1:
    raise Sorry("More than 1 experiment set")
elif len(experiments) == 1:
    imageset = experiments[0].imageset
if len(reflections) != 1:
    raise Sorry("Need 1 reflection table, got %d" % len(reflections))
else:
    reflections = reflections[0]

# Check the reflections contain the necessary stuff
assert "bbox" in reflections
assert "panel" in reflections

# Get some models
detector = imageset.get_detector()
scan = imageset.get_scan()
frame0, frame1 = scan.get_array_range()

x, y, z = reflections["xyzcal.px"].parts()

x = flex.floor(x).iround()
y = flex.floor(y).iround()
z = flex.floor(z).iround()

bbox = flex.int6(len(reflections))

dx, dy = detector[0].get_image_size()

for j, (_x, _y, _z) in enumerate(zip(x, y, z, strict=False)):
    x0 = _x - params.nx
    x1 = _x + params.nx + 1
    y0 = _y - params.ny
    y1 = _y + params.ny + 1
    z0 = _z - params.nz
    z1 = _z + params.nz + 1
    if x0 < 0:
        x0 = 0
    if x1 >= dx:
        x1 = dx - 1
    if y0 < 0:
        y0 = 0
    if y1 >= dy:
        y1 = dy - 1
    if z0 < frame0:
        z0 = frame0
    if z1 >= frame1:
        z1 = frame1 - 1
    bbox[j] = (x0, x1, y0, y1, z0, z1)

reflections["bbox"] = bbox

# beware change of variables - removing those which were a different shape because boundary conditions
x0, x1, y0, y1, z0, z1 = bbox.parts()

dx, dy, dz = (x1 - x0), (y1 - y0), (z1 - z0)

good = (dx == 2 * params.nx + 1) & (dy == 2 * params.ny + 1) & (dz == 2 * params.nz + 1)

print(f"{good.count(True)} / {good.size()} kept")

reflections = reflections.select(good)

# store shoeboxes into reflection file
reflections["shoebox"] = flex.shoebox(
    reflections["panel"], reflections["bbox"], allocate=True
)

# get shoeboxes
reflections.extract_shoeboxes(imageset)

# assign an identifier to each reflection
reflections["refl_ids"] = flex.int(np.arange(len(reflections)))

for experiment, indices in reflections.iterate_experiments_and_indices(experiments):
    print(experiment)

# get refls by index
subset = reflections.select(indices)

# mask the neighbouring reflections
modified_count = 0

for experiment, indices in reflections.iterate_experiments_and_indices(experiments):
    subset = reflections.select(indices)
    modified = subset["shoebox"].mask_neighbouring(
        subset["miller_index"],
        experiment.beam,
        experiment.detector,
        experiment.goniometer,
        experiment.scan,
        experiment.crystal,
    )
    modified_count += modified.count(True)

# mask shape is (N_samples x z_shape x x_shape x y_shape)
masks = torch.stack(
    [
        torch.tensor(
            sbox.mask.as_numpy_array().ravel(), dtype=torch.float32, requires_grad=False
        )
        for sbox in reflections["shoebox"]
    ]
)

# Find neighbouring pixels and set to 0
masks[(masks == 3)] = 0

# coordinates
coordinates = torch.stack(
    [
        torch.tensor(sbox.coords().as_numpy_array(), dtype=torch.float32)
        for sbox in reflections["shoebox"]
    ]
)

# observed intensities
i_obs = torch.stack(
    [
        torch.tensor(
            sbox.data.as_numpy_array().ravel().astype(np.float32),
            dtype=torch.float32,
            requires_grad=False,
        )
        for sbox in reflections["shoebox"]
    ]
).unsqueeze(-1)

# centroids
centroids = torch.tensor(
    reflections["xyzobs.px.value"].as_numpy_array(), dtype=torch.float32
).unsqueeze(1)

# distance from each coordinate to centroid
dxyz = torch.abs(coordinates - centroids)


# delete unused tensor
del centroids
gc.collect()

# Prepare the tensor dataset
samples = torch.cat((coordinates, dxyz, i_obs), dim=-1)


# delete unused tensors
del coordinates
del dxyz
del i_obs
gc.collect()

# filter dead panels
filter = [((samples[..., -1] < 0).sum(-1) < 700)]


# Filter panel gaps
filtered_samples = torch.clamp(samples[filter], min=0)

# delete unused tensor
del samples
gc.collect()

# filter masks for dead panel gaps
filtered_masks = masks[filter]

# delete unused tensor
del masks
gc.collect()

# Metadata Tensor
metadata = torch.stack(
    [
        torch.tensor(
            reflections["intensity.sum.value"].as_numpy_array(), dtype=torch.float32
        ),
        torch.tensor(
            reflections["intensity.sum.variance"].as_numpy_array(), dtype=torch.float32
        ),
        torch.tensor(
            reflections["intensity.prf.value"].as_numpy_array(), dtype=torch.float32
        ),
        torch.tensor(
            reflections["intensity.prf.variance"].as_numpy_array(), dtype=torch.float32
        ),
        torch.tensor(reflections["refl_ids"].as_numpy_array(), dtype=torch.float32),
    ]
).transpose(0, 1)

filtered_metadata = metadata[filter]

# delete unused tensors
del metadata
gc.collect()

# Save tensors
torch.save(filtered_masks, "masks.pt")
torch.save(filtered_samples, "samples.pt")
torch.save(filtered_metadata, "metadata.pt")

# delete shoebox column
del reflections["shoebox"]

# save reflection file
reflections.as_file("reflections_.refl")

flex.reflection_table.from_file("reflections_.refl").nrows()

torch.load("metadata.pt")[..., -1].max()
torch.load("metadata.pt")[..., -1].shape


# %%
# Check the reflection masks up to this point

# %%
idx = 80101
refl = subset.select(modified)["shoebox"][idx].data.as_numpy_array()
# mask = reflections.select(modified)['shoebox'][idx].mask.as_numpy_array()

refl = filtered_samples[idx][..., -1].reshape(3, 21, 21)
mask = filtered_masks[idx].reshape(3, 21, 21)


# subplots of the reflection and mask
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
sns.heatmap(refl[0], ax=ax[0, 0])
sns.heatmap(refl[1], ax=ax[0, 1])
sns.heatmap(refl[2], ax=ax[0, 2])
sns.heatmap(mask[0], ax=ax[1, 0])
sns.heatmap(mask[1], ax=ax[1, 1])
sns.heatmap(mask[2], ax=ax[1, 2])
plt.suptitle("Reflections and Masks", fontsize=14)
plt.show()
