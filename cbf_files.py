# %% import numpy as np
from pylab import *
import os
from torchvision.io import read_image
import pandas as pd
import cbf
import csv
import reciprocalspaceship as rs
import matplotlib.pyplot as plt
from dials.array_family import flex
import torch
from integrator.io import ImageData


# Function to write to CSV
def write_to_csv(filenames, indices, output_file_path):
    """
    Args:
        filenames ():
        indices ():
        output_file_path ():
    """
    with open(output_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Writing the header
        writer.writerow(["Filename", "Index"])

        # Writing the content
        for filename, index in zip(filenames, indices):
            writer.writerow([filename, index])


image_dir = "/Users/luis/integrator/hewl_sbgrid/816/301_helical_1_0001.cbf"
image_file = image_dir + "301_helical_1_0001.cbf"


# %% [markdown]
# I am using the [cbf](https://github.com/paulscherrerinstitute/cbf) to open
# .cbf diffraction files.

# %%

# read the image
content = cbf.read(image_dir)

# store data and column names
arr = content.data
metadata = content.metadata

# plot image
plt.imshow(arr, cmap="gray", vmax=50)
plt.show()

# %% [markdown]
# The diffraction images from this dataset (SBGRID_816) are 2500x2500.


# %% [markdown]
# Now we will create a dataset from a dials reflection prediction file.
# I need to figure out which columns specify the centroid on the image.


# %%
# reflections file
reflections = flex.reflection_table.from_file(
    "/Users/luis/integrator/hewl_sbgrid/pass1/strong.refl"
)

# column names
keys = list(reflections[0].keys())
keys = [
    "bbox",
    "intensity.sum.value",
    "intensity.sum.variance",
    "xyzobs.px.value",
    "xyzobs.px.variance",
]

# dictionary of results for each key
results = {key: [] for key in keys}

# %%

# iterating over each .refl column
for key in keys:
    for i in range(len(reflections)):
        results[key].append(reflections[key][i])


# %% [markdown]
# Now I plot some predicted centroids over the diffraction image

# %%
# image number is last index of bbox
# bbox = [x1,x2,y1,y2,z1,z2]
# https://github.com/dials/dials/blob/af67cf6d87ac2c3c136947dc5b0720ff8b5a904f/src/dials/algorithms/spot_finding/factory.py#L272

img_number = list(reflections[250]["bbox"])[-1]

# Picking random reflection centroid
x = reflections[250]["xyzobs.px.value"][0]
y = reflections[250]["xyzobs.px.value"][1]


# plot centroid over image
plt.imshow(arr, cmap="gray", vmax=50)
plt.scatter(x, y, c="red", s=0.5)
plt.show()


results["bbox"][0]

# %% [markdown]
# For each image, I need the
# 1. x,y centroid position 'bbox[-1]'
# 2. the reflection index 'xyzobs.px.value[0]'
# 3. the image index 'xyzobs.px.value[1]'

# %%
## Building reflection and centroid dataset

keys = [
    "bbox",
    "intensity.sum.value",
    "intensity.sum.variance",
    "xyzobs.px.value",
    "xyzobs.px.variance",
]

# dictionary of results for each key
results = {key: [] for key in keys}

# %%
# This is the code used to build the diffraction image dataframe

# Reflection file

reflections = flex.reflection_table.from_file(
    "/Users/luis/integrator/hewl_sbgrid/pass1/integrated.refl"
)

list(reflections[0].keys())
reflections[0]


# storing image indices
image_indices = []


for i in range(len(reflections)):
    image_indices.append(list(reflections["bbox"][i])[-1])
image_indices = np.array(image_indices)
image_indices = image_indices.reshape(image_indices.shape[0], 1)

# storing x centroid
for i in range(len(reflections)):
    xcentroid.append(list(reflections["xyzcal.px"][i])[0])
xcentroid = np.array(xcentroid)
xcentroid = xcentroid.reshape(xcentroid.shape[0], 1)

# storing x centroid
idx = []
xcentroid = []
ycentroid = []
h = []
k = []
l = []
integrated_intensity = []
sigI = []

for i in range(len(reflections)):
    idx.append(list(reflections["bbox"][i])[-1])
    xcentroid.append(list(reflections["xyzcal.px"][i])[0])
    ycentroid.append(list(reflections["xyzcal.px"][i])[1])
    h.append(list(reflections[i]["miller_index"])[0])
    k.append(list(reflections[i]["miller_index"])[1])
    l.append(list(reflections[i]["miller_index"])[2])
    integrated_intensity.append(reflections[i]["intensity.sum.value"])
    sigI.append(reflections[i]["intensity.sum.variance"])

idx = np.array(idx).reshape(len(idx), 1)
xcentroid = np.array(xcentroid).reshape(len(xcentroid), 1)
ycentroid = np.array(ycentroid).reshape(len(ycentroid), 1)
h = np.array(h).reshape(len(h), 1)
k = np.array(k).reshape(len(k), 1)
l = np.array(l).reshape(len(l), 1)
integrated_intensity = np.array(integrated_intensity).reshape(
    len(integrated_intensity), 1
)
sigI = np.array(sigI).reshape(len(sigI), 1)
ycentroid = np.array(ycentroid)
ycentroid = ycentroid.reshape(ycentroid.shape[0], 1)

columns = ["idx", "h", "k", "l", "x", "y", "I", "SigI"]

pd.DataFrame(
    columns=columns,
    data=np.column_stack(
        [idx, h, k, l, xcentroid, ycentroid, integrated_intensity, sigI]
    ),
)


# intensities

# column labels
labels = ["imgIdx", "x", "y"]

# reflection dataframe
df = pd.DataFrame(
    columns=labels, data=np.column_stack([image_indices, xcentroid, ycentroid])
)

# %%

image_directory = "./hewl_sbgrid/816/"


# %%
# DataLoader class to process diffraction images


class DiffractionData(torch.utils.data.Dataset):
    """
    Attributes:
        image_dir: Directory path where diffraction images live
        max_size: Maximum image size
        image_labels: .csv file containing diffraction image filnames and their index
    """

    def __init__(self, image_dir, image_labels, max_size=4096):
        self.image_dir = image_dir
        self.max_size = max_size
        self.image_labels = pd.read_csv(image_labels)

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_labels.iloc[idx, 0])
        content = cbf.read(img_path)
        image = content.data
        label = self.image_labels.iloc[idx, 1]
        return image, label


# %%

# list of diffraction image filenames
image_names = sorted(os.listdir(image_directory))[0:1439]

# list of indices
indices = np.linspace(1, len(image_names), len(image_names), dtype=int)

# Create .csv of image_filename, image_idx
out_csv = "./labels.csv"
write_to_csv(image_names, indices, out_csv)


# %%
# Using the DataLoader
image_dir = "/Users/luis/integrator/hewl_sbgrid/816/"


# create dataset class
dataset = DiffractionData(image_dir=image_dir, image_labels="./labels.csv")

# check number of diffraction images
num_images = dataset.__len__()
num_images

# get the first image
first_image, first_image_idx = dataset.__getitem__(0)

# plot the first image

plt.imshow(first_image, cmap="gray", vmax=50)
plt.show()


# %% [markdown]
# Think about how to handle datsets with multiple passes.
