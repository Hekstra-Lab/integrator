# %% import numpy as np
from pylab import *
import glob
import os
from torchvision.io import read_image
import pandas as pd
import csv
import reciprocalspaceship as rs
import matplotlib.pyplot as plt
from dials.array_family import flex
import torch
from integrator.io import StillData, RotationData
import fabio  # Read crystallography filetypes https://github.com/silx-kit/fabio


# %%

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

# %%
# directory containing diffraction images
image_directory = '/Users/luis/dials_out/816_sbgrid_HEWL/816/'

# file name of image we will explore
image_file = image_dir + "301_helical_1_0001.cbf"

# name to DIALS integrated.refl file
reflections = flex.reflection_table.from_file(
    "/Users/luis/dials_out/816_sbgrid_HEWL/pass1/integrated.refl"
)

# %%


# %% [markdown]
# #Description
# The purpose of this script is to build a PyTorch DataLoader class
# to handl the diffraction images.

# %% [markdown]
# ##Dataset
# I am using a HEWL dataset provided by Kevin:
# https://data.sbgrid.org/dataset/816/

# %% [markdown]
# image number is last index of bbox
# bbox = [x1,x2,y1,y2,z1,z2]
# https://github.com/dials/dials/blob/af67cf6d87ac2c3c136947dc5b0720ff8b5a904f/src/dials/algorithms/spot_finding/factory.py#L272

# %% [markdown]
# For each image, I need the
# 1. x,y centroid position 'bbox[-1]'
# 2. the reflection index 'xyzobs.px.value[0]'
# 3. the image index 'xyzobs.px.value[1]'
# 4. the directoy index
# 5. the wavelength

# %%
# This is the code used to build the diffraction image dataframe

# %%
# lists to store reflection data
idx = []
xyzcal_x = []
xyzcal_y = []
h = []
k = []
l = []
integrated_intensity = []
sigI = []


# loop over reflection values
for i in range(len(reflections)):
    idx.append(list(reflections["bbox"][i])[-1])
    xyzcal_x.append(list(reflections["xyzcal.px"][i])[0])
    xyzcal_y.append(list(reflections["xyzcal.px"][i])[1])
    h.append(list(reflections[i]["miller_index"])[0])
    k.append(list(reflections[i]["miller_index"])[1])
    l.append(list(reflections[i]["miller_index"])[2])
    integrated_intensity.append(reflections[i]["intensity.sum.value"])
    sigI.append(reflections[i]["intensity.sum.variance"])


# building pandas DataFrame from integrated.refl file
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


# %%
# This is the reflection dataset
# contains all reflections found in the diffraction dataset

# column names
columns = ["idx", "h", "k", "l", "x", "y", "I", "SigI"]

df = pd.DataFrame(
    columns=columns,
    data=np.column_stack(
        [idx, h, k, l, xcentroid, ycentroid, integrated_intensity, sigI]
    ),
)

# %% [markdown]
# The following is code used to write a .csv of directory names and their index
# %%

# Image directory
image_directory = "./datasets/"

# Building .csv of datasets
dirs = glob.glob(image_directory + "*")
dirs_idx = list(np.linspace(1, len(dirs), len(dirs), dtype=int))

# list of image filenames
ims = sorted(glob.glob(image_directory + "/*/*", recursive=True))
ims
# list of image indices
ims_idx = list(np.linspace(1, len(ims), len(ims), dtype=int))

# writing .csv with dir_name and dir_idx
write_to_csv(dirs, dirs_idx, "./dirs.csv")

# writing .csv with im_name and im_idx
write_to_csv(ims, ims_idx, "./images.csv")

# storing names of .csv
image_labels = "./images.csv"
dir_labels = "./dirs.csv"


# dfs of .csv
image_df = pd.read_csv(image_labels)
dir_df = pd.read_csv(dir_labels)


img_idx = int(bboxes[max_z_idx][-2])
img = image_df.iloc[img_idx]

plt.imshow(fabio.open(image_df.iloc[500][0]).data, cmap="gray", vmax=50)

plt.imshow(fabio.open(image_df.iloc[img_idx][0]).data, cmap="gray", vmax=50)
plt.show()


img_arr = fabio.open(image_df.iloc[img_idx][0]).data

#
max_z_shoebox = shoeboxes[max_z_idx]
max_z_coordinates = max_z_shoebox.coords().as_numpy_array()
x_shape, y_shape = max_z_coordinates.shape[0]
max_z_coordinates = max_z_coordinates.reshape(max_z, int(x_shape / max_z), int(y_shape))

img_arr[max_z_coordinates[0, :, :]]


indices = np.array(max_z_coordinates[0, :, :], dtype=int)[:, 0:2]

img_arr[591][1726]


indices = np.array(max_z_coordinates, dtype=int)


indices = np.array(max_z_coordinates[0, :, :], dtype=int)


img_arr[indices].shape

row_indices, col_indices = indices[:, 0], indices[:, 1]

img_arr[row_indices, col_indices]


img_arr[indices[:, 0]].shape


max_z_shoebox.coords().as_numpy_array()


# %%

#
max_z_shoebox = shoeboxes[max_z_idx]
max_z_shoebox_data = np.array(max_z_shoebox.data)


# shoebox coordinates
coords_max_z = np.array(max_z_shoebox.coords())

# shoebox z-stack height
n = int(len(coords_max_z) / max_z)

# reshaped coordinates as grid
coords_max_z = coords_max_z.reshape((7, n, 3))


(np.column_stack([coords_max_z, max_z_shoebox_data])).reshape(7, n, 4)[-1, :, :]

# %%


intensity
(max_z_shoebox.values().as_numpy_array())[0:110]

image_df.iloc[img_idx][0].data

img_arr = fabio.open(image_df.iloc[img_idx][0]).data

plt.imshow(img_arr[indices], cmap="gray", vmax=50)
plt.show()


# %%
# DataLoader class to process diffraction images


class RotationData(torch.utils.data.Dataset):
    """
    Attributes:
        image_dir: Directory path where diffraction images live
        max_size: Maximum image size
        image_labels: .csv file containing diffraction image filnames and their index
    """

    def __init__(
        self, image_dir, image_labels, dir_labels, prediction_file, max_size=4096
    ):
        self.image_dir = image_dir  # directory of images
        self.dir_labels = pd.read_csv(dir_labels)
        self.prediction_file = prediction_file
        self.image_labels = pd.read_csv(image_labels)
        self.max_size = max_size

    def __len__(self):
        return len(self.image_labels)

    def constructDataFrame(self, idx):
        image_file = self.image_dir[idx]

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_labels.iloc[idx, 0])
        img = self.image_labels.iloc[idx][0]
        content = fabio.open(img)
        image = content.data
        label = self.image_labels.iloc[idx, 1]
        return image, label


im = pd.read_csv(image_labels).iloc[500][0]
fabio.open(im).data
pd.read_csv(image_labels).iloc[500, 1]

data = RotationData(
    image_dir=image_dir,
    dir_labels=dir_labels,
    image_labels=image_labels,
    prediction_file=prediction_file,
)


# %%
# Using the DataLoader
image_labels = "./labels.csv"
prediction_file = "./hewl_sbgrid/pass1/integrated.refl"
image_dir = '/Users/luis/dials_out/816_sbgrid_HEWL/816/'


data = RotationData(
    image_dir=image_dir,
    dir_labels=dir_labels,
    image_labels=image_labels,
    prediction_file=prediction_file,
)


# %%
# %% [markdown]
# This is the workflow to generate a .csv of the image indices
import csv

# list of diffraction image filenames
image_names = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
# list of indices
indices = np.linspace(1, len(image_names), len(image_names), dtype=int)
# Create .csv of image_filename, image_idx
out_csv = "./labels.csv"
write_to_csv(image_names, indices, out_csv)

indices

# %%

# create dataset class
dataset = RotationData(
    image_dir=image_dir,
    dir_labels=dir_labels,
    image_labels="./labels.csv",
    prediction_file=prediction_file,
)


# get the first image
first_image, first_image_idx = dataset.__getitem__(0)


# plot the first image
plt.imshow(first_image, cmap="gray", vmax=50)
plt.show()

# %% [markdown]
# Think about how to handle datsets with multiple passes.

# %% [markdown]
# In this chunk of code I analyze how Kevins io.py `StillData` class functions

from scipy.spatial import cKDTree

# store image as an array
pix = fabio.open("./hewl_sbgrid/816/301_helical_1_0001.cbf")
pix = pix.data


# storing x,y centroids to numpy array
xy = df[["x", "y"]].to_numpy()


# getting index of each pixel as (refl x 2) array
pxy = np.indices(pix.shape).T.reshape((-1, 2))

# Nearest neighbor tree
tree = cKDTree(xy)

# get distances (d) to closest neighbor
# get index (pidx) of nearest neighbor
d, pidx = tree.query(pxy)


# %% [markdown]
# Purpose of this script is to plot reflections as a z-stack

# dfs of .csv
image_df = pd.read_csv(image_labels)
dir_df = pd.read_csv(dir_labels)

img_idx = int(bboxes[max_z_idx][-2])
img = image_df.iloc[img_idx]

plt.imshow(fabio.open(image_df.iloc[500][0]).data, cmap="gray", vmax=50)

plt.imshow(fabio.open(image_df.iloc[img_idx][0]).data, cmap="gray", vmax=50)
plt.show()


img_arr = fabio.open(image_df.iloc[img_idx][0]).data

#
max_z_shoebox = shoeboxes[max_z_idx]
max_z_coordinates = max_z_shoebox.coords().as_numpy_array()
x_shape, y_shape = max_z_coordinates.shape[0]
max_z_coordinates = max_z_coordinates.reshape(max_z, int(x_shape / max_z), int(y_shape))



indices = np.array(max_z_coordinates[0, :, :], dtype=int)[:, 0:2]



indices = np.array(max_z_coordinates, dtype=int)


# %%
# maximum z_stack
max_z
# max z shoebox
max_z_shoebox = shoeboxes[max_z_idx]
# max z bounding box
max_z_bbox = bboxes[max_z_idx]

# Image range
image_range = [int(max_z_bbox[-2]), int(max_z_bbox[-1])]

# max z coordinates
max_z_coordinates = max_z_shoebox.coords().as_numpy_array()
max_z_coordinates
#
x_shape, y_shape = max_z_coordinates.shape[0]
max_z_coordinates = max_z_coordinates.reshape(max_z, int(x_shape / max_z), int(y_shape))


# %%


indices = np.array(max_z_coordinates[0, :, :], dtype=int)

img_arr[indices].shape

indices.shape
row_indices, col_indices = indices[:, 0], indices[:, 1]

img_arr[row_indices, col_indices]


img_arr[indices[:, 0]].shape


max_z_shoebox.coords().as_numpy_array()


# %%

#
max_z_shoebox = shoeboxes[max_z_idx]
max_z_shoebox_data = np.array(max_z_shoebox.data)


# shoebox coordinates
coords_max_z = np.array(max_z_shoebox.coords())

# shoebox z-stack height
n = int(len(coords_max_z) / max_z)

# reshaped coordinates as grid
coords_max_z = coords_max_z.reshape((7, n, 3))


(np.column_stack([coords_max_z, max_z_shoebox_data])).reshape(7, n, 4)[-1, :, :]

# %%


(max_z_shoebox.values().as_numpy_array())[0:110]

image_df.iloc[img_idx][0].data

img_arr = fabio.open(image_df.iloc[img_idx][0]).data

plt.imshow(img_arr[indices], cmap="gray", vmax=50)
plt.show()



image_df.iloc[img_idx][0]

fabio.open(image_df[image_df['Index'] == img_idx]['Filename'].value())



#
img_ = image_df[image_df['Index'] == img_idx]['Filename'].values[0]



#image filename
img_filename = image_df[image_df['Index'] == img_idx-1]['Filename'].values[0]

#store image data
img_arr = fabio.open(img_filename).data




img_arr[row_indices+1,col_indices+1]

max_z_shoebox_data[660:770]


plt.imshow(img_arr[indices], cmap="gray", vmax=50)
plt.show()




np.array(max_z_shoebox)

dir(max_z_shoebox)

refl_tables[0]['entering'].size()

dir(refl_tables[0]['entering'])

np.array(refl_tables[0])





img_arr[row_indices,col_indices].reshape()

img_arr
# %%
row_indices,col_indices

plt.imshow(img_arr, cmap="gray", vmax=50)


max_z_shoebox_data
max_z_bbox

row_indices,col_indices
img_arr[]
plt.show()




plt.imshow(img_arr, cmap="gray", vmax=50)
plt.imshow(img_arr[583:594,1718:1727],cmap='gray',vmax=50)
plt.show()
# %%



# %%

# Assuming 'max_z_coordinates' is an array of shape (1, height, width, 3)
# Extract row and column indices from max_z_coordinates
indices = max_z_coordinates[0, :, :, :2].astype(int)  # We take :2 to get the row and col indices
row_indices, col_indices = indices[:, 0], indices[:, 1]

# Plot the original image
plt.imshow(img_arr, cmap="gray", vmax=50)

# Create an overlay with the extracted indices - we need to create a mask
mask = np.zeros_like(img_arr, dtype=bool)
mask[row_indices, col_indices] = True
