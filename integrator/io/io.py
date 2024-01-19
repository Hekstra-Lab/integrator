from pylab import *
import polars as pl
import torch
from scipy.spatial import cKDTree
import pandas as pd
import reciprocalspaceship as rs
import glob
import os
from dials.array_family import flex
import numpy as np


# %%
class RotationData(torch.utils.data.Dataset):
    def __init__(
        self,
        # image_dir,
        max_size=4096,
        shoebox_dir=None,
    ):
        self.max_size = max_size
        self.shoebox_dir = shoebox_dir  # directory of shoeboxes
        self.shoebox_filenames = self._get_shoebox_filenames()
        self.refl_tables = self._get_refl_tables()
        self.data, self.shoeboxes, self.centroids = self._get_rotation_data()
        self.max_voxels = max(self.data["shoebox"].map_elements(self._get_num_coords))
        self.padded_data, self.padded_masks = self._pad_data()
        self.padded_filtered_data = self._clean_data()

    def _get_rotation_data(self):
        # Create a DataFrame from the reflection tables
        refl_df = pl.DataFrame({"reflection_table": self.refl_tables})

        # Stack the shoebox arrays and other attributes
        arr_shoebox = np.vstack(
            refl_df["reflection_table"].map_elements(self._get_shoebox)
        )
        arr_miller_index = np.concatenate(
            refl_df["reflection_table"].map_elements(self._get_miller_index)
        )
        arr_xyz_cal_px = np.concatenate(
            refl_df["reflection_table"].map_elements(self._get_xyz_cal_px)
        )
        arr_xyz_obs_px_value = np.concatenate(
            refl_df["reflection_table"].map_elements(self._get_xyz_obs_px_value)
        )
        arr_intensity_sum_value = np.concatenate(
            refl_df["reflection_table"].map_elements(self._get_intensity_sum_value)
        )
        arr_intensity_sum_variance = np.concatenate(
            refl_df["reflection_table"].map_elements(self._get_intensity_sum_variance)
        )

        # number of reflections
        # Create a new DataFrame with all attributes
        result_df = pl.DataFrame(
            {
                "h": arr_miller_index[:, 0],
                "k": arr_miller_index[:, 1],
                "l": arr_miller_index[:, 2],
                "x_cal_px": arr_xyz_cal_px[:, 0],
                "y_cal_px": arr_xyz_cal_px[:, 1],
                "z_cal_px": arr_xyz_cal_px[:, 2],
                "x_obs_px_value": arr_xyz_obs_px_value[:, 0],
                "y_obs_px_value": arr_xyz_obs_px_value[:, 1],
                "z_obs_px_value": arr_xyz_obs_px_value[:, 2],
                "intensity_sum_value": arr_intensity_sum_value,
                "intensity_sum_variance": arr_intensity_sum_variance,
                "shoebox": arr_shoebox.ravel(),
            }
        )
        return result_df, arr_shoebox, arr_xyz_cal_px

    def _pad_data(self):
        coordinates = self.data["shoebox"].map_elements(self._get_coordinates)
        iobs = self.data["shoebox"].map_elements(self._get_intensity)
        pad_size = [self.max_voxels - len(coords) for coords in coordinates]
        cntroids = torch.tensor(self.centroids, dtype=torch.float32)

        dxy = [
            torch.abs(sub_tensor - centroid)
            for sub_tensor, centroid in zip(coordinates, cntroids)
        ]

        self.data = self.data.with_columns(pl.Series("padding_size", pad_size))
        self.data = self.data.with_columns(pl.Series("coordinates", coordinates))
        self.data = self.data.with_columns(pl.Series("per_pix_i_obs", iobs))
        self.data = self.data.with_columns(pl.Series("dxy", dxy))

        padded_data = [
            torch.cat(
                (
                    torch.nn.functional.pad(
                        coor, (0, 0, 0, max(pad_size, 0)), "constant", 0
                    ),
                    torch.nn.functional.pad(
                        dist, (0, 0, 0, max(pad_size, 0)), "constant", 0
                    ),
                    torch.nn.functional.pad(
                        i_obs, (0, max(pad_size, 0)), "constant", 0
                    ).unsqueeze(-1),
                ),
                dim=1,
            )
            for pad_size, coor, i_obs, dist in zip(
                self.data["padding_size"].to_list(),
                self.data["coordinates"].to_list(),
                self.data["per_pix_i_obs"].to_list(),
                self.data["dxy"].to_list(),
            )
        ]
        masks = [
            torch.nn.functional.pad(
                torch.ones_like(i_obs, dtype=torch.bool),
                (0, max(pad_size, 0)),
                "constant",
                False,
            )
            for pad_size, i_obs in zip(
                self.data["padding_size"].to_list(),
                self.data["per_pix_i_obs"].to_list(),
            )
        ]
        return torch.stack(padded_data), torch.stack(masks)

    def _get_max_(self, tens):
        return tens.max().item()

    def _clean_data(self):
        mask = self.data["coordinates"].map_elements(self._get_max_) > 50000
        coord_mask = (
            torch.tensor(mask.to_list())
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand_as(self.padded_data)
        )
        masked_data = self.padded_data.clone()
        masked_data = masked_data * ~coord_mask
        is_zero = (masked_data == 0).all(dim=2)
        is_sample_zero = is_zero.all(dim=1)
        masked_data = masked_data[~is_sample_zero]
        return masked_data

    # Define the individual functions for extracting each attribute
    def _get_shoebox(self, refl_table):
        return np.array(refl_table["shoebox"]).reshape(-1, 1)

    def _get_bbox(self, refl_table):
        return np.array(refl_table["bbox"])

    def _get_miller_index(self, refl_table):
        return np.array(refl_table["miller_index"])

    def _get_xyz_cal_px(self, refl_table):
        return np.array(refl_table["xyzcal.px"]).astype(np.float32)

    def _get_xyz_obs_px_value(self, refl_table):
        return np.array(refl_table["xyzobs.px.value"]).astype(np.float32)

    def _get_intensity_sum_value(self, refl_table):
        return torch.tensor(refl_table["intensity.sum.value"], dtype=torch.float32)

    def _get_intensity_sum_variance(self, refl_table):
        return np.array(refl_table["intensity.sum.variance"]).astype(np.float32)

    def _get_shoebox_filenames(self):
        return sorted(glob.glob(os.path.join(self.shoebox_dir, "shoebox*")))

    def _get_coordinates(self, shoebox):
        coords = torch.tensor(shoebox.coords().as_numpy_array(), dtype=torch.float32)
        return coords

    def _get_num_coords(self, shoebox_array):
        return len(shoebox_array.coords())

    def _get_intensity(self, shoebox):
        return torch.tensor(shoebox.data.as_numpy_array().ravel().astype(np.float32))

    def _get_refl_tables(self):
        refl_tables = []
        for filename in self.shoebox_filenames:
            try:
                table = flex.reflection_table.from_file(filename)
                refl_tables.append(table)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                # Handle the error appropriately, e.g., skip the file, log the error, etc.
        return refl_tables

    def __len__(self):
        """
        Returns: number of shoeboxes
        """
        return len(self.shoeboxes)

    def __getitem__(self, idx):
        """
        Args:
            idx (): index of diffraction image in `shoebox_dir`
        Returns: ([num_reflection x max_voxe_size x features] , mask)
        """
        # returns bool mask of shoeboxes that belong to idx
        return self.padded_data[idx], self.padded_masks[idx]


class StillData(torch.utils.data.Dataset):

    """
    Attributes:
        image_files: filename of diffraction image
        prediction_files: filename of prediction file
        max_size: max shoebox dimensions
    """

    def __init__(self, image_files, prediction_files, max_size=4096):
        self.image_files = image_files
        self.prediction_files = prediction_files
        self.max_size = max_size

    def __len__(self):
        return len(self.image_files)

    def get_data_set(self, idx):
        # image_file = self.image_files[idx]
        prediction_file = self.prediction_files[idx]
        ds = rs.read_precognition(prediction_file)
        ds = (
            ds.reset_index().groupby(["X", "Y"]).first().reset_index()
        )  # Needed to remove harmonics
        return ds

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        ds = self.get_data_set(idx)
        pix = imread(image_file)
        x = ds.X.to_numpy("float32")
        y = ds.Y.to_numpy("float32")

        xy = np.column_stack((x, y))
        pxy = np.indices(pix.shape).T.reshape((-1, 2))
        tree = cKDTree(xy)
        d, pidx = tree.query(pxy)
        dxy = pxy - centroids[pidx]

        df = pd.DataFrame(
            {
                "counts": pix.flatten(),
                "dist": d,
                "x": pxy[:, 0],
                "y": pxy[:, 1],
                "dx": dxy[:, 0],
                "dy": dxy[:, 1],
                "idx": pidx,
            }
        )
        df["dist_rank"] = (
            df[["idx", "dist"]].groupby("idx").rank(method="first").astype("int") - 1
        )
        m = self.max_size
        n = len(centroids)
        df = df[df.dist_rank < m]

        idx_1, idx_2 = df.idx.to_numpy(), df.dist_rank.to_numpy()
        mask = torch.zeros((n, m), dtype=torch.bool)
        mask[idx_1, idx_2] = True

        counts = torch.zeros((n, m), dtype=torch.float32)
        counts[idx_1, idx_2] = torch.tensor(df.counts.to_numpy("float32"))

        xy = torch.zeros((n, m, 2), dtype=torch.float32)
        xy[idx_1, idx_2, 0] = torch.tensor(df.x.to_numpy("float32"))
        xy[idx_1, idx_2, 1] = torch.tensor(df.y.to_numpy("float32"))

        dxy = torch.zeros((n, m, 2), dtype=torch.float32)
        dxy[idx_1, idx_2, 0] = torch.tensor(df.dx.to_numpy("float32"))
        dxy[idx_1, idx_2, 1] = torch.tensor(df.dy.to_numpy("float32"))

        # Standardized
        idx = torch.clone(xy)
        xy = self.standardize(xy)
        dxy = self.standardize(dxy)

        return idx, xy, dxy, counts, mask

    @staticmethod
    def standardize(x, center=True):
        d = x.shape[-1]
        if center:
            mu = x.reshape((-1, d)).mean(0)
        else:
            mu = 0.0
        return (x - mu) / sigma


# class RotationData(torch.utils.data.Dataset):

# """

# Attributes:
# shoebox_dir: Path to shoebox.refl files
# shoebox_filenames: filenames of .refl files in `shoebox_dir`
# """

# def __init__(
# self,
# # image_dir,
# max_size=4096,
# shoebox_dir=None,
# ):
# # self.image_dir = image_dir
# self.max_size = max_size
# self.shoebox_dir = shoebox_dir  # directory of shoeboxes
# self.shoebox_filenames = self._get_shoebox_filenames()
# self.refl_tables = self._get_refl_tables()
# self.data = self._get_rotation_data()

# def _get_rotation_data(self):
# # Creating data arrays
# arr_bbox = np.array([])
# arr_shoebox = np.array([])
# arr_miller_index = np.array([])
# arr_xyz_cal_px = np.array([])
# arr_xyz_obs_px_value = np.array([])
# arr_intensity_sum_value = np.array([])
# arr_intensity_sum_variance = np.array([])

# for s in self.refl_tables:
# arr_bbox = np.append(arr_bbox, np.array(s["bbox"]))
# arr_shoebox = np.append(arr_shoebox, np.array(s["shoebox"]))
# arr_miller_index = np.append(arr_miller_index, np.array(s["miller_index"]))
# arr_xyz_cal_px = np.append(arr_xyz_cal_px, np.array(s["xyzcal.px"]))
# arr_xyz_obs_px_value = np.append(
# arr_xyz_obs_px_value, np.array(s["xyzobs.px.value"])
# )
# arr_intensity_sum_value = np.append(
# arr_intensity_sum_value, np.array(s["intensity.sum.value"])
# )
# arr_intensity_sum_variance = np.append(
# arr_intensity_sum_variance, np.array(s["intensity.sum.variance"])
# )

# # reshaping for DataFrame
# num_refls = len(arr_shoebox)
# arr_miller_index = arr_miller_index.reshape(num_refls, 3)
# arr_xyz_cal_px = arr_xyz_cal_px.reshape(num_refls, 3)
# arr_bbox = arr_bbox.reshape(num_refls, 6)
# belongs_to_image_idx = np.floor(arr_xyz_cal_px[:, -1])
# belongs_to_image_idx[belongs_to_image_idx < 0] = 0

# # heights of shoeboxes
# z_height = arr_bbox[:, -1] - arr_bbox[:, -2]  # z_max - z_min

# data_df = pd.DataFrame(
# columns=[
# "h",
# "k",
# "l",
# "xcal.px",
# "ycal.px",
# "zcal.px",
# "z_height",
# "belongs_to",
# "shoebox",
# ],
# data=np.column_stack(
# [
# arr_miller_index,
# arr_xyz_cal_px,
# z_height,
# belongs_to_image_idx,
# arr_shoebox,
# ]
# ),
# )

# return data_df

# def _get_shoebox_filenames(self):
# return sorted(glob.glob(os.path.join(self.shoebox_dir, "shoebox*")))

# def _get_refl_tables(self):
# refl_tables = []
# for filename in self.shoebox_filenames:
# try:
# table = flex.reflection_table.from_file(filename)
# refl_tables.append(table)
# except Exception as e:
# print(f"Error loading {filename}: {e}")
# # Handle the error appropriately, e.g., skip the file, log the error, etc.
# return refl_tables

# def __len__(self):
# """
# Returns: number of diffraction images
# """
# return int(self.data["belongs_to"].max())

# def __getitem__(self, idx):
# """
# Args:
# idx (): index of diffraction image in `shoebox_dir`
# Returns: ([num_reflection x max_voxe_size x features] , mask)
# """
# # returns bool mask of shoeboxes that belong to idx
# mask = (self.data["belongs_to"].astype(int) == idx).to_numpy()

# # gets all shoeboxes of idx
# x = (self.data["xcal.px"][mask]).to_numpy()
# y = (self.data["ycal.px"][mask]).to_numpy()
# z = (self.data["zcal.px"][mask]).to_numpy()

# # centroids and shoeboxes
# pxyz = np.array([x, y, z]).T
# shoeboxes_ = self.data["shoebox"][mask]

# max_pixels = max([s.data.as_numpy_array().size for s in shoeboxes_])

# # Initialize tensors for each data type
# n = len(shoeboxes_)  # Adjust the size as per your data
# padded_data = []
# masks = []

# for i, s in enumerate(shoeboxes_):
# # Convert coords to float32 numpy array
# coords_array = s.coords().as_numpy_array().astype(np.float32)
# coords = torch.tensor(coords_array)

# # Convert pxyz to float32 and then to tensor
# pxyz_tensor = torch.tensor(np.array(pxyz[i], dtype=np.float32))
# dxyz = coords - pxyz_tensor  # d(voxel, centroid)

# # Convert I_obs to float32 numpy array
# I_obs_array = s.data.as_numpy_array().ravel().astype(np.float32)
# I_obs = torch.tensor(I_obs_array)

# # Determine the amount of padding needed
# num_pixels = I_obs.shape[0]
# padding_size = max_pixels - num_pixels

# # Pad I_obs (Intensity values) and the corresponding coordinates
# I_obs_padded = torch.nn.functional.pad(
# I_obs, (0, padding_size), "constant", 0
# )
# coords_padded = torch.nn.functional.pad(
# coords, (0, 0, 0, padding_size), "constant", 0
# )
# dxyz_padded = torch.nn.functional.pad(
# dxyz, (0, 0, 0, padding_size), "constant", 0
# )

# # Concatenate padded data and add to list
# padded_data.append(
# torch.cat(
# (coords_padded, dxyz_padded, I_obs_padded.unsqueeze(1)), dim=1
# )
# )

# # Create mask for this shoebox (1 for real data, 0 for padding)
# mask = torch.ones(num_pixels, dtype=torch.bool)
# mask_padded = torch.nn.functional.pad(mask, (0, padding_size), "constant", False)

# masks.append(mask_padded)

# return torch.stack(padded_data), torch.stack(masks)
