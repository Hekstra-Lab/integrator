from pylab import *
import polars as pl
import torch
from scipy.spatial import cKDTree
import glob
import os
from dials.array_family import flex
import numpy as np
import pytorch_lightning
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset


class RotationData(torch.utils.data.Dataset):
    def __init__(
        self,
        max_size=4096,
        shoebox_dir=None,
        max_detector_dimension=5000,
        weak_reflection_threshold=5,
    ):
        """
        Dataset class for X-ray diffraction data collected by rotation method

        Args:
            shoebox_dir (str): Directory containing the shoebox files.
            max_detector_dimension (int): Maximum detector dimension.
            train_split (float): Fraction of data to use for training.
            val_split (float): Fraction of data to use for validation.
            test_split (float): Fraction of data to use for testing.
        """
        self.refl_tables = []
        self.max_detector_dimension = max_detector_dimension
        self.mode = "train"
        self.max_size = max_size
        self.shoebox_dir = shoebox_dir
        self.shoebox_filenames = self._get_shoebox_filenames()
        (
            self.df,
            self.max_voxels,
        ) = self._get_refl_tables(weak_reflection_threshold=weak_reflection_threshold)

    def _get_shoebox_filenames(self):
        """
        Get the `shoebox_*.refl` names

        Returns:
            pl.DataFrame: DataFrame containing the shoebox filenames.
        """
        df = pl.DataFrame(
            {"shoebox_filenames": glob.glob(os.path.join(self.shoebox_dir, "shoebox*"))}
        )
        return df

    def _get_table(self, filename):
        """
        Get reflection table
        """
        return flex.reflection_table.from_file(filename)

    def _to_tens(self, element):
        """
        Convert the element to a torch tensor

        Args:
            element: element to convert to tensor

        Returns:
            torch.Tensor
        """
        return torch.tensor(element, dtype=torch.float32, requires_grad=False)

    def _get_rows(self, tbl):
        """
        Drop unused columns

        Args:
            tbl (flex.reflection_table): Reflection table

        Returns:
            pl.DataFrame: DataFrame with unused columns dropped
        """
        return pl.DataFrame(list(tbl.rows())).drop(
            [
                # "background.mean",
                # "background.sum.value",
                # "background.sum.variance",
                # "refl_ids",
                # 'tbl_id',
                "bbox",
                "partiality",
                "d",
                "num_pixels.background",
                "partial_id",
                "panel",
                "s1",
                "xyzobs.mm.value",
                "xyzcal.mm",
                # "xyzcal.px",
                "xyzcal.mm",
                "zeta",
                "num_pixels.foreground",
                "flags",
                "id",
                "entering",
                "imageset_id",
                # "intensity.sum.variance",
                # "intensity.prf.variance",
                "num_pixels.background_used",
                "num_pixels.valid",
                "xyzobs.mm.variance",
                "profile.correlation",
                # "intensity.prf.value",
                # "intensity.sum.value",
            ]
        )

    def _max_pixel_coordinate(self, coords):
        """
        Find the maximum coordinate value for each entry

        Args:
            coords (torch.Tensor): Coordinate tensor

        Returns:
            float: Maximum coordinate value
        """
        return coords.max().item()

    def _get_refl_tables(self, weak_reflection_threshold=5):
        """
        Build a dataframe of the reflection tables

        Returns:
            tuple: Train DataFrame, validation DataFrame, test DataFrame, and maximum number of voxels.
        """
        final_df = pl.DataFrame()

        max_vox = []
        for idx, filename in enumerate(self.shoebox_filenames["shoebox_filenames"]):
            # reflection table
            tbl = self._get_table(filename)

            # reflection table as polars DataFrame
            df = self._get_rows(tbl)

            # Get Intensity_observed
            iobs = [
                sbox.data.as_numpy_array().ravel().astype(np.float32).tolist()
                for sbox in df["shoebox"]
            ]

            # Get detector coordinates
            coordinates = [
                torch.tensor(sbox.coords().as_numpy_array(), dtype=torch.float32)
                for sbox in df["shoebox"]
            ]

            # distance from pixel to centroid
            dxy = [
                torch.abs(sub_tensor - centroid)
                for sub_tensor, centroid in zip(
                    coordinates, df["xyzcal.px"].map_elements(self._to_tens)
                )
            ]

            # Add columns to DataFrame
            df = df.with_columns(
                [
                    pl.Series("x", [x[:, 0].tolist() for x in coordinates]),
                    pl.Series("y", [x[:, 1].tolist() for x in coordinates]),
                    pl.Series("z", [x[:, 2].tolist() for x in coordinates]),
                    pl.Series("coordinates", coordinates),
                    pl.Series("intensity_observed", iobs),
                    pl.Series("dx", [x[:, 0].tolist() for x in dxy]),
                    pl.Series("dy", [x[:, 1].tolist() for x in dxy]),
                    pl.Series("dz", [x[:, 2].tolist() for x in dxy]),
                ]
            )

            # Mask for dead pixels
            dead_pixel_mask = df["intensity_observed"].list.eval(pl.element().ge(0))

            df = df.with_columns([pl.Series("dead_pixel_mask", dead_pixel_mask)])

            # Number of voxels
            num_pixel = df["intensity_observed"].list.len()

            max_coord = df["coordinates"].map_elements(self._max_pixel_coordinate)

            weak_shoeboxes = (
                df["intensity_observed"].list.max() < weak_reflection_threshold
            )

            df = df.with_columns(
                [pl.Series("max_coord", max_coord), pl.Series("num_pix", num_pixel)]
            )

            dead_pixels = df["intensity_observed"].list.min() < 0

            df = df.with_columns(pl.Series("weak_reflection_threshold", weak_shoeboxes))

            df = df.with_columns([pl.Series("all_pixels_dead", dead_pixels)])

            coord_mask = np.array(df["max_coord"] < 5000)

            mask = (coord_mask * (~np.array(dead_pixels).astype(bool))).astype(bool)

            df = df.filter(pl.col("max_coord") < 5000)

            df = df.filter(df["all_pixels_dead"] == False)

            max_vox.append(df.select(pl.col("num_pix").max()).item())

            # Generate ids to identify reflections
            refls = tbl.select(flex.bool(mask))

            refls["refl_ids"] = flex.int(np.arange(len(refls)))

            refls["tbl_id"] = flex.int(np.zeros(len(refls)) + idx)

            self.refl_tables.append(refls)

            # Add ids to DataFrame
            df = df.with_columns(
                [
                    pl.Series("refl_ids", refls["refl_ids"]),
                    pl.Series("tbl_id", refls["tbl_id"]),
                ]
            )

            # stack dataframe
            final_df = final_df.vstack(df) if final_df is not None else df

        # Drop shoebox and coordinates columns
        final_df = final_df.drop(["shoebox", "coordinates"])

        # Number of voxels in largest shoebox
        max_voxel = max(max_vox)

        return final_df, max_voxel

    def __len__(self):
        """
        Return the length (number of shoeboxes) of the dataset

        Returns:
            int: Length of the dataset
        """
        return self.df.height

    def __getitem__(self, idx):
        """
        Return the (idx)th shoebox

        Args:
            idx (int): Index of the shoebox

        Returns:
            tuple: Padded data tensor and mask tensor
        """

        DIALS_I_prf_val = self.df["intensity.prf.value"].gather(idx).item()

        DIALS_I_prf_var = self.df["intensity.prf.variance"].gather(idx).item()

        DIALS_I_sum_val = self.df["intensity.sum.value"].gather(idx).item()

        DIALS_I_sum_var = self.df["intensity.sum.variance"].gather(idx).item()

        # bg_sum_val = self.df["background.sum.value"].gather(idx).item()

        # bg_sum_var = self.df["background.sum.variance"].gather(idx).item()

        # bg_sum_mean = self.df["background.mean"].gather(idx).item()

        # shoebox coordinates
        coords = torch.tensor(
            [
                self.df["x"].gather(idx).item(),
                self.df["y"].gather(idx).item(),
                self.df["z"].gather(idx).item(),
            ]
        ).transpose(0, 1)

        # shape of the shoebox
        x_shape = len(coords[0].unique())
        y_shape = len(coords[1].unique())
        z_shape = len(coords[2].unique())

        shape = (x_shape, y_shape, z_shape)

        if z_shape == 1:
            is_flat = torch.tensor(True)
        else:
            is_flat = torch.tensor(False)

        # reflection id
        id = self.df["refl_ids"].gather(idx).item()

        # reflection table id
        tbl_id = self.df["tbl_id"].gather(idx).item()

        # distance from pixel to centroid
        dxy = (
            torch.tensor(
                [
                    self.df["dx"].gather(idx).to_list(),
                    self.df["dy"].gather(idx).to_list(),
                    self.df["dz"].gather(idx).to_list(),
                ]
            )
            .squeeze(1)
            .transpose(0, 1)
        )

        # padding size
        pad_size = self.max_voxels - len(coords)

        # observed intensity
        i_obs = torch.tensor(
            [self.df["intensity_observed"].gather(idx).item()]
        ).squeeze(0)

        # dead pixel mask
        dead_pixel_mask = torch.tensor(self.df["dead_pixel_mask"].gather(idx).item())

        # padded coordinates
        pad_coords = torch.nn.functional.pad(
            coords, (0, 0, 0, max(pad_size, 0)), "constant", 0
        )

        # padded offsets
        pad_dxy = torch.nn.functional.pad(
            dxy, (0, 0, 0, max(pad_size, 0)), "constant", 0
        )

        # padded observed intensity
        pad_iobs = torch.nn.functional.pad(
            i_obs, (0, max(pad_size, 0)), "constant", 0
        ).unsqueeze(-1)

        # padded dead pixel mask
        dead_pixel_mask_padded = torch.nn.functional.pad(
            dead_pixel_mask, (0, max(pad_size, 0)), "constant", 0
        )

        shoebox = torch.cat((pad_coords, pad_dxy, pad_iobs), dim=1)

        return (
            shoebox,
            dead_pixel_mask_padded,
            DIALS_I_prf_val,
            DIALS_I_prf_var,
            DIALS_I_sum_val,
            DIALS_I_sum_var,
            is_flat,
            id,
            tbl_id,
            torch.tensor(shape),
        )


class RotationDataModule(pytorch_lightning.LightningDataModule):
    def __init__(
        self,
        shoebox_dir,
        batch_size=32,
        num_workers=4,
        train_val_split=0.8,
        subset_ratio=1,
    ):
        super().__init__()
        self.shoebox_dir = shoebox_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.subset_ratio = subset_ratio

    def setup(self, stage=None):
        full_dataset = RotationData(shoebox_dir=self.shoebox_dir)
        self.full_dataset = full_dataset

        # create subset of data
        subset_size = int(self.subset_ratio * len(full_dataset))
        indices = torch.randperm(len(full_dataset)).tolist()
        subset_indices = indices[:subset_size]
        subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)

        # split data into training and validation sets
        train_size = int(self.train_val_split * subset_size)
        val_size = subset_size - train_size
        if train_size + val_size != subset_size:
            val_size = (
                subset_size - train_size
            )  # Adjust to ensure the total size matches

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            subset_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# Data modules for simulated data


class SimulatedData(torch.utils.data.Dataset):
    def __init__(self, data, max_voxels):
        self.data = data
        self.max_voxels = max_voxels
        self.true_I = data["true_I"]
        self.true_bg = data["true_bg"]
        self.true_rate = data["true_Rij"]
        self.dxy = data["dxy"]
        self.coords = data["coords"]
        self.true_counts = data["true_counts"]
        self.true_cov = data["true_cov"]
        self.x_shape = data["x_shape"]
        self.y_shape = data["y_shape"]
        self.z_shape = data["z_shape"]

    def __len__(self):
        return len(self.data["true_bg"])

    def __getitem__(self, idx):
        shoebox = torch.hstack(
            [self.coords[idx], self.dxy[idx], self.true_counts[idx].unsqueeze(-1)]
        ).to(torch.float32)
        shape = (self.x_shape[idx], self.y_shape[idx], self.z_shape[idx])
        num_voxels = shape[0] * shape[1] * shape[2]
        pad_size = self.max_voxels - num_voxels

        padded_shoebox = torch.nn.functional.pad(
            shoebox, (0, 0, 0, max(pad_size, 0)), "constant", 0
        )

        # Uncomment the following if you want to return the true_counts and the true per-pixel rate (Rij)
        # true_counts = torch.nn.functional.pad(
        # self.true_counts[idx], (0, max(pad_size, 0)), "constant", 0
        # )
        # true_Rij = torch.nn.functional.pad(
        # self.true_rate[idx], (0, max(pad_size, 0)), "constant", 0
        # )

        pad_mask = torch.nn.functional.pad(
            torch.ones_like(shoebox[:, -1], dtype=torch.bool),
            (0, max(pad_size, 0)),
            "constant",
            False,
        )

        if shape[-1] == 1:
            is_flat = torch.tensor(True)
            cov = torch.zeros(3, 3)
            if self.true_cov[idx].size(0) == 2:
                cov[:2, :2] = self.true_cov[idx]
            else:
                cov = self.true_cov[idx]
        else:
            is_flat = torch.tensor(False)
            cov = torch.zeros(3, 3)
            if self.true_cov[idx].size(0) == 2:
                cov[:2, :2] = self.true_cov[idx]
            else:
                cov = self.true_cov[idx]

        return (
            padded_shoebox,
            pad_mask,
            self.true_I[idx],
            cov,
            self.true_bg[idx],
            is_flat,
            # true_counts,
            # true_Rij,
            torch.tensor(shape),
        )


class SimulatedDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, data_df, max_voxel, batch_size=50, subset_ratio=0.1):
        super().__init__()
        self.data_df = data_df
        self.max_voxel = max_voxel
        self.batch_size = batch_size
        self.subset_ratio = subset_ratio

    def setup(self, stage=None):
        sim_data = SimulatedData(self.data_df, self.max_voxel)
        subset_size = int(len(sim_data) * self.subset_ratio)
        subset_data = torch.utils.data.Subset(sim_data, list(range(subset_size)))
        train_size = int(0.8 * subset_size)
        val_size = subset_size - train_size
        self.train_data, self.val_data = torch.utils.data.random_split(
            subset_data, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=3,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=3,
        )


import torch
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset
import pytorch_lightning as pl


# %%
# class ShoeboxDataModule(pl.LightningDataModule):
# def __init__(
# self,
# shoebox_data,
# metadata,
# is_flat,
# dead_pixel_mask,
# batch_size=32,
# val_split=0.2,
# test_split=0.1,
# include_test=False,
# subset_size=None,
# ):
# super().__init__()
# self.shoebox_data = shoebox_data
# self.metadata = metadata
# self.is_flat = is_flat
# self.dead_pixel_mask = dead_pixel_mask
# self.batch_size = batch_size
# self.val_split = val_split
# self.test_split = test_split
# self.include_test = include_test
# self.subset_size = subset_size

# def setup(self, stage=None):
# # Load the tensors
# shoeboxes = torch.load(self.shoebox_data)
# metadata = torch.load(self.metadata)
# is_flat = self.is_flat
# dead_pixel_mask = torch.load(self.dead_pixel_mask)

# # Process the shoeboxes
# # Assuming shoeboxes shape is (n_samples, 3*21*21, 7)
# n_samples, _, feature_dim = shoeboxes.shape

# # Reshape to (n_samples, 3, 21, 21, 7)
# shoeboxes_reshaped = shoeboxes.view(n_samples, 3, 21, 21, feature_dim)

# # Flatten images across z-dimension
# intensity_flattened = shoeboxes_reshaped[..., -1].sum(
# dim=1
# )  # Shape: (n_samples, 21, 21)

# # Mean of other features
# features_mean = shoeboxes_reshaped[..., :-1].mean(
# dim=1
# )  # Shape: (n_samples, 21, 21, 6)

# # Combine intensity and other features
# processed_shoeboxes = torch.cat(
# (features_mean, intensity_flattened.unsqueeze(-1)), dim=-1
# )
# # Flatten to (n_samples, 21*21, 7)
# processed_shoeboxes_flattened = processed_shoeboxes.view(
# n_samples, 21 * 21, feature_dim
# )

# # Process the mask
# # Reshape the mask from (n_samples, 3*21*21) to (n_samples, 3, 21, 21)
# dead_pixel_mask_reshaped = dead_pixel_mask.view(n_samples, 3, 21, 21)

# # Multiply the layers to get (n_samples, 21, 21)
# processed_mask = dead_pixel_mask_reshaped.prod(
# dim=1
# )  # Shape: (n_samples, 21, 21)

# processed_mask_flattened = processed_mask.view(n_samples, 21 * 21)

# valid_indices = ((processed_mask_flattened == 1).sum(-1) != 0).nonzero(
# as_tuple=True
# )[0]
# processed_shoeboxes_flattened = processed_shoeboxes_flattened[valid_indices]
# metadata = metadata[valid_indices]
# is_flat = is_flat[valid_indices]
# processed_mask_flattened = processed_mask_flattened[valid_indices]

# # Create the full dataset
# full_dataset = TensorDataset(
# processed_shoeboxes_flattened, metadata, is_flat, processed_mask_flattened
# )

# self.full_dataset = full_dataset

# # Optionally, create a subset of the dataset
# if self.subset_size is not None and self.subset_size < len(full_dataset):
# indices = torch.randperm(len(full_dataset))[: self.subset_size]
# full_dataset = Subset(full_dataset, indices)

# # Calculate lengths for train/val/test splits
# total_size = len(full_dataset)
# val_size = int(total_size * self.val_split)
# if self.include_test:
# test_size = int(total_size * self.test_split)
# train_size = total_size - val_size - test_size
# else:
# test_size = 0
# train_size = total_size - val_size

# # Split the dataset
# if self.include_test:
# self.train_dataset, self.val_dataset, self.test_dataset = random_split(
# full_dataset, [train_size, val_size, test_size]
# )
# else:
# self.train_dataset, self.val_dataset = random_split(
# full_dataset, [train_size, val_size]
# )
# self.test_dataset = None

# def train_dataloader(self):
# return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

# def val_dataloader(self):
# return DataLoader(self.val_dataset, batch_size=self.batch_size)

# def test_dataloader(self):
# if self.include_test:
# return DataLoader(self.test_dataset, batch_size=self.batch_size)
# else:
# return None


# %%


class ShoeboxDataModule(pytorch_lightning.LightningDataModule):
    def __init__(
        self,
        shoebox_data,
        metadata,
        dead_pixel_mask,
        batch_size=32,
        val_split=0.2,
        test_split=0.1,
        num_workers = 4,
        include_test=False,
        subset_size=None,
        single_sample_index=None,  # Add parameter for single sample index
    ):
        super().__init__()
        self.shoebox_data = shoebox_data
        self.metadata = metadata
        self.dead_pixel_mask = dead_pixel_mask
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.include_test = include_test
        self.subset_size = subset_size
        self.single_sample_index = single_sample_index  # Store single sample index
        self.num_workers = num_workers,

    def setup(self, stage=None):
        # Load the tensors
        shoeboxes = torch.load(self.shoebox_data)
        metadata = torch.load(self.metadata)
        dead_pixel_mask = torch.load(self.dead_pixel_mask)

        # Create the full dataset
        full_dataset = TensorDataset(shoeboxes, metadata, dead_pixel_mask)

        # If single_sample_index is specified, use only that sample
        if self.single_sample_index is not None:
            full_dataset = Subset(full_dataset, [self.single_sample_index])

        # Optionally, create a subset of the dataset
        if self.subset_size is not None and self.subset_size < len(full_dataset):
            indices = torch.randperm(len(full_dataset))[: self.subset_size]
            full_dataset = Subset(full_dataset, indices)

        # Calculate lengths for train/val/test splits
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        if self.include_test:
            test_size = int(total_size * self.test_split)
            train_size = total_size - val_size - test_size
        else:
            test_size = 0
            train_size = total_size - val_size

        # Split the dataset
        if self.include_test:
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset, [train_size, val_size, test_size]
            )
        else:
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = 3)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,shuffle=False,num_workers=3)

    def test_dataloader(self):
        if self.include_test:
            return DataLoader(self.test_dataset, batch_size=self.batch_size)
        else:
            return None


# %%
