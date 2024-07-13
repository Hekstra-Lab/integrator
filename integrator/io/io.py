from pylab import *
import polars as pl
import torch
from scipy.spatial import cKDTree
import glob
import os
from dials.array_family import flex
import numpy as np
import pytorch_lightning
from torch.utils.data import DataLoader


class RotationData(torch.utils.data.Dataset):
    def __init__(
        self,
        max_size=4096,
        shoebox_dir=None,
        max_detector_dimension=5000,
        weak_reflection_threshold=5,
    ):
        """
        Dataset class for X-ray diffraction rotation data.

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
        Get the reflection table

        Args:
            filename (str): Path to the .refl file

        Returns:
            Reflection table
        """
        return flex.reflection_table.from_file(filename)

    def _filter_very_weak_shoeboxes(self, intensities, weak_reflection_threshold=5):
        # return(intensities < 0).any()
        return intensities.max().item() <= weak_reflection_threshold

    def _get_intensity(self, sbox):
        """
        Get the observed intensity from shoebox object

        Args:
            sbox (flex.shoebox): shoebox object

        Returns:
            torch.Tensor: Observed intensity as flattened tensor
        """
        return torch.tensor(
            sbox.data.as_numpy_array().ravel().astype(np.float32),
            dtype=torch.float32,
            requires_grad=False,
        )

    def _to_tens(self, element):
        """
        Convert the element to a torch tensor

        Args:
            element: element to convert to tensor

        Returns:
            torch.Tensor
        """
        return torch.tensor(element, dtype=torch.float32, requires_grad=False)

    def _get_num_pixels(self, intensities):
        """
        Count the number of voxels in each shoebox

        Args:
            intensities (torch.Tensor): Observed intensity values of the shoebox

        Returns:
            int: Number of voxels in the shoebox
        """
        return intensities.numel()

    def _get_z_dims(self, coords):
        return len(coords[:, -1].unique())

    def _get_max_(self, tens):
        """
        Get the maximum value of the tensor

        Args:
            tens (torch.Tensor): Input tensor

        Returns:
            float: Maximum value of the tensor
        """
        return tens.max().item()

    def _filter_dead_shoeboxes(self, intensities):
        return (intensities < 0).any()

    def _get_rows(self, tbl, idx):
        """
        Drop unused columns

        Args:
            tbl (flex.reflection_table): Reflection table

        Returns:
            pl.DataFrame: DataFrame with unused columns dropped
        """

        tbl["refl_ids"] = flex.int(np.arange(len(tbl)))
        tbl["tbl_id"] = flex.int(np.zeros(len(tbl)) + idx)

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
                "xyz.px.variance",
                "xyzcal.px.value",
                "xyzcal.px.variance",
                "zeta",
                "num_pixels.foreground",
                "flags",
                "id",
                "entering",
                "imageset_id",
                # "intensity.sum.variance",
                # "intensity.prf.variance",
                "num_pixels.background_used",
                "num_pixel.foreground",
                "num_pixels.valid",
                "xyzobs.mm.variance",
                "profile.correlation",
                # "intensity.prf.value",
                # "intensity.sum.value",
            ]
        )

    def _get_coords(self, sbox):
        """
        For a shoebox, get the coordinates of each voxel

        Args:
            sbox (flex.shoebox): Shoebox object.

        Returns:
            torch.Tensor: Coordinates of each voxel as a tensor.
        """
        return torch.tensor(sbox.coords().as_numpy_array(), dtype=torch.float32)

    def _max_pixel_coordinate(self, coords):
        """
        Find the maximum coordinate value for each entry

        Args:
            coords (torch.Tensor): Coordinate tensor

        Returns:
            float: Maximum coordinate value
        """
        return coords.max().item()

    def _mask_dead_pixels(self, coords):
        return ~(coords < 0)

    def _filter_shoebox(self, max_pix):
        """
        Remove shoeboxes with coordinates outside of the detector dimensions

        Args:
            max_pix (float): Maximum pixel coordinate

        Returns:
            bool: True if the shoebox should be removed, False otherwise
        """
        return max_pix > self.max_detector_dimension

    def _get_refl_tables(self, weak_reflection_threshold=5):
        """
        Build a dataframe of the reflection tables

        Returns:
            tuple: Train DataFrame, validation DataFrame, test DataFrame, and maximum number of voxels.
        """
        final_df = pl.DataFrame()

        max_vox = []
        for idx, filename in enumerate(self.shoebox_filenames["shoebox_filenames"]):
            tbl = self._get_table(filename)  # refl table
            df = self._get_rows(tbl, idx)  # store refl table as dataframe

            # getting coordinates and observed intensity from shoeboxes
            coordinates = df["shoebox"].map_elements(self._get_coords)
            iobs = df["shoebox"].map_elements(self._get_intensity)

            # distance from pixel to centroid
            dxy = [
                torch.abs(sub_tensor - centroid)
                for sub_tensor, centroid in zip(
                    coordinates, df["xyzcal.px"].map_elements(self._to_tens)
                )
            ]
            df = df.with_columns(
                [
                    pl.Series("coordinates", coordinates),
                    pl.Series("intensity_observed", iobs),
                    pl.Series("dxy", dxy),
                ]
            )

            # masks for dead pixels
            dead_pixel_mask = df["intensity_observed"].map_elements(
                self._mask_dead_pixels
            )

            dead_pixel_mask = dead_pixel_mask.to_numpy()

            df = df.with_columns([pl.Series("dead_pixel_mask", dead_pixel_mask)])

            # get num pixels
            num_pixel = df["intensity_observed"].map_elements(self._get_num_pixels)

            max_coord = df["coordinates"].map_elements(self._max_pixel_coordinate)

            # weak_shoeboxes = df["intensity_observed"].map_elements(
            # self._filter_very_weak_shoeboxes
            # )

            weak_shoeboxes = df["intensity_observed"].map_elements(
                lambda x: self._filter_very_weak_shoeboxes(x, weak_reflection_threshold)
            )

            df = df.with_columns(
                [pl.Series("max_coord", max_coord), pl.Series("num_pix", num_pixel)]
            )

            dead_pixels = df["intensity_observed"].map_elements(
                self._filter_dead_shoeboxes
            )

            df = df.with_columns(pl.Series("weak_reflection_threshold", weak_shoeboxes))

            df = df.with_columns([pl.Series("all_pixels_dead", dead_pixels)])

            coord_mask = np.array(df["max_coord"] < 5000)

            mask = (coord_mask * (~np.array(dead_pixels).astype(bool))).astype(bool)

            df = df.filter(pl.col("max_coord") < 5000)  # returns greater than 5000

            df = df.filter(pl.col("all_pixels_dead") == 0)

            max_vox.append(df.select(pl.col("num_pix").max()).item())
            # max_vox = max(max_vox).item()

            refls = tbl.select(flex.bool(mask))
            self.refl_tables.append(refls)

            # stack dataframe
            final_df = final_df.vstack(df) if final_df is not None else df
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
        bg_sum_val = self.df["background.sum.value"].gather(idx).item()
        bg_sum_var = self.df["background.sum.variance"].gather(idx).item()
        bg_sum_mean = self.df["background.mean"].gather(idx).item()
        #        DIALS_I = (I_prf_val, I_prf_var, I_sum_val, I_sum_var)
        #        DIALS_bg = (bg_sum_val, bg_sum_var, bg_sum_mean)
        row_idx = idx
        coords = self.df["coordinates"].gather(idx).item()

        x_shape = len(coords[:, 0].unique())
        y_shape = len(coords[:, 1].unique())
        z_shape = len(coords[:, 2].unique())
        shape = (x_shape, y_shape, z_shape)

        z_coords = len(coords[:, 2].unique())  # Extract the z-coordinates
        if z_coords == 1:
            is_flat = torch.tensor(True)
        else:
            is_flat = torch.tensor(False)

        id = self.df["refl_ids"].gather(idx).item()
        tbl_id = self.df["tbl_id"].gather(idx).item()
        dxy = self.df["dxy"].gather(idx).item()
        pad_size = self.max_voxels - len(coords)
        i_obs = self.df["intensity_observed"].gather(idx).item()
        dead_pixel_mask = self.df["dead_pixel_mask"].gather(idx).item()

        pad_coords = torch.nn.functional.pad(
            coords, (0, 0, 0, max(pad_size, 0)), "constant", 0
        )
        pad_dxy = torch.nn.functional.pad(
            dxy, (0, 0, 0, max(pad_size, 0)), "constant", 0
        )
        pad_iobs = torch.nn.functional.pad(
            i_obs, (0, max(pad_size, 0)), "constant", 0
        ).unsqueeze(dim=-1)

        dead_pixel_mask_padded = torch.nn.functional.pad(
            dead_pixel_mask, (0, max(pad_size, 0)), "constant", 0
        )

        shoebox = torch.cat((pad_coords, pad_dxy, pad_iobs), dim=1)

        # pad_mask = torch.nn.functional.pad(
        # torch.ones_like(i_obs, dtype=torch.bool),
        # (0, max(pad_size, 0)),
        # "constant",
        # False,
        # )

        return (
            shoebox,
            dead_pixel_mask_padded,
            DIALS_I_prf_val,
            DIALS_I_prf_var,
            DIALS_I_sum_val,
            DIALS_I_sum_var,
            # idx,
            # pad_mask,
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
