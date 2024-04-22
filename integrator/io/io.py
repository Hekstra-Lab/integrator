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


class RotationData(torch.utils.data.Dataset):
    def __init__(
        self,
        max_size=4096,
        shoebox_dir=None,
        max_detector_dimension=5000,
        seed=60,
    ):
        """
        Dataset class for X-ray diffraction rotation data.

        Args:
            shoebox_dir (str): Directory containing the shoebox files.
            max_detector_dimension (int): Maximum detector dimension.
            train_split (float): Fraction of data to use for training.
            val_split (float): Fraction of data to use for validation.
            test_split (float): Fraction of data to use for testing.
            seed (int): Random seed for data splitting.
        """
        self.max_detector_dimension = max_detector_dimension
        self.seed = seed
        self.mode = "train"
        self.max_size = max_size
        self.shoebox_dir = shoebox_dir
        self.shoebox_filenames = self._get_shoebox_filenames()
        (
            self.df,
            self.max_voxels,
        ) = self._get_refl_tables()

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
        return (intensities < 0).all()

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

    def _get_refl_tables(self):
        """
        Build a dataframe of the reflection tables

        Returns:
            tuple: Train DataFrame, validation DataFrame, test DataFrame, and maximum number of voxels.
        """
        final_df = pl.DataFrame()

        max_vox = []
        for filename in self.shoebox_filenames["shoebox_filenames"]:
            tbl = self._get_table(filename)  # refl table
            df = self._get_rows(tbl)  # store refl table as dataframe

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
            df = df.with_columns([pl.Series("dead_pixel_mask", dead_pixel_mask)])

            # get num pixels
            num_pixel = df["intensity_observed"].map_elements(self._get_num_pixels)
            max_coord = df["coordinates"].map_elements(self._max_pixel_coordinate)
            df = df.with_columns(
                [pl.Series("max_coord", max_coord), pl.Series("num_pix", num_pixel)]
            )
            dead_pixels = df["intensity_observed"].map_elements(
                self._filter_dead_shoeboxes
            )
            df = df.with_columns([pl.Series("all_pixels_dead", dead_pixels)])
            df = df.filter(pl.col("max_coord") < 5000)  # returns greater than 5000
            df = df.filter(pl.col("all_pixels_dead") == 0)

            max_vox.append(df.select(pl.col("num_pix").max()).item())
            # max_vox = max(max_vox).item()

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

        I_prf_val = self.df["intensity.prf.value"].gather(idx).item()
        I_prf_var = self.df["intensity.prf.variance"].gather(idx).item()
        I_sum_val = self.df["intensity.sum.value"].gather(idx).item()
        I_sum_var = self.df["intensity.sum.variance"].gather(idx).item()
        DIALS_I = (I_prf_val, I_prf_var, I_sum_val, I_sum_var)
        coords = self.df["coordinates"].gather(idx).item()

        z_coords = coords[:, 2]  # Extract the z-coordinates
        z_min = z_coords.min()
        z_max = z_coords.max()

        dxy = self.df["dxy"].gather(idx).item()
        num_pix = self.df["num_pix"].gather(idx).item()
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
        ).unsqueeze(dim=-1)

        padded_data = torch.cat((pad_coords, pad_dxy, pad_iobs), dim=1)

        # mask = torch.nn.functional.pad(
        # torch.ones_like(i_obs, dtype=torch.bool),
        # (0, max(pad_size, 0)),
        # "constant",
        # False,
        # )
        return padded_data, dead_pixel_mask_padded, DIALS_I


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
