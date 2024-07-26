# The `preprocess_data` function reads .refl files
# and outputs .pt files with the shoebox tensor, padded dead pixel mask, metadata tensor, and is flat tensor.
# These tensors can be read into the DataLoader using torch.load()

import polars as pl
import torch
import glob
import os
import numpy as np
from dials.array_family import flex


# %%
def preprocess_data(shoebox_dir):
    shoebox_filenames = glob.glob(os.path.join(shoebox_dir, "shoebox*"))
    final_df = pl.DataFrame()
    max_vox = []

    for idx, filename in enumerate(shoebox_filenames):
        # reflection table
        tbl = flex.reflection_table.from_file(filename)

        # reflection table as polars DataFrame
        df = pl.DataFrame(list(tbl.rows()))

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

        centroids = [torch.tensor(x) for x in df["xyzcal.px"]]

        dxy = [
            torch.abs(sub_tensor - centroid)
            for sub_tensor, centroid in zip(coordinates, centroids)
        ]

        num_vox = [len(x) for x in coordinates]

        # Add columns to DataFrame
        df = df.with_columns(
            [
                pl.Series("x", [x[:, 0].tolist() for x in coordinates]),
                pl.Series("y", [x[:, 1].tolist() for x in coordinates]),
                pl.Series("z", [x[:, 2].tolist() for x in coordinates]),
                pl.Series("coordinates", coordinates),
                pl.Series("intensity_observed", iobs),
                pl.Series("dxyz", dxy),
                pl.Series("num_vox", num_vox),
            ]
        )

        # Mask for dead pixels
        dead_pixel_mask = df["intensity_observed"].list.eval(pl.element().ge(0))
        df = df.with_columns([pl.Series("dead_pixel_mask", dead_pixel_mask)])

        # Number of voxels
        num_pixel = df["intensity_observed"].list.len()
        # max_coord = df["coordinates"].apply(lambda x: x.max().item())
        max_coord = [x.max().item() for x in df["coordinates"]]
        weak_shoeboxes = df["intensity_observed"].list.max() < 5
        dead_pixels = df["intensity_observed"].list.min() < 0

        # Calculate additional features
        x_shape = [len(torch.unique(x[:, 0])) for x in coordinates]
        y_shape = [len(torch.unique(x[:, 1])) for x in coordinates]
        z_shape = [len(torch.unique(x[:, 2])) for x in coordinates]

        shapes = list(zip(x_shape, y_shape, z_shape))

        is_flat = [torch.tensor(z == 1).item() for z in z_shape]

        # Add additional features to DataFrame
        df = df.with_columns(
            [
                pl.Series("max_coord", max_coord),
                pl.Series("num_pix", num_pixel),
                pl.Series("weak_reflection_threshold", weak_shoeboxes),
                pl.Series("all_pixels_dead", dead_pixels),
                pl.Series("x_shape", x_shape),
                pl.Series("y_shape", y_shape),
                pl.Series("z_shape", z_shape),
                pl.Series("is_flat", is_flat),
                # pl.Series("pad_size", pad_size)
            ]
        )

        coord_mask = df["max_coord"] < 5000
        mask = coord_mask & (~dead_pixels)
        # df = df.filter((df["max_coord"] < 5000))
        # df = df.filter((df["all_pixels_dead"] == False))

        max_vox.append(df.select(pl.col("num_pix").max()).item())

        # Generate ids to identify reflections
        refls = tbl
        refls["refl_ids"] = flex.int(np.arange(len(refls)))
        refls["tbl_id"] = flex.int(np.zeros(len(refls)) + idx)

        # Add ids to DataFrame
        df = df.with_columns(
            [
                pl.Series("mask_sbox", mask),
                pl.Series("refl_ids", refls["refl_ids"]),
                pl.Series("tbl_id", refls["tbl_id"]),
            ]
        )

        # stack dataframe
        final_df = final_df.vstack(df) if final_df is not None else df

    mask = final_df["mask_sbox"]
    max_vox = max(final_df["num_vox"])
    pad_size = [max_vox - x for x in final_df["num_vox"].filter(mask)]

    padded_coordinates = [
        torch.nn.functional.pad(coords, (0, 0, 0, max(pad_size, 0)), "constant", 0)
        for coords, pad_size in zip(final_df["coordinates"].filter(mask), pad_size)
    ]

    padded_dxyz = [
        torch.nn.functional.pad(dxyz, (0, 0, 0, max(pad_size, 0)), "constant", 0)
        for dxyz, pad_size in zip(final_df["dxyz"].filter(mask), pad_size)
    ]

    padded_iobs = [
        torch.nn.functional.pad(
            torch.tensor([iobs]), (0, max(pad_size, 0)), "constant", 0
        )
        for iobs, pad_size in zip(final_df["intensity_observed"].filter(mask), pad_size)
    ]

    padded_dead_pixel_mask = torch.stack(
        [
            torch.nn.functional.pad(
                torch.tensor([dead_pixel_mask]),
                (0, max(pad_size, 0)),
                "constant",
                0,
            )
            for dead_pixel_mask, pad_size in zip(
                final_df["dead_pixel_mask"].filter(mask), pad_size
            )
        ]
    ).squeeze(1)

    shoebox_tensor = torch.cat(
        (
            torch.stack(padded_coordinates),
            torch.stack(padded_dxyz),
            torch.stack(padded_iobs).permute(0, 2, 1),
        ),
        dim=2,
    )

    metadata = (
        final_df[
            [
                "intensity.sum.value",
                "intensity.sum.variance",
                "intensity.prf.value",
                "intensity.prf.variance",
                "tbl_id",
                "refl_ids",
                "x_shape",
                "y_shape",
                "z_shape",
            ]
        ]
        .filter(mask)
        .to_torch()
    )

    is_flat_tensor = final_df["is_flat"].filter(mask).to_torch().unsqueeze(1)

    mask_sbox = final_df["mask_sbox"].to_torch().unsqueeze(1)

    torch.save(shoebox_tensor, "shoebox_tensor.pt")
    torch.save(padded_dead_pixel_mask, "padded_dead_pixel_mask_tensor.pt")
    torch.save(metadata, "metadata_tensor.pt")
    torch.save(is_flat_tensor, "is_flat_tensor.pt")

    # return final_df


# # Example usage:
# shoebox_dir = "/Users/luis/integrator/rotation_data_examples/data/"
# output_file = "processed_data.pkl"

# preprocess_data(shoebox_dir, output_file)

# torch.load("shoebox_tensor.pt").shape
# torch.load("padded_dead_pixel_mask_tensor.pt").shape
# torch.load("metadata_tensor.pt").shape
# torch.load("is_flat_tensor.pt").shape
