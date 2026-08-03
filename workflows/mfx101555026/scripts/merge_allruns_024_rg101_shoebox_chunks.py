from pathlib import Path

import numpy as np
import yaml
from numpy.lib.format import open_memmap


BASE = Path(
    "/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/"
    "mfx101555026_cctbx"
)

CHUNKS_DIR = (
    BASE / "mfx_shoebox_5000_269_289_no275_024_rg101_chunks"
)

OUT = (
    BASE / "mfx_shoebox_5000_269_289_no275_024_rg101"
)

OUT.mkdir(parents=True, exist_ok=True)


def chunk_number(path: Path) -> int:
    """Extract the Slurm task number from names such as chunk_3_start_939_n_313."""
    return int(path.name.split("_")[1])


# Sort chunks numerically: chunk_0, chunk_1, ..., chunk_15.
DATASET_DIRS = sorted(
    CHUNKS_DIR.glob("chunk_*"),
    key=chunk_number,
)

good_dirs = []

for dataset_dir in DATASET_DIRS:
    needed = [
        dataset_dir / "counts.npy",
        dataset_dir / "masks.npy",
        dataset_dir / "metadata.npy",
    ]

    if all(path.exists() for path in needed):
        good_dirs.append(dataset_dir)
    else:
        print(f"SKIP missing final files: {dataset_dir}")

if not good_dirs:
    raise SystemExit(
        f"No completed dataset folders found in {CHUNKS_DIR}"
    )

print("Datasets to merge:")

for dataset_dir in good_dirs:
    print(" ", dataset_dir)

if len(good_dirs) != 16:
    print(
        f"WARNING: expected 16 completed chunks, "
        f"but found {len(good_dirs)}"
    )

###############################################################################
# Inspect chunk shapes and metadata
###############################################################################

total_rows = 0
n_pixels = None
counts_dtype = None
all_meta = []

for dataset_dir in good_dirs:
    counts = np.load(
        dataset_dir / "counts.npy",
        mmap_mode="r",
    )

    masks = np.load(
        dataset_dir / "masks.npy",
        mmap_mode="r",
    )

    metadata = np.load(
        dataset_dir / "metadata.npy",
        allow_pickle=True,
    ).item()

    print(
        dataset_dir.name,
        "counts:",
        counts.shape,
        "masks:",
        masks.shape,
    )

    if counts.ndim != 2:
        raise ValueError(
            f"Expected 2D counts array in {dataset_dir}, "
            f"got shape {counts.shape}"
        )

    if masks.shape != counts.shape:
        raise ValueError(
            f"Mask/count shape mismatch in {dataset_dir}: "
            f"{masks.shape} versus {counts.shape}"
        )

    if n_pixels is None:
        n_pixels = counts.shape[1]
        counts_dtype = counts.dtype
    elif counts.shape[1] != n_pixels:
        raise ValueError(
            f"Pixel-count mismatch in {dataset_dir}: "
            f"{counts.shape[1]} versus {n_pixels}"
        )

    n_rows = counts.shape[0]

    for key, value in metadata.items():
        try:
            value_length = len(value)
        except TypeError:
            continue

        if value_length != n_rows:
            raise ValueError(
                f"Metadata key {key!r} in {dataset_dir} has "
                f"{value_length} rows, but counts has {n_rows}"
            )

    total_rows += n_rows
    all_meta.append(metadata)

print("Total rows:", total_rows)
print("Pixels per shoebox:", n_pixels)
print("Counts dtype:", counts_dtype)

###############################################################################
# Create merged counts and masks arrays on disk
###############################################################################

counts_out = open_memmap(
    OUT / "counts.npy",
    mode="w+",
    dtype=counts_dtype,
    shape=(total_rows, n_pixels),
)

masks_out = open_memmap(
    OUT / "masks.npy",
    mode="w+",
    dtype=np.bool_,
    shape=(total_rows, n_pixels),
)

row0 = 0

for dataset_dir in good_dirs:
    counts = np.load(
        dataset_dir / "counts.npy",
        mmap_mode="r",
    )

    masks = np.load(
        dataset_dir / "masks.npy",
        mmap_mode="r",
    )

    row1 = row0 + counts.shape[0]

    print(
        f"Copying {dataset_dir.name}: "
        f"rows {row0}:{row1}"
    )

    counts_out[row0:row1] = counts
    masks_out[row0:row1] = masks

    row0 = row1

counts_out.flush()
masks_out.flush()

del counts_out
del masks_out

###############################################################################
# Merge metadata
###############################################################################

common_keys = set(all_meta[0].keys())

for metadata in all_meta[1:]:
    common_keys &= set(metadata.keys())

common_keys = sorted(common_keys)

print("Common metadata keys:")
for key in common_keys:
    print(" ", key)

merged_meta = {}

for key in common_keys:
    arrays = [
        np.asarray(metadata[key])
        for metadata in all_meta
    ]

    merged_meta[key] = np.concatenate(
        arrays,
        axis=0,
    )

# Give every merged reflection a unique global row ID.
merged_meta["refl_ids"] = np.arange(
    total_rows,
    dtype=np.int64,
)

###############################################################################
# Rebuild global image_id and n_images
###############################################################################

if "image_index" in merged_meta:
    image_key = "image_index"
elif "image_num" in merged_meta:
    image_key = "image_num"
else:
    raise KeyError(
        "Metadata has neither image_index nor image_num"
    )

image_num = np.asarray(
    merged_meta[image_key],
    dtype=np.int64,
)

unique_images = np.unique(image_num)

image_to_id = {
    image_number: image_id
    for image_id, image_number in enumerate(unique_images)
}

merged_meta["image_num"] = image_num

merged_meta["image_id"] = np.fromiter(
    (image_to_id[value] for value in image_num),
    dtype=np.int64,
    count=total_rows,
)

merged_meta["n_images"] = np.full(
    total_rows,
    len(unique_images),
    dtype=np.int64,
)

print("Global n_images:", len(unique_images))
print(
    "image_id min/max:",
    merged_meta["image_id"].min(),
    merged_meta["image_id"].max(),
)

np.save(
    OUT / "metadata.npy",
    merged_meta,
)

###############################################################################
# Compute raw pixel statistics in chunks
###############################################################################

counts = np.load(
    OUT / "counts.npy",
    mmap_mode="r",
)

masks = np.load(
    OUT / "masks.npy",
    mmap_mode="r",
)

sum_counts = 0.0
sum_squared_counts = 0.0
number_of_valid_pixels = 0

ROW_CHUNK_SIZE = 10000

for start in range(
    0,
    total_rows,
    ROW_CHUNK_SIZE,
):
    end = min(
        start + ROW_CHUNK_SIZE,
        total_rows,
    )

    chunk_counts = counts[start:end].astype(
        np.float64,
        copy=False,
    )

    chunk_masks = masks[start:end].astype(
        bool,
        copy=False,
    )

    valid_counts = chunk_counts[chunk_masks]

    sum_counts += valid_counts.sum(
        dtype=np.float64,
    )

    sum_squared_counts += np.square(
        valid_counts,
        dtype=np.float64,
    ).sum(
        dtype=np.float64,
    )

    number_of_valid_pixels += valid_counts.size

if number_of_valid_pixels == 0:
    raise RuntimeError(
        "No valid pixels found while computing statistics"
    )

mean_counts = (
    sum_counts / number_of_valid_pixels
)

variance_counts = (
    sum_squared_counts / number_of_valid_pixels
    - mean_counts * mean_counts
)

# Protect against a tiny negative number caused by floating-point rounding.
variance_counts = max(
    variance_counts,
    0.0,
)

stats = {
    "raw": [
        float(mean_counts),
        float(variance_counts),
    ]
}

print("Raw stats:", stats)

###############################################################################
# Compute concentration in chunks
###############################################################################

concentration = open_memmap(
    OUT / "concentration.npy",
    mode="w+",
    dtype=np.float32,
    shape=(total_rows,),
)

for start in range(
    0,
    total_rows,
    ROW_CHUNK_SIZE,
):
    end = min(
        start + ROW_CHUNK_SIZE,
        total_rows,
    )

    chunk_counts = counts[start:end].astype(
        np.float32,
        copy=False,
    )

    chunk_masks = masks[start:end]

    concentration[start:end] = (
        chunk_counts * chunk_masks
    ).sum(
        axis=1,
        dtype=np.float32,
    )

concentration.flush()
del concentration

###############################################################################
# Write dataset.yaml
###############################################################################

dataset_yaml = {
    "geometry": {
        "d": 1,
        "h": 25,
        "w": 25,
    },
    "n_reflections": int(total_rows),
    "polychromatic": False,
    "anscombe": False,
    "files": {
        "counts": "counts.npy",
        "masks": "masks.npy",
        "reference": "metadata.npy",
    },
    "crystal": None,
    "stats": stats,
    "refl_file": None,
}

with open(
    OUT / "dataset.yaml",
    "w",
    encoding="utf-8",
) as stream:
    yaml.safe_dump(
        dataset_yaml,
        stream,
        sort_keys=False,
    )

###############################################################################
# Final validation
###############################################################################

final_counts = np.load(
    OUT / "counts.npy",
    mmap_mode="r",
)

final_masks = np.load(
    OUT / "masks.npy",
    mmap_mode="r",
)

final_metadata = np.load(
    OUT / "metadata.npy",
    allow_pickle=True,
).item()

if final_counts.shape != final_masks.shape:
    raise RuntimeError(
        "Final counts and masks shapes do not match"
    )

if final_counts.shape[0] != total_rows:
    raise RuntimeError(
        "Final counts row count is incorrect"
    )

if len(final_metadata["refl_ids"]) != total_rows:
    raise RuntimeError(
        "Final metadata row count is incorrect"
    )

print()
print("Merge completed successfully")
print("Output directory:", OUT)
print("Counts shape:", final_counts.shape)
print("Masks shape:", final_masks.shape)
print("Metadata rows:", len(final_metadata["refl_ids"]))
print("Unique images:", len(unique_images))
print()
print("Wrote:")
print("  counts.npy")
print("  masks.npy")
print("  metadata.npy")
print("  concentration.npy")
print("  dataset.yaml")