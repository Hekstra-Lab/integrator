from pathlib import Path

import numpy as np
import yaml
from numpy.lib.format import open_memmap


BASE = Path("/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx")

CHUNKS_DIR = BASE / "mfx_shoebox_allruns_269_289_no275_024_rg101_chunks"
OUT = BASE / "mfx_shoebox_allruns_269_289_no275_024_rg101"

OUT.mkdir(parents=True, exist_ok=True)

DATASET_DIRS = sorted(CHUNKS_DIR.glob("chunk_*"))

good_dirs = []
for d in DATASET_DIRS:
    needed = [d / "counts.npy", d / "masks.npy", d / "metadata.npy"]
    if all(p.exists() for p in needed):
        good_dirs.append(d)
    else:
        print(f"SKIP missing final files: {d}")

if not good_dirs:
    raise SystemExit("No completed dataset folders found.")

print("Datasets to merge:")
for d in good_dirs:
    print(" ", d)

total_rows = 0
n_pixels = None
counts_dtype = None
all_meta = []

for d in good_dirs:
    counts = np.load(d / "counts.npy", mmap_mode="r")
    masks = np.load(d / "masks.npy", mmap_mode="r")
    meta = np.load(d / "metadata.npy", allow_pickle=True).item()

    print(d.name, counts.shape)

    if masks.shape != counts.shape:
        raise ValueError(f"mask/count shape mismatch in {d}")

    if n_pixels is None:
        n_pixels = counts.shape[1]
        counts_dtype = counts.dtype
    elif counts.shape[1] != n_pixels:
        raise ValueError(f"pixel count mismatch in {d}")

    total_rows += counts.shape[0]
    all_meta.append(meta)

print("total rows:", total_rows)
print("pixels per shoebox:", n_pixels)
print("counts dtype:", counts_dtype)

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
for d in good_dirs:
    counts = np.load(d / "counts.npy", mmap_mode="r")
    masks = np.load(d / "masks.npy", mmap_mode="r")

    row1 = row0 + counts.shape[0]

    print(f"copying {d.name}: rows {row0}:{row1}")

    counts_out[row0:row1] = counts
    masks_out[row0:row1] = masks

    row0 = row1

counts_out.flush()
masks_out.flush()
del counts_out, masks_out

common_keys = set(all_meta[0].keys())
for meta in all_meta[1:]:
    common_keys &= set(meta.keys())

common_keys = sorted(common_keys)
print("common metadata keys:", common_keys)

merged_meta = {}
for key in common_keys:
    merged_meta[key] = np.concatenate([meta[key] for meta in all_meta], axis=0)

merged_meta["refl_ids"] = np.arange(total_rows, dtype=np.int64)

# Recompute global image_id/n_images after merging all Slurm outputs.
if "image_index" in merged_meta:
    image_key = "image_index"
elif "image_num" in merged_meta:
    image_key = "image_num"
else:
    raise KeyError("metadata has neither image_index nor image_num")

image_num = np.asarray(merged_meta[image_key]).astype(np.int64)

unique_images = np.array(sorted(set(image_num.tolist())), dtype=np.int64)
image_to_id = {img: i for i, img in enumerate(unique_images)}

merged_meta["image_num"] = image_num
merged_meta["image_id"] = np.array(
    [image_to_id[x] for x in image_num],
    dtype=np.int64,
)
merged_meta["n_images"] = np.full(
    total_rows,
    len(unique_images),
    dtype=np.int64,
)

print("global n_images:", len(unique_images))
print(
    "image_id min/max:",
    merged_meta["image_id"].min(),
    merged_meta["image_id"].max(),
)

np.save(OUT / "metadata.npy", merged_meta)

counts = np.load(OUT / "counts.npy", mmap_mode="r")
masks = np.load(OUT / "masks.npy", mmap_mode="r")

sum_c = 0.0
sumsq_c = 0.0
nel = 0
chunk = 10000

for i in range(0, total_rows, chunk):
    c = counts[i:i + chunk].astype(np.float64)
    m = masks[i:i + chunk].astype(bool)

    valid = c[m]
    sum_c += valid.sum()
    sumsq_c += (valid * valid).sum()
    nel += valid.size

if nel == 0:
    raise RuntimeError("No valid pixels found while computing stats.")

mean_c = sum_c / nel
var_c = sumsq_c / nel - mean_c * mean_c
stats = {"raw": [float(mean_c), float(var_c)]}

print("raw stats:", stats)

concentration = np.zeros(total_rows, dtype=np.float32)

for i in range(0, total_rows, chunk):
    c = counts[i:i + chunk].astype(np.float32)
    m = masks[i:i + chunk]
    concentration[i:i + chunk] = (c * m).sum(axis=1)

np.save(OUT / "concentration.npy", concentration)

dataset_yaml = {
    "geometry": {"d": 1, "h": 25, "w": 25},
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

with open(OUT / "dataset.yaml", "w") as f:
    yaml.safe_dump(dataset_yaml, f, sort_keys=False)

print("WROTE:", OUT)
print("  counts.npy")
print("  masks.npy")
print("  metadata.npy")
print("  concentration.npy")
print("  dataset.yaml")
