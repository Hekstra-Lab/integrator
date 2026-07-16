from pathlib import Path
import numpy as np

OUT = Path("/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/mfx_shoebox_r0269_018_rg070_rawcache_all_timing")

metadata_path = OUT / "metadata.npy"
counts_path = OUT / "counts.npy"
masks_path = OUT / "masks.npy"

print("Dataset:", OUT)
print("metadata exists:", metadata_path.exists())
print("counts exists:  ", counts_path.exists())
print("masks exists:   ", masks_path.exists())
print()

raw = np.load(metadata_path, allow_pickle=True)
counts = np.load(counts_path, mmap_mode="r")
masks = np.load(masks_path, mmap_mode="r")

print("counts shape:", counts.shape, counts.dtype)
print("masks shape: ", masks.shape, masks.dtype)
print("raw metadata type:", type(raw))
print("raw metadata shape:", raw.shape)
print("raw metadata dtype:", raw.dtype)
print()

if raw.shape == () and isinstance(raw.item(), dict):
    metadata = raw.item()
else:
    raise TypeError("Expected metadata.npy to be a saved dictionary object.")

print("metadata format: dictionary")
print("metadata keys:")
for key in metadata.keys():
    print("  -", key)

print()
first_key = next(iter(metadata))
n_rows = len(metadata[first_key])
print("rows:", n_rows)

print()
print("field summaries:")
for key, value in metadata.items():
    arr = np.asarray(value)

    print()
    print(f"{key}:")
    print("  shape:", arr.shape)
    print("  dtype:", arr.dtype)

    if arr.ndim > 0:
        print("  first 5:", arr[:5])
    else:
        print("  value:", arr)

    if arr.ndim > 0 and len(arr) == n_rows:
        print("  length matches rows: yes")
    else:
        print("  length matches rows: no")

    if np.issubdtype(arr.dtype, np.number):
        finite = np.isfinite(arr)
        print("  finite:", finite.sum(), "/", arr.size)
        print("  min:", np.nanmin(arr))
        print("  max:", np.nanmax(arr))
        print("  mean:", np.nanmean(arr))

    if key in ["image_num", "source_refl", "source_expt", "raw_cache_file", "panel"]:
        unique = np.unique(arr)
        print("  unique count:", len(unique))
        print("  first unique:", unique[:5])
        print("  last unique: ", unique[-5:])

print()
print("important checks:")
for key in ["image_num", "source_refl", "source_expt", "raw_cache_file", "refl_ids", "reflection_id", "d", "wavelength"]:
    if key in metadata:
        arr = np.asarray(metadata[key])
        print(f"  {key}: present, shape={arr.shape}, dtype={arr.dtype}")
    else:
        print(f"  {key}: MISSING")

print()
print("first 3 rows:")
keys = list(metadata.keys())
for i in range(min(3, n_rows)):
    print(f"row {i}:")
    for key in keys:
        arr = np.asarray(metadata[key])
        if arr.ndim > 0 and len(arr) == n_rows:
            print(f"  {key}: {arr[i]}")
