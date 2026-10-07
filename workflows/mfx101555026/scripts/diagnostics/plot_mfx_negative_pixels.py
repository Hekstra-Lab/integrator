from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Change this to the dataset you want to inspect
DATASET_DIR = Path(
    "/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/"
    "mfx101555026_cctbx/mfx_shoebox_stream_test_25"
)

OUT_DIR = Path(
    "/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/"
    "mfx101555026_cctbx/figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

counts_path = DATASET_DIR / "counts.npy"
masks_path = DATASET_DIR / "masks.npy"

counts = np.load(counts_path, mmap_mode="r")
masks = np.load(masks_path, mmap_mode="r")

print("counts path:", counts_path)
print("masks path:", masks_path)
print("counts shape:", counts.shape)
print("masks shape:", masks.shape)
print("counts dtype:", counts.dtype)
print("masks dtype:", masks.dtype)

# Collect valid pixels without loading everything at once
chunk_size = 1000
sample_values = []

valid_count = 0
negative_count = 0
below_anscombe_count = 0
valid_min = np.inf
valid_max = -np.inf

# Limit histogram sample size so plotting stays fast
max_sample_pixels = 2_000_000
rng = np.random.default_rng(42)

for start in range(0, counts.shape[0], chunk_size):
    end = min(start + chunk_size, counts.shape[0])

    c = counts[start:end]
    m = masks[start:end].astype(bool)

    valid = c[m]

    if valid.size == 0:
        continue

    valid_count += valid.size
    negative_count += np.count_nonzero(valid < 0)
    below_anscombe_count += np.count_nonzero(valid < -0.375)
    valid_min = min(valid_min, float(valid.min()))
    valid_max = max(valid_max, float(valid.max()))

    # Randomly sample pixels for histogram
    remaining = max_sample_pixels - sum(x.size for x in sample_values)
    if remaining > 0:
        if valid.size <= remaining:
            sample_values.append(np.asarray(valid))
        else:
            idx = rng.choice(valid.size, size=remaining, replace=False)
            sample_values.append(np.asarray(valid[idx]))

sample = np.concatenate(sample_values)

frac_negative = negative_count / valid_count
frac_below_anscombe = below_anscombe_count / valid_count

print("\nValid pixel statistics")
print("valid pixel count:", valid_count)
print("min valid pixel:", valid_min)
print("max valid pixel:", valid_max)
print("fraction valid pixels < 0:", frac_negative)
print("fraction valid pixels < -0.375:", frac_below_anscombe)

# Save text summary for slide notes
summary_path = OUT_DIR / "mfx_negative_pixel_summary.txt"
with open(summary_path, "w") as f:
    f.write(f"Dataset: {DATASET_DIR}\n")
    f.write(f"counts dtype: {counts.dtype}\n")
    f.write(f"counts shape: {counts.shape}\n")
    f.write(f"valid pixel count: {valid_count}\n")
    f.write(f"min valid pixel: {valid_min}\n")
    f.write(f"max valid pixel: {valid_max}\n")
    f.write(f"fraction valid pixels < 0: {frac_negative}\n")
    f.write(f"fraction valid pixels < -0.375: {frac_below_anscombe}\n")

# Histogram zoomed around the negative/low-value region
fig_path = OUT_DIR / "mfx_valid_pixel_hist_anscombe_threshold.png"

plt.figure(figsize=(8, 5))
plt.hist(sample, bins=300, range=(-10, 50), log=True)
plt.axvline(-0.375, linestyle="--", linewidth=2, label="Anscombe threshold: -0.375")
plt.xlabel("valid pixel value")
plt.ylabel("count (log scale)")
plt.title("MFX valid pixel values include negative floats")
plt.legend()
plt.tight_layout()
plt.savefig(fig_path, dpi=200)

print("\nSaved:")
print("summary:", summary_path)
print("figure:", fig_path)