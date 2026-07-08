from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
Compare real MFX input shoeboxes with the model-predicted profile qp_mean.

For each selected reflection, this script saves one side-by-side plot:

    left  = real input shoebox from counts.npy, with invalid pixels masked
    right = predicted profile from qp_mean

This helps check whether the model's predicted profile is spatially reasonable:
    - Is the predicted spot in the right location?
    - Does qp_mean look like a reasonable spot shape?
    - Does the profile behavior change from weak to strong reflections?
"""


DATA_DIR = Path(
    "/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/"
    "mfx_shoebox_r0269_012_rg058_with_d_1500"
)

RUN_DIR = Path(
    "/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/"
    "runs/run_20260702-160655_3424"
)

PRED_PATH = RUN_DIR / "predictions" / "test_preds_all.parquet"

OUT_DIR = Path(
    "/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/"
    "diagnostics_qp_mean_profiles"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# Load dataset arrays.
# counts.npy: flattened 25 x 25 shoebox pixel values, one row per reflection.
# masks.npy: matching valid-pixel masks. True = valid, False = invalid/masked/padded.
# metadata.npy: reflection metadata saved as a Python dictionary.
counts = np.load(DATA_DIR / "counts.npy", mmap_mode="r")
masks = np.load(DATA_DIR / "masks.npy", mmap_mode="r")
metadata = np.load(DATA_DIR / "metadata.npy", allow_pickle=True).item()

# Load prediction table from integrator.predict.
# This should include qp_mean if predict_keys was configured correctly.
pred = pd.read_parquet(PRED_PATH)


print("counts shape:", counts.shape)
print("masks shape:", masks.shape)
print("metadata keys:", list(metadata.keys())[:30])
print("prediction columns:", list(pred.columns)[:30])
print("number of predictions:", len(pred))


# Required prediction columns.
if "qp_mean" not in pred.columns:
    raise ValueError("qp_mean is missing from prediction output.")

if "refl_ids" not in pred.columns:
    raise ValueError("refl_ids is missing from prediction output.")


def get_profile_array(row):
    """
    Return qp_mean for one prediction row as a 25 x 25 image.
    """

    qp = np.asarray(row["qp_mean"], dtype=float)

    if qp.size != 625:
        raise ValueError(f"Expected qp_mean size 625, got {qp.size}")

    return qp.reshape(25, 25)


def get_counts_array(refl_id):
    """
    Return the real input shoebox for one reflection as a 25 x 25 image.

    This assumes refl_ids match row indices in counts.npy.
    """

    img = np.asarray(counts[int(refl_id)], dtype=float)

    if img.size != 625:
        raise ValueError(f"Expected counts size 625, got {img.size}")

    return img.reshape(25, 25)


def get_mask_array(refl_id):
    """
    Return the saved valid-pixel mask for one reflection as a 25 x 25 image.
    """

    mask_img = np.asarray(masks[int(refl_id)], dtype=bool)

    if mask_img.size != 625:
        raise ValueError(f"Expected mask size 625, got {mask_img.size}")

    return mask_img.reshape(25, 25)


def save_comparison(row, label):
    """
    Save one side-by-side comparison:

        left  = masked real input shoebox
        right = model-predicted profile qp_mean
    """

    refl_id = int(row["refl_ids"])

    # Real input shoebox, with invalid pixels hidden for visualization.
    input_img = get_counts_array(refl_id)
    mask_img = get_mask_array(refl_id)
    input_img = np.ma.masked_where(~mask_img, input_img)

    # Model-predicted profile.
    qp_img = get_profile_array(row)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im0 = axes[0].imshow(input_img)
    axes[0].set_title(f"Input shoebox\nrefl_id={refl_id}", fontsize=9)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(qp_img)
    axes[1].set_title("Predicted profile\nqp_mean", fontsize=9)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    title_parts = [label]

    if "qi_mean" in row.index:
        title_parts.append(f"qi_mean={row['qi_mean']:.2f}")

    if "intensity.sum.value" in row.index:
        title_parts.append(f"I_sum={row['intensity.sum.value']:.2f}")

    if "d" in row.index:
        title_parts.append(f"d={row['d']:.2f}")

    fig.suptitle(" | ".join(title_parts), fontsize=10)
    fig.tight_layout()

    out_path = OUT_DIR / f"{label}_refl_{refl_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out_path)


# Choose examples across weak-to-strong reflections.
# Prefer traditional cctbx/DIALS intensity if available.
if "intensity.sum.value" in pred.columns:
    score_col = "intensity.sum.value"
elif "qi_mean" in pred.columns:
    score_col = "qi_mean"
else:
    raise ValueError("Need either intensity.sum.value or qi_mean to choose examples.")


score = pred[score_col].replace([np.inf, -np.inf], np.nan)

percentiles = [10, 50, 90, 99]
chosen_rows = []

for p in percentiles:
    target = np.nanpercentile(score, p)
    idx = (score - target).abs().idxmin()
    chosen_rows.append((f"p{p}_{score_col}", pred.loc[idx]))

# Also inspect the strongest reflection.
max_idx = score.idxmax()
chosen_rows.append((f"max_{score_col}", pred.loc[max_idx]))


for label, row in chosen_rows:
    save_comparison(row, label)


print()
print("Done.")
print("Diagnostics folder:", OUT_DIR)


"""
Important assumption:

    refl_ids in the prediction table must match row indices in counts.npy.

If the input shoebox and qp_mean look mismatched, check whether refl_ids are
original cctbx/DIALS reflection IDs instead of counts.npy row numbers. If so,
this script needs a row-mapping fix before the comparison is valid.
"""