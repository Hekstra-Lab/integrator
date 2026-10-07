from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

BASE = Path(
    "/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/"
    "mfx101555026_cctbx"
)

DATA_DIR = (
    BASE
    / "mfx_shoebox_allruns_269_289_no275_024_rg101"
)

RUNS = {
    "ASINH": (
        BASE
        / "runs"
        / "run_20260805-011728_9704"
    ),
    "sqrt-squareplus": (
        BASE
        / "runs"
        / "run_20260805-183129_a8b9"
    ),
}

OUT_DIR = (
    BASE
    / "diagnostics_qp_mean_profiles_allruns_cctbx_thresholds"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# cctbx intensity categories
#
# Thresholds from Luis:
#
#   Weak   : I < 20
#   Medium : 20 <= I <= 200
#   Strong : I > 200
#
# The "target" values below are ONLY used to choose one
# representative example inside each category.
#
# They do NOT define the categories.
# ============================================================

CATEGORIES = {
    "Weak": {
        "target": 10.0,
    },
    "Medium": {
        "target": 100.0,
    },
    "Strong": {
        "target": 500.0,
    },
}


# ============================================================
# Load observed shoebox arrays
#
# mmap_mode="r" keeps the huge arrays on disk instead of
# loading the entire 115M-reflection dataset into RAM.
# ============================================================

counts = np.load(
    DATA_DIR / "counts.npy",
    mmap_mode="r",
)

masks = np.load(
    DATA_DIR / "masks.npy",
    mmap_mode="r",
)

print("counts shape:", counts.shape)
print("masks shape :", masks.shape)


# ============================================================
# Helpers
# ============================================================

def prediction_files(run_dir):
    """
    Return all prediction parquet shards for epoch 24.
    """

    pred_dir = (
        run_dir
        / "predictions"
        / "epoch_0024"
    )

    files = sorted(
        pred_dir.glob("*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {pred_dir}"
        )

    return files


def category_mask(values, label):
    """
    Return a Boolean mask for Luis's cctbx intensity ranges.

    Weak:
        I < 20

    Medium:
        20 <= I <= 200

    Strong:
        I > 200
    """

    finite = np.isfinite(values)

    if label == "Weak":
        return (
            finite
            & (values < 20.0)
        )

    if label == "Medium":
        return (
            finite
            & (values >= 20.0)
            & (values <= 200.0)
        )

    if label == "Strong":
        return (
            finite
            & (values > 200.0)
        )

    raise ValueError(
        f"Unknown category: {label}"
    )


def validate_category(label, intensity):
    """
    Make sure the selected reflection actually satisfies
    Luis's threshold definition.
    """

    if label == "Weak":
        valid = intensity < 20.0

    elif label == "Medium":
        valid = (
            20.0 <= intensity <= 200.0
        )

    elif label == "Strong":
        valid = intensity > 200.0

    else:
        raise ValueError(label)

    if not valid:
        raise RuntimeError(
            f"{label} reflection has invalid cctbx "
            f"intensity: {intensity}"
        )


# ============================================================
# PASS 1
#
# Select Weak / Medium / Strong using ONLY cctbx intensity.
#
# We search the ASINH prediction files because they contain:
#
#   refl_ids
#   cctbx intensity
#   qi_mean
#   qp_mean
#   d
#
# ASINH qi_mean does NOT determine the category.
# ============================================================

def find_cctbx_selected_reflections(run_dir):

    files = prediction_files(run_dir)

    print(
        "\n=== PASS 1: selecting reflections "
        "using cctbx intensity ==="
    )

    print(
        "Thresholds:"
    )

    print(
        "  Weak   : I < 20"
    )

    print(
        "  Medium : 20 <= I <= 200"
    )

    print(
        "  Strong : I > 200"
    )

    best = {
        label: {
            "distance": np.inf,
            "refl_id": None,
            "cctbx_I": None,
            "qi_mean": None,
            "qp_mean": None,
            "d": None,
        }
        for label in CATEGORIES
    }

    columns = [
        "refl_ids",
        "qi_mean",
        "qp_mean",
        "d",
        "intensity.sum.value",
    ]

    for i, path in enumerate(
        files,
        start=1,
    ):

        df = pd.read_parquet(
            path,
            columns=columns,
        )

        cctbx_I = (
            df["intensity.sum.value"]
            .to_numpy(
                dtype=np.float64,
                copy=False,
            )
        )

        for label, config in CATEGORIES.items():

            mask = category_mask(
                cctbx_I,
                label,
            )

            if not np.any(mask):
                continue

            valid_indices = np.flatnonzero(
                mask
            )

            valid_I = cctbx_I[mask]

            target = config["target"]

            # Pick the reflection inside the category
            # whose cctbx intensity is nearest to the
            # representative target.
            local_pos = int(
                np.argmin(
                    np.abs(
                        valid_I - target
                    )
                )
            )

            dataframe_pos = int(
                valid_indices[local_pos]
            )

            selected_I = float(
                valid_I[local_pos]
            )

            distance = abs(
                selected_I - target
            )

            if (
                distance
                < best[label]["distance"]
            ):

                row = df.iloc[
                    dataframe_pos
                ]

                best[label] = {
                    "distance": float(
                        distance
                    ),
                    "refl_id": int(
                        row["refl_ids"]
                    ),
                    "cctbx_I": float(
                        row[
                            "intensity.sum.value"
                        ]
                    ),
                    "qi_mean": float(
                        row["qi_mean"]
                    ),
                    "qp_mean": np.asarray(
                        row["qp_mean"],
                        dtype=np.float32,
                    ),
                    "d": float(
                        row["d"]
                    ),
                }

        if (
            i % 500 == 0
            or i == len(files)
        ):
            print(
                f"  searched "
                f"{i:,} / {len(files):,}"
            )

        del df

    # --------------------------------------------------------
    # Validate results
    # --------------------------------------------------------

    for label, info in best.items():

        if info["refl_id"] is None:
            raise RuntimeError(
                f"No reflection found for "
                f"{label}"
            )

        validate_category(
            label,
            info["cctbx_I"],
        )

    return best


# ============================================================
# PASS 2
#
# Find those exact SAME reflection IDs in another Integrator
# model.
#
# IMPORTANT:
#
# We intentionally do NOT read cctbx intensity here.
#
# The original cctbx intensity from PASS 1 is preserved later.
# This fixes the previous mismatch bug.
# ============================================================

def find_same_reflections_in_model(
    run_dir,
    selected_ids,
):

    files = prediction_files(run_dir)

    wanted = set(
        selected_ids
    )

    found = {}

    print(
        "\n=== PASS 2: finding same reflections "
        "in second model ==="
    )

    columns = [
        "refl_ids",
        "qi_mean",
        "qp_mean",
        "d",
    ]

    for i, path in enumerate(
        files,
        start=1,
    ):

        df = pd.read_parquet(
            path,
            columns=columns,
        )

        ids = (
            df["refl_ids"]
            .to_numpy(
                dtype=np.int64,
                copy=False,
            )
        )

        for target_id in list(
            wanted
        ):

            positions = np.flatnonzero(
                ids == target_id
            )

            if len(positions) == 0:
                continue

            row = df.iloc[
                int(positions[0])
            ]

            found[target_id] = {
                "refl_id": target_id,
                "qi_mean": float(
                    row["qi_mean"]
                ),
                "qp_mean": np.asarray(
                    row["qp_mean"],
                    dtype=np.float32,
                ),
                "d": float(
                    row["d"]
                ),
            }

            wanted.remove(
                target_id
            )

        if (
            i % 500 == 0
            or not wanted
        ):
            print(
                f"  searched "
                f"{i:,} / {len(files):,} "
                f"| remaining IDs: "
                f"{len(wanted)}"
            )

        del df

        if not wanted:
            break

    if wanted:
        raise RuntimeError(
            "Could not find these reflection IDs "
            "in the second model: "
            f"{sorted(wanted)}"
        )

    return found


# ============================================================
# Run selection
# ============================================================

asinh_selected = (
    find_cctbx_selected_reflections(
        RUNS["ASINH"]
    )
)


# ------------------------------------------------------------
# Store the selected IDs by category.
# ------------------------------------------------------------

selected_ids = {
    label: info["refl_id"]
    for label, info
    in asinh_selected.items()
}


# ------------------------------------------------------------
# Find those SAME IDs in sqrt-squareplus.
# ------------------------------------------------------------

sqrt_by_id = (
    find_same_reflections_in_model(
        RUNS["sqrt-squareplus"],
        selected_ids.values(),
    )
)


# ============================================================
# CRITICAL BUG FIX
#
# Build sqrt_selected while preserving:
#
#   1. same refl_id
#   2. same original cctbx intensity
#   3. same Weak / Medium / Strong classification
#
# Only the Integrator prediction changes between models.
# ============================================================

sqrt_selected = {}

for label, refl_id in selected_ids.items():

    sqrt_info = (
        sqrt_by_id[
            refl_id
        ].copy()
    )

    # Preserve EXACT cctbx intensity
    # used for classification.
    sqrt_info["cctbx_I"] = (
        asinh_selected[
            label
        ]["cctbx_I"]
    )

    sqrt_selected[label] = (
        sqrt_info
    )


# ============================================================
# Sanity checks
# ============================================================

print(
    "\n=== SELECTED REFLECTIONS ==="
)

for label in [
    "Weak",
    "Medium",
    "Strong",
]:

    a = asinh_selected[label]
    s = sqrt_selected[label]

    validate_category(
        label,
        a["cctbx_I"],
    )

    # Make sure both models use the
    # exact same reflection.
    if (
        a["refl_id"]
        != s["refl_id"]
    ):
        raise RuntimeError(
            f"{label}: ASINH and sqrt "
            "reflection IDs do not match."
        )

    # Make sure displayed cctbx intensity
    # is identical in both models.
    if not np.isclose(
        a["cctbx_I"],
        s["cctbx_I"],
    ):
        raise RuntimeError(
            f"{label}: cctbx intensity "
            "changed between models."
        )

    print(
        f"{label:6s} | "
        f"refl_id={a['refl_id']} | "
        f"cctbx I={a['cctbx_I']:.2f} | "
        f"ASINH I={a['qi_mean']:.2f} | "
        f"sqrt I={s['qi_mean']:.2f} | "
        f"d={a['d']:.2f} A"
    )


# ============================================================
# Extra threshold check
# ============================================================

weak_I = (
    asinh_selected[
        "Weak"
    ]["cctbx_I"]
)

medium_I = (
    asinh_selected[
        "Medium"
    ]["cctbx_I"]
)

strong_I = (
    asinh_selected[
        "Strong"
    ]["cctbx_I"]
)

print(
    "\n=== THRESHOLD CHECK ==="
)

print(
    f"Weak   : {weak_I:.2f}  "
    "(must be < 20)"
)

print(
    f"Medium : {medium_I:.2f}  "
    "(must be 20-200)"
)

print(
    f"Strong : {strong_I:.2f}  "
    "(must be > 200)"
)


# ============================================================
# Image helpers
# ============================================================

def get_observed(refl_id):
    """
    Get the real experimental 25x25 detector shoebox.
    """

    img = np.asarray(
        counts[refl_id],
        dtype=float,
    ).reshape(
        25,
        25,
    )

    mask = np.asarray(
        masks[refl_id],
        dtype=bool,
    ).reshape(
        25,
        25,
    )

    return np.ma.masked_where(
        ~mask,
        img,
    )


def get_profile(info):
    """
    Convert Integrator qp_mean into a 25x25 learned profile.
    """

    qp = np.asarray(
        info["qp_mean"],
        dtype=float,
    )

    if qp.size != 625:
        raise ValueError(
            "Expected qp_mean size 625, "
            f"got {qp.size}"
        )

    return qp.reshape(
        25,
        25,
    )


# ============================================================
# Make figure
# ============================================================

def make_figure(
    model_name,
    model_selected,
):

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11, 7),
    )

    labels = [
        "Weak",
        "Medium",
        "Strong",
    ]

    for col, label in enumerate(
        labels
    ):

        info = (
            model_selected[
                label
            ]
        )

        refl_id = (
            info["refl_id"]
        )

        observed = get_observed(
            refl_id
        )

        profile = get_profile(
            info
        )

        cctbx_I = (
            info["cctbx_I"]
        )

        integrator_I = (
            info["qi_mean"]
        )

        d = info["d"]

        # ----------------------------------------------------
        # Top row:
        # real experimental shoebox
        # ----------------------------------------------------

        ax = axes[
            0,
            col,
        ]

        ax.imshow(
            observed,
            interpolation="nearest",
        )

        ax.set_title(
            f"{label}\n"
            f"cctbx I = {cctbx_I:.1f}",
            fontsize=15,
            fontweight="bold",
        )

        ax.set_xticks([])
        ax.set_yticks([])

        # ----------------------------------------------------
        # Bottom row:
        # Integrator learned profile
        # ----------------------------------------------------

        ax = axes[
            1,
            col,
        ]

        ax.imshow(
            profile,
            interpolation="nearest",
        )

        ax.set_title(
            f"Integrator I = "
            f"{integrator_I:.1f}\n"
            f"d = {d:.2f} A",
            fontsize=14,
        )

        ax.set_xticks([])
        ax.set_yticks([])


    # --------------------------------------------------------
    # Row labels
    # --------------------------------------------------------

    axes[
        0,
        0,
    ].set_ylabel(
        "Observed shoebox",
        fontsize=16,
        fontweight="bold",
        labelpad=15,
    )

    axes[
        1,
        0,
    ].set_ylabel(
        "Learned profile",
        fontsize=16,
        fontweight="bold",
        labelpad=15,
    )


    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    fig.suptitle(
        f"{model_name}: "
        "Observed Shoeboxes and Learned Profiles",
        fontsize=18,
        fontweight="bold",
    )


    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )


    safe_name = (
        model_name.lower()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )


    out = (
        OUT_DIR
        / (
            f"cctbx_threshold_"
            f"{safe_name}_"
            f"weak_medium_strong.png"
        )
    )


    fig.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "saved:",
        out,
    )


# ============================================================
# Generate ASINH and sqrt-squareplus figures
# ============================================================

make_figure(
    "ASINH",
    asinh_selected,
)

make_figure(
    "sqrt-squareplus",
    sqrt_selected,
)


print(
    "\nDone."
)

print(
    "Output folder:",
    OUT_DIR,
)