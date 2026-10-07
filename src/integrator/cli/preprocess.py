"""
integrator.preprocess — one-time preprocessing: filter, transform, save chunks.

Two-pass design
───────────────
Pass 1  (_pass1_scan)
    • load raw mmaps and metadata lazily
    • read dataset.yaml for stats; --transform is an explicit CLI arg
    • compute valid_indices (same logic as RotationDataModule.setup)
    • verify required metadata columns; warn about optional ones
    • check for duplicate refl_ids (raises ValueError)
    • split assignment — two mutually exclusive modes:

        Mode B (preferred for new datasets):
          --validation-split FLOAT  --split-seed INT
          Generates a deterministic split from sorted refl_ids + seeded RNG.
          Independent of input row order.  Exports train_labels.csv and
          val_labels.csv to <out-dir> for later reuse.
          Every valid reflection is assigned to exactly one split.

        Mode A (backward compatible):
          --train-labels PATH  --val-labels PATH
          Reads existing CSV split files.  Rows absent from both CSVs
          (e.g. test reflections) get is_train=False is_val=False and
          appear in predict-only; they are NOT an error.

    • compute group_label for ALL n_bins >= 1 (required by _step())
    • append _EXTRA_SAVE_COLS (reflection_id, source_refl, source_expt,
      wavelength) when present in reference
    • n_images from image_id only — image_num is not suitable for embeddings

Pass 2  (_pass2_write_chunks)
    • iterate chunks of valid_indices
    • copy rows from mmaps, apply vectorised transform (NO global clip —
      only anscombe and log1p clamp to min=0, matching _transform_counts)
    • write counts.npy, shoebox.npy, mask.npy (large pixel arrays, separate)
    • write metadata.npz  — all numeric + bool columns in one file
    • write strings.parquet — all string columns (omitted when none exist)
    • write manifest.yaml

Chunk layout
────────────
chunk_NNNNN/
  counts.npy        float32  (N, pixels)   raw pixel counts
  shoebox.npy       float32  (N, pixels)   transform-applied (model input)
  mask.npy          bool     (N, pixels)   valid-pixel mask
  metadata.npz      all numeric + bool metadata columns:
                      row_id, is_train, is_val, image_id, d, lp,
                      group_label, profile_group_label, refl_ids, ...
  strings.parquet   all string columns (source_refl, source_expt, ...);
                    omitted entirely when no string columns exist

Usage — Mode B (preferred)
──────────────────────────
integrator.preprocess \
    --data-dir /data/mfx \
    --validation-split 0.2 \
    --split-seed 42 \
    [--transform asinh] [--chunk-size 50000] [--n-bins 10] \
    [--out-dir /data/mfx/chunks] [--min-valid-pixels 10]

Usage — Mode A (existing split labels)
───────────────────────────────────────
integrator.preprocess \
    --data-dir /data/mfx \
    --train-labels /runs/run_abc/train_labels.csv \
    --val-labels   /runs/run_abc/val_labels.csv \
    [--transform asinh] [--chunk-size 50000] [--n-bins 10]
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml

logger = logging.getLogger(__name__)

_VALID_TRANSFORMS = (
    "anscombe",
    "log1p",
    "standardization",
    "asinh",
    "log_softplus",
    "sqrt_squareplus",
)

# Columns that must be present for the pipeline to work correctly.
_REQUIRED_COLS = ["refl_ids", "d"]
# At least one image identifier must exist.
_REQUIRED_ONE_OF_IMAGE = ["image_id", "image_num"]
# Optional columns worth warning about when absent.
_VERIFIED_OPTIONAL = [
    "reflection_id",
    "wavelength",
    "lp",
    "source_refl",
    "source_expt",
]
# Columns critical for MFX write-back and polychromatic loss that are absent
# from DEFAULT_DS_COLS.  Saved when present in reference.
# Note: lp IS in DEFAULT_DS_COLS (line 154) and does not appear here.
_EXTRA_SAVE_COLS = [
    "reflection_id",  # MFX write-back: row index within source .refl
    "source_refl",  # MFX write-back: source .refl filename (string)
    "source_expt",  # MFX write-back: source .expt filename (string)
    "wavelength",  # polychromatic loss
]

# Mode B defaults
_DEFAULT_VALIDATION_SPLIT = 0.2
_DEFAULT_SPLIT_SEED = 42


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="integrator.preprocess",
        description=(
            "Preprocess shoeboxes into fixed-size chunks.\n\n"
            "Split modes (mutually exclusive):\n"
            "  Mode B (preferred): --validation-split / --split-seed\n"
            "  Mode A (compat):    --train-labels / --val-labels"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        required=True,
        help="Directory with counts.npy, masks.npy, metadata.npy, dataset.yaml",
    )
    # ── Mode A args ──
    p.add_argument(
        "--train-labels",
        default=None,
        metavar="PATH",
        help=(
            "Mode A: CSV with column 'train_ids' (refl_ids of training "
            "reflections).  Mutually exclusive with --validation-split."
        ),
    )
    p.add_argument(
        "--val-labels",
        default=None,
        metavar="PATH",
        help=(
            "Mode A: CSV with column 'val_ids'.  "
            "Must be given together with --train-labels."
        ),
    )
    # ── Mode B args ──
    p.add_argument(
        "--validation-split",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            f"Mode B: fraction of valid reflections held out for validation "
            f"(default {_DEFAULT_VALIDATION_SPLIT} when neither mode is "
            "specified).  Must be in (0, 1).  "
            "Mutually exclusive with --train-labels / --val-labels."
        ),
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=_DEFAULT_SPLIT_SEED,
        metavar="INT",
        help=(
            f"Mode B: RNG seed for the deterministic split "
            f"(default {_DEFAULT_SPLIT_SEED})."
        ),
    )
    # ── Shared args ──
    p.add_argument(
        "--transform",
        default=None,
        choices=_VALID_TRANSFORMS,
        help=(
            "Pixel transform applied before saving shoeboxes. "
            "Must match the transform used for training. "
            "Default: inferred from dataset.yaml "
            "('anscombe' if anscombe:true, else 'standardization')."
        ),
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Rows per chunk.  Default 50 000; benchmark before locking in.",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=1,
        help=(
            "Resolution bins for group_label.  group_label is always saved "
            "regardless of this value (required by the training step). "
            "Must be >= 1."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Chunk output directory (default: <data-dir>/chunks)",
    )
    p.add_argument("--min-valid-pixels", type=int, default=10)
    p.add_argument("--resolution-cutoff", type=float, default=None)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


# ── Transform — vectorised numpy, one full chunk at a time ────────────────────


def _apply_transform_np(
    counts_f32: np.ndarray,  # (N, pixels)  float32
    masks_bool: np.ndarray,  # (N, pixels)  bool
    mean: float,
    var: float,
    transform: str,
) -> np.ndarray:
    """Vectorised equivalent of IntegratorDataset._transform_counts.

    Clamping policy (matches _transform_counts exactly):
      anscombe, log1p  → clamp input to min=0 before the formula
      all others       → use raw pixel values (no pre-clamp)

    Masking fix for standardisation: ((x-mean)/std)*m so that masked pixels
    are 0.0, not -mean/std.  All other transforms already multiply by m last.
    """
    x = counts_f32.astype(np.float32, copy=False)  # no global clip
    m = masks_bool.astype(np.float32)

    if transform == "anscombe":
        xc = np.clip(x, 0.0, None)  # clamp only here
        t = 2.0 * np.sqrt(xc + 0.375)
        return ((t - mean) / np.sqrt(var)) * m

    if transform == "log1p":
        return np.log1p(np.clip(x, 0.0, None)) * m  # clamp only here

    if transform == "asinh":
        # scale = sqrt(var).clamp(min=1e-8) — raw x, no clamp
        scale = max(float(np.sqrt(var)), 1e-8)
        return np.arcsinh(x / scale) * m

    if transform == "log_softplus":
        # numerically stable softplus(x) = log(1+exp(x)) — raw x, no clamp
        sp = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
        return np.log(sp + 1e-8) * m

    if transform == "sqrt_squareplus":
        # raw x, no clamp — negative x gives a small positive squareplus value
        b = 4.0
        sqp = 0.5 * (x + np.sqrt(x * x + b))
        return np.sqrt(sqp + 1e-8) * m

    # standardization — masking fix: multiply by m after normalisation
    return ((x - mean) / np.sqrt(var)) * m


# ── Column helpers ────────────────────────────────────────────────────────────


def _is_string_col(v) -> bool:
    arr = (
        v.detach().cpu().numpy()
        if isinstance(v, torch.Tensor)
        else np.asarray(v)
    )
    return arr.dtype.kind in ("O", "U", "S")


def _to_np(v) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def _vectorized_isin(values: np.ndarray, sorted_ids: np.ndarray) -> np.ndarray:
    """Binary-search membership test.  O(N log M), no Python objects, no hash table."""
    pos = np.searchsorted(sorted_ids, values)
    valid = pos < len(sorted_ids)
    out = np.zeros(len(values), dtype=bool)
    out[valid] = sorted_ids[pos[valid]] == values[valid]
    return out


# ── Split helpers ─────────────────────────────────────────────────────────────


def _make_deterministic_split(
    refl_ids: np.ndarray,  # int64, (N,) — must be duplicate-free
    validation_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each refl_id to train or val deterministically.

    Assignment is based on position in sorted refl_id order combined with a
    seeded RNG — independent of input row order.  The same set of refl_ids
    with the same seed always produces identical is_train / is_val arrays.

    Returns:
        is_train: bool (N,)
        is_val:   bool (N,)
    """
    # Sort to a canonical order, making assignment row-order-independent
    sort_order = np.argsort(refl_ids, kind="stable")

    rng = np.random.default_rng(seed)
    fracs = rng.random(len(refl_ids))  # one float per sorted position

    is_val_sorted = fracs < validation_split
    is_train_sorted = ~is_val_sorted

    # Build inverse permutation: sorted-rank → original position
    inverse = np.empty_like(sort_order)
    inverse[sort_order] = np.arange(len(sort_order))

    return is_train_sorted[inverse], is_val_sorted[inverse]


def _save_split_csvs(
    out_dir: Path,
    refl_ids_valid: np.ndarray,  # int64 (N_valid,)
    is_train: np.ndarray,  # bool  (N_valid,)
    is_val: np.ndarray,  # bool  (N_valid,)
) -> None:
    """Export train_labels.csv and val_labels.csv to out_dir.

    These files follow the same schema as those produced by assign_labels()
    in prediction_writer.py and can be used as Mode A inputs on a future run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"train_ids": refl_ids_valid[is_train]}).write_csv(
        out_dir / "train_labels.csv"
    )
    pl.DataFrame({"val_ids": refl_ids_valid[is_val]}).write_csv(
        out_dir / "val_labels.csv"
    )
    logger.info(
        "Exported train_labels.csv (%d rows) and val_labels.csv (%d rows) to %s",
        int(is_train.sum()),
        int(is_val.sum()),
        out_dir,
    )


# ── ScanResult ────────────────────────────────────────────────────────────────


@dataclass
class ScanResult:
    valid_indices: np.ndarray  # int64,  (N_valid,)
    refl_ids_valid: np.ndarray  # int64,  (N_valid,) — for CSV export
    is_train: np.ndarray  # bool,   (N_valid,)
    is_val: np.ndarray  # bool,   (N_valid,)
    group_label: np.ndarray  # int64,  (N_all_rows,) — always set
    profile_group_label: np.ndarray | None  # int64, (N_all_rows,) or None
    stats: tuple[float, float]  # (mean, variance)
    transform: str
    numeric_columns: list[str]  # ordered list of numeric/bool column names
    string_columns: list[str]  # ordered list of string column names
    n_images: int | None
    n_valid: int
    n_train: int
    n_val: int
    split_meta: dict  # recorded in manifest.yaml


# ── Pass 1: scan, validate, plan ─────────────────────────────────────────────


def _pass1_scan(
    data_dir: Path,
    train_labels_path: str | None,  # Mode A; None triggers Mode B
    val_labels_path: str | None,  # Mode A
    validation_split: float | None,  # Mode B
    split_seed: int | None,  # Mode B
    min_valid_pixels: int,
    resolution_cutoff: float | None,
    n_bins: int,
    transform: str,  # resolved by caller before entering pass 1
    masks,  # numpy mmap
    reference: dict,
    spec: dict,
) -> ScanResult:
    from integrator.data_loaders.data_module import (
        DEFAULT_DS_COLS,
        _compute_valid_indices,
    )
    from integrator.io import data_path, load_data
    from integrator.utils.prepare_priors import _bin_by_resolution, _nbins_path

    # ── Stats from dataset.yaml ───────────────────────────────────────────────
    stats_key = "anscombe" if transform == "anscombe" else "raw"
    stats_arr = spec.get("stats", {}).get(stats_key)
    if stats_arr is None:
        raise KeyError(
            f"dataset.yaml is missing 'stats.{stats_key}'. "
            "Regenerate the dataset with integrator.make_shoeboxes."
        )
    mean, var = float(stats_arr[0]), float(stats_arr[1])
    if var <= 0:
        raise ValueError(f"stats.{stats_key} variance={var} is non-positive.")

    # ── Required-column verification ──────────────────────────────────────────
    for col in _REQUIRED_COLS:
        if col not in reference:
            raise KeyError(
                f"Required metadata column '{col}' is missing from metadata.npy. "
                "Re-run integrator.make_shoeboxes."
            )
    if not any(c in reference for c in _REQUIRED_ONE_OF_IMAGE):
        raise KeyError(
            f"At least one of {_REQUIRED_ONE_OF_IMAGE} must be present in metadata.npy."
        )
    for col in _VERIFIED_OPTIONAL:
        if col not in reference:
            logger.warning(
                "Optional metadata column '%s' not found; will not be saved.",
                col,
            )

    # ── Valid indices ─────────────────────────────────────────────────────────
    logger.info("Pass 1: computing valid indices...")
    valid_indices = _compute_valid_indices(
        masks=masks,
        reference=reference,
        min_valid_pixels=min_valid_pixels,
        resolution_cutoff=resolution_cutoff,
    )
    n_valid = len(valid_indices)
    logger.info("  valid rows: %d / %d total", n_valid, len(masks))
    if n_valid == 0:
        raise RuntimeError(
            "No valid rows after filtering. "
            "Check --min-valid-pixels and --resolution-cutoff."
        )

    # ── Duplicate refl_ids check (both modes) ─────────────────────────────────
    refl_ids_valid = _to_np(reference["refl_ids"])[valid_indices].astype(
        np.int64
    )
    unique_ids, id_counts = np.unique(refl_ids_valid, return_counts=True)
    dups = id_counts > 1
    if dups.any():
        examples = unique_ids[dups][:3].tolist()
        raise ValueError(
            f"metadata.npy contains {int(dups.sum())} duplicate refl_id(s) "
            f"in the valid set (examples: {examples}). "
            "This indicates a data integrity issue in the make_shoeboxes output."
        )

    # ── Split assignment ───────────────────────────────────────────────────────
    if train_labels_path is not None:
        # Mode A: read existing CSV split files.
        # Rows absent from both CSVs (e.g. test reflections) receive
        # is_train=False is_val=False and appear in predict-only; not an error.
        logger.info("Pass 1: reading train/val split labels (Mode A — CSV)...")
        train_ids_sorted = np.sort(
            pl.read_csv(train_labels_path)["train_ids"]
            .to_numpy()
            .astype(np.int64)
        )
        val_ids_sorted = np.sort(
            pl.read_csv(val_labels_path)["val_ids"].to_numpy().astype(np.int64)
        )
        overlap = np.intersect1d(train_ids_sorted, val_ids_sorted)
        if len(overlap) > 0:
            raise ValueError(
                f"train_labels and val_labels share {len(overlap)} refl_ids. "
                "They must not overlap. "
                "Regenerate train_labels.csv / val_labels.csv."
            )
        is_train = _vectorized_isin(refl_ids_valid, train_ids_sorted)
        is_val = _vectorized_isin(refl_ids_valid, val_ids_sorted)

        n_train_missing = len(train_ids_sorted) - int(is_train.sum())
        n_val_missing = len(val_ids_sorted) - int(is_val.sum())
        if n_train_missing > 0:
            logger.warning(
                "%d train refl_ids absent from valid_indices "
                "(filtered out by min_valid_pixels / resolution_cutoff).",
                n_train_missing,
            )
        if n_val_missing > 0:
            logger.warning(
                "%d val refl_ids absent from valid_indices "
                "(filtered out by min_valid_pixels / resolution_cutoff).",
                n_val_missing,
            )
        split_meta: dict = {
            "mode": "csv",
            "train_labels": str(Path(train_labels_path).resolve()),
            "val_labels": str(Path(val_labels_path).resolve()),
        }

    else:
        # Mode B: generate deterministic split.
        # Every valid reflection is assigned to exactly one split by construction
        # (is_train = ~is_val).
        logger.info(
            "Pass 1: generating deterministic split "
            "(validation_split=%.3f  seed=%d) (Mode B)...",
            validation_split,
            split_seed,
        )
        is_train, is_val = _make_deterministic_split(
            refl_ids_valid,
            float(validation_split),
            int(split_seed),
        )
        split_meta = {
            "mode": "generated",
            "validation_split": float(validation_split),
            "seed": int(split_seed),
        }

    n_train = int(is_train.sum())
    n_val = int(is_val.sum())

    # Overlap guard (both modes)
    if (is_train & is_val).any():
        raise RuntimeError(
            "Internal error: train and val sets overlap after split assignment."
        )

    # Coverage check — mode-specific behaviour
    n_unassigned = int((~is_train & ~is_val).sum())
    if train_labels_path is not None:
        # Mode A: unlabelled rows are permitted (test reflections etc.)
        if n_unassigned > 0:
            logger.info(
                "  %d reflection(s) in neither train nor val "
                "(not present in CSV files — will appear in predict only).",
                n_unassigned,
            )
    else:
        # Mode B: full coverage is mathematically guaranteed (is_train = ~is_val).
        # Any non-zero count indicates a bug in _make_deterministic_split.
        if n_unassigned != 0:
            raise RuntimeError(
                f"Generated split left {n_unassigned} valid reflections "
                "unassigned.  This should be impossible — please report a bug."
            )

    logger.info(
        "  train: %d  val: %d  other: %d",
        n_train,
        n_val,
        n_unassigned,
    )

    # ── Resolution-bin labels — always computed (n_bins >= 1) ─────────────────
    # group_label is required by _step() even when n_bins=1 (all-zeros array).
    logger.info("Pass 1: computing group_labels (%d bin(s))...", n_bins)
    group_label_all: np.ndarray
    profile_group_label_all: np.ndarray | None = None

    gl_path = _nbins_path("group_labels.npy", n_bins, data_dir)
    if data_path(gl_path) is not None:
        logger.info("  loading existing group_labels from %s", gl_path.name)
        group_label_all = _to_np(load_data(gl_path)).astype(np.int64)
    else:
        gl_tensor, _, actual = _bin_by_resolution(
            torch.as_tensor(_to_np(reference["d"]), dtype=torch.float32),
            n_bins,
        )
        group_label_all = gl_tensor.numpy().astype(np.int64)
        if actual != n_bins:
            logger.warning(
                "n_bins reduced %d -> %d (sparse resolution shells)",
                n_bins,
                actual,
            )

    pgl_path = _nbins_path("profile_group_labels.npy", n_bins, data_dir)
    if data_path(pgl_path) is not None:
        logger.info("  loading profile_group_labels from %s", pgl_path.name)
        profile_group_label_all = _to_np(load_data(pgl_path)).astype(np.int64)

    # ── Classify metadata columns: numeric vs. string ─────────────────────────
    # group_label and profile_group_label are handled separately (see below).
    base_cols = [
        c
        for c in DEFAULT_DS_COLS
        if c in reference and c not in ("group_label", "profile_group_label")
    ]
    numeric_columns: list[str] = []
    string_columns: list[str] = []
    for col in base_cols:
        if _is_string_col(reference[col]):
            string_columns.append(col)
        else:
            numeric_columns.append(col)

    for col in _EXTRA_SAVE_COLS:
        if (
            col in reference
            and col not in numeric_columns
            and col not in string_columns
        ):
            if _is_string_col(reference[col]):
                string_columns.append(col)
            else:
                numeric_columns.append(col)

    # group_label always saved (required by training step even for n_bins=1)
    numeric_columns.append("group_label")
    if profile_group_label_all is not None:
        numeric_columns.append("profile_group_label")

    # ── n_images: image_id only ───────────────────────────────────────────────
    # image_num is not suitable: it may have gaps or repeats across runs.
    # nn.Embedding requires a compact 0-indexed integer (image_id).
    # If image_id is absent n_images=None; train.py will raise a clear error
    # at startup if image_level_wilson=True is configured.
    n_images = None
    if "image_id" in reference:
        n_images = int(_to_np(reference["image_id"]).max()) + 1
    else:
        logger.warning(
            "image_id not found in metadata.npy; n_images will be None. "
            "Training will fail at startup if image_level_wilson=True."
        )

    return ScanResult(
        valid_indices=valid_indices,
        refl_ids_valid=refl_ids_valid,
        is_train=is_train,
        is_val=is_val,
        group_label=group_label_all,
        profile_group_label=profile_group_label_all,
        stats=(mean, var),
        transform=transform,
        numeric_columns=numeric_columns,
        string_columns=string_columns,
        n_images=n_images,
        n_valid=n_valid,
        n_train=n_train,
        n_val=n_val,
        split_meta=split_meta,
    )


# ── Pass 2: write chunks ──────────────────────────────────────────────────────


def _pass2_write_chunks(
    scan: ScanResult,
    counts,  # numpy mmap
    masks,  # numpy mmap
    reference: dict,
    chunk_size: int,
    out_dir: Path,
) -> list[dict]:
    n_valid = scan.n_valid
    n_chunks = (n_valid + chunk_size - 1) // chunk_size
    mean, var = scan.stats
    infos: list[dict] = []

    has_image_id = "image_id" in reference

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_valid)
        rows = scan.valid_indices[start:end]
        n = len(rows)

        logger.info("Pass 2: chunk %d / %d  (%d rows)...", i + 1, n_chunks, n)

        counts_c = np.array(counts[rows], copy=True)
        if counts_c.dtype == np.uint16:
            counts_c = counts_c.astype(np.int32)
        counts_f32 = counts_c.astype(np.float32)
        masks_bool = np.array(masks[rows], copy=True).astype(bool)
        shoebox_f32 = _apply_transform_np(
            counts_f32, masks_bool, mean, var, scan.transform
        )

        d = out_dir / f"chunk_{i:05d}"
        d.mkdir(parents=True, exist_ok=True)

        # ── Large pixel arrays — separate files (unchanged) ───────────────────
        np.save(d / "counts.npy", counts_f32)
        np.save(d / "shoebox.npy", shoebox_f32)
        np.save(d / "mask.npy", masks_bool)

        # ── All numeric + bool metadata — one consolidated file ───────────────
        meta_arrays: dict[str, np.ndarray] = {
            "row_id": rows.astype(np.int64),
            "is_train": scan.is_train[start:end],
            "is_val": scan.is_val[start:end],
        }
        for col in scan.numeric_columns:
            if col == "group_label":
                meta_arrays[col] = scan.group_label[rows]
            elif col == "profile_group_label":
                meta_arrays[col] = scan.profile_group_label[rows]
            else:
                meta_arrays[col] = _to_np(reference[col])[rows]
        np.savez(d / "metadata.npz", **meta_arrays)

        # ── String columns — one consolidated Parquet file ────────────────────
        if scan.string_columns:
            str_data: dict[str, list[str]] = {}
            for col in scan.string_columns:
                raw = _to_np(reference[col])
                str_data[col] = [str(raw[r]) for r in rows]
            pl.DataFrame(str_data).write_parquet(d / "strings.parquet")

        # ── Per-chunk image statistics ────────────────────────────────────────
        if has_image_id:
            chunk_img_ids = _to_np(reference["image_id"])[rows]
            unique_imgs = np.unique(chunk_img_ids)
            n_unique_images: int | None = int(len(unique_imgs))
            min_image_id: int | None = int(unique_imgs.min())
            max_image_id: int | None = int(unique_imgs.max())
        else:
            n_unique_images = None
            min_image_id = None
            max_image_id = None

        n_tr = int(scan.is_train[start:end].sum())
        n_va = int(scan.is_val[start:end].sum())
        infos.append(
            {
                "dir": f"chunk_{i:05d}",
                "n_rows": n,
                "n_train": n_tr,
                "n_val": n_va,
                "n_unique_images": n_unique_images,
                "min_image_id": min_image_id,
                "max_image_id": max_image_id,
            }
        )

    return infos


# ── Manifest ──────────────────────────────────────────────────────────────────


def _write_manifest(
    scan: ScanResult,
    chunk_infos: list[dict],
    chunk_size: int,
    n_bins: int,
    out_dir: Path,
    data_dir: Path,
) -> Path:
    """Write manifest.yaml to out_dir.

    data_dir is the original shoebox input directory (absolute path).
    It is stored as source_data_dir so integrator.predict can locate
    metadata.npy and dataset.yaml without a manual --data-dir argument.
    """
    manifest = {
        "version": 2,
        "source_data_dir": str(data_dir.resolve()),
        "chunk_size": chunk_size,
        "n_chunks": len(chunk_infos),
        "n_valid_rows": scan.n_valid,
        "n_train": scan.n_train,
        "n_val": scan.n_val,
        "n_images": scan.n_images,
        "transform": scan.transform,
        "n_bins": n_bins,
        "numeric_columns": scan.numeric_columns,
        "string_columns": scan.string_columns,
        "split": scan.split_meta,
        "chunks": chunk_infos,
    }
    path = out_dir / "manifest.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return path


# ── Background prior ─────────────────────────────────────────────────────────


def _write_bg_prior(
    data_dir: Path,
    out_dir: Path,
    scan: "ScanResult",
    reference: dict,
    n_bins: int,
) -> Path | None:
    """Compute and save the empirical per-bin background Gamma prior.

    Uses the same fitting function (_fit_per_bin_gamma) and math as
    prepare_per_bin_priors() in prepare_priors.py, ensuring that
    chunked training uses an identical prior to the rotation_data workflow.

    Background column priority (matches prepare_per_bin_priors exactly):
        1. background.mean
        2. background.sum.value

    group_label indexing: uses scan.group_label (all rows, matching the
    d-spacing arrays), consistent with how prepare_per_bin_priors operates
    on all rows of metadata["d"] before valid-row filtering.

    Saves bg_prior_{n_bins}.npy to out_dir (= chunk_dir).

    Returns:
        Path to the saved file, or None if no usable background column found.
    """
    import torch as _torch

    from integrator.io import save_data
    from integrator.utils.prepare_priors import _fit_per_bin_gamma, _nbins_path

    # Try background.mean first, then background.sum.value — same priority
    # as prepare_per_bin_priors (prepare_priors.py).
    bg_raw = None
    for key in ("background.mean", "background.sum.value"):
        v = reference.get(key)
        if v is not None:
            arr = _to_np(v).astype("float32")
            if arr.ndim == 1 and len(arr) > 0:
                bg_raw = arr
                logger.info(
                    "bg_prior: using column '%s' (%d rows)", key, len(arr)
                )
                break

    if bg_raw is None:
        logger.warning(
            "bg_prior: no usable background column found in metadata "
            "(tried 'background.mean', 'background.sum.value'). "
            "bg_prior_%d.npy will NOT be written. "
            "Add bg_rate and bg_concentration explicitly to loss.args in your "
            "config YAML to avoid a FileNotFoundError at training time.",
            n_bins,
        )
        return None

    bg_vals = _torch.as_tensor(bg_raw, dtype=_torch.float32)

    if int((bg_vals > 0).sum()) < 10:
        logger.warning(
            "bg_prior: fewer than 10 positive background values found. "
            "bg_prior_%d.npy will NOT be written.",
            n_bins,
        )
        return None

    # actual n_bins after potential reduction in _pass1_scan
    actual_n_bins = int(scan.group_label.max()) + 1
    group_labels_tensor = _torch.as_tensor(
        scan.group_label.astype("int64"), dtype=_torch.long
    )

    alphas, rates = _fit_per_bin_gamma(
        bg_vals, group_labels_tensor, actual_n_bins
    )

    payload: dict = (
        {"bg_concentration": alphas[0], "bg_rate": rates[0]}
        if actual_n_bins == 1
        else {"bg_concentration": alphas, "bg_rate": rates}
    )
    payload["n_bins"] = actual_n_bins

    prior_path = _nbins_path("bg_prior.npy", actual_n_bins, out_dir)
    saved = save_data(payload, prior_path)
    logger.info(
        "bg_prior: saved bg_prior_%d.npy to %s  "
        "(concentration=%.4f  rate=%.4f%s)",
        actual_n_bins,
        out_dir,
        float(alphas[0]),
        float(rates[0]),
        f"  [{actual_n_bins} bins]" if actual_n_bins > 1 else "",
    )
    return saved


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    from integrator.data_loaders.data_module import (
        _load_shoebox_array,
        _squeeze_last_axis,
    )
    from integrator.io import load_metadata, read_dataset_spec

    args = parse_args()

    # ── Guard: n_bins ─────────────────────────────────────────────────────────
    if args.n_bins < 1:
        raise ValueError("--n-bins must be at least 1")

    # ── Determine split mode ──────────────────────────────────────────────────
    has_csv = bool(args.train_labels) or bool(args.val_labels)
    has_gen = args.validation_split is not None

    if has_csv and has_gen:
        raise ValueError(
            "Specify either (--train-labels + --val-labels) "
            "OR --validation-split [--split-seed], not both."
        )
    if bool(args.train_labels) != bool(args.val_labels):
        raise ValueError(
            "--train-labels and --val-labels must be provided together."
        )

    if has_csv:
        split_mode = "csv"
    else:
        split_mode = "generated"
        if args.validation_split is None:
            args.validation_split = _DEFAULT_VALIDATION_SPLIT
        if not (0.0 < args.validation_split < 1.0):
            raise ValueError(
                f"--validation-split must be in (0, 1), got {args.validation_split}"
            )

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][
            min(args.verbose, 2)
        ],
        format="%(levelname)s %(name)s %(message)s",
    )

    data_dir = Path(args.data_dir).resolve()
    out_dir = (
        Path(args.out_dir).resolve() if args.out_dir else data_dir / "chunks"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("data_dir:    %s", data_dir)
    logger.info("out_dir:     %s", out_dir)
    logger.info("chunk_size:  %d", args.chunk_size)
    logger.info("split_mode:  %s", split_mode)

    counts = _squeeze_last_axis(_load_shoebox_array(data_dir / "counts.npy"))
    masks = _squeeze_last_axis(_load_shoebox_array(data_dir / "masks.npy"))
    logger.info("mmaps: counts=%s  masks=%s", counts.shape, masks.shape)

    spec = read_dataset_spec(data_dir)
    if spec is None:
        raise FileNotFoundError(
            f"dataset.yaml not found in {data_dir}. "
            "Run integrator.make_shoeboxes first."
        )

    transform = args.transform or (
        "anscombe" if spec.get("anscombe") else "standardization"
    )
    logger.info("transform:   %s", transform)

    reference = load_metadata(data_dir / "metadata.npy")

    # ── Pass 1 ───────────────────────────────────────────────────────────────
    scan = _pass1_scan(
        data_dir=data_dir,
        train_labels_path=args.train_labels,
        val_labels_path=args.val_labels,
        validation_split=args.validation_split,
        split_seed=args.split_seed,
        min_valid_pixels=args.min_valid_pixels,
        resolution_cutoff=args.resolution_cutoff,
        n_bins=args.n_bins,
        transform=transform,
        masks=masks,
        reference=reference,
        spec=spec,
    )

    # Mode B: export generated split labels alongside the chunks
    if split_mode == "generated":
        _save_split_csvs(
            out_dir=out_dir,
            refl_ids_valid=scan.refl_ids_valid,
            is_train=scan.is_train,
            is_val=scan.is_val,
        )

    n_chunks = (scan.n_valid + args.chunk_size - 1) // args.chunk_size
    logger.info(
        "Pass 1 complete: %d valid -> %d chunk(s)  (train=%d  val=%d  other=%d)",
        scan.n_valid,
        n_chunks,
        scan.n_train,
        scan.n_val,
        scan.n_valid - scan.n_train - scan.n_val,
    )

    # ── Pass 2 ───────────────────────────────────────────────────────────────
    chunk_infos = _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=args.chunk_size,
        out_dir=out_dir,
    )

    manifest_path = _write_manifest(
        scan=scan,
        chunk_infos=chunk_infos,
        chunk_size=args.chunk_size,
        n_bins=args.n_bins,
        out_dir=out_dir,
        data_dir=data_dir,
    )

    # ── Empirical background Gamma prior ──────────────────────────────────────
    # Fits the same Gamma MLE used by prepare_per_bin_priors() for rotation_data,
    # preserving Luis's original Wilson ELBO background prior math.
    # Saved as bg_prior_{n_bins}.npy in out_dir so the factory can load it at
    # training time via _get_loss_module() in factory_utils.py.
    _write_bg_prior(
        data_dir=data_dir,
        out_dir=out_dir,
        scan=scan,
        reference=reference,
        n_bins=args.n_bins,
    )

    print(f"\nDone.  {len(chunk_infos)} chunk(s) written to {out_dir}")
    print(
        f"Train: {scan.n_train:,}   Val: {scan.n_val:,}   "
        f"Other: {scan.n_valid - scan.n_train - scan.n_val:,}"
    )
    if split_mode == "generated":
        print(f"Split labels: {out_dir}/train_labels.csv  and  val_labels.csv")
    print(f"Manifest: {manifest_path}")
    print(f"\nTo train, set in your config:")
    print(f"  data_loader:")
    print(f"    name: chunked_rotation_data")
    print(f"    args:")
    print(f"      chunk_dir: {out_dir}")


if __name__ == "__main__":
    main()
