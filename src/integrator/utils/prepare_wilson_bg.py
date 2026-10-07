"""Per-image Wilson B/G initialization from cctbx summation intensities.

Fits one (B, G) pair per image via ordinary least squares on the Wilson
plot linearization:

    log(I) = log(G) - B / (2 * d^2)
    y = a + m*x    where  x = 1/d^2,  y = log(I)
    B = -2 * m
    G = exp(a)

The result is cached as chunk_dir/wilson_bg_running_sums.npy and loaded by
MonochromaticWilsonLoss to initialize per-image nn.Embedding weights.
"""

import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Fallback values used for images with too few reflections or invalid fits.
# These are NOT controlled by init_log_B / init_log_G in the YAML — those
# govern the scalar init path (wilson_bg_init: null).
GLOBAL_B: float = 20.0
GLOBAL_G: float = 1000.0
MIN_REFLECTIONS: int = 10


def fit_wilson_bg_from_chunks(cfg: dict, *, force: bool = False) -> None:
    """Fit per-image Wilson B and G from cctbx summation intensities.

    Reads chunk metadata.npz files, accumulates running sums for OLS across
    all chunks (vectorized per chunk with np.add.at), solves the regression
    for all images simultaneously, and saves the result.

    Only runs when ALL of:
      - loss.name == "monochromatic_wilson"
      - loss.args.image_level_wilson == True
      - loss.args.wilson_bg_init == "cctbx"

    The result is cached; subsequent calls with force=False are no-ops.

    Args:
        cfg:   Full YAML config dict.
        force: Recompute even if the output file already exists.
    """
    from integrator.utils.factory_utils import _get_data_dir

    # ── Early-return for configs that don't request cctbx init ───────────────
    loss_name = cfg.get("loss", {}).get("name", "")
    if loss_name != "monochromatic_wilson":
        return
    loss_args = cfg.get("loss", {}).get("args", {})
    if not loss_args.get("image_level_wilson", False):
        return
    if loss_args.get("wilson_bg_init") != "cctbx":
        return

    chunk_dir = Path(_get_data_dir(cfg))
    out_path = chunk_dir / "wilson_bg_running_sums.npy"

    if out_path.exists() and not force:
        logger.info(
            "fit_wilson_bg_from_chunks: %s already exists, skipping.",
            out_path,
        )
        return

    # ── Read n_images from manifest.yaml ─────────────────────────────────────
    manifest_path = chunk_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.yaml found in {chunk_dir}.\n"
            "Run  integrator.preprocess  first."
        )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    n_images = int(manifest["n_images"])

    # ── Verify intensity column is present ───────────────────────────────────
    intensity_col = "intensity.sum.value"
    numeric_cols = manifest.get("numeric_columns", [])
    if isinstance(numeric_cols, dict):
        numeric_cols = list(numeric_cols.keys())
    if intensity_col not in numeric_cols:
        raise RuntimeError(
            f"Column '{intensity_col}' not found in manifest numeric_columns: "
            f"{numeric_cols}.\n"
            "Re-run integrator.preprocess with intensity.sum.value kept as a "
            "numeric column (add it to the --numeric-columns list)."
        )

    # ── Pre-allocate running sum arrays (float64 for numerical precision) ────
    count = np.zeros(n_images, dtype=np.float64)
    sum_x = np.zeros(n_images, dtype=np.float64)
    sum_y = np.zeros(n_images, dtype=np.float64)
    sum_x2 = np.zeros(n_images, dtype=np.float64)
    sum_xy = np.zeros(n_images, dtype=np.float64)

    # ── Scan chunks ──────────────────────────────────────────────────────────
    n_chunks_processed = 0
    for chunk_path in sorted(chunk_dir.glob("chunk_*")):
        meta_path = chunk_path / "metadata.npz"
        if not meta_path.exists():
            continue

        meta = np.load(meta_path, allow_pickle=False)

        image_id = meta["image_id"].astype(np.int64)
        d = meta["d"].astype(np.float64)
        I = meta[intensity_col].astype(np.float64)

        # Vectorized validity filter
        valid = np.isfinite(d) & np.isfinite(I) & (d > 0.0) & (I > 0.0)
        image_id = image_id[valid]
        d = d[valid]
        I = I[valid]

        if len(d) == 0:
            continue

        # Wilson linearization
        x = 1.0 / (d**2)
        y = np.log(I)

        # Accumulate running sums — np.add.at handles repeated image_ids
        # within a chunk without buffering (unlike +=).
        np.add.at(count, image_id, 1.0)
        np.add.at(sum_x, image_id, x)
        np.add.at(sum_y, image_id, y)
        np.add.at(sum_x2, image_id, x * x)
        np.add.at(sum_xy, image_id, x * y)

        n_chunks_processed += 1

    logger.info(
        "fit_wilson_bg_from_chunks: scanned %d chunks for %d images",
        n_chunks_processed,
        n_images,
    )

    # ── Batch OLS for all images simultaneously ───────────────────────────────
    denom = count * sum_x2 - sum_x**2
    valid_fit = (count >= MIN_REFLECTIONS) & (denom != 0.0)
    safe_denom = np.where(denom != 0.0, denom, 1.0)
    safe_count = np.where(count > 0.0, count, 1.0)

    m = np.where(
        valid_fit,
        (count * sum_xy - sum_x * sum_y) / safe_denom,
        0.0,
    )
    a = np.where(
        valid_fit,
        (sum_y - m * sum_x) / safe_count,
        0.0,
    )

    B_array = np.where(valid_fit, -2.0 * m, GLOBAL_B)
    G_array = np.where(valid_fit, np.exp(np.clip(a, -20.0, 20.0)), GLOBAL_G)

    # Fallback any non-finite or non-positive result to globals.
    # B > 0 check is intentional: the fitted Wilson B parameter is required
    # to be positive for this fixed-prior experiment.
    B_array = np.where(
        np.isfinite(B_array) & (B_array > 0.0), B_array, GLOBAL_B
    )
    G_array = np.where(
        np.isfinite(G_array) & (G_array > 0.0), G_array, GLOBAL_G
    )

    n_fallback = int((~valid_fit).sum())
    logger.info(
        "fit_wilson_bg_from_chunks: %d/%d images used fallback "
        "(B=%.1f, G=%.1f)",
        n_fallback,
        n_images,
        GLOBAL_B,
        GLOBAL_G,
    )
    logger.info(
        "B: min=%.3f  median=%.3f  max=%.3f",
        float(B_array.min()),
        float(np.median(B_array)),
        float(B_array.max()),
    )
    logger.info(
        "G: min=%.3f  median=%.3f  max=%.3f",
        float(G_array.min()),
        float(np.median(G_array)),
        float(G_array.max()),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        "B_array": B_array.astype(np.float32),
        "G_array": G_array.astype(np.float32),
        "n_images": n_images,
    }
    np.save(out_path, payload)
    logger.info("fit_wilson_bg_from_chunks: saved %s", out_path)
