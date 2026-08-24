"""Scatter plot: model intensity vs DIALS intensity.prf.value, per HKL.

For each unique HKL (asu_id) in the model's buffer, compares:
    x: model posterior mean    E[I_h] = α_h / β_h
    y: DIALS weighted-mean     sigma w_i · intensity.prf.value_i / sigma w_i
                               (w_i = 1 / intensity.prf.variance_i)

Log-log Pearson correlation tells you whether the model's per-HKL intensities
track classical DIALS profile-fitted integration. The slope (~1 expected if
both are on the same scale) and the median ratio (= effective scale offset)
flag whether the model's MLP scale has absorbed magnitude.

Usage:
    uv run python scripts/compare_dials_intensities.py RUN_DIR

Output: RUN_DIR/diagnostics/dials_vs_model.png + console stats.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ====================================================================
# Reused helpers (kept inline so this script has no cross-script deps)
# ====================================================================


def load_run_metadata(run_dir: Path) -> tuple[dict, dict]:
    meta_path = run_dir / "run_paths.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_paths.yaml not found in {run_dir}")
    meta = yaml.safe_load(meta_path.read_text())
    cfg_path = Path(meta["config"])
    cfg = yaml.safe_load(cfg_path.read_text())
    return cfg, meta


def find_last_checkpoint(meta: dict) -> Path:
    log_dir = Path(meta["wandb"]["log_dir"])
    ckpt_dir = log_dir / "checkpoints"
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last.resolve()
    epoch_ckpts = sorted(ckpt_dir.glob("epoch=*.ckpt")) or sorted(ckpt_dir.glob("*.ckpt"))
    if not epoch_ckpts:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir}")
    return epoch_ckpts[-1].resolve()


def load_integrator(cfg: dict, checkpoint_path: Path):
    from integrator.utils.factory_utils import construct_integrator

    model = construct_integrator(cfg)
    ckpt = torch.load(
        checkpoint_path, weights_only=False, map_location="cpu"
    )
    state = ckpt["state_dict"]
    model_state = model.state_dict()
    compat = {
        k: v
        for k, v in state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    model.load_state_dict(compat, strict=False)
    model.eval()
    return model


def extract_model_intensity(integrator) -> tuple[np.ndarray, np.ndarray]:
    """Returns (E[I_h], seen) per asu_id from the merged posterior."""
    with torch.no_grad():
        q = integrator.get_merged_qi()
    alpha = q.concentration.detach().cpu().numpy().astype(np.float64)
    beta = q.rate.detach().cpu().numpy().astype(np.float64)
    seen = (
        integrator.buffer_seen.detach().cpu().numpy().astype(bool)
        if hasattr(integrator, "buffer_seen")
        else np.ones(len(alpha), dtype=bool)
    )
    I_model = np.where(
        seen & (beta > 0),
        alpha / np.clip(beta, 1e-12, None),
        np.nan,
    )
    return I_model, seen


# ====================================================================
# DIALS aggregation
# ====================================================================


def compute_dials_per_hkl(
    metadata_path: Path, n_hkl: int
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean of intensity.prf.value per asu_id."""
    logger.info("Loading metadata: %s", metadata_path)
    md = torch.load(metadata_path, weights_only=False, map_location="cpu")
    for k in ("asu_id", "intensity.prf.value", "intensity.prf.variance"):
        if k not in md:
            raise KeyError(f"metadata missing '{k}'")

    asu_ids = md["asu_id"].long().numpy()
    I_obs = md["intensity.prf.value"].float().numpy()
    var_obs = md["intensity.prf.variance"].float().numpy()

    good = (var_obs > 0) & np.isfinite(I_obs) & np.isfinite(var_obs)
    asu_ids = asu_ids[good]
    I_obs = I_obs[good]
    var_obs = var_obs[good]
    w = 1.0 / var_obs

    weights_sum = np.zeros(n_hkl, dtype=np.float64)
    weighted_I = np.zeros(n_hkl, dtype=np.float64)
    n_obs = np.zeros(n_hkl, dtype=np.int64)

    np.add.at(weights_sum, asu_ids, w)
    np.add.at(weighted_I, asu_ids, w * I_obs)
    np.add.at(n_obs, asu_ids, 1)

    I_dials = np.full(n_hkl, np.nan, dtype=np.float64)
    mask = weights_sum > 0
    I_dials[mask] = weighted_I[mask] / weights_sum[mask]
    return I_dials, n_obs


# ====================================================================
# Plot
# ====================================================================


def make_scatter(
    model_I: np.ndarray,
    dials_I: np.ndarray,
    n_obs: np.ndarray,
    out_path: Path,
    title_prefix: str = "",
) -> dict:
    valid = (
        np.isfinite(model_I)
        & np.isfinite(dials_I)
        & (model_I > 0)
        & (dials_I > 0)
    )
    if valid.sum() < 100:
        logger.warning(
            "Too few valid (positive, finite) points: %d", int(valid.sum())
        )
        return {}

    x = model_I[valid]
    y = dials_I[valid]
    nn = n_obs[valid]

    log_x = np.log10(x)
    log_y = np.log10(y)
    log_corr = float(np.corrcoef(log_x, log_y)[0, 1])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    median_ratio = float(np.median(y / x))

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        x, y,
        c=np.clip(nn, 1, None),
        s=4, alpha=0.3,
        cmap="viridis",
        norm=matplotlib.colors.LogNorm(),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model intensity  E[I_h] = α/β  (model units)")
    ax.set_ylabel("DIALS intensity.prf.value  (weighted mean per HKL)")

    lo = float(min(x.min(), y.min())) * 0.5
    hi = float(max(x.max(), y.max())) * 2.0
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    diag = np.array([lo, hi])
    ax.plot(diag, diag, "r--", alpha=0.5, label="y = x")

    fit_x = np.logspace(np.log10(lo), np.log10(hi), 100)
    fit_y = 10 ** (slope * np.log10(fit_x) + intercept)
    ax.plot(
        fit_x, fit_y, "g--", alpha=0.7,
        label=f"log y = {slope:.3f} · log x + {intercept:+.3f}",
    )

    plt.colorbar(sc, ax=ax, label="# observations per HKL")

    title = (
        f"{title_prefix}N = {int(valid.sum())} HKLs   "
        f"log-Pearson = {log_corr:.4f}\n"
        f"log-log slope = {slope:.3f} (1.0 ⇒ same scale)   "
        f"median(DIALS/model) = {median_ratio:.3g}"
    )
    ax.set_title(title)
    ax.legend(loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", out_path)

    return {
        "log_corr": log_corr,
        "log_slope": slope,
        "log_intercept": float(intercept),
        "median_ratio": median_ratio,
        "n_valid": int(valid.sum()),
    }


# ====================================================================
# Main
# ====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Scatter plot: model vs DIALS per-HKL intensities"
    )
    parser.add_argument("run_dir", type=Path, help="Training run directory")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Override checkpoint path (default: last.ckpt)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output PNG path (default: RUN_DIR/diagnostics/dials_vs_model.png)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_path = args.out or (run_dir / "diagnostics" / "dials_vs_model.png")

    logger.info("Run dir: %s", run_dir)
    cfg, meta = load_run_metadata(run_dir)
    ckpt = args.checkpoint or find_last_checkpoint(meta)
    logger.info("Checkpoint: %s", ckpt)

    integrator = load_integrator(cfg, ckpt)
    logger.info("Loaded %s", type(integrator).__name__)
    I_model, seen = extract_model_intensity(integrator)
    logger.info(
        "Model: %d HKLs seen / %d total (%.1f%%)",
        int(seen.sum()), len(seen), 100 * seen.mean(),
    )

    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])
    metadata_path = data_dir / cfg["data_loader"]["args"]["shoebox_file_names"][
        "reference"
    ]
    I_dials, n_obs = compute_dials_per_hkl(metadata_path, len(I_model))
    logger.info(
        "DIALS: %d HKLs have ≥1 observation", int(np.isfinite(I_dials).sum())
    )

    stats = make_scatter(I_model, I_dials, n_obs, out_path)
    print("\n=== Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
