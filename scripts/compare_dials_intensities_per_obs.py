"""Per-observation scatter: model vs DIALS, intensity and background.

For each shoebox in the dataset:

  Intensity panel (left)
      x: model integrated intensity  =  s_i · I_h_buffer[asu_id_i]
      y: DIALS intensity.prf.value

  Background panel (right)
      x: model background  =  qbg.mean  (per-obs from bg encoder)
      y: DIALS background.mean  (per-obs from metadata)

Linear axes by default (matches DIALS convention). The model's integrated
intensity `s · I_h` is *invariant under scale absorption* - if training
drove `s` up and `I_h` down by the same factor, this product is unchanged.
So slope ≈ 1 means model fits the data correctly, independent of how the
parameterization absorbed magnitude.

Usage:
    uv run python scripts/compare_dials_intensities_per_obs.py RUN_DIR
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ====================================================================
# Helpers
# ====================================================================


def load_run_metadata(run_dir: Path) -> tuple[dict, dict]:
    meta_path = run_dir / "run_paths.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_paths.yaml not found in {run_dir}")
    meta = yaml.safe_load(meta_path.read_text())
    cfg = yaml.safe_load(Path(meta["config"]).read_text())
    return cfg, meta


def find_last_checkpoint(meta: dict) -> Path:
    log_dir = Path(meta["wandb"]["log_dir"])
    ckpt_dir = log_dir / "checkpoints"
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last.resolve()
    ckpts = sorted(ckpt_dir.glob("epoch=*.ckpt")) or sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir}")
    return ckpts[-1].resolve()


def load_integrator(cfg, ckpt_path, device):
    from integrator.utils.factory_utils import construct_integrator

    model = construct_integrator(cfg)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    state = ckpt["state_dict"]
    model_state = model.state_dict()
    compat = {
        k: v for k, v in state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    model.load_state_dict(compat, strict=False)
    model.eval()
    model.to(device)
    return model


# ====================================================================
# Predict pass
# ====================================================================


@torch.no_grad()
def predict_per_obs(integrator, dataloader, device, max_batches=None):
    """Per-obs model predictions vs DIALS observations.

    Returns dict with arrays:
        I_model:  s · I_h_buffer  (per-obs predicted integrated intensity)
        I_dials:  intensity.prf.value
        bg_model: qbg.mean
        bg_dials: background.mean
        d:        d-spacing
        asu_id, scale, I_h: components (saved for inspection)
    """
    has_buffer = hasattr(integrator, "alpha_buffer") and hasattr(
        integrator, "buffer_seen"
    )

    rows: dict[str, list[np.ndarray]] = {
        k: [] for k in (
            "asu_id", "I_model", "I_dials", "bg_model", "bg_dials",
            "d", "scale", "I_h",
        )
    }

    n_done = 0
    for batch in tqdm(dataloader, desc="predict"):
        counts, shoebox, mask, metadata = batch
        shoebox = shoebox.to(device)
        mask = mask.to(device).float()

        B = shoebox.shape[0]
        shoebox_masked = shoebox * mask
        shoebox_reshaped = shoebox_masked.reshape(B, 1, *integrator.shoebox_shape)

        # Background encoder pass for qbg.mean
        x_k_bg = integrator.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = integrator.encoders["r_bg"](shoebox_reshaped)
        qbg = integrator.surrogates["qbg"](x_k_bg, x_r_bg)
        bg_model = qbg.mean.cpu().numpy().astype(np.float64)

        # Scale + per-HKL I_h from buffer
        scale = integrator._get_scale(metadata, device)  # (B,)

        asu_ids = metadata["asu_id"].long().to(device)
        d_per_obs = metadata["d"].to(device).float()

        if has_buffer:
            a = integrator.alpha_buffer[asu_ids]
            b = integrator.beta_buffer[asu_ids].clamp(min=1e-12)
            seen = integrator.buffer_seen[asu_ids]
            tau = integrator._wilson_tau(d_per_obs)
            wilson_mean = 1.0 / tau.clamp(min=1e-12)
            I_h = torch.where(seen, a / b, wilson_mean)
        else:
            tau = integrator._wilson_tau(d_per_obs)
            I_h = 1.0 / tau.clamp(min=1e-12)

        I_model = (scale * I_h).cpu().numpy().astype(np.float64)

        rows["asu_id"].append(asu_ids.cpu().numpy().astype(np.int64))
        rows["I_model"].append(I_model)
        rows["I_dials"].append(
            metadata["intensity.prf.value"].cpu().numpy().astype(np.float64)
        )
        rows["bg_model"].append(bg_model)
        if "background.mean" in metadata:
            rows["bg_dials"].append(
                metadata["background.mean"].cpu().numpy().astype(np.float64)
            )
        else:
            rows["bg_dials"].append(np.full(B, np.nan, dtype=np.float64))
        rows["d"].append(d_per_obs.cpu().numpy().astype(np.float64))
        rows["scale"].append(scale.cpu().numpy().astype(np.float64))
        rows["I_h"].append(I_h.cpu().numpy().astype(np.float64))

        n_done += 1
        if max_batches is not None and n_done >= max_batches:
            break

    return {k: np.concatenate(v) for k, v in rows.items()}


# ====================================================================
# Plot
# ====================================================================


def _scatter_panel(ax, x, y, c, xlabel, ylabel, title, log=False):
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    if log:
        valid &= (x > 0) & (y > 0)
    x, y, c = x[valid], y[valid], c[valid]
    if len(x) < 100:
        ax.set_title(f"{title}\n(too few points: {len(x)})")
        return {}

    sc = ax.scatter(x, y, c=c, s=1, alpha=0.25, cmap="viridis_r")
    plt.colorbar(sc, ax=ax, label="d-spacing (Å)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    pearson = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    median_ratio = float(np.median(y / np.clip(x, 1e-12, None)))

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        lo = max(min(x.min(), y.min()), 1e-6) * 0.5
        hi = max(x.max(), y.max()) * 2.0
    else:
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.02 * (hi - lo)
        lo -= pad
        hi += pad
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.plot([lo, hi], [lo, hi], "r--", alpha=0.5, lw=1, label="y = x  (perfect)")

    ax.set_title(
        f"{title}\nN={len(x):,}   r={pearson:.4f}   "
        f"slope={slope:.3f}   med(y/x)={median_ratio:.3g}"
    )
    ax.legend(loc="upper left", fontsize=9)

    return {
        "pearson": pearson,
        "slope": float(slope),
        "intercept": float(intercept),
        "median_ratio": median_ratio,
        "n_plotted": len(x),
    }


def make_two_panel(
    data: dict, out_path: Path, max_points: int | None = 200_000
) -> dict:
    if max_points and len(data["I_model"]) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(data["I_model"]), size=max_points, replace=False)
        data = {k: v[idx] for k, v in data.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Intensity: log-log (spans many orders of magnitude)
    stats_I = _scatter_panel(
        ax1,
        x=data["I_model"], y=data["I_dials"], c=data["d"],
        xlabel="Model intensity  s · I_h  (DIALS units)",
        ylabel="DIALS intensity.prf.value",
        title="Integrated intensity per observation (log-log)",
        log=True,
    )

    # Background: linear (narrow value range)
    stats_bg = _scatter_panel(
        ax2,
        x=data["bg_model"], y=data["bg_dials"], c=data["d"],
        xlabel="Model background  qbg.mean",
        ylabel="DIALS background.mean",
        title="Background per observation (linear)",
        log=False,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return {"intensity": stats_I, "background": stats_bg}


# ====================================================================
# Main
# ====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Per-obs scatter: model vs DIALS, intensity and background"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument(
        "--save-parquet", action="store_true",
        help="Write per_obs_predictions.parquet alongside the PNG",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_path = args.out or (run_dir / "diagnostics" / "model_vs_dials_per_obs.png")

    cfg, meta = load_run_metadata(run_dir)
    ckpt = args.checkpoint or find_last_checkpoint(meta)
    logger.info("Run dir: %s", run_dir)
    logger.info("Checkpoint: %s", ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    integrator = load_integrator(cfg, ckpt, device)
    logger.info("Loaded %s", type(integrator).__name__)

    from integrator.utils.factory_utils import construct_data_loader
    dm = construct_data_loader(cfg)
    dm.setup("fit")
    dl = dm.predict_dataloader()
    logger.info("Dataloader: %d batches", len(dl))

    data = predict_per_obs(
        integrator, dl, device, max_batches=args.max_batches
    )
    logger.info("Collected %d per-obs estimates", len(data["I_model"]))

    if args.save_parquet:
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            pq_path = out_path.with_name("per_obs_predictions.parquet")
            df.to_parquet(pq_path)
            logger.info("Wrote %s (%d rows)", pq_path, len(df))
        except ImportError:
            logger.warning("pandas/pyarrow not available - skipping parquet")

    stats = make_two_panel(data, out_path, max_points=args.max_points)
    print("\n=== Stats ===")
    for panel, s in stats.items():
        print(f"  [{panel}]")
        for k, v in s.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
