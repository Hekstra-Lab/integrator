"""Compare predicted profiles vs raw observed shoeboxes.

Takes the run-dir (containing run_paths.yaml and config_log.yaml)
and loads predictions + raw counts to compare.

Usage:
    uv run python scripts/compare_profiles.py <run_dir> \
        [--epoch EPOCH] [--n-samples 16] [--out profiles.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare predicted profiles vs raw shoeboxes"
    )
    parser.add_argument(
        "run_dir", type=Path, help="Run directory with run_paths.yaml"
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Epoch to visualize (default: latest)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=16,
        help="Number of random reflections to plot",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def load_run_info(run_dir: Path) -> tuple[dict, Path]:
    """Load config and find prediction directory from run_paths.yaml."""
    meta = yaml.safe_load((run_dir / "run_paths.yaml").read_text())
    cfg = yaml.safe_load((run_dir / "config_log.yaml").read_text())
    log_dir = Path(meta["wandb"]["log_dir"])
    pred_dir = log_dir.parent / "predictions"
    return cfg, pred_dir


def find_epoch_dir(pred_dir: Path, epoch: int | None) -> Path:
    epoch_dirs = sorted(d for d in pred_dir.glob("epoch_*") if d.is_dir())
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch_* dirs in {pred_dir}")
    if epoch is not None:
        target = pred_dir / f"epoch_{epoch:04d}"
        if not target.exists():
            raise FileNotFoundError(f"{target} not found")
        return target
    return epoch_dirs[-1]


def load_predictions(epoch_dir: Path, n_samples: int, seed: int):
    """Load prediction parquets and sample random reflections."""
    parquets = sorted(epoch_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquets in {epoch_dir}")

    df = pl.read_parquet(parquets)

    if "qp_mean" not in df.columns:
        raise ValueError(
            "No qp_mean column in parquets. "
            "Ensure 'qp_mean' is in predict_keys in the config."
        )

    np.random.seed(seed)
    idx = np.random.choice(len(df), size=min(n_samples, len(df)), replace=False)
    idx.sort()

    sub = df[idx]

    # qp_mean may be a single list/array column or exploded into qp_mean.0, etc.
    qp_col = sub["qp_mean"]
    if qp_col.dtype == pl.List:
        profiles = np.stack(qp_col.to_list())
    elif hasattr(qp_col.dtype, "fields"):
        profiles = sub.select(pl.col("qp_mean").struct.unnest()).to_numpy()
    else:
        profiles = qp_col.to_numpy()

    refl_ids = sub["refl_ids"].to_numpy() if "refl_ids" in df.columns else idx

    # Extract scalar predictions
    qi_mean = sub["qi_mean"].to_numpy() if "qi_mean" in df.columns else None
    dials_I = sub["intensity.sum.value"].to_numpy() if "intensity.sum.value" in df.columns else None
    d_vals = sub["d"].to_numpy() if "d" in df.columns else None
    wavelength = sub["wavelength"].to_numpy() if "wavelength" in df.columns else None

    return profiles, refl_ids, idx, qi_mean, dials_I, d_vals, wavelength


def load_raw_counts(cfg: dict, refl_ids):
    """Load raw counts and masks for specific reflection IDs."""
    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])
    sfn = cfg["data_loader"]["args"]["shoebox_file_names"]

    counts_file = data_dir / sfn["counts"]
    masks_file = data_dir / sfn["masks"]

    refl_ids = np.asarray(refl_ids, dtype=int)

    if counts_file.suffix == ".npy":
        counts = np.load(counts_file, mmap_mode="r")[refl_ids].copy()
        masks = np.load(masks_file, mmap_mode="r")[refl_ids].copy()
    else:
        counts = torch.load(counts_file, weights_only=False)[refl_ids].numpy()
        masks = torch.load(masks_file, weights_only=False)[refl_ids].numpy()

    if counts.ndim > 2:
        counts = counts.squeeze(-1)
    if masks.ndim > 2:
        masks = masks.squeeze(-1)
    counts = counts.astype(np.float32)
    masks = masks.astype(np.float32)
    return counts, masks


def _chebyshev(x, degree):
    T = [np.ones_like(x)]
    if degree >= 1:
        T.append(x)
    for k in range(2, degree + 1):
        T.append(2 * x * T[k - 1] - T[k - 2])
    return T


def compute_tau_from_checkpoint(run_dir, d_vals, wavelength_vals):
    """Compute per-reflection Wilson prior rate τ from the latest checkpoint."""
    import torch.nn.functional as F

    meta = yaml.safe_load((run_dir / "run_paths.yaml").read_text())
    log_dir = Path(meta["wandb"]["log_dir"])
    ckpt_dir = log_dir / "checkpoints"

    last_ckpt = ckpt_dir / "last.ckpt"
    if not last_ckpt.exists():
        ckpts = sorted(ckpt_dir.glob("epoch*.ckpt"))
        if not ckpts:
            return None, None
        last_ckpt = ckpts[-1]

    sd = torch.load(last_ckpt, map_location="cpu", weights_only=False)["state_dict"]

    # Get B
    if "loss.raw_B" in sd:
        B = F.softplus(sd["loss.raw_B"]).item()
    elif "loss.q_log_B_loc" in sd:
        B = F.softplus(sd["loss.q_log_B_loc"]).item()
    else:
        return None, None

    s_sq = 1.0 / (4.0 * np.clip(d_vals, 1e-6, None) ** 2)

    # Get G per reflection
    if "loss.spectrum.c" in sd or "loss.spectrum.coeff_loc" in sd:
        c = sd.get("loss.spectrum.c", sd.get("loss.spectrum.coeff_loc")).numpy()
        lam_mid = sd["loss.spectrum.lam_mid"].item()
        lam_scale = sd["loss.spectrum.lam_scale"].item()
        degree = len(c) - 1
        x = (wavelength_vals - lam_mid) / lam_scale
        phi = np.stack(_chebyshev(x, degree), axis=-1)
        log_G = phi @ c
        G = np.exp(log_G)
    elif "loss.q_log_K_loc" in sd:
        # Binned - need bin edges
        if "loss.wavelength_bin_edges" in sd:
            edges = sd["loss.wavelength_bin_edges"].numpy()
            loc = sd["loss.q_log_K_loc"].numpy()
            bins = np.searchsorted(edges, wavelength_vals, side="right") - 1
            bins = np.clip(bins, 0, len(loc) - 1)
            G = np.exp(loc[bins])
        else:
            G = np.exp(sd["loss.q_log_K_loc"].item())
    else:
        return None, None

    tau = (1.0 / G) * np.exp(2.0 * B * s_sq)
    prior_mean = 1.0 / tau  # E[I] under Exp(tau) prior

    return tau, prior_mean


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()

    cfg, pred_dir = load_run_info(run_dir)
    epoch_dir = find_epoch_dir(pred_dir, args.epoch)
    epoch_num = epoch_dir.name.replace("epoch_", "")

    profiles, refl_ids, _, qi_mean, dials_I, d_vals, wavelength = load_predictions(
        epoch_dir, args.n_samples, args.seed
    )
    counts, masks = load_raw_counts(cfg, refl_ids)

    # Compute Wilson prior τ and expected intensity
    tau, prior_mean = (None, None)
    if d_vals is not None and wavelength is not None:
        tau, prior_mean = compute_tau_from_checkpoint(run_dir, d_vals, wavelength)

    H = int(cfg["data_loader"]["args"].get("H", cfg["data_loader"]["args"].get("h", 21)))
    W = int(cfg["data_loader"]["args"].get("W", cfg["data_loader"]["args"].get("w", 21)))
    shape = (H, W)

    n = len(profiles)

    # Normalize observed counts to profiles
    obs = counts * masks
    obs_sum = obs.sum(axis=-1, keepdims=True).clip(min=1)
    obs_norm = obs / obs_sum

    # Plot grid: observed | predicted | residual
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols * 3, figsize=(2.5 * ncols * 3, 3.2 * nrows)
    )
    if nrows == 1:
        axes = [axes]

    for i in range(n):
        r = i // ncols
        c = (i % ncols) * 3

        raw_img = (counts[i] * masks[i]).reshape(shape)
        obs_img = obs_norm[i].reshape(shape)
        pred_img = profiles[i].reshape(shape)
        resid_img = obs_img - pred_img

        rmax = max(abs(resid_img.min()), abs(resid_img.max()), 1e-8)

        ax_obs = axes[r][c]
        ax_pred = axes[r][c + 1]
        ax_res = axes[r][c + 2]

        ax_obs.imshow(raw_img, cmap="viridis", origin="lower")
        title_parts = [f"id={int(refl_ids[i])}"]
        if qi_mean is not None:
            title_parts.append(f"qi={qi_mean[i]:.1f}")
        if dials_I is not None:
            title_parts.append(f"dials={dials_I[i]:.1f}")
        if prior_mean is not None:
            title_parts.append(f"prior E[I]={prior_mean[i]:.1f}")
        if tau is not None:
            title_parts.append(f"τ={tau[i]:.4f}")
        ax_obs.set_title("\n".join(title_parts), fontsize=5)

        ax_pred.imshow(pred_img, cmap="viridis", origin="lower")
        pred_title = "predicted"
        if d_vals is not None and wavelength is not None:
            pred_title += f"\nd={d_vals[i]:.1f}Å  λ={wavelength[i]:.3f}Å"
        ax_pred.set_title(pred_title, fontsize=6)

        ax_res.imshow(resid_img, cmap="RdBu_r", vmin=-rmax, vmax=rmax, origin="lower")
        ax_res.set_title("residual", fontsize=7)

        for ax in (ax_obs, ax_pred, ax_res):
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide unused
    for i in range(n, nrows * ncols):
        r = i // ncols
        c = (i % ncols) * 3
        for j in range(3):
            axes[r][c + j].set_visible(False)

    fig.suptitle(
        f"Profile comparison - epoch {epoch_num} - {n} reflections",
        fontsize=11,
    )
    fig.tight_layout()

    out = args.out or f"profile_comparison_epoch{epoch_num}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
