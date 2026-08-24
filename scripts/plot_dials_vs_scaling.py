"""Plot DIALS integrated intensities vs scaling model predicted intensities.

For each checkpoint, reconstructs the model's predicted observed intensity:
    I_model = s(frame) / lp * E[F²_hkl]
and compares against DIALS intensity.prf.value per observation.

Usage
-----
    python scripts/plot_dials_vs_scaling.py --run-dir <run_dir> [--epochs 4,49,98]
"""

import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _detect_table_type(sd):
    if "hkl_table.raw_fano.weight" in sd:
        return "gamma"
    if "hkl_table.raw_sigma.weight" in sd:
        return "amplitude"
    raise KeyError("Cannot detect HKL table type.")


def _extract_F_sq_per_hkl(sd, table_type):
    if table_type == "gamma":
        raw_mu = sd["hkl_table.raw_mu.weight"].cpu().squeeze(-1)
        raw_fano = sd["hkl_table.raw_fano.weight"].cpu().squeeze(-1)
        mu = torch.exp(raw_mu)
        fano = F.softplus(raw_fano) + 1e-6
        rate = 1.0 / fano
        k = mu * rate + 0.1
        return (k / rate).numpy()
    else:
        raw_mu = sd["hkl_table.raw_mu.weight"].cpu().squeeze(-1)
        raw_sigma = sd["hkl_table.raw_sigma.weight"].cpu().squeeze(-1)
        mu = torch.exp(raw_mu)
        sigma = F.softplus(raw_sigma) + 1e-6
        return (mu.pow(2) + sigma.pow(2)).numpy()


def _chebyshev(x, degree):
    T = [torch.ones_like(x)]
    if degree >= 1:
        T.append(x)
    for k in range(2, degree + 1):
        T.append(2 * x * T[k - 1] - T[k - 2])
    return T


def _eval_scale(sd, frame, x_det=None, y_det=None):
    """Evaluate the learned scale function from checkpoint parameters."""
    c = sd["scale_fn.c"].cpu()
    frame_mid = sd["scale_fn.frame_mid"].cpu()
    frame_half = sd["scale_fn.frame_half"].cpu()

    t = ((frame - frame_mid) / frame_half).clamp(-1.0, 1.0)

    if c.dim() == 1:
        phi = torch.stack(_chebyshev(t, len(c) - 1), dim=-1)
        return F.softplus(phi @ c).numpy()
    elif c.dim() == 2:
        d_frame, d_radius = c.shape
        beam_cx = sd["scale_fn.beam_cx"].cpu()
        beam_cy = sd["scale_fn.beam_cy"].cpu()
        r_mid = sd["scale_fn.r_mid"].cpu()
        r_half = sd["scale_fn.r_half"].cpu()

        r = torch.sqrt((x_det - beam_cx).pow(2) + (y_det - beam_cy).pow(2))
        rn = ((r - r_mid) / r_half).clamp(-1.0, 1.0)

        phi_t = torch.stack(_chebyshev(t, d_frame - 1), dim=-1)
        phi_r = torch.stack(_chebyshev(rn, d_radius - 1), dim=-1)
        out = (phi_t @ c * phi_r).sum(-1)
        return F.softplus(out).numpy()


def _load_obs_data(metadata_path):
    """Load per-observation data from metadata."""
    meta = torch.load(metadata_path, weights_only=False, map_location="cpu")
    return {
        "asu_id": meta["asu_id"].long().numpy(),
        "I_prf": meta["intensity.prf.value"].numpy(),
        "lp": meta["lp"].numpy(),
        "frame": meta["xyzcal.px.2"].float(),
        "x_det": meta["xyzcal.px.0"].float(),
        "y_det": meta["xyzcal.px.1"].float(),
        "d": meta["d"].numpy(),
    }


def plot_epoch(obs, F_sq_hkl, scale, epoch, out_dir, table_type):
    asu_id = obs["asu_id"]
    I_dials = obs["I_prf"]
    lp = obs["lp"]
    d = obs["d"]

    F_sq_obs = F_sq_hkl[asu_id]
    I_model = scale * F_sq_obs / np.maximum(lp, 1e-8)

    # Filter valid
    mask = (
        (I_dials > 0) & (I_model > 0)
        & np.isfinite(I_dials) & np.isfinite(I_model)
    )
    x = np.log10(I_model[mask])
    y = np.log10(I_dials[mask])
    d_masked = d[mask]

    cc = np.corrcoef(x, y)[0, 1]
    ratio = np.median(I_model[mask] / I_dials[mask])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: scatter
    ax = axes[0]
    ax.scatter(x, y, s=0.5, alpha=0.05, c="k", rasterized=True)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, "r--", lw=1, alpha=0.7)
    ax.set_xlabel("log10(Model s * F²)")
    ax.set_ylabel("log10(DIALS intensity.prf.value)")
    ax.set_title(f"Epoch {epoch}  CC={cc:.4f}  ratio={ratio:.3f}")
    ax.set_aspect("equal")

    # Panel 2: ratio vs resolution
    ax = axes[1]
    log_ratio = y - x
    ax.hexbin(1 / d_masked ** 2, log_ratio, gridsize=100, cmap="viridis", mincnt=1)
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("1/d² (Å⁻²)")
    ax.set_ylabel("log10(Model / DIALS)")
    ax.set_title("Scale ratio vs resolution")

    # Panel 3: ratio histogram
    ax = axes[2]
    ax.hist(log_ratio, bins=100, edgecolor="none", alpha=0.8)
    ax.axvline(0, color="r", ls="--", lw=1)
    ax.axvline(np.median(log_ratio), color="orange", ls="-", lw=1,
               label=f"median={np.median(log_ratio):.3f}")
    ax.set_xlabel("log10(Model / DIALS)")
    ax.set_ylabel("Count")
    ax.set_title(f"Ratio distribution")
    ax.legend()

    plt.tight_layout()
    out_path = out_dir / f"dials_vs_model_epoch_{epoch:04d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(
        "  Epoch %d: CC=%.4f  median_ratio=%.3f  n_obs=%d  -> %s",
        epoch, cc, ratio, mask.sum(), out_path.name,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot DIALS vs scaling model intensities."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--epochs", type=str, default=None,
        help="Comma-separated epochs (default: all checkpoints)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory for plots (default: run_dir/plots)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    meta = yaml.safe_load((run_dir / "run_paths.yaml").read_text())
    config_path = meta["config"]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data_loader"]["args"]["data_dir"])
    ref_name = (
        config["data_loader"]["args"]
        .get("shoebox_file_names", {})
        .get("reference", "metadata.pt")
    )
    metadata_path = data_dir / ref_name

    wandb_info = meta["wandb"]
    log_dir = Path(wandb_info["log_dir"])

    out_dir = args.out_dir or (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(log_dir.glob("**/epoch*.ckpt"))
    epoch_re = re.compile(r"epoch=(\d+)")

    if args.epochs:
        requested = {int(e) for e in args.epochs.split(",")}
        checkpoints = [
            c for c in checkpoints
            if int(epoch_re.search(c.name).group(1)) in requested
        ]

    logger.info("Loading observation data from %s", metadata_path)
    obs = _load_obs_data(metadata_path)
    logger.info("  %d observations, %d unique HKLs", len(obs["asu_id"]), obs["asu_id"].max() + 1)

    for ckpt in checkpoints:
        m = epoch_re.search(ckpt.name)
        epoch = int(m.group(1))

        sd = torch.load(ckpt, weights_only=False, map_location="cpu")["state_dict"]
        table_type = _detect_table_type(sd)
        F_sq_hkl = _extract_F_sq_per_hkl(sd, table_type)

        scale = _eval_scale(
            sd, obs["frame"],
            x_det=obs["x_det"], y_det=obs["y_det"],
        )

        plot_epoch(obs, F_sq_hkl, scale, epoch, out_dir, table_type)

    logger.info("Done. Plots saved to %s", out_dir)


if __name__ == "__main__":
    main()
