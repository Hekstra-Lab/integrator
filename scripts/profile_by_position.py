"""Sample reflections from different detector positions and compare
raw counts vs predicted profiles.

Selects strong reflections (qi > threshold) from different quadrants/regions
of the detector to visualize how the profile model captures position-dependent
spot shapes (e.g., radial elongation away from beam center).

Usage:
    uv run python scripts/profile_by_position.py <run_dir> \
        [--n-batches 50] [--qi-min 50] [--n-per-region 3]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile comparison by detector position"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-batches", type=int, default=50)
    parser.add_argument(
        "--qi-min",
        type=float,
        default=50,
        help="Minimum qi_mean to select (want visible spots)",
    )
    parser.add_argument("--n-per-region", type=int, default=3)
    parser.add_argument("--ckpt", type=str, default="last")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()

    meta = yaml.safe_load((run_dir / "run_paths.yaml").read_text())
    log_dir = Path(meta["wandb"]["log_dir"])

    from integrator.utils import (
        construct_data_loader,
        construct_integrator,
        inject_binning_labels,
        load_config,
    )

    cfg = load_config(str(run_dir / "config_log.yaml"))

    ckpt_dir = log_dir / "checkpoints"
    if args.ckpt == "last":
        ckpt_path = ckpt_dir / "last.ckpt"
        if not ckpt_path.exists():
            ckpts = sorted(ckpt_dir.glob("epoch*.ckpt"))
            ckpt_path = ckpts[-1]
    else:
        ckpt_path = ckpt_dir / f"epoch={int(args.ckpt):04d}.ckpt"

    print(f"Checkpoint: {ckpt_path.name}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = construct_integrator(cfg, skip_warmstart=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    data_loader = construct_data_loader(cfg)
    data_loader.setup()
    inject_binning_labels(data_loader, cfg)

    loader = data_loader.val_dataloader()

    H = int(
        cfg["data_loader"]["args"].get(
            "H", cfg["data_loader"]["args"].get("h", 25)
        )
    )
    W = int(
        cfg["data_loader"]["args"].get(
            "W", cfg["data_loader"]["args"].get("w", 25)
        )
    )
    shape = (H, W)

    # Collect data
    all_qi = []
    all_counts = []
    all_masks = []
    all_profiles = []
    all_xcal = []
    all_ycal = []
    all_dials = []
    all_d = []
    all_wl = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.n_batches:
                break
            counts, shoebox, mask, metadata = batch
            if torch.cuda.is_available():
                counts = counts.cuda()
                shoebox = shoebox.cuda()
                mask = mask.cuda()
                metadata = {
                    k: v.cuda() if isinstance(v, torch.Tensor) else v
                    for k, v in metadata.items()
                }

            outputs = model(counts, shoebox, mask, metadata)
            qi = outputs["qi"]
            fwd = outputs["forward_out"]

            all_qi.append(qi.mean.cpu().numpy())
            all_counts.append(counts.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            if "qp_mean" in fwd:
                all_profiles.append(fwd["qp_mean"].cpu().numpy())
            if "xyzcal.px.0" in metadata:
                all_xcal.append(metadata["xyzcal.px.0"].cpu().numpy())
                all_ycal.append(metadata["xyzcal.px.1"].cpu().numpy())
            if "intensity.sum.value" in metadata:
                all_dials.append(metadata["intensity.sum.value"].cpu().numpy())
            if "d" in metadata:
                all_d.append(metadata["d"].cpu().numpy())
            if "wavelength" in metadata:
                all_wl.append(metadata["wavelength"].cpu().numpy())

    qi_arr = np.concatenate(all_qi)
    counts_arr = np.concatenate(all_counts)
    masks_arr = np.concatenate(all_masks)
    profiles_arr = np.concatenate(all_profiles) if all_profiles else None
    dials_arr = np.concatenate(all_dials) if all_dials else None
    d_arr = np.concatenate(all_d) if all_d else None
    wl_arr = np.concatenate(all_wl) if all_wl else None

    if not all_xcal:
        print("No xyzcal.px columns in metadata - cannot group by position")
        return

    xcal = np.concatenate(all_xcal)
    ycal = np.concatenate(all_ycal)

    # Detector center (approximate)
    cx, cy = np.median(xcal), np.median(ycal)
    print(f"Detector center (median): ({cx:.0f}, {cy:.0f})")

    # Filter to strong reflections
    strong = qi_arr > args.qi_min
    print(
        f"Strong reflections (qi > {args.qi_min}): {strong.sum()} / {len(qi_arr)}"
    )

    if strong.sum() < 10:
        print("Not enough strong reflections. Lower --qi-min.")
        return

    # Define regions: 3x3 grid + inner/outer rings
    regions = {}
    dx = xcal - cx
    dy = ycal - cy
    r = np.sqrt(dx**2 + dy**2)
    r_med = np.median(r)

    # Quadrants at inner and outer radii
    for rname, rmask in [("inner", r < r_med), ("outer", r >= r_med)]:
        for qname, qmask in [
            ("top-left", (dx < 0) & (dy > 0)),
            ("top-right", (dx > 0) & (dy > 0)),
            ("bot-left", (dx < 0) & (dy < 0)),
            ("bot-right", (dx > 0) & (dy < 0)),
        ]:
            label = f"{rname} {qname}"
            combined = strong & rmask & qmask
            if combined.sum() >= args.n_per_region:
                regions[label] = combined

    print(f"Regions with enough reflections: {len(regions)}")

    # Sample from each region
    np.random.seed(args.seed)
    samples = []
    for label, mask in sorted(regions.items()):
        idx = np.where(mask)[0]
        chosen = np.random.choice(
            idx, size=min(args.n_per_region, len(idx)), replace=False
        )
        for c in chosen:
            samples.append((label, c))

    n_show = len(samples)
    print(f"Total reflections to plot: {n_show}")

    if n_show == 0:
        return

    # Plot
    ncols = min(n_show, 4)
    nrows = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        axes = [axes]

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    for plot_i, (label, idx) in enumerate(samples):
        r_val = plot_i // ncols
        c_val = (plot_i % ncols) * 2

        raw = (counts_arr[idx] * masks_arr[idx]).reshape(shape)
        prof = (
            profiles_arr[idx].reshape(shape)
            if profiles_arr is not None
            else np.zeros(shape)
        )

        ax_raw = axes[r_val][c_val]
        ax_prof = axes[r_val][c_val + 1]

        im_raw = ax_raw.imshow(raw, cmap="cividis", origin="lower")
        div_raw = make_axes_locatable(ax_raw)
        cax_raw = div_raw.append_axes("right", size="5%", pad=0.03)
        fig.colorbar(im_raw, cax=cax_raw)

        im_prof = ax_prof.imshow(prof, cmap="cividis", origin="lower")
        div_prof = make_axes_locatable(ax_prof)
        cax_prof = div_prof.append_axes("right", size="5%", pad=0.03)
        fig.colorbar(im_prof, cax=cax_prof)

        # Info
        lines = [label]
        lines.append(f"qi={qi_arr[idx]:.0f}")
        if dials_arr is not None:
            lines.append(f"dials={dials_arr[idx]:.0f}")
        lines.append(f"x={xcal[idx]:.0f} y={ycal[idx]:.0f}")
        if d_arr is not None and wl_arr is not None:
            lines.append(f"d={d_arr[idx]:.1f}Å λ={wl_arr[idx]:.3f}Å")

        ax_raw.set_title("\n".join(lines), fontsize=6)
        ax_prof.set_title("profile", fontsize=7)
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])
        ax_prof.set_xticks([])
        ax_prof.set_yticks([])

    for plot_i in range(n_show, nrows * ncols):
        r_val = plot_i // ncols
        c_val = (plot_i % ncols) * 2
        axes[r_val][c_val].set_visible(False)
        axes[r_val][c_val + 1].set_visible(False)

    fig.suptitle(
        f"Profiles by detector position - {n_show} strong reflections (qi > {args.qi_min})",
        fontsize=11,
    )
    fig.tight_layout()

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    out_path = plots_dir / "profiles_by_position.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
