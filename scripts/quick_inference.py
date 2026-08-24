"""Quick inference on a checkpoint without running full prediction.

Loads the model from a run directory, runs a few batches, and reports
intensity and detection statistics.

Usage:
    uv run python scripts/quick_inference.py <run_dir> [--n-batches 5]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick inference from checkpoint"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument(
        "--ckpt", type=str, default="last", help="'last' or epoch number"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()

    meta = yaml.safe_load((run_dir / "run_paths.yaml").read_text())
    cfg = yaml.safe_load((run_dir / "config_log.yaml").read_text())
    log_dir = Path(meta["wandb"]["log_dir"])

    from integrator.utils import (
        construct_data_loader,
        construct_integrator,
        inject_binning_labels,
        load_config,
    )

    cfg = load_config(str(run_dir / "config_log.yaml"))

    # Find checkpoint
    ckpt_dir = log_dir / "checkpoints"
    if args.ckpt == "last":
        ckpt_path = ckpt_dir / "last.ckpt"
        if not ckpt_path.exists():
            ckpts = sorted(ckpt_dir.glob("epoch*.ckpt"))
            ckpt_path = ckpts[-1]
    else:
        ckpt_path = ckpt_dir / f"epoch={int(args.ckpt):04d}.ckpt"

    print(f"Checkpoint: {ckpt_path.name}")

    # Load model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = construct_integrator(cfg, skip_warmstart=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    # Load data
    data_loader = construct_data_loader(cfg)
    data_loader.setup()
    inject_binning_labels(data_loader, cfg)

    loader = data_loader.val_dataloader()

    # Get spectrum G and B for Wilson prior computation
    loss = model.loss
    has_spectrum = hasattr(loss, "spectrum")
    has_B = hasattr(loss, "get_B") or hasattr(loss, "raw_B")

    # Run inference
    all_qi = []
    all_pi = []
    all_qbg = []
    all_dials = []
    all_counts = []
    all_masks = []
    all_profiles = []
    all_d = []
    all_wl = []
    all_refl_ids = []

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
            qbg = outputs["qbg"]
            fwd = outputs["forward_out"]

            all_qi.append(qi.mean.cpu().numpy())
            all_qbg.append(qbg.mean.cpu().numpy())
            all_counts.append(counts.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

            if "qp_mean" in fwd:
                all_profiles.append(fwd["qp_mean"].cpu().numpy())

            if hasattr(qi, "pi"):
                all_pi.append(qi.pi.cpu().numpy())

            if "intensity.sum.value" in metadata:
                all_dials.append(metadata["intensity.sum.value"].cpu().numpy())
            if "d" in metadata:
                all_d.append(metadata["d"].cpu().numpy())
            if "wavelength" in metadata:
                all_wl.append(metadata["wavelength"].cpu().numpy())
            if "refl_ids" in metadata:
                all_refl_ids.append(metadata["refl_ids"].cpu().numpy())

    qi_arr = np.concatenate(all_qi)
    qbg_arr = np.concatenate(all_qbg)
    n = len(qi_arr)
    print(f"\nN reflections: {n:,}")

    print("\n=== qi_mean ===")
    for p in [0, 1, 5, 10, 25, 50, 75, 90, 99, 100]:
        print(f"  {p:>3d}%: {np.percentile(qi_arr, p):.2f}")

    print("\n=== qbg_mean ===")
    for p in [0, 25, 50, 75, 100]:
        print(f"  {p:>3d}%: {np.percentile(qbg_arr, p):.2f}")

    if all_pi:
        pi_arr = np.concatenate(all_pi)
        print("\n=== π (detection probability) ===")
        print(f"  mean:   {pi_arr.mean():.4f}")
        print(f"  std:    {pi_arr.std():.4f}")
        print(f"  min:    {pi_arr.min():.4f}")
        print(f"  max:    {pi_arr.max():.4f}")
        print("  Distribution:")
        for edge in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]:
            print(f"    π < {edge:.2f}: {(pi_arr < edge).mean() * 100:.1f}%")

        # Bimodality check
        below_02 = (pi_arr < 0.2).mean()
        above_08 = (pi_arr > 0.8).mean()
        middle = 1 - below_02 - above_08
        print("\n  Bimodality:")
        print(f"    π < 0.2: {below_02 * 100:.1f}%")
        print(f"    0.2 ≤ π ≤ 0.8: {middle * 100:.1f}%")
        print(f"    π > 0.8: {above_08 * 100:.1f}%")

        if all_dials:
            dials_arr = np.concatenate(all_dials)
            print("\n=== π vs DIALS ===")
            for label, mask in [
                ("dials < 0", dials_arr < 0),
                ("dials 0-10", (dials_arr >= 0) & (dials_arr < 10)),
                ("dials 10-100", (dials_arr >= 10) & (dials_arr < 100)),
                ("dials > 100", dials_arr >= 100),
            ]:
                if mask.sum() > 0:
                    print(
                        f"  {label:>12s}: N={mask.sum():>6,}  "
                        f"π_mean={pi_arr[mask].mean():.3f}  "
                        f"π_median={np.median(pi_arr[mask]):.3f}"
                    )

    # Plot: qi_mean vs DIALS intensity
    if all_dials:
        import matplotlib.pyplot as plt

        dials_arr = np.concatenate(all_dials) if not all_pi else dials_arr

        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 7))
        if all_pi:
            sc = ax.scatter(
                qi_arr,
                dials_arr,
                c=pi_arr,
                s=1,
                alpha=0.3,
                cmap="coolwarm",
                vmin=0,
                vmax=1,
                edgecolors="none",
            )
            fig.colorbar(sc, ax=ax, label="π", shrink=0.8)
        else:
            ax.scatter(
                qi_arr,
                dials_arr,
                s=1,
                alpha=0.3,
                c="steelblue",
                edgecolors="none",
            )

        lims = [
            min(
                np.percentile(qi_arr, 0.1), np.percentile(dials_arr, 0.1), -10
            ),
            max(np.percentile(qi_arr, 99.9), np.percentile(dials_arr, 99.9)),
        ]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="x = y")

        ax.set_xlabel("Predicted qi_mean")
        ax.set_ylabel("DIALS intensity.sum.value")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f"Quick inference - {n:,} reflections")

        out_path = plots_dir / "qi_vs_dials.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved plot to {out_path}")
        plt.close(fig)

    # Diagnostic shoebox plots for weak/boundary reflections
    if all_profiles and all_dials:
        import matplotlib.pyplot as plt

        dials_arr = (
            np.concatenate(all_dials)
            if "dials_arr" not in dir()
            else dials_arr
        )
        counts_arr = np.concatenate(all_counts)
        masks_arr = np.concatenate(all_masks)
        profiles_arr = np.concatenate(all_profiles)
        d_arr = np.concatenate(all_d) if all_d else None
        wl_arr = np.concatenate(all_wl) if all_wl else None
        pi_arr = np.concatenate(all_pi) if all_pi else None

        H = cfg["data_loader"]["args"].get(
            "H", cfg["data_loader"]["args"].get("h", 25)
        )
        W = cfg["data_loader"]["args"].get(
            "W", cfg["data_loader"]["args"].get("w", 25)
        )
        shape = (int(H), int(W))

        # Compute Wilson prior τ per reflection
        tau_arr = None
        if (
            d_arr is not None
            and wl_arr is not None
            and hasattr(loss, "spectrum")
        ):
            import torch.nn.functional as Fn

            wl_t = torch.tensor(wl_arr, dtype=torch.float32)
            device = next(loss.parameters()).device
            log_G = (
                loss.spectrum.get_log_G(wl_t.to(device)).detach().cpu().numpy()
            )
            G = np.exp(log_G)
            if hasattr(loss, "get_B"):
                B = loss.get_B().item()
            elif hasattr(loss, "raw_B"):
                B = Fn.softplus(loss.raw_B).item()
            else:
                B = 0
            s_sq = 1.0 / (4.0 * np.clip(d_arr, 1e-6, None) ** 2)
            tau_arr = (1.0 / G) * np.exp(2.0 * B * s_sq)

        # Select: weak (qi < 5) and boundary (5 < qi < 20)
        weak_idx = np.where(qi_arr < 5)[0]
        boundary_idx = np.where((qi_arr >= 5) & (qi_arr < 20))[0]

        np.random.seed(42)
        n_each = 8
        if len(weak_idx) > n_each:
            weak_idx = np.random.choice(weak_idx, n_each, replace=False)
        if len(boundary_idx) > n_each:
            boundary_idx = np.random.choice(
                boundary_idx, n_each, replace=False
            )

        sample_idx = np.concatenate([weak_idx, boundary_idx])
        sample_idx.sort()
        n_show = len(sample_idx)

        if n_show > 0:
            ncols = min(n_show, 4)
            nrows = (n_show + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols * 2, figsize=(4 * ncols, 3.5 * nrows)
            )
            if nrows == 1:
                axes = [axes]

            for plot_i, idx in enumerate(sample_idx):
                r = plot_i // ncols
                c = (plot_i % ncols) * 2

                raw = (counts_arr[idx] * masks_arr[idx]).reshape(shape)
                prof = profiles_arr[idx].reshape(shape)

                ax_raw = axes[r][c]
                ax_prof = axes[r][c + 1]

                from mpl_toolkits.axes_grid1 import make_axes_locatable

                im_raw = ax_raw.imshow(raw, cmap="cividis", origin="lower")
                div_raw = make_axes_locatable(ax_raw)
                cax_raw = div_raw.append_axes("right", size="5%", pad=0.03)
                fig.colorbar(im_raw, cax=cax_raw)

                im_prof = ax_prof.imshow(prof, cmap="cividis", origin="lower")
                div_prof = make_axes_locatable(ax_prof)
                cax_prof = div_prof.append_axes("right", size="5%", pad=0.03)
                fig.colorbar(im_prof, cax=cax_prof)

                # Build info text
                lines = [f"qi={qi_arr[idx]:.1f}  dials={dials_arr[idx]:.1f}"]
                if pi_arr is not None:
                    lines.append(f"π={pi_arr[idx]:.3f}")
                if d_arr is not None and wl_arr is not None:
                    lines.append(f"d={d_arr[idx]:.1f}Å  λ={wl_arr[idx]:.3f}Å")
                if tau_arr is not None:
                    lines.append(
                        f"E[I]={1 / tau_arr[idx]:.1f}  τ={tau_arr[idx]:.4f}"
                    )
                lines.append(f"bg={qbg_arr[idx]:.1f}")

                ax_raw.set_title("\n".join(lines), fontsize=5)
                ax_prof.set_title("profile", fontsize=7)
                ax_raw.set_xticks([])
                ax_raw.set_yticks([])
                ax_prof.set_xticks([])
                ax_prof.set_yticks([])

            for plot_i in range(n_show, nrows * ncols):
                r = plot_i // ncols
                c = (plot_i % ncols) * 2
                axes[r][c].set_visible(False)
                axes[r][c + 1].set_visible(False)

            fig.suptitle(
                f"Weak (qi<5) and boundary (5<qi<20) reflections - {n_show} shown",
                fontsize=10,
            )
            fig.tight_layout()

            plots_dir = run_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            out_path = plots_dir / "weak_boundary_shoeboxes.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved diagnostic plot to {out_path}")
            plt.close(fig)


if __name__ == "__main__":
    main()
