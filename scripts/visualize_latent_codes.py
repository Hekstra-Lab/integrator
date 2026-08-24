"""Visualize profile latent codes h vs detector position.

Shows whether the positional encoding produces spatially structured
latent codes - i.e., do reflections at different detector positions
use the basis columns differently?

Usage:
    uv run python scripts/visualize_latent_codes.py \
        --run-dir /path/to/run_dir \
        --metadata /path/to/pytorch_data/metadata.pt

    uv run python scripts/visualize_latent_codes.py \
        --parquet /path/to/preds.parquet \
        --metadata /path/to/metadata.pt

Options:
    --epoch     Which epoch (default: latest)
    --out       Output directory (default: cwd)
    --gridsize  Hexbin resolution (default: 30)
    --image     Single image number (default: all)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from matplotlib.colors import Normalize


def parse_args():
    p = argparse.ArgumentParser(
        description="Hexbin of profile latent codes over detector position"
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--wandb-dir", type=Path)
    source.add_argument("--parquet", type=Path)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--image", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--gridsize", type=int, default=30)
    return p.parse_args()


def resolve_wandb_dir(args):
    if args.wandb_dir is not None:
        return args.wandb_dir
    if args.run_dir is not None:
        meta_path = args.run_dir / "run_paths.yaml"
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        return Path(meta["wandb"]["log_dir"])
    return None


def load_predictions(args):
    if args.parquet is not None:
        files = [args.parquet]
        label = args.parquet.stem
    else:
        wandb_dir = resolve_wandb_dir(args)
        pred_dir = (
            wandb_dir.parent / "predictions"
            if wandb_dir.name == "files"
            else wandb_dir / "predictions"
        )
        if not pred_dir.exists():
            pred_dir = wandb_dir / "predictions"

        epoch_dirs = sorted(pred_dir.glob("epoch_*"))
        if not epoch_dirs:
            raise FileNotFoundError(f"No epoch_* dirs in {pred_dir}")

        if args.epoch is not None:
            epoch_dirs = [
                d for d in epoch_dirs if d.name == f"epoch_{args.epoch:04d}"
            ]
        epoch_dir = epoch_dirs[-1]
        files = sorted(epoch_dir.glob("*.parquet"))
        label = epoch_dir.name

    available = pl.scan_parquet(files[0]).collect_schema().names()

    # Find qp_mean columns (the profile latent codes)
    # They may be stored as qp_mean (flattened) or individual columns
    select = ["refl_ids"]
    qp_cols = [c for c in available if c.startswith("qp_mean")]
    if not qp_cols:
        raise ValueError(
            "No qp_mean columns in predictions. "
            "Add 'qp_mean' to predict_keys in the config."
        )
    select.extend(sorted(qp_cols))

    df = pl.scan_parquet(files).select(select).collect()
    return df, label, qp_cols


def main():
    import torch

    args = parse_args()

    meta = torch.load(args.metadata, weights_only=False)
    df, label, qp_cols = load_predictions(args)

    refl_ids = df["refl_ids"].to_numpy().astype(np.int64)
    x = meta["xyzcal.px.0"][refl_ids].numpy().astype(np.float64)
    y = meta["xyzcal.px.1"][refl_ids].numpy().astype(np.float64)

    if args.image is not None:
        image_num = meta["image_num"][refl_ids].numpy().astype(np.int64)
        mask = image_num == args.image
        x, y = x[mask], y[mask]
        refl_ids = refl_ids[mask]
        df = df.filter(pl.col("refl_ids").is_in(refl_ids.tolist()))
        title_extra = f" - image {args.image}"
    else:
        title_extra = ""

    # Check if qp_mean is a single column (flattened profile) or latent code
    if len(qp_cols) == 1 and qp_cols[0] == "qp_mean":
        # It's the full profile mean, not the latent code
        # We can still visualize it via PCA
        profile_data = df["qp_mean"].to_numpy()
        if hasattr(profile_data[0], '__len__'):
            # It's a list/array column
            profiles = np.stack(profile_data)
        else:
            print("qp_mean is scalar - need latent codes or full profiles")
            return

        from sklearn.decomposition import PCA
        n_components = min(8, profiles.shape[1])
        pca = PCA(n_components=n_components)
        h = pca.fit_transform(profiles)
        component_labels = [
            f"PC{i} ({pca.explained_variance_ratio_[i]:.1%})"
            for i in range(n_components)
        ]
        n_latent = n_components
        suptitle_prefix = "Profile PCA"
    else:
        # Individual latent code columns
        h_cols = sorted(qp_cols)
        n_latent = len(h_cols)
        h = np.column_stack([
            df[c].to_numpy().astype(np.float64) for c in h_cols
        ])
        component_labels = [f"h[{i}]" for i in range(n_latent)]
        suptitle_prefix = "Profile latent codes"

    valid = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(h), axis=1)
    x, y, h = x[valid], y[valid], h[valid]

    # Plot hexbins of each latent dimension
    n_show = min(n_latent, 12)
    ncols = min(n_show, 4)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)

    for i in range(n_show):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        vals = h[:, i]
        vmax = np.percentile(np.abs(vals), 95)
        hb = ax.hexbin(
            x, y,
            C=vals,
            reduce_C_function=np.mean,
            gridsize=args.gridsize,
            mincnt=3,
            cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
        )
        fig.colorbar(hb, ax=ax)
        ax.set_title(component_labels[i], fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

    for i in range(n_show, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"{suptitle_prefix} over detector - {label}{title_extra}  "
        f"({len(x)} refls)",
        fontsize=12,
    )
    fig.tight_layout()

    out_dir = args.out or Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    img_tag = f"_img{args.image}" if args.image is not None else ""
    fname = out_dir / f"{label}_latent_codes{img_tag}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}")


if __name__ == "__main__":
    main()
