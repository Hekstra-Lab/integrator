"""Find reflections with bright-pixel outliers and dump their shoeboxes
as image montages. Lets us eyeball whether those pixels are saturation/
ice/hot-pixel artifacts, or real high-intensity signal we don't want to
throw away.

For each candidate reflection we plot:
  - one panel per z-frame (D panels in a row)
  - log color scale to make tail values readable alongside background
  - a red 'X' on every pixel above `--threshold`
  - overlay of the DIALS mask (Valid & ~Overlapped) as a translucent contour
A title strip on top reports refl_id, chunk index, shape, max raw count,
DIALS valid-pixel count, and resolution (if metadata.pt exists).

Run:
    uv run python scripts/visualize_outlier_shoeboxes.py \
        --chunks  /n/.../pytorch_data_dials/chunks \
        --metadata /n/.../pytorch_data_dials/metadata.pt \
        --threshold 65535 \
        --out-dir /n/.../pytorch_data_dials/outlier_montages \
        --max-shown 30
"""

import argparse
from pathlib import Path

import numpy as np


def find_outlier_refls(chunks_dir: Path, threshold: float, max_shown: int):
    """Return a list of (chunk_idx, local_idx, max_value, n_above)
    for reflections that contain at least one pixel > threshold,
    sorted by max_value descending. Caps at max_shown for plotting."""
    out = []
    chunk_paths = sorted(chunks_dir.glob("chunk_*.npz"))
    for ci, p in enumerate(chunk_paths):
        with np.load(p) as npz:
            data = npz["data"]
            offsets = npz["offsets"]
        n = len(offsets) - 1
        for i in range(n):
            s, e = int(offsets[i]), int(offsets[i + 1])
            sl = data[s:e]
            mx = sl.max() if sl.size else 0
            if mx > threshold:
                n_above = int((sl > threshold).sum())
                out.append((ci, i, int(mx), n_above))
    out.sort(key=lambda t: -t[2])
    return out[:max_shown]


def load_one(chunk_path: Path, local_idx: int):
    """Load a single reflection's 3D data + masks from a chunk."""
    with np.load(chunk_path) as npz:
        offsets = npz["offsets"]
        shapes = npz["shapes"]
        data = npz["data"]
        mask = npz["mask"]
        fg = npz["foreground"] if "foreground" in npz.files else None
        bg = npz["background"] if "background" in npz.files else None
        bbox = npz["bboxes"][local_idx]
        refl_id = int(npz["refl_ids"][local_idx])
    s, e = int(offsets[local_idx]), int(offsets[local_idx + 1])
    D, H, W = (int(x) for x in shapes[local_idx])
    arr = data[s:e].reshape(D, H, W)
    m = mask[s:e].reshape(D, H, W).astype(bool)
    f = fg[s:e].reshape(D, H, W).astype(bool) if fg is not None else None
    b = bg[s:e].reshape(D, H, W).astype(bool) if bg is not None else None
    return arr, m, f, b, bbox, refl_id, (D, H, W)


def plot_one(arr, mask, fg, refl_id, chunk_idx, shape_dhw, bbox, threshold,
             d_resolution, out_path):
    """Plot all z-frames of one shoebox in a row, with overlays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    D, H, W = shape_dhw
    fig, axes = plt.subplots(1, D, figsize=(2.6 * D, 3.2),
                             squeeze=False)
    axes = axes[0]

    vmin = max(1.0, float(arr[arr > 0].min()) if (arr > 0).any() else 1.0)
    vmax = max(2.0, float(arr.max()))

    for k in range(D):
        ax = axes[k]
        im = ax.imshow(np.maximum(arr[k], 1).astype(float),
                       norm=LogNorm(vmin=vmin, vmax=vmax),
                       cmap="viridis")
        # Mark hot pixels with a red x
        ys, xs = np.where(arr[k] > threshold)
        if ys.size:
            ax.scatter(xs, ys, marker="x", c="red", s=40, linewidths=1.4)
        # Outline DIALS valid mask as contour
        ax.contour(mask[k].astype(float), levels=[0.5],
                   colors="white", linewidths=0.8)
        # Foreground bit as dashed contour, if present
        if fg is not None:
            ax.contour(fg[k].astype(float), levels=[0.5],
                       colors="orange", linewidths=0.6, linestyles="--")
        ax.set_title(f"frame {k}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="counts (log)")

    title = (f"refl_id={refl_id}  chunk={chunk_idx}  shape=(D={D}, H={H}, W={W})  "
             f"max={int(arr.max())}  valid_px={int(mask.sum())}/{arr.size}  "
             f"px_above_{threshold:g}={int((arr > threshold).sum())}")
    if d_resolution is not None:
        title += f"  d={d_resolution:.2f} Å"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", required=True, help="path to chunks/ dir")
    ap.add_argument("--metadata", default=None,
                    help="optional metadata.pt for resolution lookup")
    ap.add_argument("--threshold", type=float, default=65535.0,
                    help="pixel-count threshold; reflections with any pixel "
                         "above this are flagged")
    ap.add_argument("--max-shown", type=int, default=30,
                    help="cap on number of montages saved (sorted by brightest)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    chunks_dir = Path(args.chunks)
    out_dir = Path(args.out_dir)

    print(f"Scanning chunks under {chunks_dir} for pixels > {args.threshold} ...")
    flagged = find_outlier_refls(chunks_dir, args.threshold, args.max_shown)

    print(f"Found {len(flagged)} reflections (showing brightest {args.max_shown}).")
    for ci, li, mx, n in flagged:
        print(f"  chunk {ci} local {li}: max={mx}, n_above={n}")

    # Optional resolution lookup
    d_table = None
    refl_id_table = None
    if args.metadata:
        import torch
        md = torch.load(args.metadata, weights_only=True)
        if "d" in md:
            d_table = md["d"].numpy()
        if "refl_ids" in md:
            refl_id_table = md["refl_ids"].numpy().astype(np.int64)

    chunk_paths = sorted(chunks_dir.glob("chunk_*.npz"))
    for rank, (ci, li, mx, n_above) in enumerate(flagged):
        arr, mask, fg, _bg, bbox, refl_id, shape = load_one(chunk_paths[ci], li)
        # Look up resolution by refl_id
        d_res = None
        if d_table is not None and refl_id_table is not None:
            hit = np.where(refl_id_table == refl_id)[0]
            if hit.size:
                d_res = float(d_table[int(hit[0])])
        out_path = out_dir / f"outlier_{rank:03d}_chunk{ci}_refl{refl_id}_max{mx}.png"
        plot_one(arr, mask, fg, refl_id, ci, shape, bbox, args.threshold,
                 d_res, out_path)
        print(f"  wrote {out_path.name}")

    print(f"\nDone. Open the PNGs under {out_dir}.")


if __name__ == "__main__":
    main()
