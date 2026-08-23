"""Plot the learned profile basis from a learned-basis-profile run.

The profile surrogate decodes `prf = softmax(W @ h + b)`, where `W` is the
decoder weight of shape `(K, latent_dim)` and `b` the decoder bias of shape
`(K,)`, with `K = D*H*W` pixels. Latent dimension `i` corresponds to basis
column `W[:, i]`. For 3D shoeboxes (e.g. `3x21x21`) only the middle z-slice is
shown, plus the bias as a final panel.

Usage:
    uv run python plot_basis.py --run-dir <run-dir>
    uv run python plot_basis.py --ckpt path/to/model.ckpt --out basis.png
"""

import argparse
import logging
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the learned profile basis (middle slice) from a run"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run dir with run_paths.yaml (checkpoints resolved from log_dir)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Specific .ckpt to read; overrides --run-dir discovery",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG (default: <run-dir>/plots/learned_basis.png)",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default=None,
        help="Shoebox shape 'D,H,W'; inferred from K and --depth when omitted",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Shoebox depth, for shape inference and the middle-slice index",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="RdBu_r",
        help="Diverging colormap (basis weights are signed, centered at 0)",
    )
    return parser.parse_args()


def _latest_epoch(candidates: list[Path]) -> Path:
    """Checkpoint with the highest epoch number, else the most recent file."""

    def epoch_of(p: Path) -> int:
        m = re.search(r"epoch[=_](\d+)", p.name)
        return int(m.group(1)) if m else -1

    if any(epoch_of(p) >= 0 for p in candidates):
        return max(candidates, key=epoch_of)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_checkpoint(run_dir: Path | None, ckpt: str | None) -> Path:
    """Resolve the checkpoint to read from `--ckpt` or the run dir layout."""
    if ckpt is not None:
        return Path(ckpt)
    if run_dir is None:
        raise ValueError("Requires --run-dir or --ckpt")

    ckpt_dir = None
    rp = run_dir / "run_paths.yaml"
    if rp.is_file():
        log_dir = load_config(rp).get("log_dir")
        if log_dir:
            ckpt_dir = Path(log_dir) / "checkpoints"

    candidates = []
    if ckpt_dir is not None and ckpt_dir.is_dir():
        candidates = list(ckpt_dir.glob("*.ckpt"))
    if not candidates:  # non-W&B runs keep everything under the run dir
        candidates = list(run_dir.glob("**/*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt found under {run_dir}")
    return _latest_epoch(candidates)


def _load_decoder(ckpt_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (W, b, key) for the profile surrogate's learned decoder.

    W has shape `(K, latent_dim)` and b shape `(K,)`. Raises if the run has no
    learned-basis profile (a Dirichlet profile has no decoder to plot).
    """
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj

    weight_keys = [
        k
        for k in state
        if k.startswith("surrogates.") and k.endswith(".decoder.weight")
    ]
    if not weight_keys:
        raise KeyError(
            "No 'surrogates.*.decoder.weight' in the checkpoint; this run does "
            "not use a learned-basis profile (Dirichlet profiles have no basis)."
        )

    wkey = weight_keys[0]
    bkey = wkey[: -len("weight")] + "bias"
    W = state[wkey].float().cpu().numpy()  # (K, latent_dim)
    b = state[bkey].float().cpu().numpy()  # (K,)
    logger.info("Loaded %s: W%s, b%s", wkey, W.shape, b.shape)
    return W, b, wkey


def _infer_shape(k: int, shape_arg: str | None, depth: int) -> tuple[int, int, int]:
    """Resolve the (D, H, W) shoebox shape from `--shape` or K and `--depth`."""
    if shape_arg:
        d, h, w = (int(x) for x in re.split(r"[,x]", shape_arg))
        if d * h * w != k:
            raise ValueError(f"--shape {d}x{h}x{w} does not multiply to K={k}")
        return d, h, w

    hw = k // depth
    side = int(round(math.sqrt(hw)))
    if depth * side * side != k:
        raise ValueError(
            f"Cannot infer H, W from K={k} with depth={depth}; pass --shape D,H,W"
        )
    return depth, side, side


def _plot_basis(W, b, shape, out_path, cmap) -> None:
    """Plot each basis column and the bias as middle-slice images in a grid."""
    D, H, Wd = shape
    mid = D // 2

    panels = [
        (f"h{i}", W[:, i].reshape(D, H, Wd)[mid]) for i in range(W.shape[1])
    ]
    panels.append(("bias", b.reshape(D, H, Wd)[mid]))

    n = len(panels)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows)
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, img) in zip(axes, panels):
        vmax = float(np.abs(img).max()) or 1.0  # per-panel symmetric scale
        ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Learned profile basis (slice z={mid} of {D})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else None

    ckpt = _find_checkpoint(run_dir, args.ckpt)
    logger.info("Using checkpoint: %s", ckpt)

    W, b, _ = _load_decoder(ckpt)
    shape = _infer_shape(W.shape[0], args.shape, args.depth)

    if args.out:
        out = Path(args.out)
    elif run_dir is not None:
        out = run_dir / "plots" / "learned_basis.png"
    else:
        out = ckpt.with_name("learned_basis.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    _plot_basis(W, b, shape, out, args.cmap)


if __name__ == "__main__":
    main()
