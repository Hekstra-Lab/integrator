"""Plot predicted profiles for a few reflections from a trained run.

Runs the model forward on a few batches and plots the mean profile
(`qp_mean = softmax(W @ h + b)`) for a sample of reflections: the strongest
ones and one-per-resolution-ring. Profiles are `D*H*W` pixels; for 3D shoeboxes
only the middle z-slice is shown.

Usage:
    uv run python plot_profiles.py --run-dir <run-dir>
    uv run python plot_profiles.py --config cfg.yaml --ckpt model.ckpt --mode strong
"""

import argparse
import logging
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot predicted profiles for a few reflections"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run dir with run_paths.yaml (config + checkpoints)",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Config YAML (overrides run-dir)"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="A .ckpt (overrides run-dir)"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output dir (default: <run-dir>/plots, else next to the ckpt)",
    )
    parser.add_argument(
        "--mode",
        choices=["strong", "rings", "both"],
        default="both",
        help="Which reflections to sample",
    )
    parser.add_argument(
        "--n", type=int, default=9, help="Reflections per plot"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=4,
        help="Batches to forward for the candidate pool",
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
        default="rdbu+",
        help="Colormap; 'rdbu+' = positive (white->red) half of the basis "
        "RdBu_r, so 0=white and the peak=red (any Matplotlib name also works)",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# model + data loading (mirrors integrator.cli.predict)
# --------------------------------------------------------------------------- #
def _resolve_config_ckpt(args):
    """Resolve (config, ckpt_path, out_dir) from --run-dir and/or overrides."""
    import yaml

    from integrator.utils import apply_dataset_defaults, load_config

    meta = {}
    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is not None:
        meta = yaml.safe_load((run_dir / "run_paths.yaml").read_text())

    config_path = args.config or meta.get("config")
    if not config_path:
        raise SystemExit("plot_profiles: provide --config or --run-dir")
    config = apply_dataset_defaults(load_config(config_path))

    if args.ckpt:
        ckpt = Path(args.ckpt)
    else:
        log_dir = meta.get("log_dir") or meta.get("wandb", {}).get("log_dir")
        if not log_dir:
            raise SystemExit("plot_profiles: provide --ckpt or a run-dir with log_dir")
        candidates = sorted(Path(log_dir).glob("**/epoch*.ckpt"))
        if not candidates:
            raise SystemExit(f"plot_profiles: no checkpoints under {log_dir}")
        ckpt = _latest_epoch(candidates)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif run_dir is not None:
        out_dir = run_dir / "plots"
    else:
        out_dir = ckpt.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return config, ckpt, out_dir


def _latest_epoch(candidates: list[Path]) -> Path:
    def epoch_of(p: Path) -> int:
        m = re.search(r"epoch[=_](\d+)", p.name)
        return int(m.group(1)) if m else -1

    return max(candidates, key=epoch_of)


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy()


def _collect(integrator, dataloader, max_batches, device):
    """Forward a few batches and gather (profiles, strength, d, hkl)."""
    profiles, strength, dvals, hkls = [], [], [], []
    strength_key = None
    integrator.eval()
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            if bi >= max_batches:
                break
            fo = integrator(*_to_device(batch, device))["forward_out"]
            if strength_key is None:
                strength_key = (
                    "intensity.prf.value"
                    if "intensity.prf.value" in fo
                    else "qi_mean"
                )
            prof = _np(fo["qp_mean"])
            if prof.ndim == 3:  # (S, B, K) -> average over MC samples
                prof = prof.mean(axis=0)
            n = prof.shape[0]

            profiles.append(prof)
            strength.append(_np(fo[strength_key]).reshape(-1)[:n])
            dvals.append(
                _np(fo["d"]).reshape(-1)[:n] if "d" in fo else np.full(n, np.nan)
            )
            if all(k in fo for k in ("H", "K", "L")):
                hkls.append(
                    np.stack([_np(fo[k]).reshape(-1)[:n] for k in "HKL"], axis=1)
                )
            else:
                hkls.append(np.full((n, 3), np.nan))

    return (
        np.concatenate(profiles),
        np.concatenate(strength),
        np.concatenate(dvals),
        np.concatenate(hkls),
    )


# --------------------------------------------------------------------------- #
# selection + plotting (pure)
# --------------------------------------------------------------------------- #
def _select_strong(strength, n):
    """Indices of the n strongest reflections, brightest first."""
    return np.argsort(strength)[::-1][:n]


def _select_rings(strength, d, n):
    """One reflection per resolution ring: the strongest in each d quantile bin."""
    finite = np.isfinite(d)
    if not finite.any():
        return _select_strong(strength, n)
    edges = np.quantile(d[finite], np.linspace(0, 1, n + 1))
    picks = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (d >= lo) & (d <= hi if i == n - 1 else d < hi)
        cand = np.where(in_bin)[0]
        if cand.size:
            picks.append(cand[np.argmax(strength[cand])])
    # sort by resolution, high-res (small d) first
    return sorted(dict.fromkeys(picks), key=lambda i: d[i])


def _resolve_cmap(name):
    """Colormap for non-negative profiles.

    'rdbu+' is the positive (white->red) half of RdBu_r, so a profile value
    gets the same color it would have in the signed basis/bias plots (0=white,
    peak=red); there is no blue because profiles have no negative values.
    """
    if name == "rdbu+":
        base = plt.get_cmap("RdBu_r")
        return LinearSegmentedColormap.from_list(
            "RdBu_r+", base(np.linspace(0.5, 1.0, 256))
        )
    return plt.get_cmap(name)


def _infer_shape(k, shape_arg, depth):
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


def _title(strength, d, hkl) -> str:
    parts = [f"I={strength:.0f}"]
    if np.isfinite(d):
        parts.append(f"d={d:.2f}Å")
    line = ", ".join(parts)
    if np.all(np.isfinite(hkl)):
        h, k, l = hkl.astype(int)
        line += f"\n({h} {k} {l})"
    return line


def _plot_profiles(profiles, idx, strength, dvals, hkl, shape, out_path, title, cmap):
    """Plot the middle slice of each selected profile in a grid."""
    D, H, Wd = shape
    mid = D // 2
    n = len(idx)
    if n == 0:
        logger.warning("No reflections selected for %s; skipping", title)
        return

    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, i in zip(axes, idx):
        img = profiles[i].reshape(D, H, Wd)[mid]
        ax.imshow(img, cmap=cmap, vmin=0.0, vmax=float(img.max()) or 1.0)
        ax.set_title(_title(strength[i], dvals[i], hkl[i]), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"{title} (slice z={mid} of {D})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    from integrator.utils import (
        construct_data_loader,
        construct_integrator,
        inject_binning_labels,
    )

    config, ckpt, out_dir = _resolve_config_ckpt(args)
    logger.info("config + checkpoint: %s", ckpt)

    data_loader = construct_data_loader(config)
    data_loader.setup()
    inject_binning_labels(data_loader, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    integrator = construct_integrator(config)
    integrator.load_state_dict(torch.load(ckpt.as_posix())["state_dict"])
    integrator.to(device)

    profiles, strength, dvals, hkl = _collect(
        integrator, data_loader.predict_dataloader(), args.max_batches, device
    )
    logger.info("Collected %d reflections", len(profiles))

    shape = _infer_shape(profiles.shape[1], args.shape, args.depth)
    cmap = _resolve_cmap(args.cmap)

    if args.mode in ("strong", "both"):
        idx = _select_strong(strength, args.n)
        _plot_profiles(
            profiles, idx, strength, dvals, hkl, shape,
            out_dir / "profiles_strong.png", "Strongest reflections", cmap,
        )
    if args.mode in ("rings", "both"):
        idx = _select_rings(strength, dvals, args.n)
        _plot_profiles(
            profiles, idx, strength, dvals, hkl, shape,
            out_dir / "profiles_by_resolution.png",
            "Reflections across resolution rings", cmap,
        )


if __name__ == "__main__":
    main()
