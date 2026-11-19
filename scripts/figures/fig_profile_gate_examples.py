"""One reflection per row: the counts, the profile drawn for it, the gate.

Rows run from low resolution to high, because that is the axis the answer
should vary along. A spot weakens with resolution and the background does
not, so by the outer shells the gate has progressively less to work with.
A model that draws the same shape and the same gate at 1.8 A as at 4 A is
not reading the shoebox.

Three panels per row, on their own scales because they are three different
quantities:

- **counts**, what the detector recorded, on the ink ramp;
- **profile**, `q(p)` mean, a distribution over the box summing to one;
- **signal prob**, the per-pixel gate: the posterior probability that a
  count at that pixel is signal rather than background. Fixed 0 to 1, so
  a saturated gate is visible as a flat bright box rather than rescaled
  into looking structured.

The gate panel is omitted for an integrator that has none -- the
hierarchical family -- rather than drawn empty.

Reads `profile_maps_*.npz` from `scripts/figures/dump_profile_maps.py`,
which stores `<arm>_ex_counts`, `_ex_profile`, `_ex_gate` and `_ex_d`.

Run:
    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_profile_gate_examples.py \\
            --data docs/figures/data/profile_maps_pdz2.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def figstyle():
    """figTeX, imported at the call: it is not a dependency of this project."""
    import figstyle as _figstyle

    return _figstyle


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA = ROOT / "docs" / "figures" / "data"
OUT = ROOT / "docs" / "figures" / "cmp"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path,
                   default=DATA / "profile_maps_pdz2.npz")
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--prefix", default="profile_gate")
    return p.parse_args()


def arms_in(blob) -> list[str]:
    return sorted({k[: -len("_ex_counts")] for k in blob
                   if k.endswith("_ex_counts")})


def square(vec, shape) -> np.ndarray:
    """A flat shoebox as an image, depth summed if there is depth."""
    h, w = int(shape[0]), int(shape[1])
    if h and w and vec.size == h * w:
        return vec.reshape(h, w)
    side = int(round(float(np.sqrt(vec.size))))
    if side * side == vec.size:
        return vec.reshape(side, side)
    return vec.reshape(1, -1)


def draw_arm(blob, arm, out_dir: Path, prefix: str) -> Path | None:
    counts = blob.get(f"{arm}_ex_counts")
    profile = blob.get(f"{arm}_ex_profile")
    gate = blob.get(f"{arm}_ex_gate")
    shape = blob.get(f"{arm}_shape", np.array([0, 0]))
    d = blob.get(f"{arm}_ex_d")
    if counts is None or profile is None:
        logger.warning("  %s: no examples stored", arm)
        return None

    n = counts.shape[0]
    cols = 3 if gate is not None else 2
    titles = ["counts", "profile", "signal prob"][:cols]
    fig, axes = plt.subplots(
        n, cols, squeeze=False, layout="constrained",
        figsize=figstyle().figsize("half", aspect=0.42 * n),
    )
    for i in range(n):
        images = [square(counts[i], shape), square(profile[i], shape)]
        if gate is not None:
            images.append(square(gate[i], shape))
        for j, img in enumerate(images):
            ax = axes[i][j]
            if titles[j] == "signal prob":
                # fixed 0-1: a saturated gate must look saturated, not be
                # rescaled into looking structured
                ax.imshow(img, cmap=figstyle().cmap(), vmin=0.0, vmax=1.0,
                          interpolation="nearest")
            else:
                ax.imshow(img, cmap=figstyle().cmap(),
                          interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize="small")
        if d is not None and i < len(d):
            axes[i][0].set_ylabel(f"{float(d[i]):.2f}", fontsize="small")
    if d is not None:
        axes[n // 2][0].set_ylabel(
            f"{float(d[n // 2]):.2f}\n" + r"resolution (\AA)",
            fontsize="small")
    path = out_dir / f"{prefix}_{arm}"
    figstyle().save(fig, path)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(
            f"no {args.data}. It comes from "
            "scripts/figures/dump_profile_maps.py"
        )
    blob = np.load(args.data, allow_pickle=False)
    figstyle().use()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    arms = arms_in(blob)
    if not arms:
        raise SystemExit(f"{args.data} holds no `<arm>_ex_counts`")
    drawn = 0
    for arm in arms:
        path = draw_arm(blob, arm, args.out_dir, args.prefix)
        if path is None:
            continue
        drawn += 1
        d = blob.get(f"{arm}_ex_d")
        gate = blob.get(f"{arm}_ex_gate")
        span = (f"{float(np.max(d)):.2f}-{float(np.min(d)):.2f} A"
                if d is not None else "resolution unknown")
        if gate is not None:
            logger.info("  %-16s %s | gate mean %.3f, saturated pixels "
                        "(>0.99) %.1f%%", arm, span, float(gate.mean()),
                        100 * float((gate > 0.99).mean()))
        else:
            logger.info("  %-16s %s | no gate (not an SVAE)", arm, span)
        logger.info("     saved %s", path)
    logger.info("%d figures", drawn)


if __name__ == "__main__":
    main()
