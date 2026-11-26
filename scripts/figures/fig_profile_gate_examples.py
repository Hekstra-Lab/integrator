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
import os
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
    p.add_argument("--profiles-only", action="store_true",
                   help="combined grid without the gate columns: the counts, then one "
                        "profile per arm (each row labelled with d and detector x, y)")
    p.add_argument("--labels", nargs="*", default=[],
                   help="ARM=NAME column titles for the combined grid, overriding the shortened arm names")
    p.add_argument("--rows", type=int, default=6,
                   help="reflections drawn, spanning the resolution range "
                        "(only for a dump of NAMED reflections)")
    return p.parse_args()


def arms_in(blob) -> list[str]:
    return sorted({k[: -len("_ex_counts")] for k in blob
                   if k.endswith("_ex_counts")})


def named_as_examples(blob, n_rows: int):
    """Rows matched BY REFLECTION across arms, in the `_ex_*` names.

    `dump_profile_maps.py` run with `REFL_IDS` stores `<arm>_named_*` for
    the same named reflections in every arm. Its `_ex_*` arrays are a
    different thing: each arm picks its own examples out of its own first
    batch, and those are not the same reflections -- measured, the first
    row came out at 12.36, 8.40, 12.36 and 12.36 A in four arms -- so
    `draw_combined` refuses them, correctly, because a grid with one
    reflection per row invites exactly the comparison they cannot bear.

    This selects reflections every arm has, ordered by resolution and
    evenly spaced across the range, and presents them under the `_ex_*`
    names so the drawing code is unchanged.
    """
    arms = sorted({k[: -len("_named_refl_id")] for k in blob
                   if k.endswith("_named_refl_id")})
    if not arms:
        return None, []
    ids = {a: blob[f"{a}_named_refl_id"].tolist() for a in arms}
    common = set.intersection(*[set(v) for v in ids.values()])
    if len(common) < n_rows:
        logger.warning("  only %d reflections common to all arms",
                       len(common))
        return None, []
    first = arms[0]
    order = {int(r): i for i, r in enumerate(ids[first])}
    d_first = blob[f"{first}_named_d"]
    # by resolution, then evenly spaced over the range so the rows span it
    by_d = sorted(common, key=lambda r: float(d_first[order[int(r)]]),
                  reverse=True)
    pick = [by_d[i] for i in
            np.unique(np.linspace(0, len(by_d) - 1, n_rows)
                      .round().astype(int))]
    out = {}
    for a in arms:
        idx = {int(r): i for i, r in enumerate(ids[a])}
        take = np.array([idx[int(r)] for r in pick])
        for src, dst in (("counts", "counts"), ("profile", "profile"),
                         ("gate", "gate"), ("d", "d"), ("x", "x"), ("y", "y")):
            key = f"{a}_named_{src}"
            if key in blob:
                out[f"{a}_ex_{dst}"] = blob[key][take]
        if f"{a}_shape" in blob:
            out[f"{a}_shape"] = blob[f"{a}_shape"]
    logger.info("  %d arms matched on %d reflections, %d drawn (%.2f-%.2f A)",
                len(arms), len(common), len(pick),
                float(out[f"{first}_ex_d"][0]),
                float(out[f"{first}_ex_d"][-1]))
    return out, arms


def column_labels(arms) -> dict[str, str]:
    """Arm names short enough for one column of a nine-column grid.

    The dataset prefix goes first, then whatever the arms have in common:
    four arms of one family are `svae`, `svae_bg`, `svae_calib` and
    `svae_conjugate`, and at a column width of about 50 pt the full names
    collide with their neighbours and the header is unreadable. Dropping
    the shared `svae_` leaves `svae`, `bg`, `calib`, `conjugate`, which is
    the part that distinguishes them. The arm that IS the common prefix
    keeps its own name rather than becoming empty.
    """
    base = {a: a.replace("hewl1118_", "").replace("hewl9b7c_", "")
               .replace("pdz2_", "") for a in arms}
    names = list(base.values())
    if len(names) < 2:
        return base
    # The family name is usually one of the arms -- `svae` beside
    # `svae_bg`, `svae_calib`, `svae_conjugate` -- so the shared part has
    # no trailing separator of its own and `commonprefix` alone strips
    # nothing. Take the shortest name as the candidate family.
    root = min(names, key=len)
    if not all(n == root or n.startswith(f"{root}_") for n in names):
        prefix = os.path.commonprefix(names)
        prefix = prefix[: prefix.rfind("_") + 1] if "_" in prefix else ""
        return ({a: (n[len(prefix):] or n) for a, n in base.items()}
                if prefix else base)
    # the family arm keeps its own name rather than becoming empty
    return {a: (n if n == root else n[len(root) + 1:])
            for a, n in base.items()}


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


def draw_combined(blob, arms, out_dir: Path, prefix: str, profiles_only: bool = False,
                  label_override: dict | None = None) -> Path | None:
    """All arms on one grid, one row per reflection.

    Only valid when the arms picked the SAME reflections, which they do
    when they share a dataset and a dataloader seed. That is checked on
    the stored resolutions rather than assumed: rows that are not the same
    reflection cannot be compared across columns, and the figure would
    invite exactly that comparison.
    """
    ds = [blob.get(f"{a}_ex_d") for a in arms]
    if any(d is None for d in ds):
        return None
    ref = ds[0]
    if not all(np.allclose(d, ref, rtol=1e-4) for d in ds):
        logger.warning("  combined: arms picked different reflections "
                       "(%s), drawing separately only",
                       ", ".join(f"{a}:{float(d[0]):.2f}"
                                 for a, d in zip(arms, ds, strict=True)))
        return None

    n = len(ref)
    counts = blob[f"{arms[0]}_ex_counts"]
    shape = blob.get(f"{arms[0]}_shape", np.array([0, 0]))
    # counts once, then profile and gate per arm
    blocks = []
    for a in arms:
        blocks.append((a, blob.get(f"{a}_ex_profile"),
                       None if profiles_only else blob.get(f"{a}_ex_gate")))
    cols = 1 + sum(1 + (1 if g is not None else 0) for _a, _p, g in blocks)
    labels = column_labels(arms)
    labels.update(label_override or {})
    xs, ys = blob.get(f"{arms[0]}_ex_x"), blob.get(f"{arms[0]}_ex_y")

    fig, axes = plt.subplots(
        n, cols, squeeze=False, layout="constrained",
        figsize=figstyle().figsize("full", aspect=0.17 * n * 7 / cols),
    )
    for i in range(n):
        j = 0
        ax = axes[i][j]
        ax.imshow(square(counts[i], shape), cmap=figstyle().cmap(),
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title("counts", fontsize="x-small")
        row = f"{float(ref[i]):.2f}" + (r" \AA" if profiles_only else "")
        if profiles_only and xs is not None and ys is not None:
            row += f"\n({float(xs[i]):.0f}, {float(ys[i]):.0f})"
        ax.set_ylabel(row, fontsize="x-small")
        j += 1
        for arm, prof, gate in blocks:
            short = labels[arm]
            ax = axes[i][j]
            ax.imshow(square(prof[i], shape), cmap=figstyle().cmap(),
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(short if profiles_only else f"{short}\nprofile", fontsize="x-small")
            j += 1
            if gate is not None:
                ax = axes[i][j]
                ax.imshow(square(gate[i], shape), cmap=figstyle().cmap(),
                          vmin=0.0, vmax=1.0, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"{short}\ngate", fontsize="x-small")
                j += 1
    if profiles_only:
        fig.supylabel(r"resolution (\AA), detector (x, y) px", fontsize="x-small")
    path = out_dir / (f"{prefix}_combined" + ("_profiles" if profiles_only else ""))
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

    # a dump of NAMED reflections carries the same rows in every arm, which
    # is what the combined grid needs; the per-arm `_ex_*` examples do not
    matched, named_arms = named_as_examples(blob, args.rows)
    if matched is not None:
        blob, arms = matched, named_arms
    else:
        arms = arms_in(blob)
    if not arms:
        raise SystemExit(f"{args.data} holds no `<arm>_ex_counts`")
    combined = draw_combined(blob, arms, args.out_dir, args.prefix, profiles_only=args.profiles_only,
                             label_override=dict(x.split("=", 1) for x in args.labels))
    if combined is not None:
        logger.info("  combined: %s", combined)
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
