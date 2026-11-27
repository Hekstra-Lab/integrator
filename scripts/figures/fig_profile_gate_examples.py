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
    p.add_argument("--laue-profiles", type=Path, default=None,
                   help="npz of scripts/sim_truth/laue_profiles.py: laue.integrate's fitted profile per reflection, "
                        "drawn as a column after the counts (joined on image, h, k, l) and in --regions")
    p.add_argument("--laue-label", default="laue.integrate")
    p.add_argument("--regions", action="store_true",
                   help="also draw the mean profile per detector region (dump_profile_maps' 3x3 grid): one row "
                        "per region, one column per arm (and laue, from --laue-profiles, on the same region edges)")
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
                         ("gate", "gate"), ("d", "d"), ("x", "x"), ("y", "y"),
                         ("image", "image"), ("h", "h"), ("k", "k"), ("l", "l")):
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


def laue_rows(laue: dict, blob, arm: str):
    """laue.integrate's profiles for the drawn rows, by (image, h, k, l); None where laue has no fit."""
    need = [f"{arm}_ex_{k}" for k in ("image", "h", "k", "l")]
    if laue is None or any(k not in blob for k in need):
        return None
    key = {(int(i), int(h), int(k), int(l)): j for j, (i, h, k, l) in
           enumerate(zip(laue["image_num"], laue["h"], laue["k"], laue["l"]))}
    out = []
    for i, h, k, l in zip(*(blob[k] for k in need)):
        j = key.get((int(i), int(h), int(k), int(l)))
        out.append(None if j is None else laue["profile_box"][j])
    logger.info("  laue profiles found for %d of %d rows", sum(o is not None for o in out), len(out))
    return out


def draw_combined(blob, arms, out_dir: Path, prefix: str, profiles_only: bool = False,
                  label_override: dict | None = None, laue: dict | None = None,
                  laue_label: str = "laue.integrate") -> Path | None:
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
    lrows = laue_rows(laue, blob, arms[0]) if laue is not None else None
    cols = 1 + (1 if lrows is not None else 0) + sum(1 + (1 if g is not None else 0) for _a, _p, g in blocks)
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
        if lrows is not None:
            ax = axes[i][j]
            if lrows[i] is not None:
                ax.imshow(square(lrows[i], shape), cmap=figstyle().cmap(), interpolation="nearest")
            else:
                ax.text(0.5, 0.5, "no fit", ha="center", va="center", fontsize="x-small", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(laue_label, fontsize="x-small")
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


def draw_regions(raw, arms, out_dir: Path, prefix: str, label_override: dict | None = None,
                 laue: dict | None = None, laue_label: str = "laue.integrate") -> Path | None:
    """Mean profile per detector region: rows the 3x3 cells of dump_profile_maps, columns the arms.

    Each arm's `<arm>_cells` is its mean profile over the reflections in that cell (cut on
    the observed xyzcal range, `<arm>_extent`); laue's column averages its fitted
    profiles over the reflections falling in the same cell edges.
    """
    have = [a for a in arms if f"{a}_cells" in raw]
    if not have:
        return None
    shape = raw.get(f"{have[0]}_shape", np.array([0, 0]))
    x0, x1, y0, y1 = (float(v) for v in raw[f"{have[0]}_extent"])
    grid = raw[f"{have[0]}_cells"].shape[0]
    labels = column_labels(have)
    labels.update(label_override or {})
    lcells = None
    if laue is not None and "cells" in laue:
        # laue_profiles.py --reduce-from: cell means computed on the cluster over every laue row
        if not np.allclose(np.asarray(laue["cells_extent"], float), [x0, x1, y0, y1]):
            logger.warning("laue cells were cut on extent %s, the arms on %s",
                           np.asarray(laue["cells_extent"]).tolist(), [x0, x1, y0, y1])
        lc = np.asarray(laue["cells"], dtype=np.float64)
        lcells = {(r, c): None if np.isnan(lc[r, c]).all() else lc[r, c]
                  for r in range(grid) for c in range(grid)}
    elif laue is not None:
        x, y = np.asarray(laue["x"]), np.asarray(laue["y"])
        ix = np.clip(((x - x0) / max(x1 - x0, 1e-9) * grid).astype(int), 0, grid - 1)
        iy = np.clip(((y - y0) / max(y1 - y0, 1e-9) * grid).astype(int), 0, grid - 1)
        prof = np.asarray(laue["profile_box"], dtype=np.float64)
        lcells = {(r, c): prof[(iy == r) & (ix == c)].mean(0) if ((iy == r) & (ix == c)).any() else None
                  for r in range(grid) for c in range(grid)}
    cols = len(have) + (1 if lcells is not None else 0)
    n = grid * grid
    fig, axes = plt.subplots(n, cols, squeeze=False, layout="constrained",
                             figsize=figstyle().figsize("full", aspect=0.17 * n * 7 / max(cols, 4)))
    for r in range(grid):
        for c in range(grid):
            i = r * grid + c
            j = 0
            panels = []
            if lcells is not None:
                panels.append((laue_label, lcells[(r, c)]))
            for a in have:
                panels.append((labels[a], raw[f"{a}_cells"][r, c]))
            for lab, img in panels:
                ax = axes[i][j]
                if img is not None and np.isfinite(img).all() and img.sum() > 0:
                    ax.imshow(square(np.asarray(img), shape), cmap=figstyle().cmap(), interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(lab, fontsize="x-small")
                j += 1
            xc = x0 + (c + 0.5) * (x1 - x0) / grid
            yc = y0 + (r + 0.5) * (y1 - y0) / grid
            ncount = int(raw[f"{have[0]}_counts"][r, c]) if f"{have[0]}_counts" in raw else 0
            axes[i][0].set_ylabel(f"({xc:.0f}, {yc:.0f})\nn {ncount}", fontsize="x-small")
    fig.supylabel("detector region centre (x, y) px; reflections averaged", fontsize="x-small")
    path = out_dir / f"{prefix}_regions"
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
    raw = blob  # the dump as written, for --regions (blob may be replaced by the matched rows)
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
    laue = dict(np.load(args.laue_profiles)) if args.laue_profiles is not None else None
    overrides = dict(x.split("=", 1) for x in args.labels)
    combined = draw_combined(blob, arms, args.out_dir, args.prefix, profiles_only=args.profiles_only,
                             label_override=overrides, laue=laue, laue_label=args.laue_label)
    if args.regions:
        reg = draw_regions(raw, arms, args.out_dir, args.prefix, label_override=overrides, laue=laue,
                           laue_label=args.laue_label)
        if reg is not None:
            logger.info("  regions: %s", reg)
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
