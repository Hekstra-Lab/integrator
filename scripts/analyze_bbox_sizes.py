"""Shoebox size vs clipping-loss tradeoff analysis.

Reads DIALS integrated.refl, computes per-axis bbox extents, and reports
clipping rates for a set of candidate (H, W, D) window sizes. Also breaks
down loss by resolution shell and reports memory footprint per candidate.

Run with:
    dials.python analyze_bbox_sizes.py /path/to/integrated.refl [--expt /path/to/integrated.expt]
"""

import argparse
import sys

import numpy as np


CANDIDATES = [
    # (H, W, D, label)
    (21, 21, 3, "current"),
    (21, 21, 5, "current+z5"),
    (27, 27, 5, "medium-5"),
    (27, 27, 7, "medium-7"),
    (33, 33, 7, "large-7"),
    (33, 33, 9, "large-9"),
    (33, 33, 11, "large-11"),
    (33, 33, 13, "p99-safe"),
    (41, 41, 13, "xl"),
]


def resolution_from_expt(refl, expt_path):
    """Compute d-spacing per reflection using the experiment's crystal + scan.
    Returns None if we can't (script works fine without resolution breakdown)."""
    try:
        from dxtbx.model.experiment_list import ExperimentListFactory
        experiments = ExperimentListFactory.from_json_file(expt_path)
        crystal = experiments[0].crystal
        uc = crystal.get_unit_cell()
        miller = refl["miller_index"]
        d = np.array([uc.d(h) for h in miller])
        return d
    except Exception as e:
        print(f"[resolution disabled] {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refl", help="Path to integrated.refl")
    ap.add_argument("--expt", help="Path to integrated.expt (enables resolution breakdown)")
    ap.add_argument("--n-shells", type=int, default=10, help="Resolution shells")
    args = ap.parse_args()

    from dials.array_family import flex
    refl = flex.reflection_table.from_file(args.refl)

    # Keep only reflections DIALS actually integrated
    n_before = len(refl)
    refl = refl.select(refl.get_flags(refl.flags.integrated_sum))
    print(f"Integrated reflections: {len(refl):,}  (from {n_before:,} total)")

    bbox = np.asarray(refl["bbox"]).reshape(-1, 6)  # x0,x1,y0,y1,z0,z1
    dx = bbox[:, 1] - bbox[:, 0]
    dy = bbox[:, 3] - bbox[:, 2]
    dz = bbox[:, 5] - bbox[:, 4]
    n = len(dx)

    # ---------- 1) Distributions ----------
    print("\n=== Per-axis distributions ===")
    print(f"{'axis':6s}{'min':>6s}{'p25':>6s}{'p50':>6s}{'p75':>6s}"
          f"{'p90':>6s}{'p95':>6s}{'p99':>6s}{'p99.9':>7s}{'max':>6s}{'mean':>7s}")
    for name, d in [("x", dx), ("y", dy), ("z", dz)]:
        qs = np.percentile(d, [0, 25, 50, 75, 90, 95, 99, 99.9, 100])
        print(f"{name:6s}" + "".join(f"{q:6.0f}" for q in qs) + f"{d.mean():7.1f}")

    # ---------- 2) Per-axis reflection clipping curve ----------
    print("\n=== Reflection clipping by window size (per axis) ===")
    print("  (what fraction of reflections have bbox > window along that axis)")
    candidate_sizes_xy = sorted({h for h, _, _, _ in CANDIDATES} | {w for _, w, _, _ in CANDIDATES})
    candidate_sizes_z = sorted({d for _, _, d, _ in CANDIDATES})
    print(f"{'W/H':>5s}" + "".join(f"{s:>7d}" for s in candidate_sizes_xy))
    print(f"{'x%':>5s}" + "".join(f"{(dx > s).mean()*100:7.2f}" for s in candidate_sizes_xy))
    print(f"{'y%':>5s}" + "".join(f"{(dy > s).mean()*100:7.2f}" for s in candidate_sizes_xy))
    print(f"{'D':>5s}" + "".join(f"{s:>7d}" for s in candidate_sizes_z))
    print(f"{'z%':>5s}" + "".join(f"{(dz > s).mean()*100:7.2f}" for s in candidate_sizes_z))

    # ---------- 3) Combined clipping per candidate ----------
    print("\n=== Combined loss by candidate window (any-axis clipping) ===")
    header = (f"{'H':>3s}{'W':>4s}{'D':>4s}  {'label':10s}  "
              f"{'clipped_refl_%':>14s}  {'voxels/refl':>12s}  "
              f"{'tot_voxels_GB(u16)':>20s}  {'voxel_loss_%':>13s}")
    print(header)
    for h, w, d, label in CANDIDATES:
        clip_any = ((dx > w) | (dy > h) | (dz > d)).mean() * 100

        vox_per_refl = h * w * d
        bytes_total = vox_per_refl * n * 2  # uint16 = 2 bytes
        gb = bytes_total / 1e9

        # voxel-level loss: fraction of DIALS-bbox voxels falling OUTSIDE our window
        # approx: voxels in DIALS bbox = dx*dy*dz, voxels covered = min(dx,w)*min(dy,h)*min(dz,d)
        dials_vox = dx * dy * dz
        covered = np.minimum(dx, w) * np.minimum(dy, h) * np.minimum(dz, d)
        voxel_loss = 100 * (1 - covered.sum() / dials_vox.sum())

        print(f"{h:>3d}{w:>4d}{d:>4d}  {label:10s}  {clip_any:>13.2f}%  "
              f"{vox_per_refl:>12,d}  {gb:>18.2f}    {voxel_loss:>12.3f}%")

    # ---------- 4) Per-resolution-shell breakdown ----------
    d_spacing = None
    if args.expt:
        d_spacing = resolution_from_expt(refl, args.expt)

    if d_spacing is not None:
        print("\n=== Clipping by resolution shell (any-axis) ===")
        # Bin by d-spacing (low d = high res, etc.)
        qs = np.quantile(d_spacing, np.linspace(0, 1, args.n_shells + 1))
        print(f"{'d_max':>7s}{'d_min':>7s}{'n':>8s}  ", end="")
        for h, w, d, label in CANDIDATES:
            print(f"{label[:8]:>9s}", end="")
        print()
        for i in range(args.n_shells):
            lo, hi = qs[i], qs[i + 1]
            in_shell = (d_spacing >= lo) & (d_spacing <= hi)
            ns = in_shell.sum()
            if ns == 0:
                continue
            print(f"{hi:>7.2f}{lo:>7.2f}{ns:>8,d}  ", end="")
            for h, w, d, _ in CANDIDATES:
                clipped = ((dx[in_shell] > w) | (dy[in_shell] > h) | (dz[in_shell] > d)).mean() * 100
                print(f"{clipped:>8.2f}%", end="")
            print()

    # ---------- 5) Recommendation heuristic ----------
    print("\n=== Heuristic recommendation ===")
    print("  Rule: pick smallest window with any-axis clipping < 1% at p99 level.")
    for h, w, d, label in CANDIDATES:
        any_clip = ((dx > w) | (dy > h) | (dz > d)).mean() * 100
        if any_clip < 1.0:
            gb = h * w * d * n * 2 / 1e9
            print(f"  -> ({h}, {w}, {d})  [{label}]  "
                  f"clipping={any_clip:.2f}%  memory={gb:.2f} GB uint16")
            return
    print("  No candidate under 1% clip; go larger or filter outliers first.")


if __name__ == "__main__":
    main()
