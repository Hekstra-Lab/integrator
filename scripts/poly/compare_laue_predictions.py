"""Compare a rerun laue-dials prediction with the reflection list an integrator dataset was cut from.

Reads two DIALS reflection tables -- the new `integrated_*.refl` and the dataset's
own (e.g. pytorch_data/reflections_.refl) -- keeps the experiments (images) with
id < --max-images in each, matches reflections on (experiment id, h, k, l) and
reports how many match, how many each table has that the other lacks, and for the
matched rows the distance between predicted centroids (pixels) and the wavelength
difference. If the two agree, the integrator's shoeboxes stand and only the
reference intensities change.

It also writes the new MTZ restricted to the same images (--out-mtz): the rows whose
BATCH is among the first --max-images distinct BATCH values, so the classical
reference covers exactly the images the integrator was trained on.

Needs dials (flex reflection tables) and reciprocalspaceship.

usage: python compare_laue_predictions.py --new-refl NEW.refl --ours OURS.refl [--new-mtz NEW.mtz
           --out-mtz OUT.mtz] [--max-images 1000] [--out-csv summary.csv]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def table(path: Path, max_images: int):
    from dials.array_family import flex

    t = flex.reflection_table.from_file(str(path))
    eid = np.asarray(t["id"])
    hkl = np.asarray(t["miller_index"].as_vec3_double()).astype(np.int64)
    xyz = np.asarray(t["xyzcal.px"].as_numpy_array())
    lam = np.asarray(t["wavelength"]) if "wavelength" in t else np.full(len(eid), np.nan)
    keep = eid < max_images
    return eid[keep], hkl[keep], xyz[keep], lam[keep], len(t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--new-refl", type=Path, required=True)
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--new-mtz", type=Path, default=None)
    ap.add_argument("--out-mtz", type=Path, default=None)
    ap.add_argument("--max-images", type=int, default=1000)
    ap.add_argument("--out-csv", type=Path, default=None)
    a = ap.parse_args()

    e1, h1, x1, l1, n1 = table(a.new_refl, a.max_images)
    e0, h0, x0, l0, n0 = table(a.ours, a.max_images)
    k1 = {(int(e), *map(int, h)): i for i, (e, h) in enumerate(zip(e1, h1))}
    k0 = {(int(e), *map(int, h)): i for i, (e, h) in enumerate(zip(e0, h0))}
    both = sorted(set(k1) & set(k0))
    i1 = np.array([k1[k] for k in both], dtype=np.int64)
    i0 = np.array([k0[k] for k in both], dtype=np.int64)
    dxy = np.hypot(*(x1[i1, :2] - x0[i0, :2]).T) if len(both) else np.array([])
    dlam = np.abs(l1[i1] - l0[i0]) if len(both) else np.array([])
    rows = {
        "images": a.max_images, "ours_total": n0, "new_total": n1,
        "ours_in_images": len(k0), "new_in_images": len(k1), "matched": len(both),
        "only_ours": len(set(k0) - set(k1)), "only_new": len(set(k1) - set(k0)),
        "centroid_dxy_median_px": float(np.median(dxy)) if dxy.size else float("nan"),
        "centroid_dxy_p95_px": float(np.percentile(dxy, 95)) if dxy.size else float("nan"),
        "frac_dxy_over_1px": float(np.mean(dxy > 1.0)) if dxy.size else float("nan"),
        "wavelength_abs_diff_median": float(np.nanmedian(dlam)) if dlam.size else float("nan"),
    }
    for k, v in rows.items():
        print(f"{k:28s} {v}")
    if a.out_csv:
        with open(a.out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["quantity", "value"])
            w.writerows(rows.items())

    if a.new_mtz is not None and a.out_mtz is not None:
        import reciprocalspaceship as rs

        m = rs.read_mtz(str(a.new_mtz))
        batches = np.sort(np.unique(m["BATCH"].to_numpy()))[: a.max_images]
        sub = m[m["BATCH"].isin(batches)]
        sub.write_mtz(str(a.out_mtz))
        print(f"wrote {a.out_mtz}: {len(sub)} of {len(m)} rows, BATCH {batches.min()}..{batches.max()}")


if __name__ == "__main__":
    main()
