"""How many reflections survive each laue-dials stage on a datsimx row, beside how many the simulation holds.

Per dataset given as NAME=ROW_DIR:LAUE_SUBDIR:SHOEBOX_PREFIX, one long-format row per
stage (method `laue_dials`, dataset NAME, stage, n):

  truth_all, truth_ge10, truth_ge100   rows of truth/truth_reflections.csv, all and with
                                       i_true >= 10 and >= 100 photons
  strong          laue.find_spots (strong.refl)
  indexed         laue.optimize_indexing (optimized.refl)
  refined         laue.refine (poly_refined.refl)
  experiments     images kept by laue.refine (poly_refined.expt)
  predicted       laue.predict (predicted.refl)
  integrated      laue.integrate (integrated.refl)
  shoeboxes_<S>   rows of shoeboxes/<SHOEBOX_PREFIX>_<S>/counts.npy for each --sizes S

The shoeboxes are cut from integrated.refl, so the predicted list sets how many
reflections every method integrates. Needs dials (integrator-cuda-dev).

    python laue_stage_counts.py --out stage_counts.csv \\
        mos010=$P/mos010:laue_dials_pc:pc_default mos025=$P/mos025:laue_dials_pc_m0:pc_m0_default
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="+", help="NAME=ROW_DIR:LAUE_SUBDIR:SHOEBOX_PREFIX")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", default="25,41")
    a = ap.parse_args()

    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory

    rows = []
    for spec in a.datasets:
        name, rest = spec.split("=", 1)
        row, laue, prefix = rest.split(":")
        row = Path(row)
        t = pd.read_csv(row / "truth" / "truth_reflections.csv", usecols=["i_true"])
        counts = {"truth_all": len(t), "truth_ge10": int((t.i_true >= 10).sum()),
                  "truth_ge100": int((t.i_true >= 100).sum())}
        L = row / laue
        for stage, f in (("strong", "strong.refl"), ("indexed", "optimized.refl"), ("refined", "poly_refined.refl"),
                         ("predicted", "predicted.refl"), ("integrated", "integrated.refl")):
            counts[stage] = len(flex.reflection_table.from_file(str(L / f))) if (L / f).is_file() else None
        e = L / "poly_refined.expt"
        counts["experiments"] = len(ExperimentListFactory.from_json_file(str(e), check_format=False)) if e.is_file() else None
        for s in a.sizes.split(","):
            c = row / "shoeboxes" / f"{prefix}_{s}" / "counts.npy"
            counts[f"shoeboxes_{s}"] = int(np.load(c, mmap_mode="r").shape[0]) if c.is_file() else None
        for stage, n in counts.items():
            rows.append(dict(method="laue_dials", dataset=name, stage=stage, n=n))
        print(name, counts)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("->", a.out)


if __name__ == "__main__":
    main()
