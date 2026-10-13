#!/usr/bin/env python
"""Write what a merging model produces, for the statistics that apply to it.

The model's learned intensity IS the merged value: `qi_per_obs` is the
per-HKL posterior `Gamma(alpha_h, beta_h)` indexed back onto observations.
The per-observation scale is a nuisance parameter that explains why
observation i of reflection h differs from I_h -- it fits the model to the
data and is NOT a factor to multiply back out. There is no unmerged scaled
intensity to recover, because the model never estimates one.

That rules out the classical merging table. CC(1/2), R-merge, R-meas and
R-pim all measure agreement between INDEPENDENT measurements of one
reflection, and this model has none to offer: every observation of a
reflection in a batch carries that batch's merged estimate. Splitting those
in half and correlating them measures batch-to-batch reproducibility of an
already-averaged quantity, which is why it returned CC(1/2) = 0.996 for a
model whose intensities were 800x mis-scaled.

The statistic that does apply is the one careless computes with its xval
files: merge two INDEPENDENT halves of the data and correlate the results.
`--half-by` does that split here.

An integrate-then-scale pipeline hands DIALS or careless a file of unmerged
observations, and the merging table everyone reports -- CC(1/2), R-merge,
R-meas, R-pim, completeness, multiplicity, <I/sigma> per shell -- is computed
from that file by standard software. A model that merges internally goes from
pixels to merged structure factors in one pass and writes no such file, so
those statistics cannot be produced at all, by us or by anyone checking.

That is a reporting gap, not a modelling one: the observations exist, they are
just never written down. The model's likelihood is

    rate = scale * I * profile + background

so `scale * qi_mean` is the per-observation intensity on the measured gauge,
which is exactly what belongs in an unmerged file.

The output is a standard unmerged MTZ. Feed it to `phenix.merging_statistics`
(or `dials.merge`, or `aimless`) and the usual table comes back, computed by
the usual tool rather than by us -- which is the point, since our own half-set
statistics put CC(1/2) at 0.996 for a model whose intensities were 800x
mis-scaled.

Usage:
    uv run python scripts/write_unmerged_mtz.py <pred_dir> <out.mtz> \
        --spacegroup "P 43 21 2" --cell 79.41 79.41 37.84 90 90 90
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("pred_dir", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--spacegroup", default=None, help="e.g. 'P 43 21 2'")
    p.add_argument(
        "--cell", nargs=6, type=float, default=None, metavar=("A", "B", "C", "AL", "BE", "GA")
    )
    p.add_argument(
        "--expt",
        default=None,
        help="DIALS experiments file to read cell and space group from",
    )
    p.add_argument(
        "--unscaled",
        action="store_true",
        help="write qi_mean without the per-observation scale, to show what "
        "the statistics look like when the scale is left in the data",
    )
    return p.parse_args()


def main() -> None:
    a = parse_args()
    import polars as pl
    import reciprocalspaceship as rs

    files = sorted(glob.glob(f"{a.pred_dir}/**/*.parquet", recursive=True))
    if not files:
        raise SystemExit(f"no parquets under {a.pred_dir}")

    have = set(pl.scan_parquet(files[0]).collect_schema().names())
    need = {"H", "K", "L", "qi_mean", "qi_var"}
    missing = need - have
    if missing:
        raise SystemExit(
            f"predictions lack {sorted(missing)}. Add them to the config's "
            f"predict_keys and re-run predict; have {sorted(have)}"
        )
    cols = sorted(need | ({"scale"} & have))
    df = pl.read_parquet(files, columns=cols)

    i = df["qi_mean"].to_numpy().astype(np.float32)
    v = df["qi_var"].to_numpy().astype(np.float32)
    if "scale" in df.columns and not a.unscaled:
        s = df["scale"].to_numpy().astype(np.float32)
        # the likelihood multiplies them, so the measured-gauge intensity is
        # scale * I and its variance scales by the square
        i, v = s * i, (s**2) * v
        print(f"  applied the per-observation scale (median {np.median(s):.4g})")
    elif "scale" not in df.columns:
        print("  no `scale` column: writing the de-scaled intensities, which "
              "inflates every merging statistic that depends on agreement")

    ds = rs.DataSet(
        {
            "H": df["H"].to_numpy().astype(np.int32),
            "K": df["K"].to_numpy().astype(np.int32),
            "L": df["L"].to_numpy().astype(np.int32),
            "I": i,
            "SIGI": np.sqrt(np.clip(v, 1e-12, None)),
        }
    )
    ds = ds.set_index(["H", "K", "L"])
    ds["I"] = ds["I"].astype("J")
    ds["SIGI"] = ds["SIGI"].astype("Q")

    if a.expt:
        import json

        e = json.loads(Path(a.expt).read_text())
        cry = e["crystal"][0]
        import gemmi

        uc = gemmi.UnitCell(*cry["unit_cell"]) if "unit_cell" in cry else None
        if uc:
            ds.cell = uc
        ds.spacegroup = gemmi.SpaceGroup(cry["space_group_hall_symbol"])
    if a.cell:
        import gemmi

        ds.cell = gemmi.UnitCell(*a.cell)
    if a.spacegroup:
        import gemmi

        ds.spacegroup = gemmi.SpaceGroup(a.spacegroup)
    if ds.cell is None or ds.spacegroup is None:
        raise SystemExit("need --cell and --spacegroup (or --expt)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    ds.write_mtz(str(a.out))
    print(f"  {len(ds):,} unmerged observations -> {a.out}")
    print(f"  spacegroup {ds.spacegroup.short_name()}  "
          f"cell {[round(float(x), 2) for x in ds.cell.parameters]}")
    print("\n  merging statistics, by the standard tool:")
    print(f"    phenix.merging_statistics {a.out} ")


if __name__ == "__main__":
    main()
