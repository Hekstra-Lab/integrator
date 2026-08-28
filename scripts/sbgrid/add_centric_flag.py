"""Add a `centric` flag to a dataset's metadata.

Wilson statistics are not the same for every reflection: an acentric |F|^2 is
exponential, a centric one is a chi-squared with one degree of freedom. The
prior in `wilson_loss.py` needs to tell them apart, and whether a reflection
is centric depends on the space group, which the shoebox datasets do not
record -- only H, K, L.

The space group is read from the DIALS experiments the shoeboxes were cut
from, so the flag agrees with the symmetry the data were actually integrated
under rather than one supplied by hand.

Existing datasets are updated in place; the flag is additive, and a dataset
without it still trains (the loss falls back to all-acentric).

Usage:
    python scripts/sbgrid/add_centric_flag.py --dataset-dir <combined dataset> \
        --expt <path to an integrated.expt>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Add a centric flag to metadata")
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument(
        "--expt",
        type=Path,
        default=None,
        help="a DIALS .expt carrying the crystal symmetry; defaults to the "
        "first integrated.expt under the dataset's dials_reference",
    )
    p.add_argument("--space-group", default=None, help="override, e.g. 'P 21 21 21'")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def space_group_from_expt(path: Path) -> str:
    from dxtbx.model.experiment_list import ExperimentListFactory

    experiments = ExperimentListFactory.from_json_file(str(path), check_format=False)
    return str(experiments[0].crystal.get_space_group().info())


def main():
    args = parse_args()
    meta_path = args.dataset_dir / "metadata.npy"
    if not meta_path.exists():
        raise SystemExit(f"no metadata.npy in {args.dataset_dir}")

    name = args.space_group
    if name is None:
        expt = args.expt
        if expt is None:
            root = args.dataset_dir.parent.parent
            found = sorted(root.glob("dials_reference/*/integrated.expt"))
            if not found:
                raise SystemExit(
                    "no --expt given and no integrated.expt found; pass "
                    "--expt or --space-group"
                )
            expt = found[0]
        name = space_group_from_expt(expt)
        print(f"space group {name}  (from {expt.parent.name})")

    from integrator.io import load_metadata, save_data

    meta = load_metadata(meta_path)
    for key in ("H", "K", "L"):
        if key not in meta:
            raise SystemExit(f"metadata has no {key}; cannot label centrics")
    hkl = np.column_stack(
        [np.asarray(meta[k]).ravel().astype(np.int32) for k in ("H", "K", "L")]
    )

    import gemmi

    ops = gemmi.SpaceGroup(name).operations()
    # unique indices first: a dataset has millions of observations but far
    # fewer distinct reflections, and the symmetry test is per-index
    unique, inverse = np.unique(hkl, axis=0, return_inverse=True)
    flags = np.fromiter(
        (ops.is_reflection_centric(tuple(int(v) for v in h)) for h in unique),
        dtype=bool,
        count=len(unique),
    )
    centric = flags[inverse]

    print(f"{len(hkl):,} observations, {len(unique):,} distinct reflections")
    print(f"centric: {centric.sum():,} ({100 * centric.mean():.1f}%)")
    if "d" in meta:
        d = np.asarray(meta["d"]).ravel()
        for hi, lo in ((99, 3.0), (3.0, 2.2), (2.2, 1.9), (1.9, 0)):
            m = (d <= hi) & (d > lo)
            if m.sum():
                print(f"  {hi:5.1f}-{lo:4.1f} A: {100 * centric[m].mean():5.1f}%")

    if args.dry_run:
        print("\ndry run, metadata not written")
        return 0
    meta["centric"] = centric
    save_data(meta, meta_path)
    print(f"\nwrote {meta_path} with a centric column")
    return 0


if __name__ == "__main__":
    sys.exit(main())
