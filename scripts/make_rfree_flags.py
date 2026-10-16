"""Write one canonical R-free flag set for a dataset, complete to `--dmin`.

Every refinement chain in this project generated its own R-free set: both
`laue_refine_pass1.eff` and the 1.20 eff it descends from set
`r_free_flags.generate = True`, so each merged MTZ was held out on a different
subset, and `refinement.main.random_seed` moved that subset too.  Two numbers
produced that way differ by the flag draw as well as by the data, which on weak
Laue data is worth about 0.008 in `R_free` -- the same size as the model
effects being measured.

The fix is a flag file that is complete over the ASU, so no MTZ can miss a
reflection and no two MTZs can disagree about which reflections are free.
Pass it to phenix with `generate = False` and refinements become comparable
across arms, epochs, resolution shells and phenix versions.

Flags are a uniform random subset of the complete set at a fixed seed, which
makes the file reproducible from this command alone.  Value 1 is the test set,
matching the `test_flag_value = 1` the effs already ask for.  The set is not
anomalous: phenix applies a flag to both Bijvoet mates, which is what keeps a
Friedel pair out of the working set together.

Usage
-----
    uv run python scripts/make_rfree_flags.py \\
        --cell 78.3121 78.3121 37.2343 90 90 90 \\
        --space-group "P 43 21 2" --dmin 1.3 \\
        -o reference_data/hewl_1118_rfree.mtz

    uv run python scripts/make_rfree_flags.py --from-mtz merged.mtz --dmin 1.3 -o flags.mtz
"""

import argparse
from pathlib import Path

import gemmi
import numpy as np

FLAG_LABEL = "R-free-flags"
TEST_FLAG_VALUE = 1


def complete_flags(
    cell: gemmi.UnitCell,
    spacegroup: gemmi.SpaceGroup,
    dmin: float,
    fraction: float = 0.1,
    seed: int = 2026,
) -> gemmi.Mtz:
    """A complete ASU to `dmin` with `fraction` of it flagged as the test set."""
    hkl = gemmi.make_miller_array(cell, spacegroup, dmin, 0.0, True)
    n_free = int(round(fraction * len(hkl)))
    rng = np.random.default_rng(seed)
    flags = np.zeros(len(hkl), dtype=np.int32)
    flags[rng.choice(len(hkl), size=n_free, replace=False)] = TEST_FLAG_VALUE

    mtz = gemmi.Mtz(with_base=True)
    mtz.cell = cell
    mtz.spacegroup = spacegroup
    mtz.add_dataset("free")
    mtz.add_column(FLAG_LABEL, "I")
    mtz.set_data(np.hstack([hkl.astype(np.float32), flags.reshape(-1, 1).astype(np.float32)]))
    return mtz


def _symmetry_from_mtz(path: Path) -> tuple[gemmi.UnitCell, gemmi.SpaceGroup]:
    mtz = gemmi.read_mtz_file(str(path))
    return mtz.cell, mtz.spacegroup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write one canonical R-free flag set, complete to --dmin."
    )
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument(
        "--from-mtz",
        type=Path,
        default=None,
        help="Read the cell and space group off this MTZ instead of naming them.",
    )
    parser.add_argument("--cell", type=float, nargs=6, default=None)
    parser.add_argument("--space-group", type=str, default=None)
    parser.add_argument(
        "--dmin",
        type=float,
        required=True,
        help="Generate past the finest shell any refinement will ask for; a "
        "reflection with no flag is one phenix would have to invent.",
    )
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.from_mtz is not None:
        cell, spacegroup = _symmetry_from_mtz(args.from_mtz)
    elif args.cell is not None and args.space_group is not None:
        cell = gemmi.UnitCell(*args.cell)
        spacegroup = gemmi.SpaceGroup(args.space_group)
    else:
        parser.error("give --from-mtz, or both --cell and --space-group")

    mtz = complete_flags(
        cell, spacegroup, args.dmin, fraction=args.fraction, seed=args.seed
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mtz.write_to_file(str(args.out))

    data = np.array(gemmi.read_mtz_file(str(args.out)))
    flags = data[:, 3].astype(int)
    print(
        f"{args.out}: {len(flags)} reflections to {args.dmin} A in "
        f"{spacegroup.hm}, {int((flags == TEST_FLAG_VALUE).sum())} free "
        f"({(flags == TEST_FLAG_VALUE).mean():.4f}), seed {args.seed}"
    )


if __name__ == "__main__":
    main()
