"""Create crystal.yaml from a PDB file.

Usage
-----
    uv run python scripts/create_crystal_yaml.py <pdb_file> <output_dir>
"""

import argparse
from pathlib import Path

import gemmi
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Create crystal.yaml from a PDB file."
    )
    parser.add_argument("pdb", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    st = gemmi.read_structure(str(args.pdb))
    cell = st.cell
    sg = st.spacegroup_hm

    crystal = {
        "cell": [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma],
        "space_group": sg,
    }

    out_path = args.output_dir / "crystal.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(crystal, f, sort_keys=False)

    print(f"Wrote {out_path}")
    print(f"  cell: {crystal['cell']}")
    print(f"  space_group: {sg}")


if __name__ == "__main__":
    main()
