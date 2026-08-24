"""Diagnose asu_id -> SFcalculator HKL mapping failures.

Usage
-----
    uv run python scripts/diagnose_asu_mapping.py \
        --data-dir /path/to/pytorch_data \
        --pdb /path/to/9b7c.pdb \
        --space-group "P 43 21 2" \
        --dmin 1.1 \
        --anomalous
"""

import argparse
from pathlib import Path

import gemmi
import numpy as np
import torch

if not hasattr(gemmi.UnitCell, "fractionalization_matrix"):
    gemmi.UnitCell.fractionalization_matrix = property(
        lambda self: self.frac.mat
    )
if not hasattr(gemmi.UnitCell, "orthogonalization_matrix"):
    gemmi.UnitCell.orthogonalization_matrix = property(
        lambda self: self.orth.mat
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--space-group", type=str, required=True)
    parser.add_argument("--dmin", type=float, default=1.1)
    parser.add_argument("--wavelength", type=float, default=1.0375)
    parser.add_argument("--anomalous", action="store_true")
    args = parser.parse_args()

    # 1. Load the asu_id_to_hkl mapping
    id_to_hkl = torch.load(
        args.data_dir / "asu_id_to_hkl.pt", weights_only=False, map_location="cpu"
    )
    n_asu_ids = len(id_to_hkl)
    print(f"asu_id_to_hkl.pt: {n_asu_ids} unique asu_ids")

    # 2. Load metadata to check how it was prepared
    meta = torch.load(
        args.data_dir / "metadata.pt", weights_only=False, map_location="cpu"
    )
    n_obs = len(meta["H"])
    has_asu_id = "asu_id" in meta
    print(f"metadata.pt: {n_obs} observations, has asu_id: {has_asu_id}")
    if has_asu_id:
        n_unique = int(meta["asu_id"].max()) + 1
        print(f"  max(asu_id)+1 = {n_unique}")

    # 3. Check resolution range of the data
    sg = gemmi.SpaceGroup(args.space_group)
    cell_yaml = args.data_dir / "crystal.yaml"
    if cell_yaml.exists():
        import yaml
        crystal = yaml.safe_load(cell_yaml.read_text())
        cell = gemmi.UnitCell(*crystal["cell"])
    else:
        structure = gemmi.read_structure(str(args.pdb))
        cell = structure.cell

    H = meta["H"].numpy()
    K = meta["K"].numpy()
    L = meta["L"].numpy()

    d_values = np.array([
        cell.calculate_d([int(H[i]), int(K[i]), int(L[i])])
        for i in range(min(n_obs, 100000))
    ])
    print(f"\nData resolution range (first 100k obs):")
    print(f"  d_min = {d_values.min():.3f} A")
    print(f"  d_max = {d_values.max():.3f} A")
    print(f"  Observations with d < {args.dmin}: {(d_values < args.dmin).sum()}")
    print(f"  Observations with d >= {args.dmin}: {(d_values >= args.dmin).sum()}")

    # 4. Check resolution of unmapped asu_ids
    asu_d_min = {}
    for i in range(min(n_obs, n_obs)):
        aid = int(meta["asu_id"][i]) if has_asu_id else None
        if aid is not None and aid not in asu_d_min:
            h, k, l = int(H[i]), int(K[i]), int(L[i])
            asu_d_min[aid] = cell.calculate_d([h, k, l])

    # 5. Build SFcalculator lookup
    from SFC_Torch import SFcalculator as SFC

    sfc = SFC(
        pdbmodel=str(args.pdb),
        dmin=args.dmin,
        anomalous=args.anomalous,
        wavelength=args.wavelength,
    )
    sfc.inspect_data()
    print(f"\nSFcalculator:")
    print(f"  Hasu_array: {len(sfc.Hasu_array)} HKLs (dmin={args.dmin}, anomalous={args.anomalous})")
    print(f"  solvent_pct: {sfc.solventpct:.1%}")

    # Build lookup (always include Friedel mates - SFcalculator's Hasu_array
    # contains one member per Friedel pair even with anomalous=True)
    op_list = list(sg.operations())
    lookup = {}
    for idx in range(len(sfc.Hasu_array)):
        h, k, l = int(sfc.Hasu_array[idx, 0]), int(sfc.Hasu_array[idx, 1]), int(sfc.Hasu_array[idx, 2])
        for op in op_list:
            hkl_rot = op.apply_to_hkl([h, k, l])
            lookup[tuple(hkl_rot)] = idx
            lookup[(-hkl_rot[0], -hkl_rot[1], -hkl_rot[2])] = idx

    # 6. Check each asu_id
    mapped = 0
    unmapped_beyond_dmin = 0
    unmapped_within_dmin = 0
    unmapped_hkls = []

    for aid in range(n_asu_ids):
        h, k, l = int(id_to_hkl[aid, 0]), int(id_to_hkl[aid, 1]), int(id_to_hkl[aid, 2])
        d = cell.calculate_d([h, k, l]) if (h, k, l) != (0, 0, 0) else 999.0
        hkl_key = (h, k, l)

        if hkl_key in lookup:
            mapped += 1
        elif d < args.dmin:
            unmapped_beyond_dmin += 1
        else:
            unmapped_within_dmin += 1
            if len(unmapped_hkls) < 10:
                unmapped_hkls.append((h, k, l, d))

    print(f"\nMapping results:")
    print(f"  Mapped:                     {mapped}")
    print(f"  Unmapped (d < dmin={args.dmin}):   {unmapped_beyond_dmin}")
    print(f"  Unmapped (d >= dmin={args.dmin}):  {unmapped_within_dmin}")

    if unmapped_within_dmin > 0:
        print(f"\n  First unmapped HKLs within resolution (should be 0!):")
        for h, k, l, d in unmapped_hkls:
            # Check if Friedel mate is in lookup
            friedel = (-h, -k, -l)
            friedel_found = friedel in lookup
            print(f"    ({h:3d}, {k:3d}, {l:3d})  d={d:.3f}  Friedel in lookup: {friedel_found}")

    # 7. Check for anomalous flag issue
    if args.anomalous:
        n_friedel_only = 0
        for aid in range(min(n_asu_ids, 10000)):
            h, k, l = int(id_to_hkl[aid, 0]), int(id_to_hkl[aid, 1]), int(id_to_hkl[aid, 2])
            if (h, k, l) not in lookup and (-h, -k, -l) in lookup:
                n_friedel_only += 1
        if n_friedel_only > 0:
            print(f"\n  WARNING: {n_friedel_only} asu_ids (of first 10k) match "
                  f"only via Friedel mate - likely prepared without --anomalous")


if __name__ == "__main__":
    main()
