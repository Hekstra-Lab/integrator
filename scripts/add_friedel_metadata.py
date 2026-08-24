"""Add Friedel-pair metadata for the anomalous-preserving merge terms.

Adds three per-reflection keys to an existing metadata/reference file, in the
same row order (so it stays aligned through the data module's load-time
filtering):

    nonanom_id   - Friedel-POOLED reflection id: both mates (H and -H) share an
                   id (no anomalous split). Used as the loader `group_by_key` so
                   I(+) and I(-) of a reflection land in the same batch, and as
                   the grouping for the Friedel-pooled consistency target.
    friedel_plus - bool, the I(+) member of the pair (ISYM odd, the hasu form),
                   matching scripts/prepare_asu_ids.py's --anomalous convention.
    centric      - bool, centric reflection (I(+)==I(-) by symmetry), the
                   zero-anomalous control for the centric-anchor loss.

This complements the existing `asu_id` (the Friedel-SEPARATE anomalous id from
prepare_asu_ids.py --anomalous), which is left untouched: the model still merges
I(+) and I(-) separately, but the new keys let it (a) co-locate mates, (b) pin
the scale on centrics, and (c) couple mates with a double-Wilson prior.

Run on the cluster in the dials / refltorch env, pointing --ref at whatever
metadata the config already loads (e.g. the absorption_sh-augmented file so the
output keeps that key too):

    uv run python scripts/add_friedel_metadata.py \
        /n/.../pytorch_data "P 43 21 2" \
        --ref metadata_sh.pt --out metadata_sh_friedel.pt

Then point the loader's `reference:` at the --out file and set
`group_by_key: nonanom_id`.
"""

import argparse
from pathlib import Path

import gemmi
import numpy as np
import torch

import reciprocalspaceship as rs


def main():
    ap = argparse.ArgumentParser(description="Add Friedel-pair metadata.")
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("space_group", type=str)
    ap.add_argument(
        "--ref",
        default="metadata.pt",
        help="Input reference/metadata file name (default: metadata.pt).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output file name (default: overwrite --ref in place).",
    )
    ap.add_argument(
        "--cell",
        type=float,
        nargs=6,
        default=None,
        metavar=("A", "B", "C", "ALPHA", "BETA", "GAMMA"),
        help="Unit cell. Centric labeling is cell-independent, so a placeholder "
        "is used when omitted; pass the real cell only if rs complains.",
    )
    args = ap.parse_args()

    sg = gemmi.SpaceGroup(args.space_group)
    ref_path = args.data_dir / args.ref
    out_path = args.data_dir / (args.out or args.ref)
    reference = torch.load(ref_path, weights_only=False)

    H = reference["H"].long().numpy()
    K = reference["K"].long().numpy()
    L = reference["L"].long().numpy()
    n_obs = len(H)
    hkl = np.stack([H, K, L], axis=1).astype(np.int32)

    # ASU mapping (same convention as prepare_asu_ids.py / SFcalculator). ISYM
    # odd = F(+)/hasu form, even = F(-). The pooled canonical is the asu
    # representative with NO sign flip, so both mates share it.
    asu_hkl, isym = rs.utils.hkl_to_asu(hkl, sg)
    friedel_plus = (isym % 2 == 1)

    pooled_to_id: dict[tuple[int, int, int], int] = {}
    nonanom_id = np.empty(n_obs, dtype=np.int64)
    for i in range(n_obs):
        key = (int(asu_hkl[i, 0]), int(asu_hkl[i, 1]), int(asu_hkl[i, 2]))
        if key not in pooled_to_id:
            pooled_to_id[key] = len(pooled_to_id)
        nonanom_id[i] = pooled_to_id[key]

    # Centric flags (cell-independent; rs needs a cell to build the DataSet).
    cell = args.cell if args.cell is not None else [1.0, 1.0, 1.0, 90.0, 90.0, 90.0]
    ds = rs.DataSet(
        {"H": H, "K": K, "L": L},
        cell=gemmi.UnitCell(*cell),
        spacegroup=sg,
    ).set_index(["H", "K", "L"])
    centric = ds.label_centrics()["CENTRIC"].to_numpy().astype(bool)

    reference["nonanom_id"] = torch.from_numpy(nonanom_id)
    reference["friedel_plus"] = torch.from_numpy(friedel_plus.astype(bool))
    reference["centric"] = torch.from_numpy(centric)
    torch.save(reference, out_path)

    n_pooled = len(pooled_to_id)
    print(f"Wrote {out_path}")
    print(f"  {n_obs} observations -> {n_pooled} Friedel-pooled reflections")
    print(f"  centric fraction: {centric.mean():.3f}")
    print(f"  F(+) fraction:    {friedel_plus.mean():.3f}")
    print("  Set the loader's reference to this file and group_by_key: nonanom_id")


if __name__ == "__main__":
    main()
