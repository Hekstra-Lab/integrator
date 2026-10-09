"""Add ``asu_id`` to an existing metadata file for the scaling model.

Uses gemmi to map each (H, K, L) to its canonical asymmetric-unit
representative, then assigns a contiguous integer ID per unique
representative.

Usage
-----
    uv run python scripts/prepare_asu_ids.py <data_dir> <space_group> [--ref metadata.pt]

Example
-------
    uv run python scripts/prepare_asu_ids.py /path/to/pytorch_data "P 43 21 2"

After running, the metadata file will contain a new ``asu_id`` key
(int64 tensor, same length as H/K/L) and ``hkl_meta.pt`` will be
written with ``n_hkl`` (int) for the YAML config.
"""

import argparse
from pathlib import Path

import gemmi
import numpy as np
import torch

import reciprocalspaceship as rs


def main():
    parser = argparse.ArgumentParser(
        description="Add asu_id to a metadata/reference file."
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("space_group", type=str)
    parser.add_argument(
        "--ref",
        default="metadata.pt",
        help="Name of the reference/metadata file (default: metadata.pt)",
    )
    parser.add_argument(
        "--anomalous",
        action="store_true",
        help="Keep Friedel pairs separate (for anomalous data).",
    )
    args = parser.parse_args()

    sg = gemmi.SpaceGroup(args.space_group)

    ref_path = args.data_dir / args.ref
    reference = torch.load(ref_path, weights_only=False)

    H = reference["H"].long().numpy()
    K = reference["K"].long().numpy()
    L = reference["L"].long().numpy()
    n_obs = len(H)

    hkl_obs = np.stack([H, K, L], axis=1).astype(np.int32)

    # Map to ASU using reciprocalspaceship (same convention as SFcalculator)
    asu_hkl, isym = rs.utils.hkl_to_asu(hkl_obs, sg)

    if args.anomalous:
        # ISYM even = Friedel minus; negate to get the anomalous ASU form
        is_minus = (isym % 2 == 0)
        canon_hkl = asu_hkl.copy()
        canon_hkl[is_minus] = -canon_hkl[is_minus]
    else:
        canon_hkl = asu_hkl

    # Assign contiguous integer IDs
    canon_to_id: dict[tuple[int, int, int], int] = {}
    asu_ids = np.empty(n_obs, dtype=np.int64)

    for i in range(n_obs):
        key = (int(canon_hkl[i, 0]), int(canon_hkl[i, 1]), int(canon_hkl[i, 2]))
        if key not in canon_to_id:
            canon_to_id[key] = len(canon_to_id)
        asu_ids[i] = canon_to_id[key]

    n_hkl = len(canon_to_id)

    reference["asu_id"] = torch.from_numpy(asu_ids)
    torch.save(reference, ref_path)

    meta_path = args.data_dir / "hkl_meta.pt"
    torch.save({"n_hkl": n_hkl}, meta_path)

    id_to_hkl = torch.zeros(n_hkl, 3, dtype=torch.long)
    for canon, aid in canon_to_id.items():
        id_to_hkl[aid] = torch.tensor(canon)
    hkl_map_path = args.data_dir / "asu_id_to_hkl.pt"
    torch.save(id_to_hkl, hkl_map_path)

    print(f"Added asu_id to {ref_path}")
    print(f"  {n_obs} observations -> {n_hkl} unique ASU reflections")
    print(f"  Average {n_obs / n_hkl:.1f} observations per reflection")
    print(f"  Saved n_hkl={n_hkl} to {meta_path}")
    print(f"  Saved asu_id->(H,K,L) mapping to {hkl_map_path}")
    print(f"  Put n_hkl: {n_hkl} in your YAML config under integrator.args")


if __name__ == "__main__":
    main()
