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
import torch


def asu_key(op_list, h, k, l, anomalous=False):
    """Return the canonical (h, k, l) under space-group symmetry.

    When ``anomalous=False`` (default), Friedel mates are merged:
    (h,k,l) and (-h,-k,-l) map to the same key.

    When ``anomalous=True``, Friedel mates are kept separate so that
    F(+) and F(-) get independent variational parameters — preserving
    Bijvoet differences.
    """
    candidates = []
    for op in op_list:
        hkl_rot = op.apply_to_hkl([h, k, l])
        candidates.append(tuple(hkl_rot))
        if not anomalous:
            candidates.append((-hkl_rot[0], -hkl_rot[1], -hkl_rot[2]))
    return min(candidates)


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
    op_list = list(sg.operations())

    ref_path = args.data_dir / args.ref
    reference = torch.load(ref_path, weights_only=False)

    H = reference["H"].long()
    K = reference["K"].long()
    L = reference["L"].long()
    n_obs = len(H)

    canon_to_id: dict[tuple[int, int, int], int] = {}
    asu_ids = torch.empty(n_obs, dtype=torch.long)

    for i in range(n_obs):
        canon = asu_key(op_list, int(H[i]), int(K[i]), int(L[i]), args.anomalous)
        if canon not in canon_to_id:
            canon_to_id[canon] = len(canon_to_id)
        asu_ids[i] = canon_to_id[canon]

    n_hkl = len(canon_to_id)

    reference["asu_id"] = asu_ids
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
