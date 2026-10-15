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

If the metadata already carries ``asu_id``, the ids are left alone and only
``asu_id_to_hkl.pt`` and ``hkl_meta.pt`` are (re)written, because an id is
the index into a trained model's ``hkl_table`` and renumbering it silently
invalidates every checkpoint trained on the file.  Pass ``--reassign`` to
renumber on purpose.
"""

import argparse
from pathlib import Path

import gemmi
import numpy as np
import torch

from integrator.utils.asu import canonical_hkl, resolve_asu_id_table


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
    parser.add_argument(
        "--reassign",
        action="store_true",
        help=(
            "Renumber asu_id even though the file already has one. "
            "Invalidates checkpoints trained on the existing ids."
        ),
    )
    args = parser.parse_args()

    sg = gemmi.SpaceGroup(args.space_group)

    ref_path = args.data_dir / args.ref
    # this project writes metadata as .npy (numpy pickle) while the merging
    # branch wrote .pt; accept either so the script works on both layouts
    is_npy = str(ref_path).endswith(".npy")
    if is_npy:
        reference = np.load(ref_path, allow_pickle=True).item()
    else:
        reference = torch.load(ref_path, weights_only=False)

    # .npy metadata holds numpy arrays, .pt holds tensors
    def _as_int(a):
        if hasattr(a, "numpy"):
            a = a.numpy()
        return np.rint(np.asarray(a, dtype=np.float64)).astype(np.int64)

    H = _as_int(reference["H"])
    K = _as_int(reference["K"])
    L = _as_int(reference["L"])
    n_obs = len(H)

    hkl_obs = np.stack([H, K, L], axis=1).astype(np.int32)

    existing = reference.get("asu_id")
    if existing is not None and not args.reassign:
        # Keep the ids a trained model already indexes by; resolve only the
        # Miller index each one stands for, and let the data say which
        # Friedel grouping that is.
        id_to_hkl_np, info = resolve_asu_id_table(
            _as_int(existing), hkl_obs, sg
        )
        n_hkl = info["n_hkl"]
        print(f"{ref_path} already has asu_id; keeping it (no renumbering)")
        print(f"  grouping read off the data: {info['grouping']}")
        print(f"  {n_obs} observations -> {n_hkl} ids")
        if info["n_unobserved_ids"]:
            print(
                f"  WARNING: {info['n_unobserved_ids']} ids never occur; "
                "their rows are (0, 0, 0)"
            )
        id_to_hkl = torch.from_numpy(id_to_hkl_np.astype(np.int64))
    else:
        canon_hkl = canonical_hkl(hkl_obs, sg, anomalous=args.anomalous)

        # Assign contiguous integer IDs
        canon_to_id: dict[tuple[int, int, int], int] = {}
        asu_ids = np.empty(n_obs, dtype=np.int64)

        for i in range(n_obs):
            key = (
                int(canon_hkl[i, 0]),
                int(canon_hkl[i, 1]),
                int(canon_hkl[i, 2]),
            )
            if key not in canon_to_id:
                canon_to_id[key] = len(canon_to_id)
            asu_ids[i] = canon_to_id[key]

        n_hkl = len(canon_to_id)

        if existing is not None:
            print(f"  --reassign: renumbering the existing asu_id in {ref_path}")
        reference["asu_id"] = torch.from_numpy(asu_ids)
        if is_npy:
            np.save(ref_path, reference, allow_pickle=True)
        else:
            torch.save(reference, ref_path)
        print(f"Added asu_id to {ref_path}")
        print(
            f"  {n_obs} observations -> {n_hkl} unique ASU reflections"
            f" ({'anomalous' if args.anomalous else 'pooled'})"
        )

        id_to_hkl = torch.zeros(n_hkl, 3, dtype=torch.long)
        for canon, aid in canon_to_id.items():
            id_to_hkl[aid] = torch.tensor(canon)

    meta_path = args.data_dir / "hkl_meta.pt"
    torch.save({"n_hkl": n_hkl}, meta_path)

    hkl_map_path = args.data_dir / "asu_id_to_hkl.pt"
    torch.save(id_to_hkl, hkl_map_path)

    print(f"  Average {n_obs / n_hkl:.1f} observations per reflection")
    print(f"  Saved n_hkl={n_hkl} to {meta_path}")
    print(f"  Saved asu_id->(H,K,L) mapping to {hkl_map_path}")
    print(f"  Put n_hkl: {n_hkl} in your YAML config under integrator.args")


if __name__ == "__main__":
    main()
