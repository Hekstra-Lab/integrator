"""Generate R-free flags by unique HKL (asu_id), not by observation.

Randomly selects a fraction of unique asu_ids as the R-free set.
ALL observations of a flagged asu_id are held out from the training
NLL, so the atomic model has never seen those reflections.

Adds ``rfree_flag`` to metadata.pt: 1 = R-free (test), 0 = R-work (train).

Usage
-----
    uv run python scripts/generate_rfree_flags.py \
        --data-dir /path/to/pytorch_data \
        --fraction 0.05 \
        --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ref", default="metadata.pt",
        help="Name of the metadata file (default: metadata.pt)",
    )
    args = parser.parse_args()

    ref_path = args.data_dir / args.ref
    meta = torch.load(ref_path, weights_only=False, map_location="cpu")

    asu_ids = meta["asu_id"].long()
    n_unique = int(asu_ids.max()) + 1
    n_test = max(1, int(n_unique * args.fraction))

    rng = np.random.RandomState(args.seed)
    test_hkls = set(rng.choice(n_unique, n_test, replace=False).tolist())

    rfree_per_asu = torch.zeros(n_unique, dtype=torch.bool)
    for aid in test_hkls:
        rfree_per_asu[aid] = True

    rfree_per_obs = rfree_per_asu[asu_ids]
    meta["rfree_flag"] = rfree_per_obs

    torch.save(meta, ref_path)

    n_obs_test = rfree_per_obs.sum().item()
    n_obs_total = len(asu_ids)
    print(f"R-free flags added to {ref_path}")
    print(f"  {n_test} / {n_unique} unique HKLs flagged ({100*n_test/n_unique:.1f}%)")
    print(f"  {n_obs_test} / {n_obs_total} observations flagged ({100*n_obs_test/n_obs_total:.1f}%)")


if __name__ == "__main__":
    main()
