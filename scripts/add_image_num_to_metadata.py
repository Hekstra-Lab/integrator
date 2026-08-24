"""Add image_num and is_test columns to an existing metadata.pt.

Usage:
    python scripts/add_image_num_to_metadata.py \
        --metadata /n/.../pytorch_data/metadata.pt \
        --refl /n/.../pytorch_data/reflections_.refl
"""

import argparse

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--refl", type=str, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()

    from dials.array_family import flex

    refl = flex.reflection_table.from_file(args.refl)
    image_id = np.array(refl["id"]).astype(np.int32)
    n = len(image_id)
    print(f"loaded {n} refls, {len(np.unique(image_id))} unique images")

    meta = torch.load(args.metadata, weights_only=True, map_location="cpu")
    assert n == len(next(iter(meta.values()))), (
        f"length mismatch: refl has {n}, "
        f"metadata has {len(next(iter(meta.values())))}"
    )

    meta["image_num"] = torch.from_numpy(image_id).float()

    rng = np.random.default_rng(42)
    is_test = rng.random(n) < args.test_fraction
    meta["is_test"] = torch.from_numpy(is_test)
    print(f"is_test: {is_test.sum()} / {n} ({100 * is_test.mean():.1f}%)")

    torch.save(meta, args.metadata)
    print(f"wrote image_num + is_test to {args.metadata}")


if __name__ == "__main__":
    main()
