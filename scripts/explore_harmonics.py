"""Explore harmonic structure in a Laue dataset.

Deduplicates shoeboxes by harmonic family (gcd_reduce on Miller indices),
reports statistics, and visualizes a few examples showing that harmonics
share identical pixel data.

Usage:
    uv run python scripts/explore_harmonics.py \
        --metadata /n/.../pytorch_data/metadata.pt \
        --counts   /n/.../pytorch_data/counts.npy \
        --out      /n/.../pytorch_data/harmonic_exploration.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=str, required=True)
    p.add_argument("--counts", type=str, required=True)
    p.add_argument("--refl", type=str, default=None,
                   help="reflections_.refl file to get per-refl image id")
    p.add_argument("--out", type=str, default="harmonic_exploration.png")
    p.add_argument("--n-examples", type=int, default=5)
    return p.parse_args()


def harmonic_families(hkl: np.ndarray, image_id: np.ndarray):
    """Group reflections into harmonic families.

    Two reflections are harmonics iff they are on the SAME image AND their
    primitive Miller indices match: (nh,nk,nl)/gcd = (h,k,l)/gcd.

    Args:
        hkl: (N, 3) int Miller indices.
        image_id: (N,) int per-reflection image/experiment identifier.
    """
    hkl = hkl.astype(np.int64)
    gcd = np.gcd.reduce(np.abs(hkl), axis=-1)
    gcd = np.where(gcd == 0, 1, gcd)
    primitive = hkl // gcd[..., None]

    dtype = np.dtype([
        ("img", "i4"),
        ("ph", "i4"), ("pk", "i4"), ("pl", "i4"),
    ])
    keys = np.empty(len(hkl), dtype=dtype)
    keys["img"] = image_id.astype(np.int32)
    keys["ph"] = primitive[:, 0]
    keys["pk"] = primitive[:, 1]
    keys["pl"] = primitive[:, 2]

    _, family_id = np.unique(keys, return_inverse=True)
    return family_id, gcd


def main():
    args = parse_args()
    meta = torch.load(args.metadata, weights_only=True, map_location="cpu")

    hkl = np.stack([meta["H"].numpy(), meta["K"].numpy(), meta["L"].numpy()], axis=-1).astype(int)

    x = meta["xyzcal.px.0"].numpy()
    y = meta["xyzcal.px.1"].numpy()

    if args.refl is not None:
        from dials.array_family import flex
        refl = flex.reflection_table.from_file(args.refl)
        image_id = np.array(refl["id"]).astype(int)
        print(f"loaded image_id from {args.refl}: "
              f"{len(np.unique(image_id))} unique images")
    elif "image_num" in meta:
        image_id = meta["image_num"].numpy().astype(int)
    else:
        raise ValueError(
            "no image identifier available. Pass --refl reflections_.refl "
            "or add image_num to metadata.pt via mksbox-laue."
        )

    n_refls = len(hkl)
    print(f"total reflections: {n_refls}")

    family_id, order = harmonic_families(hkl, image_id)
    n_families = int(family_id.max()) + 1
    print(f"unique harmonic families: {n_families}")
    print(f"dedup ratio: {n_refls / n_families:.2f}x")

    sizes = np.bincount(family_id)
    print(f"\nfamily size distribution:")
    for s in range(1, min(sizes.max() + 1, 8)):
        count = (sizes == s).sum()
        print(f"  size {s}: {count} families ({100 * count / n_families:.1f}%)")
    if sizes.max() >= 8:
        count = (sizes >= 8).sum()
        print(f"  size 8+: {count} families")

    max_order = int(order.max())
    print(f"\nharmonic order (n = gcd) distribution (max n={max_order}):")
    for n in range(1, max_order + 1):
        count = (order == n).sum()
        if count > 0:
            print(f"  n={n}: {count} refls ({100 * count / n_refls:.1f}%)")

    n_pixels = None
    hw = None

    # Find families with 2-4 members for visualization
    multi_families = np.where((sizes >= 2) & (sizes <= 4))[0]
    if len(multi_families) == 0:
        print("\nno multi-member families found for visualization")
        return

    # Pick families with high total intensity for interesting visuals
    intensities = meta.get("intensity.sum.value", torch.zeros(n_refls)).numpy()
    # Vectorized: sum intensities per family, then pick top multi-member ones
    family_intensity = np.bincount(family_id, weights=intensities, minlength=n_families)
    top_families = multi_families[np.argsort(-family_intensity[multi_families])]
    examples = top_families[: args.n_examples]

    # Build member lists for examples only
    sort_idx = np.argsort(family_id)
    sorted_fids = family_id[sort_idx]
    boundaries = np.searchsorted(sorted_fids, examples, side="left")
    boundaries_r = np.searchsorted(sorted_fids, examples, side="right")

    all_member_ids = []
    for l, r in zip(boundaries, boundaries_r):
        all_member_ids.extend(sort_idx[l:r].tolist())

    counts_mm = np.load(args.counts, mmap_mode="r")
    n_pixels = counts_mm.shape[1]
    hw = int(np.sqrt(n_pixels))
    loaded = {idx: counts_mm[idx].copy() for idx in all_member_ids}
    del counts_mm

    n_ex = len(examples)
    max_members = max(sizes[fid] for fid in examples)
    fig, axes = plt.subplots(n_ex, max_members + 1, figsize=(3 * (max_members + 1), 3 * n_ex))
    if n_ex == 1:
        axes = axes[np.newaxis, :]

    for row, (fid, bl, br) in enumerate(zip(examples, boundaries, boundaries_r)):
        members = sort_idx[bl:br]
        members = members[np.argsort(order[members])]

        for col in range(max_members + 1):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])

            if col < len(members):
                idx = members[col]
                sbox = loaded[idx].reshape(hw, hw)
                ax.imshow(sbox, cmap="viridis", origin="lower")
                h, k, l = hkl[idx]
                lam = meta["wavelength"][idx].item() if "wavelength" in meta else 0
                d_val = meta["d"][idx].item() if "d" in meta else 0
                xi, yi = x[idx], y[idx]
                img = image_id[idx]
                ax.set_title(
                    f"({h},{k},{l}) n={order[idx]}\n"
                    f"λ={lam:.4f} d={d_val:.2f}\n"
                    f"img={img} xy=({xi:.1f},{yi:.1f})",
                    fontsize=7,
                )
            elif col == len(members) and len(members) >= 2:
                sbox0 = loaded[members[0]].astype(float)
                sbox1 = loaded[members[1]].astype(float)
                diff = np.abs(sbox0 - sbox1)
                ax.imshow(diff.reshape(hw, hw), cmap="hot", origin="lower")
                ax.set_title(
                    f"|Δpixels| (0 vs 1)\n"
                    f"max={diff.max():.0f} sum={diff.sum():.0f}",
                    fontsize=8,
                )
            else:
                ax.axis("off")

        axes[row, 0].set_ylabel(f"family {fid}\n{len(members)} harmonics", fontsize=9)

    fig.suptitle(
        f"Harmonic families: {n_families} unique / {n_refls} total "
        f"({n_refls / n_families:.1f}x dedup)\n"
        f"Right column: absolute pixel difference between primary and 2nd harmonic",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
