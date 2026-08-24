"""Compute Wilson-mean F per ASU reflection for amplitude table initialization.

Uses the resolution (d-spacing) of each unique HKL and the Wilson model
E[F] = sqrt(G * exp(-2Bs^2)) to compute the expected amplitude per
reflection. Saves a file that HKLAmplitudeTable can load via
``init_from_wilson``.

Usage
-----
    python scripts/prepare_wilson_init.py <data_dir> [--B 20.0] [--G 1.0]

The B-factor and G scale can be estimated from the data or set manually.
A typical protein B-factor is 15-30 A^2.
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Compute Wilson-mean F initialization for amplitude table."
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument(
        "--ref",
        default="metadata.pt",
        help="Reference/metadata file name (default: metadata.pt)",
    )
    parser.add_argument(
        "--B", type=float, default=20.0,
        help="Overall B-factor in A^2 (default: 20.0)",
    )
    parser.add_argument(
        "--G", type=float, default=1.0,
        help="Overall scale factor (default: 1.0)",
    )
    args = parser.parse_args()

    meta = torch.load(args.data_dir / args.ref, weights_only=False)
    asu_id = meta["asu_id"].long()
    d = meta["d"].float()

    n_hkl = int(asu_id.max()) + 1

    d_per_hkl = torch.zeros(n_hkl)
    seen = torch.zeros(n_hkl, dtype=torch.bool)
    for i in range(len(asu_id)):
        aid = asu_id[i].item()
        if not seen[aid]:
            d_per_hkl[aid] = d[i]
            seen[aid] = True

    s_sq = 1.0 / (4.0 * d_per_hkl.clamp(min=1e-6).pow(2))

    # Wilson model: E[I] = G * exp(-2*B*s^2), E[F] = sqrt(E[I])
    wilson_I = args.G * torch.exp(-2.0 * args.B * s_sq)
    wilson_F = torch.sqrt(wilson_I.clamp(min=1e-12))

    out_path = args.data_dir / "wilson_F_init.pt"
    torch.save({"wilson_F_mean": wilson_F}, out_path)

    print(f"Saved {out_path}: {n_hkl} reflections")
    print(f"  B={args.B:.1f}, G={args.G:.1f}")
    print(f"  F range: {wilson_F.min():.4f} to {wilson_F.max():.4f}")
    print(f"  F median: {wilson_F.median():.4f}")
    print(f"  Use in YAML: init_from_wilson: wilson_F_init.pt")


if __name__ == "__main__":
    main()
