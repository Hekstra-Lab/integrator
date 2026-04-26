"""Stratified-sample test reflections by resolution and add an `is_test`
boolean to the metadata.pt file the data_module reads.

5% per resolution bin (configurable). Saves either in-place into the
existing metadata.pt or to a sibling file via --out.

Run:
    uv run python scripts/add_is_test.py \
        --metadata /n/.../pytorch_data/metadata.pt \
        --frac 0.05 \
        --n-bins 20 \
        --seed 55
"""

import argparse
from pathlib import Path

import torch


def _bin_by_resolution(d: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Quantile-bin reflections by d. Returns (N,) integer bin label in [0, n_bins).
    Uses interior quantiles so each bin has roughly equal count."""
    qs = torch.linspace(0, 1, n_bins + 1)
    edges = torch.quantile(d.float(), qs)
    # searchsorted on interior edges places each refl in [0, n_bins)
    labels = torch.searchsorted(edges[1:-1], d).long()
    labels.clamp_(0, n_bins - 1)
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", required=True, help="path to metadata.pt")
    ap.add_argument("--frac", type=float, default=0.05,
                    help="fraction of each resolution bin to flag as test")
    ap.add_argument("--n-bins", type=int, default=20,
                    help="number of resolution bins for stratified sampling")
    ap.add_argument("--seed", type=int, default=55)
    ap.add_argument("--out", default=None,
                    help="output path (default: in-place over --metadata)")
    args = ap.parse_args()

    md_path = Path(args.metadata)
    md = torch.load(md_path, weights_only=False)
    if "d" not in md:
        raise SystemExit("metadata.pt has no 'd' column — can't stratify by resolution.")

    d = md["d"]
    if d.dtype != torch.float32:
        d = d.float()
    n = d.numel()

    print(f"Loaded {md_path}: N={n:,} reflections")

    bin_labels = _bin_by_resolution(d, args.n_bins)

    # Stratified sample within each bin
    g = torch.Generator().manual_seed(args.seed)
    is_test = torch.zeros(n, dtype=torch.bool)
    for b in range(args.n_bins):
        idx = torch.where(bin_labels == b)[0]
        n_b = idx.numel()
        if n_b == 0:
            continue
        n_pick = int(round(n_b * args.frac))
        if n_pick == 0:
            continue
        perm = torch.randperm(n_b, generator=g)[:n_pick]
        is_test[idx[perm]] = True
        print(f"  bin {b:>2d}  n={n_b:>7d}  picked={n_pick}")

    # data_module uses is_test as a boolean mask for index selection,
    # so it must be a bool (or int/long) tensor — not float.
    md["is_test"] = is_test.bool()
    print(f"\nTotal: {int(is_test.sum()):,} / {n:,} reflections flagged is_test "
          f"({100*is_test.float().mean():.2f}%)")

    out_path = Path(args.out) if args.out else md_path
    torch.save(md, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
