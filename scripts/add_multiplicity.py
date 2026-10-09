"""Add a `multiplicity` column so the Wilson prior can be tempered by 1/N.

The Wilson prior describes ONE unique reflection, but the loss applies it once per
observation. With N observations sharing a latent it therefore enters N times, and
the product over observations only reconstructs the correct joint posterior if each
copy is raised to the power 1/N:

    prod_i [ p(I)^(1/N) L_i(I) ]  =  p(I) prod_i L_i(I)

Since the KL weight is the prior's exponent in the ELBO, that is a per-reflection
weight of 1/N. This writes N for each observation.

Which symmetry N is counted under matters, and the safe choice is the LOWEST one the
signal lives in. For an electric-field experiment the field breaks C2 to P1, so a
reflection and its 2-fold mate are physically distinct and each needs its own prior;
counting them as one latent would shrink the pair toward a common value and destroy
exactly the difference being measured -- the same error as applying a Wilson prior per
Bijvoet hemisphere. Friedel mates are folded together, since I(h) = I(-h) holds to the
size of the anomalous signal, which is negligible for a native protein.

N is also counted WITHIN a timepoint by default: different pump-probe delays are
different physical states, so they do not share a latent.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml


def fold_friedel(hkl: np.ndarray) -> np.ndarray:
    """Map each index triple to a canonical representative of the Friedel pair."""
    flip = (hkl[:, 2] < 0) | (
        (hkl[:, 2] == 0)
        & ((hkl[:, 1] < 0) | ((hkl[:, 1] == 0) & (hkl[:, 0] < 0)))
    )
    return np.where(flip[:, None], -hkl, hkl)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="directory holding metadata.pt")
    ap.add_argument(
        "--group-key",
        default="sweep_id",
        help="metadata column whose value separates physical states (use 'none' to pool)",
    )
    ap.add_argument(
        "--group-map",
        default=None,
        help="YAML mapping group-key values to state labels, e.g. sweep_id -> timepoint. "
        "Without it each distinct group-key value is its own state.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    meta_path = data_dir / "metadata.pt"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found")
    meta = torch.load(meta_path, weights_only=False)

    hkl = np.stack(
        [np.asarray(meta[k]).astype(np.int64).reshape(-1) for k in ("H", "K", "L")],
        axis=1,
    )
    n = len(hkl)
    canon = fold_friedel(hkl)

    if args.group_key.lower() == "none":
        state = np.zeros(n, dtype=np.int64)
        print("pooling every observation into one state")
    else:
        if args.group_key not in meta:
            raise SystemExit(
                f"metadata has no '{args.group_key}'; pass --group-key none to pool"
            )
        raw = np.asarray(meta[args.group_key]).astype(np.int64).reshape(-1)
        if args.group_map:
            mapping = yaml.safe_load(Path(args.group_map).read_text())
            labels = sorted({str(v) for v in mapping.values()})
            lut = {int(k): labels.index(str(v)) for k, v in mapping.items()}
            missing = sorted(set(raw.tolist()) - set(lut))
            if missing:
                raise SystemExit(f"{args.group_key} values absent from the map: {missing}")
            state = np.array([lut[int(v)] for v in raw], dtype=np.int64)
            print(f"grouped {len(set(raw.tolist()))} {args.group_key} values into "
                  f"{len(labels)} states: {labels}")
        else:
            _, state = np.unique(raw, return_inverse=True)
            print(f"each distinct {args.group_key} is its own state "
                  f"({state.max() + 1} states)")

    # pack (folded hkl, state) into one integer key
    off = 512
    span = 1024
    if np.abs(canon).max() >= off:
        raise SystemExit(f"Miller index {np.abs(canon).max()} exceeds the packing range")
    key = (
        ((canon[:, 0] + off) * span + (canon[:, 1] + off)) * span + (canon[:, 2] + off)
    ) * (state.max() + 1) + state

    uniq, inv = np.unique(key, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq))
    mult = counts[inv].astype(np.int32)

    print(f"{n:,} observations, {len(uniq):,} unique (reflection, state) latents")
    print(f"multiplicity: min {mult.min()} median {int(np.median(mult))} "
          f"max {mult.max()} mean {mult.mean():.2f}")
    qs = np.percentile(mult, [10, 25, 50, 75, 90]).astype(int)
    print(f"  percentiles 10/25/50/75/90: {qs.tolist()}")
    print(f"  observations whose latent is seen once (no tempering): "
          f"{(mult == 1).sum():,} ({(mult == 1).mean() * 100:.2f}%)")

    if args.dry_run:
        print("dry run; metadata not written")
        return

    meta["multiplicity"] = torch.as_tensor(mult, dtype=torch.int32)
    torch.save(meta, meta_path)
    print(f"wrote 'multiplicity' into {meta_path}")


if __name__ == "__main__":
    main()
