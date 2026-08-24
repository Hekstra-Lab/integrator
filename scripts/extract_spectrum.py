"""Extract learned spectrum, B factor, and concentration from a checkpoint.

Usage:
    uv run python scripts/extract_spectrum.py <checkpoint.ckpt>
    uv run python scripts/extract_spectrum.py <checkpoint.ckpt> --save prior_init.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    epoch = ckpt.get("epoch", "?")

    print(f"Checkpoint: {args.checkpoint.name}")
    print(f"Epoch: {epoch}")
    print()

    out = {}

    # Spectrum coefficients
    c = sd.get("loss.spectrum.c", sd.get("loss.spectrum.coeff_loc"))
    if c is not None:
        out["c"] = c
        print(f"Spectrum coefficients ({c.shape[0]} terms):")
        print(f"  {c.tolist()}")

    # Spectrum buffers
    lam_mid = sd.get("loss.spectrum.lam_mid")
    lam_scale = sd.get("loss.spectrum.lam_scale")
    if lam_mid is not None:
        out["lam_mid"] = lam_mid
        out["lam_scale"] = lam_scale
        print(f"  lam_mid: {lam_mid.item():.4f}")
        print(f"  lam_scale: {lam_scale.item():.4f}")

    # B factor
    if "loss.raw_B" in sd:
        out["raw_B"] = sd["loss.raw_B"]
        B = F.softplus(sd["loss.raw_B"]).item()
        print(f"\nB factor: {B:.2f} (raw_B: {sd['loss.raw_B'].item():.4f})")
    elif "loss.q_log_B_loc" in sd:
        out["raw_B"] = sd["loss.q_log_B_loc"]
        B = F.softplus(sd["loss.q_log_B_loc"]).item()
        print(f"\nB factor: {B:.2f}")

    # Learned concentration (alpha per bin)
    if "loss.log_alpha_per_group" in sd:
        out["log_alpha_per_group"] = sd["loss.log_alpha_per_group"]
        alpha = F.softplus(sd["loss.log_alpha_per_group"])
        print(f"\nLearned concentration ({alpha.shape[0]} bins):")
        print(f"  alpha: {alpha.tolist()}")
        print(
            f"  min={alpha.min():.3f}  max={alpha.max():.3f}  mean={alpha.mean():.3f}"
        )

    # Save
    if args.save:
        torch.save(out, args.save)
        print(f"\nSaved {list(out.keys())} to {args.save}")


if __name__ == "__main__":
    main()
