"""Recalculate stats.pt and anscombe_stats.pt using only valid (masked) pixels."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate standardization stats"
    )
    parser.add_argument(
        "data_dir", type=Path, help="Directory with counts.pt and masks.pt"
    )
    args = parser.parse_args()

    counts = torch.load(args.data_dir / "counts.pt", weights_only=False)
    masks = torch.load(args.data_dir / "masks.pt", weights_only=False)
    valid = masks.bool()

    n_valid = valid.sum().item()
    n_total = valid.numel()
    print(
        f"Valid pixels: {n_valid:,} / {n_total:,} ({100 * n_valid / n_total:.2f}%)"
    )

    # Raw stats (mean, variance) over valid pixels
    valid_counts = counts[valid].float()
    raw_mean = valid_counts.mean()
    raw_var = valid_counts.var()
    raw_stats = torch.tensor([raw_mean, raw_var])
    print(f"\nRaw stats:     mean={raw_mean:.6f}, var={raw_var:.6f}")

    # Anscombe stats over valid pixels
    anscombe = 2.0 * torch.sqrt(counts[valid].float().clamp(min=0) + 0.375)
    ans_mean = anscombe.mean()
    ans_var = anscombe.var()
    ans_stats = torch.tensor([ans_mean, ans_var])
    print(f"Anscombe stats: mean={ans_mean:.6f}, var={ans_var:.6f}")

    # Save
    torch.save(raw_stats, args.data_dir / "stats.pt")
    torch.save(ans_stats, args.data_dir / "anscombe_stats.pt")
    print(f"\nSaved stats.pt and anscombe_stats.pt to {args.data_dir}")


if __name__ == "__main__":
    main()
