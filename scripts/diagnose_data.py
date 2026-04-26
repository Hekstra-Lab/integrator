#!/usr/bin/env python
"""Diagnose data directory for numerical issues.

Prints statistics for prior buffers (tau, s², bg_rate, concentration),
raw counts/masks, and Wilson tau at default vs auto-init K,B values.

Usage:
    python scripts/diagnose_data.py /path/to/data_dir
    python scripts/diagnose_data.py /path/to/experimental /path/to/simulated
"""

import sys
from pathlib import Path

import torch


def tensor_stats(t: torch.Tensor, name: str, full: bool = False) -> None:
    """Print summary statistics for a tensor."""
    t = t.float()
    flat = t.flatten()
    finite = flat[flat.isfinite()]

    print(f"\n  {name}:")
    print(f"    shape: {list(t.shape)}  dtype: {t.dtype}")
    print(f"    min: {flat.min().item():.6g}  max: {flat.max().item():.6g}")
    print(f"    mean: {flat.mean().item():.6g}  std: {flat.std().item():.6g}")

    n_nan = flat.isnan().sum().item()
    n_inf = flat.isinf().sum().item()
    n_zero = (flat == 0).sum().item()
    n_neg = (flat < 0).sum().item()
    if n_nan or n_inf or n_neg:
        print(f"    *** NaN: {n_nan}  Inf: {n_inf}  negative: {n_neg}")
    print(
        f"    zeros: {n_zero}/{flat.numel()} ({100 * n_zero / flat.numel():.1f}%)"
    )

    if full and finite.numel() > 0:
        qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        # Subsample if tensor is too large for torch.quantile
        if finite.numel() > 10_000_000:
            idx = torch.randperm(finite.numel())[:10_000_000]
            sample = finite[idx]
        else:
            sample = finite
        vals = torch.quantile(sample, torch.tensor(qs))
        qstr = "  ".join(
            f"q{int(q * 100):02d}={v:.4g}"
            for q, v in zip(qs, vals, strict=False)
        )
        print(f"    {qstr}")


def diagnose_priors(data_dir: Path) -> dict:
    """Load and report on per-bin prior .pt files."""
    info = {}
    for name in [
        "tau_per_group",
        "s_squared_per_group",
        "bg_rate_per_group",
        "concentration_per_group",
    ]:
        path = data_dir / f"{name}.pt"
        if path.exists():
            t = torch.load(path, weights_only=True)
            if isinstance(t, dict):
                t = next(iter(t.values()))
            t = t.float()
            tensor_stats(t, name, full=True)
            info[name] = t
        else:
            print(f"\n  {name}: NOT FOUND")
    return info


def diagnose_counts(data_dir: Path) -> None:
    """Report on raw count and mask data."""
    for name in ["counts", "masks"]:
        path = data_dir / f"{name}.pt"
        if path.exists():
            t = torch.load(path, weights_only=True).float()
            tensor_stats(t, name, full=True)
        else:
            print(f"\n  {name}: NOT FOUND")

    # metadata
    meta_path = data_dir / "metadata.pt"
    if meta_path.exists():
        meta = torch.load(meta_path, weights_only=True)
        if isinstance(meta, dict):
            print(f"\n  metadata keys: {list(meta.keys())}")
            for k, v in meta.items():
                if isinstance(v, torch.Tensor):
                    tensor_stats(v, f"metadata[{k}]", full=False)
        else:
            tensor_stats(meta, "metadata", full=False)


def diagnose_wilson(info: dict) -> None:
    """Simulate Wilson tau at default and auto-init K,B."""
    s_sq = info.get("s_squared_per_group")
    tau = info.get("tau_per_group")
    if s_sq is None:
        print("\n  (skipping Wilson analysis — no s_squared_per_group)")
        return

    print("\n  --- Wilson tau analysis ---")

    # Default init: K=1 (log_K=0), B=exp(3.4)≈30
    K_default, B_default = 1.0, torch.exp(torch.tensor(3.4)).item()
    exponent_default = 2.0 * B_default * s_sq
    tau_default = (1.0 / K_default) * torch.exp(exponent_default)
    print(f"\n  Default init (K=1.0, B={B_default:.1f}):")
    print(
        f"    exponent 2*B*s²: min={exponent_default.min():.4g}  max={exponent_default.max():.4g}"
    )
    print(f"    tau: min={tau_default.min():.4g}  max={tau_default.max():.4g}")
    if tau_default.max() > 1e8:
        print("    *** WARNING: max tau > 1e8 — will overflow KL gradient!")

    # Auto-init from empirical tau (if available)
    if tau is not None:
        y = torch.log(tau.clamp(min=1e-12))
        x = s_sq
        x_mean, y_mean = x.mean(), y.mean()
        b = ((x - x_mean) * (y - y_mean)).sum() / (x - x_mean).pow(
            2
        ).sum().clamp(min=1e-12)
        a = y_mean - b * x_mean
        K_fit = torch.exp(-a).item()
        B_fit = (b / 2.0).clamp(min=1e-6).item()

        exponent_fit = 2.0 * B_fit * s_sq
        tau_fit = (1.0 / K_fit) * torch.exp(exponent_fit)
        print(
            f"\n  Auto-init from empirical tau (K={K_fit:.4g}, B={B_fit:.4g}):"
        )
        print(
            f"    exponent 2*B*s²: min={exponent_fit.min():.4g}  max={exponent_fit.max():.4g}"
        )
        print(f"    tau: min={tau_fit.min():.4g}  max={tau_fit.max():.4g}")

        # Compare empirical vs fitted
        residual = (
            torch.log(tau.clamp(min=1e-12))
            - torch.log(tau_fit.clamp(min=1e-12))
        ).abs()
        print(
            f"    |log(tau_empirical) - log(tau_fit)|: mean={residual.mean():.4g}  max={residual.max():.4g}"
        )

        if tau_fit.max() > 1e8:
            print("    *** WARNING: max tau > 1e8 even with auto-init!")


def diagnose_one(data_dir: Path) -> None:
    """Full diagnostic for one data directory."""
    print(f"\n{'=' * 60}")
    print(f"DATA DIR: {data_dir}")
    print(f"{'=' * 60}")

    # List .pt files
    pt_files = sorted(data_dir.glob("*.pt"))
    print(f"\n  .pt files: {[f.name for f in pt_files]}")

    print("\n--- Per-bin prior buffers ---")
    info = diagnose_priors(data_dir)

    print("\n--- Raw data ---")
    diagnose_counts(data_dir)

    diagnose_wilson(info)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    for path_str in sys.argv[1:]:
        data_dir = Path(path_str)
        if not data_dir.is_dir():
            print(f"WARNING: {data_dir} is not a directory, skipping")
            continue
        diagnose_one(data_dir)

    print()


if __name__ == "__main__":
    main()
