"""Inspect Wilson G and B factors from a run directory or checkpoint.

Supports both monochromatic (WilsonLoss: scalar G) and polychromatic
(PolyWilsonLoss: per-wavelength-bin G_k) models.

Prints a text summary and saves a plot to the same directory as the
checkpoint (or current directory if writing fails).

Usage:
    uv run python scripts/inspect_wilson_params.py <run_dir_or_ckpt>

Examples:
    uv run python scripts/inspect_wilson_params.py /path/to/run-dir/
    uv run python scripts/inspect_wilson_params.py /path/to/epoch=0049.ckpt
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def find_checkpoint(path: Path) -> Path:
    if path.suffix == ".ckpt":
        return path
    candidates = sorted(path.rglob("*.ckpt"))
    if not candidates:
        print(f"No .ckpt files found under {path}", file=sys.stderr)
        sys.exit(1)
    best = [c for c in candidates if "last" in c.name]
    if best:
        return best[-1]
    return candidates[-1]


def _lognormal_std(mu: float, sigma: float) -> float:
    var = (math.exp(sigma**2) - 1) * math.exp(2 * mu + sigma**2)
    return math.sqrt(var)


def _tau_curve(K: float, B: float, d_values: np.ndarray) -> np.ndarray:
    s_sq = 1.0 / (4.0 * d_values**2)
    exponent = 2.0 * B * s_sq
    exponent = np.clip(exponent, None, 700)
    return (1.0 / K) * np.exp(exponent)


def _print_tau_curve(K: float, B: float, label: str = ""):
    header = f"=== Wilson prior rate tau(d){label} ==="
    print(header)
    print(f"  {'d (Å)':>8s}  {'s²':>10s}  {'tau':>12s}  {'E[I]=1/tau':>12s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*12}")
    for d in [10.0, 5.0, 3.0, 2.0, 1.5, 1.2, 1.0, 0.8]:
        s_sq = 1.0 / (4.0 * d * d)
        exponent = 2.0 * B * s_sq
        if exponent > 700:
            tau_str = "       +inf"
            ei_str = "     0.0000"
        else:
            tau = (1.0 / K) * math.exp(exponent)
            expected_I = 1.0 / tau if tau > 0 else float("inf")
            tau_str = f"{tau:12.4f}"
            ei_str = f"{expected_I:12.4f}"
        print(f"  {d:8.2f}  {s_sq:10.4f}  {tau_str}  {ei_str}")


def _plot_mono(K_loc, K_scale, B_loc, B_scale, out_path: Path):
    import matplotlib.pyplot as plt

    K = math.exp(K_loc)
    B = math.exp(B_loc)
    d_vals = np.linspace(0.8, 10.0, 200)
    tau = _tau_curve(K, B, d_vals)
    expected_I = 1.0 / tau

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: E[I] vs d
    ax = axes[0]
    ax.semilogy(d_vals, expected_I, "b-", linewidth=2)
    ax.set_xlabel("d (Å)")
    ax.set_ylabel("E[I] = K · exp(−2B·s²)")
    ax.set_title(f"Wilson expected intensity\nK={K:.1f}, B={B:.2f}")
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)

    # Right: tau vs d
    ax = axes[1]
    ax.semilogy(d_vals, tau, "r-", linewidth=2)
    ax.set_xlabel("d (Å)")
    ax.set_ylabel("τ = (1/K) · exp(2B·s²)")
    ax.set_title(f"Wilson prior rate τ(d)\nK={K:.1f}, B={B:.2f}")
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close(fig)


def _plot_poly(k_locs, k_scales, B_loc, B_scale, edges, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    B = math.exp(B_loc)
    n_bins = len(k_locs)
    d_vals = np.linspace(0.8, 10.0, 200)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: E[I] vs d for each wavelength bin
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0, 1, n_bins))
    for i in range(n_bins):
        K_i = math.exp(k_locs[i].item())
        expected_I = K_i * np.exp(-2.0 * B / (4.0 * d_vals**2))
        if edges is not None and i < len(edges) - 1:
            label = f"λ=[{edges[i]:.3f},{edges[i+1]:.3f})"
        else:
            label = f"bin {i}"
        ax.semilogy(d_vals, expected_I, color=cmap[i], linewidth=1.5, label=label)
    ax.set_xlabel("d (Å)")
    ax.set_ylabel("E[I] = G_k · exp(−2B·s²)")
    ax.set_title(f"Wilson expected intensity per λ-bin\nB={B:.2f}")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Middle: G_k bar chart
    ax = axes[1]
    K_means = [(k_locs[i] + 0.5 * k_scales[i] ** 2).exp().item() for i in range(n_bins)]
    K_modes = [math.exp(k_locs[i].item()) for i in range(n_bins)]

    if edges is not None:
        centers = [(edges[i].item() + edges[i + 1].item()) / 2 for i in range(n_bins)]
        widths = [(edges[i + 1].item() - edges[i].item()) * 0.8 for i in range(n_bins)]
        ax.bar(centers, K_modes, width=widths, color=cmap, alpha=0.7, label="exp(loc)")
        ax.set_xlabel("λ (Å)")
    else:
        ax.bar(range(n_bins), K_modes, color=cmap, alpha=0.7, label="exp(loc)")
        ax.set_xlabel("Wavelength bin")
    ax.set_ylabel("G_k")
    ax.set_title("Scale factor per wavelength bin")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: tau vs d for min/median/max G bins
    ax = axes[2]
    K_arr = np.array(K_modes)
    idx_min = int(np.argmin(K_arr))
    idx_max = int(np.argmax(K_arr))
    idx_med = int(np.argsort(K_arr)[n_bins // 2])

    for idx, color, label in [
        (idx_min, "blue", f"min G (bin {idx_min}, G={K_arr[idx_min]:.0f})"),
        (idx_med, "green", f"med G (bin {idx_med}, G={K_arr[idx_med]:.0f})"),
        (idx_max, "red", f"max G (bin {idx_max}, G={K_arr[idx_max]:.0f})"),
    ]:
        tau = _tau_curve(K_arr[idx], B, d_vals)
        ax.semilogy(d_vals, tau, color=color, linewidth=2, label=label)
    ax.set_xlabel("d (Å)")
    ax.set_ylabel("τ(d)")
    ax.set_title(f"Wilson prior rate τ(d)\nB={B:.2f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Run directory or .ckpt file")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    ckpt_path = find_checkpoint(args.path)
    print(f"Checkpoint: {ckpt_path.name}\n")

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]

    # Find Wilson parameters - match by key ending
    log_K_loc = log_K_log_scale = log_B_loc = log_B_log_scale = None
    edges = None
    for k, v in sd.items():
        if k.endswith("q_log_K_loc"):
            log_K_loc = v
        elif k.endswith("q_log_K_log_scale"):
            log_K_log_scale = v
        elif k.endswith("q_log_B_loc"):
            log_B_loc = v
        elif k.endswith("q_log_B_log_scale"):
            log_B_log_scale = v
        elif k.endswith("wavelength_bin_edges"):
            edges = v

    if log_K_loc is None or log_B_loc is None:
        wilson_keys = [k for k in sd if "log_K" in k or "log_B" in k]
        if not wilson_keys:
            print("This checkpoint does not use WilsonLoss.")
        else:
            print("Wilson-related keys found in checkpoint:")
            for wk in wilson_keys:
                print(f"  {wk}: shape={sd[wk].shape}")
            found = {
                "q_log_K_loc": log_K_loc is not None,
                "q_log_K_log_scale": log_K_log_scale is not None,
                "q_log_B_loc": log_B_loc is not None,
                "q_log_B_log_scale": log_B_log_scale is not None,
            }
            missing = [k for k, v in found.items() if not v]
            print(f"Missing: {missing}")
        sys.exit(1)

    is_poly = log_K_loc.dim() >= 1 and log_K_loc.numel() > 1

    # B is always scalar
    b_loc = log_B_loc.item()
    b_scale = F.softplus(log_B_log_scale).item()
    B_mean = math.exp(b_loc + 0.5 * b_scale**2)

    print("=== B (temperature factor) ===")
    print(f"  q(log B):  loc={b_loc:.4f}  scale={b_scale:.4f}")
    print(f"  E[B] = {B_mean:.4f}   (exp(loc) = {math.exp(b_loc):.4f})")
    print(f"  std[B] = {_lognormal_std(b_loc, b_scale):.4f}")
    print()

    if not is_poly:
        # Monochromatic - scalar G
        k_loc = log_K_loc.item()
        k_scale = F.softplus(log_K_log_scale).item()
        K_mean = math.exp(k_loc + 0.5 * k_scale**2)

        print("=== G (scale factor) - monochromatic ===")
        print(f"  q(log G):  loc={k_loc:.4f}  scale={k_scale:.4f}")
        print(f"  E[G] = {K_mean:.4f}   (exp(loc) = {math.exp(k_loc):.4f})")
        print(f"  std[G] = {_lognormal_std(k_loc, k_scale):.4f}")
        print()

        _print_tau_curve(math.exp(k_loc), math.exp(b_loc))

        if not args.no_plot:
            out_path = ckpt_path.parent / "wilson_prior.png"
            try:
                _plot_mono(k_loc, k_scale, b_loc, b_scale, out_path)
            except Exception as e:
                print(f"\nCould not save plot: {e}")

    else:
        # Polychromatic - per-wavelength-bin G_k
        n_bins = log_K_loc.numel()
        k_locs = log_K_loc.detach()
        k_scales = F.softplus(log_K_log_scale.detach())
        K_means = (k_locs + 0.5 * k_scales**2).exp()

        print(f"=== G_k (scale factors) - polychromatic, {n_bins} wavelength bins ===")

        if edges is not None:
            print(f"  Wavelength bin edges: {edges.tolist()}")
            print()

        print(f"  {'Bin':>4s}  {'λ range (Å)':>16s}  {'log_G loc':>10s}  {'scale':>8s}  {'E[G_k]':>12s}  {'exp(loc)':>12s}")
        print(f"  {'─'*4}  {'─'*16}  {'─'*10}  {'─'*8}  {'─'*12}  {'─'*12}")

        for i in range(n_bins):
            loc_i = k_locs[i].item()
            scale_i = k_scales[i].item()
            mean_i = K_means[i].item()
            exp_loc_i = math.exp(loc_i)

            if edges is not None and i < len(edges) - 1:
                lam_lo = edges[i].item()
                lam_hi = edges[i + 1].item()
                lam_str = f"[{lam_lo:.3f}, {lam_hi:.3f})"
            else:
                lam_str = ""

            print(f"  {i:4d}  {lam_str:>16s}  {loc_i:10.4f}  {scale_i:8.4f}  {mean_i:12.4f}  {exp_loc_i:12.4f}")

        print()
        print(f"  Summary: E[G_k] mean={K_means.mean():.4f}  min={K_means.min():.4f}  max={K_means.max():.4f}")
        print()

        # Show tau curve for min, median, max G bins
        B = math.exp(b_loc)
        idx_min = K_means.argmin().item()
        idx_max = K_means.argmax().item()
        idx_med = K_means.argsort()[n_bins // 2].item()

        for idx, label in [(idx_min, "min G"), (idx_med, "median G"), (idx_max, "max G")]:
            K_val = math.exp(k_locs[idx].item())
            bin_label = f" (bin {idx}, {label}, G={K_val:.1f})"
            _print_tau_curve(K_val, B, label=bin_label)
            print()

        if not args.no_plot:
            out_path = ckpt_path.parent / "wilson_prior.png"
            try:
                _plot_poly(k_locs, k_scales, b_loc, b_scale, edges, out_path)
            except Exception as e:
                print(f"\nCould not save plot: {e}")

    # Learned concentration (alpha) if present
    alpha_keys = [k for k in sd if "log_alpha_per_group" in k]
    if alpha_keys:
        alpha_raw = sd[alpha_keys[0]]
        alpha = F.softplus(alpha_raw)
        print(f"=== Learned concentration (alpha per group) ===")
        print(f"  n_bins={alpha.shape[0]}  mean={alpha.mean():.4f}  min={alpha.min():.4f}  max={alpha.max():.4f}")


if __name__ == "__main__":
    main()
