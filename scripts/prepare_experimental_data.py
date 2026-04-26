"""Prepare experimental DIALS data for hierarchical per-bin models.

Takes raw .pt files (counts, masks, metadata) from DIALS processing and
generates the per-bin prior buffers needed by PerBinLoss / WilsonLoss.

Usage:
    python scripts/prepare_experimental_data.py \
        --data-dir /path/to/hewl_9b7c \
        --n-bins 20 \
        --min-intensity 0.01

Reads:
    counts.pt             (N, 1323)
    masks.pt              (N, 1323)
    metadata.pt           dict with 'd', 'background.mean', etc.

Writes (in same directory):
    metadata.pt           updated with 'group_label' added
    tau_per_group.pt      (n_bins,)  — optional, for PerBinLoss or Wilson init
    bg_rate_per_group.pt  (n_bins,)
    concentration_per_group.pt  (n_bins, 441)
    s_squared_per_group.pt      (n_bins,)
"""

import argparse
from pathlib import Path

import torch


def bin_by_resolution(
    d: torch.Tensor,
    n_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign reflections to resolution bins via quantiles.

    Returns:
        group_labels: (N,) integer bin index per reflection
        bin_edges: (n_bins + 1,) bin boundaries in d-spacing
    """
    quantiles = torch.linspace(0, 1, n_bins + 1)
    bin_edges = torch.quantile(d.float(), quantiles)

    # searchsorted: find bin index for each reflection
    # Use right=True so that the minimum d falls into bin 0
    group_labels = torch.searchsorted(bin_edges[1:-1], d).long()

    return group_labels, bin_edges


def compute_bg_rate_per_group(
    bg_mean: torch.Tensor,
    group_labels: torch.Tensor,
    n_bins: int,
) -> torch.Tensor:
    """Exponential rate for background prior: lambda_k = 1 / mean(bg_k)."""
    bg_rate = torch.zeros(n_bins)
    for b in range(n_bins):
        sel = bg_mean[group_labels == b]
        if len(sel) > 0:
            bg_rate[b] = 1.0 / sel.mean().clamp(min=1e-6)
    return bg_rate


def compute_tau_per_group(
    intensity: torch.Tensor,
    group_labels: torch.Tensor,
    n_bins: int,
    min_intensity: float = 0.01,
) -> torch.Tensor:
    """Exponential rate for intensity prior: tau_k = 1 / mean(I_k)."""
    tau = torch.zeros(n_bins)
    for b in range(n_bins):
        sel = intensity[group_labels == b]
        sel = sel[sel > min_intensity]
        if len(sel) > 0:
            tau[b] = 1.0 / sel.mean()
        else:
            tau[b] = 1.0  # fallback
    return tau


def compute_s_squared_per_group(
    d: torch.Tensor,
    group_labels: torch.Tensor,
    n_bins: int,
) -> torch.Tensor:
    """Wilson resolution parameter: s_k^2 = 1 / (4 * mean_d_k^2)."""
    s_sq = torch.zeros(n_bins)
    for b in range(n_bins):
        sel = d[group_labels == b]
        if len(sel) > 0:
            mean_d = sel.mean()
            s_sq[b] = 1.0 / (4.0 * mean_d**2)
    return s_sq


def fit_dirichlet_per_group(
    counts: torch.Tensor,
    masks: torch.Tensor,
    group_labels: torch.Tensor,
    n_bins: int,
) -> torch.Tensor:
    """Fit Dirichlet concentration per bin via method of moments.

    Uses the central frame (frame 1 of 3) of each 3x21x21 shoebox,
    giving a 441-dimensional simplex.

    Returns: (n_bins, 441)
    """
    # Extract central frame: (N, 1323) -> (N, 3, 441) -> (N, 441)
    central_counts = counts.reshape(-1, 3, 441)[:, 1, :].float()
    central_masks = masks.reshape(-1, 3, 441)[:, 1, :].float()

    # Apply mask
    central_counts = central_counts * central_masks

    concentration = torch.zeros(n_bins, 441)

    for b in range(n_bins):
        sel = central_counts[group_labels == b]
        if len(sel) < 2:
            concentration[b] = 1e-6
            continue

        # Normalize to simplex
        totals = sel.sum(dim=1, keepdim=True).clamp(min=1)
        sel_norm = sel / totals

        # Method of moments for Dirichlet
        p_bar = sel_norm.mean(dim=0)
        var_p = sel_norm.var(dim=0)

        # Estimate kappa from the ratio p(1-p)/var - 1
        valid = p_bar > 1e-6
        if valid.sum() > 0:
            ratio = (p_bar[valid] * (1 - p_bar[valid])) / var_p[valid].clamp(
                min=1e-12
            ) - 1
            kappa = ratio.median().clamp(min=1.0)
        else:
            kappa = torch.tensor(1.0)

        concentration[b] = (kappa * p_bar).clamp(min=1e-6)

    return concentration


def main():
    parser = argparse.ArgumentParser(
        description="Prepare experimental data for hierarchical per-bin models"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with counts.pt, masks.pt, metadata.pt",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=20,
        help="Number of resolution bins (default: 20)",
    )
    parser.add_argument(
        "--min-intensity",
        type=float,
        default=0.01,
        help="Minimum intensity for tau estimation (default: 0.01)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    n_bins = args.n_bins

    # Load data
    print(f"Loading data from {data_dir}...")
    counts = torch.load(data_dir / "counts.pt", weights_only=True)
    masks = torch.load(data_dir / "masks.pt", weights_only=True)
    metadata = torch.load(data_dir / "metadata.pt", weights_only=False)

    d = metadata["d"]
    bg_mean = metadata["background.mean"]
    N = len(d)
    print(f"  {N} reflections, d range: [{d.min():.2f}, {d.max():.2f}] A")

    # Bin by resolution
    print(f"Binning into {n_bins} resolution shells...")
    group_labels, bin_edges = bin_by_resolution(d, n_bins)

    for b in range(n_bins):
        n_in_bin = (group_labels == b).sum()
        d_sel = d[group_labels == b]
        print(
            f"  bin {b:2d}: n={n_in_bin:7d}, d=[{d_sel.min():.2f}, {d_sel.max():.2f}]"
        )

    # Compute per-bin priors
    print("Computing per-bin priors...")
    bg_rate = compute_bg_rate_per_group(bg_mean, group_labels, n_bins)
    s_squared = compute_s_squared_per_group(d, group_labels, n_bins)

    # Tau (optional, for PerBinLoss or Wilson init)
    intensity = metadata.get(
        "intensity.prf.value", metadata.get("intensity.sum.value")
    )
    if intensity is not None:
        tau = compute_tau_per_group(
            intensity, group_labels, n_bins, args.min_intensity
        )
        torch.save(tau, data_dir / "tau_per_group.pt")
        print(f"  tau_per_group: [{tau.min():.4f}, {tau.max():.4f}]")

    # Dirichlet concentration
    print("Fitting Dirichlet profiles per bin (this may take a moment)...")
    concentration = fit_dirichlet_per_group(
        counts, masks, group_labels, n_bins
    )

    # Add group_label to metadata and save
    metadata["group_label"] = group_labels
    torch.save(metadata, data_dir / "metadata.pt")
    print("  Added 'group_label' to metadata.pt")

    # Save per-bin buffers
    torch.save(bg_rate, data_dir / "bg_rate_per_group.pt")
    torch.save(s_squared, data_dir / "s_squared_per_group.pt")
    torch.save(concentration, data_dir / "concentration_per_group.pt")

    print(f"\nSaved to {data_dir}:")
    print(f"  bg_rate_per_group.pt     ({n_bins},)")
    print(f"  s_squared_per_group.pt   ({n_bins},)")
    print(f"  concentration_per_group.pt ({n_bins}, 441)")
    if intensity is not None:
        print(f"  tau_per_group.pt         ({n_bins},)")
    print("  metadata.pt              (updated with group_label)")
    print("\nDone!")


if __name__ == "__main__":
    main()
