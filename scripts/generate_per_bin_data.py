"""Generate simulated per-resolution-bin shoebox data from pre-fitted parameters.

Usage:
    python scripts/generate_per_bin_data.py \
        --params simulation_params.pt \
        --save-dir /path/to/output \
        --N 1000 \
        --profile-model dirichlet

The params file is a small .pt dict produced by save_simulation_params() in the
simulation notebook. It contains per-bin fitted values (mus, covs, kappa, etc.)
but not the large raw dataset. This script regenerates the full simulated
dataset on the cluster without needing the original experimental data.

Output files (compatible with SimulatedShoeboxLoader + PerBinLoss):
    counts.pt                  (N*n_bins, 441)
    masks.pt                   (N*n_bins, 441)
    stats_anscombe.pt          [mean, var]
    reference.pt               dict with group_label, intensity, background, ...
    tau_per_group.pt           (n_bins,)
    bg_rate_per_group.pt       (n_bins,)
    concentration_per_group.pt (n_bins, 441)
"""

import argparse
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Profile evaluation
# ---------------------------------------------------------------------------

def evaluate_gaussian_profile(
    mus: torch.Tensor, covs: torch.Tensor, shape: tuple[int, int] = (21, 21)
) -> torch.Tensor:
    """Normalized 2D Gaussian on pixel grid per bin. Returns (n_bins, H, W)."""
    y, x = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij")
    grid = torch.stack([x.flatten(), y.flatten()], dim=1).float()
    prec = torch.linalg.inv(covs)
    diff = grid[None] - mus[:, None, :]
    exponent = -0.5 * ((diff @ prec) * diff).sum(-1)
    profiles = exponent.exp().reshape(-1, *shape)
    return profiles / profiles.sum(dim=(-2, -1), keepdim=True)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    mean_intensities: torch.Tensor,
    bg_per_bin: torch.Tensor,
    N_per_bin: int,
    shape: tuple[int, int] = (21, 21),
    profile_model: str = "gaussian",
    # gaussian
    mus: torch.Tensor | None = None,
    covs: torch.Tensor | None = None,
    # dirichlet
    mean_profile: torch.Tensor | None = None,
    kappa: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Simulate shoebox counts per resolution bin.

    For each bin b and reflection n:
        I_n ~ Exponential(mean=mean_intensity_b)
        p   ~ profile model
        rate_ij = I_n * p_ij + bg_b
        counts_ij ~ Poisson(rate_ij)
    """
    n_bins = mean_intensities.shape[0]

    # profiles
    if profile_model == "gaussian":
        assert mus is not None and covs is not None
        profile = evaluate_gaussian_profile(mus, covs, shape)
        profile = profile[None].expand(N_per_bin, -1, -1, -1)
    elif profile_model == "dirichlet":
        assert mean_profile is not None and kappa is not None
        alpha = (kappa[:, None] * mean_profile).clamp(min=1e-6)
        profile = torch.distributions.Dirichlet(alpha).sample((N_per_bin,))
        profile = profile.reshape(N_per_bin, n_bins, *shape)
    else:
        raise ValueError(f"Unknown profile_model: {profile_model}")

    # Wilson prior intensities
    rates = 1.0 / mean_intensities
    I_true = torch.distributions.Exponential(rates).sample((N_per_bin,))

    # Poisson rate
    lam = I_true[:, :, None, None] * profile + bg_per_bin[None, :, None, None]

    # sample counts
    sim_counts = torch.poisson(lam)

    return {
        "I_true": I_true,
        "profile": profile,
        "bg": bg_per_bin,
        "lam": lam,
        "counts": sim_counts,
    }


# ---------------------------------------------------------------------------
# Save for integrator
# ---------------------------------------------------------------------------

def save_for_integrator(
    sim: dict[str, torch.Tensor],
    mean_intensities: torch.Tensor,
    bg_per_bin: torch.Tensor,
    dir_mean_profile: torch.Tensor,
    dir_kappa: torch.Tensor,
    save_dir: Path,
    test_frac: float = 0.05,
) -> None:
    """Reshape and save simulation output as .pt files."""
    save_dir.mkdir(parents=True, exist_ok=True)

    N_per_bin, n_bins = sim["I_true"].shape

    counts_flat = sim["counts"].reshape(N_per_bin * n_bins, -1).float()
    n_total = counts_flat.shape[0]

    group_label = torch.arange(n_bins).unsqueeze(0).expand(N_per_bin, -1).reshape(-1)
    intensity = sim["I_true"].reshape(-1)
    background = bg_per_bin.unsqueeze(0).expand(N_per_bin, -1).reshape(-1)

    is_test = torch.zeros(n_total, dtype=torch.bool)
    n_test = int(n_total * test_frac)
    is_test[torch.randperm(n_total)[:n_test]] = True

    masks = torch.ones_like(counts_flat)

    reference = {
        "shoebox_median": counts_flat.median(-1).values,
        "shoebox_var": counts_flat.var(-1),
        "shoebox_mean": counts_flat.mean(-1),
        "shoebox_min": counts_flat.min(-1).values,
        "shoebox_max": counts_flat.max(-1).values,
        "intensity": intensity,
        "background": background,
        "refl_ids": torch.arange(1, n_total + 1),
        "is_test": is_test,
        "group_label": group_label,
    }

    ans = 2.0 * (counts_flat + 0.375).sqrt()
    stats_anscombe = torch.tensor([ans.mean(), ans.var()])

    tau_per_group = 1.0 / mean_intensities
    bg_rate_per_group = 1.0 / bg_per_bin
    concentration_per_group = (dir_kappa[:, None] * dir_mean_profile).clamp(min=1e-6)

    torch.save(counts_flat, save_dir / "counts.pt")
    torch.save(masks, save_dir / "masks.pt")
    torch.save(stats_anscombe, save_dir / "stats_anscombe.pt")
    torch.save(reference, save_dir / "reference.pt")
    torch.save(tau_per_group, save_dir / "tau_per_group.pt")
    torch.save(bg_rate_per_group, save_dir / "bg_rate_per_group.pt")
    torch.save(concentration_per_group, save_dir / "concentration_per_group.pt")

    print(f"Saved {n_total} reflections ({N_per_bin} x {n_bins} bins) to {save_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-bin simulated shoebox data from pre-fitted params"
    )
    parser.add_argument(
        "--params", type=str, required=True,
        help="Path to simulation_params.pt (from save_simulation_params())",
    )
    parser.add_argument(
        "--save-dir", type=str, required=True,
        help="Output directory for .pt files",
    )
    parser.add_argument(
        "--N", type=int, default=1000,
        help="Number of reflections per bin (default: 1000)",
    )
    parser.add_argument(
        "--profile-model", type=str, default="dirichlet",
        choices=["gaussian", "dirichlet"],
        help="Profile model to use (default: dirichlet)",
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.05,
        help="Fraction of reflections to flag as test (default: 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load pre-fitted parameters
    params = torch.load(args.params, weights_only=True)
    print(f"Loaded params: {params['n_bins']} bins")

    # Simulate
    print(f"Simulating {args.N} reflections/bin with {args.profile_model} profiles...")
    sim = simulate(
        mean_intensities=params["mean_intensities"],
        bg_per_bin=params["bg_per_bin"],
        N_per_bin=args.N,
        profile_model=args.profile_model,
        mus=params["mus"],
        covs=params["covs"],
        mean_profile=params["dir_mean_profile"],
        kappa=params["dir_kappa"],
    )

    # Save
    save_dir = Path(args.save_dir)
    save_for_integrator(
        sim=sim,
        mean_intensities=params["mean_intensities"],
        bg_per_bin=params["bg_per_bin"],
        dir_mean_profile=params["dir_mean_profile"],
        dir_kappa=params["dir_kappa"],
        save_dir=save_dir,
        test_frac=args.test_frac,
    )

    print("Done!")


if __name__ == "__main__":
    main()
