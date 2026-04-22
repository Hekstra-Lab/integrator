"""Core simulation: sample shoeboxes from the generative model.

Generative model per reflection n in bin b:
    prf_n ~ sample_profiles(...)          # 2D Gaussian on pixel grid
    I_n   ~ Exponential(rate=tau[b])      # intensity
    bg_n  ~ Exponential(rate=bg_rate[b])  # background per pixel
    counts_frame ~ Poisson(I_n * prf_n + bg_n)   # per frame
"""

import logging
from pathlib import Path

import torch
from torch import Tensor

from .profiles import sample_profiles

logger = logging.getLogger(__name__)


def simulate(
    n_per_bin: int,
    n_bins: int,
    tau: Tensor,
    bg_rate: Tensor,
    *,
    H: int = 21,
    W: int = 21,
    n_frames: int = 3,
    profile_kwargs: dict | None = None,
    seed: int | None = None,
) -> dict:
    """Simulate shoeboxes with 2D Gaussian profiles.

    Args:
        n_per_bin: Number of reflections per resolution bin.
        n_bins: Number of resolution bins.
        tau: Exponential rate for intensity per bin, shape (n_bins,).
        bg_rate: Exponential rate for background per bin, shape (n_bins,).
        H: Spatial height (default 21).
        W: Spatial width (default 21).
        n_frames: Number of frames per shoebox (default 3, same profile per frame).
        profile_kwargs: Extra kwargs passed to :func:`sample_profiles`.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: counts (N, n_frames * H * W), profiles (N, H * W),
        intensity (N,), background (N,), and group_label (N,).
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = n_per_bin * n_bins
    pkw = profile_kwargs or {}

    logger.info(
        "Simulating %d reflections (%d bins x %d per bin)",
        N,
        n_bins,
        n_per_bin,
    )

    # Sample all profiles at once
    profiles = sample_profiles(N, H=H, W=W, **pkw)  # (N, K)

    # Assign bin labels
    group_label = (
        torch.arange(n_bins).unsqueeze(1).expand(n_bins, n_per_bin).reshape(-1)
    )  # (N,)

    # Sample intensity and background per reflection from bin-specific priors
    tau_per_refl = tau[group_label]  # (N,)
    bg_rate_per_refl = bg_rate[group_label]  # (N,)

    intensity = torch.distributions.Exponential(tau_per_refl).sample()  # (N,)
    background = torch.distributions.Exponential(
        bg_rate_per_refl
    ).sample()  # (N,)

    # Poisson rate per pixel: I * profile + bg
    # profiles: (N, K), intensity: (N,), background: (N,)
    lam = intensity[:, None] * profiles + background[:, None]  # (N, K)

    # Sample counts per frame, stack frames
    frames = []
    for _ in range(n_frames):
        frames.append(torch.poisson(lam))
    counts = torch.cat(frames, dim=1)  # (N, n_frames * K)

    logger.info(
        "Mean pixel count: %.1f, max: %.0f",
        counts.mean().item(),
        counts.max().item(),
    )

    return {
        "counts": counts,
        "profiles": profiles,
        "intensity": intensity,
        "background": background,
        "group_label": group_label,
        "profile_kwargs": pkw,
    }


def save_dataset(
    sim: dict,
    tau: Tensor,
    bg_rate: Tensor,
    save_dir: Path,
    *,
    s_squared: Tensor | None = None,
    K_true: float | None = None,
    B_true: float | None = None,
    test_frac: float = 0.05,
) -> None:
    """Write .pt files consumable by SimulatedShoeboxLoader.

    Args:
        sim: Output of :func:`simulate`.
        tau: Per-bin prior rate for intensity.
        bg_rate: Per-bin prior rate for background.
        save_dir: Output directory (created if needed).
        s_squared: Wilson parameter per bin. Saved if provided.
        K_true: Ground truth Wilson K parameter. Saved to `ground_truth.pt`
            for diagnostics (hyperparameter recovery checks).
        B_true: Ground truth Wilson B parameter. Saved to `ground_truth.pt`
            for diagnostics (hyperparameter recovery checks).
        test_frac: Fraction of reflections to mark as test.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    counts = sim["counts"].float()
    N = counts.shape[0]

    # Test split
    is_test = torch.zeros(N, dtype=torch.bool)
    n_test = int(N * test_frac)
    is_test[torch.randperm(N)[:n_test]] = True

    # Masks: all valid
    masks = torch.ones_like(counts)

    # Shoebox statistics (computed from central frame for multi-frame data)
    reference = {
        "shoebox_median": counts.median(dim=-1).values,
        "shoebox_var": counts.var(dim=-1),
        "shoebox_mean": counts.mean(dim=-1),
        "shoebox_min": counts.min(dim=-1).values,
        "shoebox_max": counts.max(dim=-1).values,
        "intensity": sim["intensity"],
        "background": sim["background"],
        "refl_ids": torch.arange(1, N + 1),
        "is_test": is_test,
        "group_label": sim["group_label"],
    }

    # Anscombe transform statistics
    ans = 2.0 * (counts + 0.375).sqrt()
    stats_anscombe = torch.tensor([ans.mean(), ans.var()])

    # Save core files
    torch.save(counts, save_dir / "counts.pt")
    torch.save(masks, save_dir / "masks.pt")
    torch.save(stats_anscombe, save_dir / "stats_anscombe.pt")
    torch.save(reference, save_dir / "reference.pt")
    torch.save(sim["profiles"], save_dir / "profiles.pt")

    # Per-bin prior files
    torch.save(tau, save_dir / "tau_per_group.pt")
    torch.save(bg_rate, save_dir / "bg_rate_per_group.pt")

    if s_squared is not None:
        torch.save(s_squared, save_dir / "s_squared_per_group.pt")

    # Physical Gaussian profile basis (for physical_gaussian_surrogate)
    # Use profile_kwargs from simulation if available, otherwise defaults
    pkw = sim.get("profile_kwargs", {})
    center_base = pkw.get(
        "center_base", (sim["profiles"].shape[1] ** 0.5 - 1) / 2.0
    )
    profile_basis = {
        "basis_type": "physical_gaussian",
        "d": 5,
        "sigma_prior": 1.0,
        "center_base": float(center_base),
        "center_scale": float(pkw.get("center_scale", 1.5)),
        "log_sigma_base": float(pkw.get("log_sigma_base", 0.7)),
        "width_scale": float(pkw.get("width_scale", 0.4)),
    }
    torch.save(profile_basis, save_dir / "profile_basis.pt")

    # Ground truth Wilson parameters (for hyperparameter recovery checks)
    if K_true is not None or B_true is not None:
        ground_truth = {
            "K": K_true,
            "B": B_true,
            "tau": tau,
            "bg_rate": bg_rate,
        }
        if s_squared is not None:
            ground_truth["s_squared"] = s_squared
        torch.save(ground_truth, save_dir / "ground_truth.pt")
        logger.info("Saved ground_truth.pt (K=%.4f, B=%.4f)", K_true, B_true)

    logger.info("Saved %d reflections to %s", N, save_dir)
