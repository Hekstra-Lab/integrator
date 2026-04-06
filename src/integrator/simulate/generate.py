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


def _fit_concentration_from_profiles(
    profiles: Tensor,
    group_label: Tensor,
) -> Tensor:
    """Fit Dirichlet concentration per bin from simulated profiles.

    Uses method-of-moments: kappa = median((p(1-p)/var(p)) - 1),
    then alpha_k = kappa * p_bar.

    Returns: (n_bins, n_pixels)
    """
    n_bins = int(group_label.max().item()) + 1
    n_pixels = profiles.shape[1]
    concentration = torch.zeros(n_bins, n_pixels)

    for b in range(n_bins):
        sel = profiles[group_label == b]
        if len(sel) < 2:
            concentration[b] = 1e-6
            continue
        p_bar = sel.mean(dim=0)
        var_p = sel.var(dim=0)
        valid = p_bar > 1e-6
        if valid.sum() > 0:
            ratio = (
                p_bar[valid] * (1 - p_bar[valid])
            ) / var_p[valid].clamp(min=1e-12) - 1
            kappa = ratio.median().clamp(min=1.0)
        else:
            kappa = torch.tensor(1.0)
        concentration[b] = (kappa * p_bar).clamp(min=1e-6)

    return concentration


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
) -> dict[str, Tensor]:
    """Simulate shoeboxes with 2D Gaussian profiles.

    Parameters
    ----------
    n_per_bin : int
        Number of reflections per resolution bin.
    n_bins : int
        Number of resolution bins.
    tau : Tensor, shape (n_bins,)
        Exponential rate for intensity per bin.
    bg_rate : Tensor, shape (n_bins,)
        Exponential rate for background per bin.
    H, W : int
        Spatial dimensions (default 21 x 21).
    n_frames : int
        Number of frames per shoebox (default 3, same profile per frame).
    profile_kwargs : dict, optional
        Extra kwargs passed to :func:`sample_profiles`.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        counts      : (N, n_frames * H * W)
        profiles    : (N, H * W)
        intensity   : (N,)
        background  : (N,)
        group_label : (N,) long
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = n_per_bin * n_bins
    pkw = profile_kwargs or {}

    logger.info(
        "Simulating %d reflections (%d bins x %d per bin)", N, n_bins, n_per_bin
    )

    # Sample all profiles at once
    profiles = sample_profiles(N, H=H, W=W, **pkw)  # (N, K)

    # Assign bin labels
    group_label = (
        torch.arange(n_bins)
        .unsqueeze(1)
        .expand(n_bins, n_per_bin)
        .reshape(-1)
    )  # (N,)

    # Sample intensity and background per reflection from bin-specific priors
    tau_per_refl = tau[group_label]          # (N,)
    bg_rate_per_refl = bg_rate[group_label]  # (N,)

    intensity = torch.distributions.Exponential(tau_per_refl).sample()  # (N,)
    background = torch.distributions.Exponential(bg_rate_per_refl).sample()  # (N,)

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
    }


def save_dataset(
    sim: dict[str, Tensor],
    tau: Tensor,
    bg_rate: Tensor,
    save_dir: Path,
    *,
    s_squared: Tensor | None = None,
    concentration: Tensor | None = None,
    K_true: float | None = None,
    B_true: float | None = None,
    test_frac: float = 0.05,
) -> None:
    """Write .pt files consumable by SimulatedShoeboxLoader.

    Parameters
    ----------
    sim : dict
        Output of :func:`simulate`.
    tau, bg_rate : Tensor
        Per-bin prior rates.
    save_dir : Path
        Output directory (created if needed).
    s_squared : Tensor, optional
        Wilson parameter per bin. Saved if provided.
    concentration : Tensor, optional
        Dirichlet concentration per bin. Saved if provided.
    K_true, B_true : float, optional
        Ground truth Wilson parameters. Saved to ``ground_truth.pt``
        for diagnostics (hyperparameter recovery checks).
    test_frac : float
        Fraction of reflections to mark as test.
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

    if concentration is not None:
        torch.save(concentration, save_dir / "concentration_per_group.pt")
    else:
        # Fit Dirichlet concentration from simulated profiles (method of moments)
        concentration = _fit_concentration_from_profiles(
            sim["profiles"], sim["group_label"]
        )
        torch.save(concentration, save_dir / "concentration_per_group.pt")
        logger.info("Auto-generated concentration_per_group.pt from profiles")

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
