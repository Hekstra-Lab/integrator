"""Construct per-bin prior tensors for simulated data.

Two modes:
  - `fit_priors_from_experimental`: fit from real DIALS data
  - `make_config_priors`: build directly from YAML config values
"""

import logging
from pathlib import Path

import torch
from torch import Tensor

from integrator.utils.prepare_priors import (
    _bin_by_resolution,
    _compute_bg_rate_per_group,
    _compute_s_squared_per_group,
    _compute_tau_per_group,
    _fit_dirichlet_per_group,
    _load_raw_data,
)

logger = logging.getLogger(__name__)

# Default 25-bin resolution edges from HEWL PDB:9b7c (d-spacing in Angstroms).
# 26 edges define 25 bins, from high resolution (small d) to low resolution.
DEFAULT_BIN_EDGES = torch.tensor([
    1.0960, 1.1877, 1.2291, 1.2613, 1.2876, 1.3097, 1.3323, 1.3566,
    1.3831, 1.4117, 1.4427, 1.4786, 1.5188, 1.5637, 1.6091, 1.6606,
    1.7192, 1.7885, 1.8707, 1.9731, 2.1159, 2.2782, 2.4995, 2.8518,
    3.6124, 79.4244,
])

# Mean d-spacing per bin (midpoints of edges)
DEFAULT_MEAN_D = (DEFAULT_BIN_EDGES[:-1] + DEFAULT_BIN_EDGES[1:]) / 2

# Default per-bin background rates from HEWL PDB:9b7c
DEFAULT_BG_RATE = torch.tensor([
    4.8917, 3.8758, 3.3546, 2.9835, 2.7017, 2.6042, 2.4695, 2.3401,
    2.2026, 2.0765, 1.9536, 1.8289, 1.6834, 1.5416, 1.4174, 1.2937,
    1.1733, 1.0304, 0.8658, 0.7062, 0.5984, 0.5486, 0.4867, 0.3635,
    0.3514,
])


def fit_priors_from_experimental(
    data_dir: Path,
    cfg: dict,
    n_bins: int = 20,
    min_intensity: float = 0.01,
) -> dict:
    """Fit per-bin priors from experimental DIALS data.

    Parameters
    ----------
    data_dir : Path
        Directory containing counts.pt, masks.pt, reference.pt.
    cfg : dict
        Integrator-style config dict (needs `data_loader.args` keys).
    n_bins : int
        Target number of resolution bins.
    min_intensity : float
        Floor for intensity when computing tau.

    Returns
    -------
    dict with keys:
        tau         : (n_bins,) Exponential rate for intensity
        bg_rate     : (n_bins,) Exponential rate for background
        s_squared   : (n_bins,) Wilson parameter 1/(4d²)
        concentration : (n_bins, 441) Dirichlet concentration
        n_bins      : int, actual number of bins used
    """
    counts, masks, metadata = _load_raw_data(data_dir, cfg)

    d = metadata["d"]
    group_labels, _bin_edges, n_bins = _bin_by_resolution(d, n_bins)
    logger.info(
        "Binned %d reflections into %d resolution shells", len(d), n_bins
    )

    # Intensity rate
    intensity = metadata.get(
        "intensity.prf.value",
        metadata.get("intensity.sum.value"),
    )
    if intensity is not None:
        tau = _compute_tau_per_group(
            intensity, group_labels, n_bins, min_intensity
        )
    else:
        logger.warning("No intensity column; using tau=1.0 for all bins")
        tau = torch.ones(n_bins)

    # Background rate
    bg_mean = metadata["background.mean"]
    bg_rate = _compute_bg_rate_per_group(bg_mean, group_labels, n_bins)

    # Wilson s**2
    s_squared = _compute_s_squared_per_group(d, group_labels, n_bins)

    # Dirichlet concentration
    concentration = _fit_dirichlet_per_group(
        counts, masks, group_labels, n_bins
    )

    return {
        "tau": tau,
        "bg_rate": bg_rate,
        "s_squared": s_squared,
        "concentration": concentration,
        "n_bins": n_bins,
    }


def make_config_priors(prior_cfg: dict) -> dict:
    """Build prior tensors from YAML config values.

    Supports two ways of specifying the intensity prior:

    1. **Direct tau**: provide ``tau`` (scalar or list).
    2. **Wilson model**: provide ``K``, ``B``, and ``s_squared`` (or
       ``mean_d``).  tau is derived as ``(1/K) * exp(2*B*s²)``.
       Both ``tau_per_group.pt`` and ``s_squared_per_group.pt`` are
       saved so that ``PerBinLoss`` and ``WilsonPerBinLoss`` produce
       equivalent effective priors.

    Parameters
    ----------
    prior_cfg : dict
        Must contain ``n_bins`` and ``bg_rate``.
        Intensity: either ``tau`` directly, or ``K``, ``B`` with
        ``s_squared`` (or ``mean_d``).

    Returns
    -------
    dict with same structure as :func:`fit_priors_from_experimental`.
    """
    n_bins = prior_cfg.get("n_bins", len(DEFAULT_MEAN_D))

    if "bg_rate" in prior_cfg:
        bg_rate = _to_tensor(prior_cfg["bg_rate"], n_bins)
    elif n_bins == len(DEFAULT_BG_RATE):
        bg_rate = DEFAULT_BG_RATE.clone()
        logger.info("Using default 25-bin HEWL background rates")
    else:
        raise ValueError("Must provide 'bg_rate' or use default n_bins=25")

    # Resolve s_squared: explicit, from mean_d, or from defaults
    s_squared = None
    if "s_squared" in prior_cfg:
        s_squared = _to_tensor(prior_cfg["s_squared"], n_bins)
    elif "mean_d" in prior_cfg:
        mean_d = _to_tensor(prior_cfg["mean_d"], n_bins)
        s_squared = 1.0 / (4.0 * mean_d**2)
    elif n_bins == len(DEFAULT_MEAN_D):
        # Use default HEWL bin edges
        s_squared = 1.0 / (4.0 * DEFAULT_MEAN_D**2)
        logger.info("Using default 25-bin HEWL resolution edges")

    # Resolve tau: explicit, or derived from Wilson (K, B)
    if "K" in prior_cfg and "B" in prior_cfg:
        if s_squared is None:
            raise ValueError(
                "Wilson model requires 's_squared', 'mean_d', "
                "or default n_bins=25 to derive tau per bin"
            )
        K = float(prior_cfg["K"])
        B = float(prior_cfg["B"])
        tau = (1.0 / K) * torch.exp(2.0 * B * s_squared)
        logger.info(
            "Wilson priors: K=%.4f, B=%.4f -> tau range [%.4f, %.4f]",
            K,
            B,
            tau.min().item(),
            tau.max().item(),
        )
    elif "tau" in prior_cfg:
        tau = _to_tensor(prior_cfg["tau"], n_bins)
    else:
        raise ValueError("Must provide either 'tau' or 'K'+'B' in priors")

    result: dict[str, Tensor | int | float] = {
        "tau": tau,
        "bg_rate": bg_rate,
        "n_bins": n_bins,
    }

    if s_squared is not None:
        result["s_squared"] = s_squared

    # Store ground truth Wilson params if used
    if "K" in prior_cfg and "B" in prior_cfg:
        result["K_true"] = float(prior_cfg["K"])
        result["B_true"] = float(prior_cfg["B"])

    return result


def _to_tensor(value: float | list[float], n_bins: int) -> Tensor:
    """Convert a scalar or list to a (n_bins,) tensor."""
    if isinstance(value, (int, float)):
        return torch.full((n_bins,), float(value))
    return torch.tensor(value, dtype=torch.float32)
