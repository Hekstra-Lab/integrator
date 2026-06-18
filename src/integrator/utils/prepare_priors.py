import logging
from pathlib import Path

import torch
from torch import Tensor

from integrator.io import data_path, load_data, save_data

logger = logging.getLogger(__name__)


def _nbins_path(filename: str, n_bins: int, data_dir: Path) -> Path:
    """Resolve a prior filename with n_bins suffix."""
    p = Path(filename)
    suffixed = f"{p.stem}_{n_bins}{p.suffix}"
    if p.is_absolute():
        return p.parent / suffixed
    return data_dir / suffixed


def prepare_per_bin_priors(
    cfg: dict,
    *,
    n_bins: int = 0,
    force: bool = False,
) -> None:
    """Generate resolution-bin group labels and the empirical background prior.

    Args:
        cfg: Full YAML config dict.
        n_bins: Number of resolution bins.
    """
    loss_name = cfg.get("loss", {}).get("name", "")
    if loss_name not in ("monochromatic_wilson", "polychromatic_wilson"):
        return

    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])
    loss_args = cfg["loss"].get("args", {})

    if n_bins <= 0:
        n_bins = int(loss_args.get("n_bins", 1))

    # Auto-compute d_min/d_max for concentration_cfg from data
    conc_cfg = loss_args.get("concentration_cfg")
    if isinstance(conc_cfg, dict) and (
        "d_min" not in conc_cfg or "d_max" not in conc_cfg
    ):
        ref_path = _resolve_reference_path(data_dir, cfg)
        ref = load_data(ref_path)
        d = None
        if isinstance(ref, dict) and "d" in ref:
            d = ref["d"]
        if d is not None:
            conc_cfg.setdefault("d_min", float(d.min()))
            conc_cfg.setdefault("d_max", float(d.max()))
            logger.info(
                "concentration_cfg: auto d_min=%.4f d_max=%.4f",
                conc_cfg["d_min"],
                conc_cfg["d_max"],
            )

    # Regenerate group_labels if missing or binned at a different n_bins.
    need_group_labels = False
    gl_path = _nbins_path("group_labels.npy", n_bins, data_dir)
    if data_path(gl_path) is None:
        need_group_labels = True
    else:
        existing_gl = load_data(gl_path)
        if int(existing_gl.max().item()) + 1 != n_bins:
            need_group_labels = True

    bg_prior_path = _nbins_path("bg_prior.npy", n_bins, data_dir)
    need_bg_prior = (
        "bg_rate" not in loss_args and "bg_concentration" not in loss_args
    ) and (force or data_path(bg_prior_path) is None)

    if not need_group_labels and not need_bg_prior:
        return

    metadata = load_data(_resolve_reference_path(data_dir, cfg))
    d = metadata["d"]
    N = len(d)

    requested = n_bins
    group_labels, _, n_bins = _bin_by_resolution(d, n_bins)
    if n_bins != requested:
        logger.warning(
            "Reduced n_bins %d -> %d (sparse shells); updating config so "
            "label lookup and the loss use the actual bin count",
            requested,
            n_bins,
        )

    cfg["loss"].setdefault("args", {})["n_bins"] = n_bins
    logger.info("Binned %d reflections into %d resolution shells", N, n_bins)

    gl_path = save_data(
        group_labels, _nbins_path("group_labels.npy", n_bins, data_dir)
    )
    logger.info("Saved %s (%d bins)", gl_path.name, n_bins)

    # Per-resolution-bin background Gamma prior (a single global fit when n_bins == 1)
    if need_bg_prior:
        bg_vals = metadata.get(
            "background.mean",
            metadata.get("background.sum.value"),
        )
        if bg_vals is not None and int((bg_vals > 0).sum()) >= 10:
            alphas, rates = _fit_per_bin_gamma(bg_vals, group_labels, n_bins)
            payload = (
                {"bg_concentration": alphas[0], "bg_rate": rates[0]}
                if n_bins == 1
                else {"bg_concentration": alphas, "bg_rate": rates}
            )
            payload["n_bins"] = n_bins
            p = save_data(
                payload, _nbins_path("bg_prior.npy", n_bins, data_dir)
            )
            logger.info(
                "Saved %s (%d-bin background Gamma MLE)", p.name, n_bins
            )
        else:
            logger.warning(
                "No usable background column for MLE; "
                "using default bg_rate=1.0, bg_concentration=1.0"
            )


def inject_binning_labels(data_loader, cfg: dict) -> None:
    """Load binning label files and inject into the dataset's metadata."""

    loss_args = cfg.get("loss", {}).get("args", {})
    n_bins = int(loss_args.get("n_bins", 1))
    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])

    ref = data_loader.full_dataset.reference

    gl_path = _nbins_path("group_labels.npy", n_bins, data_dir)
    if data_path(gl_path) is not None:
        ref["group_label"] = load_data(gl_path)
        logger.debug("Injected group_label from %s", gl_path.name)

    pgl_path = _nbins_path("profile_group_labels.npy", n_bins, data_dir)
    if data_path(pgl_path) is not None:
        ref["profile_group_label"] = load_data(pgl_path)
        logger.debug("Injected profile_group_label from %s", pgl_path.name)


def _resolve_reference_path(data_dir: Path, cfg: dict) -> Path:
    """Find the metadata/reference .pt file from the config."""
    sfn = cfg["data_loader"]["args"].get("shoebox_file_names", {})
    ref_name = sfn.get("reference", "metadata.npy")
    ref_path = data_dir / ref_name
    if data_path(ref_path) is not None:
        return ref_path
    return data_dir / "metadata.npy"


def _bin_by_resolution(
    d: Tensor,
    n_bins: int,
    min_per_bin: int = 50,
) -> tuple[Tensor, Tensor, int]:
    """Assign reflections to resolution bins via quantiles.

    Returns:
        group_labels: (N,) integer bin index per reflection
        bin_edges: (n_bins_actual + 1,) bin boundaries
        n_bins_actual: final number of bins used
    """
    while n_bins > 1:
        quantiles = torch.linspace(0, 1, n_bins + 1)
        bin_edges = torch.quantile(d.float(), quantiles)
        group_labels = torch.searchsorted(bin_edges[1:-1], d).long()

        # Check minimum bin occupancy
        counts_per_bin = torch.bincount(group_labels, minlength=n_bins)
        if counts_per_bin.min() >= min_per_bin:
            return group_labels, bin_edges, n_bins

        old_n = n_bins
        n_bins = max(1, n_bins - 1)
        logger.warning(
            "Bin with <%d reflections detected; reducing n_bins %d -> %d",
            min_per_bin,
            old_n,
            n_bins,
        )

    # n_bins == 1: single bin fallback
    group_labels = torch.zeros(len(d), dtype=torch.long)
    bin_edges = torch.tensor([d.min(), d.max()])
    return group_labels, bin_edges, 1


def _fit_gamma_mle(
    x: Tensor,
    n_iter: int = 100,
) -> tuple[Tensor, Tensor]:
    """Fit Gamma distribution via MLE on the profile log-likelihood.

    Args:
        x: Positive-valued samples (1-D).
        n_iter: Newton iterations.

    Returns:
        Tuple of (alpha, beta) -- MLE shape (concentration) and rate parameters.
    """
    xbar = x.mean()
    s = (xbar.log() - x.log().mean()).clamp(min=1e-6)

    # Method-of-moments init, kept finite (var==0 for constant x would give inf)
    var = x.var().clamp(min=1e-12)
    alpha = (xbar**2 / var).clamp(1e-3, 1e6)

    for _ in range(n_iter):
        grad = alpha.log() - torch.digamma(alpha) - s
        hess = 1.0 / alpha - torch.polygamma(1, alpha)
        alpha = (alpha - grad / hess).clamp(1e-3, 1e6)

    beta = alpha / xbar
    return alpha, beta


def _fit_per_bin_gamma(
    bg_vals: Tensor,
    group_labels: Tensor,
    n_bins: int,
    min_samples: int = 10,
) -> tuple[list[float], list[float]]:
    """Fit a Gamma MLE to the positive background values in each resolution bin."""
    pos_all = bg_vals[bg_vals > 0]
    g_alpha, g_rate = _fit_gamma_mle(pos_all)  # global fallback
    alphas: list[float] = []
    rates: list[float] = []
    for k in range(n_bins):
        pos_k = bg_vals[(group_labels == k) & (bg_vals > 0)]
        if pos_k.numel() >= min_samples:
            a, r = _fit_gamma_mle(pos_k)
        else:
            a, r = g_alpha, g_rate
        alphas.append(float(a))
        rates.append(float(r))
    return alphas, rates
