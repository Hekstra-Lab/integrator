import logging
from pathlib import Path

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _nbins_path(filename: str, n_bins: int, data_dir: Path) -> Path:
    """Resolve a prior filename with n_bins suffix: 'foo.pt' -> data_dir/'foo_30.pt'."""
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
    events_out: list[dict] | None = None,
) -> None:
    """Generate resolution-bin group labels and the empirical background prior.

    Args:
        cfg: Full YAML config dict.  If `loss.args.n_bins` is set in the
            config, it is used as the number of resolution bins (unless
            overridden by the *n_bins* argument).
        n_bins: Number of resolution bins.  When <= 0 (default), reads from
            `cfg["loss"]["args"]["n_bins"]`, falling back to 20.
            force: Regenerate even if files already exist.
        events_out: Optional list that will be appended with structured
            event dicts describing each file action ("created",
            "regenerated", "reused") and the reason. Lets the CLI echo
            a summary and record what happened in run_metadata.yaml.
    """
    loss_name = cfg.get("loss", {}).get("name", "")
    if loss_name not in ("monochromatic_wilson", "polychromatic_wilson"):
        return

    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])
    loss_args = cfg["loss"].get("args", {})

    if n_bins <= 0:
        n_bins = int(loss_args.get("n_bins", 20))

    # Spectral Wilson: auto-compute lambda_min/lambda_max from data
    if loss_name == "polychromatic_wilson":
        if "lambda_min" not in loss_args or "lambda_max" not in loss_args:
            ref_path = _resolve_reference_path(data_dir, cfg)
            ref = torch.load(ref_path, weights_only=False)
            wl = None
            if isinstance(ref, dict) and "wavelength" in ref:
                wl = ref["wavelength"]
            elif isinstance(ref, dict) and "column_names" in ref:
                col_names = ref["column_names"]
                if "wavelength" in col_names:
                    wl_idx = col_names.index("wavelength")
                    wl = ref["reference"][:, wl_idx]
            if wl is not None:
                pad = 0.01
                loss_args.setdefault("lambda_min", float(wl.min()) - pad)
                loss_args.setdefault("lambda_max", float(wl.max()) + pad)
                logger.info(
                    "polychromatic_wilson: auto lambda_min=%.4f lambda_max=%.4f",
                    loss_args["lambda_min"],
                    loss_args["lambda_max"],
                )

    # Auto-compute d_min/d_max for concentration_cfg from data
    conc_cfg = loss_args.get("concentration_cfg")
    if isinstance(conc_cfg, dict) and (
        "d_min" not in conc_cfg or "d_max" not in conc_cfg
    ):
        ref_path = _resolve_reference_path(data_dir, cfg)
        ref = torch.load(ref_path, weights_only=False)
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
    gl_path = _nbins_path("group_labels.pt", n_bins, data_dir)
    if not gl_path.exists():
        need_group_labels = True
    else:
        existing_gl = torch.load(gl_path, weights_only=True)
        if int(existing_gl.max().item()) + 1 != n_bins:
            need_group_labels = True

    bg_prior_path = data_dir / "bg_prior.pt"
    need_bg_prior = (
        "bg_rate" not in loss_args and "bg_concentration" not in loss_args
    ) and (force or not bg_prior_path.exists())

    if not need_group_labels and not need_bg_prior:
        return

    metadata = torch.load(
        _resolve_reference_path(data_dir, cfg), weights_only=False
    )
    d = metadata["d"]
    N = len(d)

    group_labels, _, n_bins = _bin_by_resolution(d, n_bins)
    logger.info("Binned %d reflections into %d resolution shells", N, n_bins)

    gl_path = _nbins_path("group_labels.pt", n_bins, data_dir)
    torch.save(group_labels, gl_path)
    logger.info("Saved %s (%d bins)", gl_path.name, n_bins)

    # Global background prior via Gamma MLE
    if need_bg_prior:
        bg_vals = metadata.get(
            "background.mean",
            metadata.get("background.sum.value"),
        )
        if bg_vals is not None:
            pos = bg_vals[bg_vals > 0]
            if pos.numel() >= 10:
                alpha, rate = _fit_gamma_mle(pos)
                bg_prior = {
                    "bg_concentration": float(alpha.item()),
                    "bg_rate": float(rate.item()),
                    "n_samples": int(pos.numel()),
                }
                torch.save(bg_prior, bg_prior_path)
                logger.info(
                    "Saved bg_prior.pt (Gamma MLE: alpha=%.3f, rate=%.3f, n=%d)",
                    bg_prior["bg_concentration"],
                    bg_prior["bg_rate"],
                    bg_prior["n_samples"],
                )
            else:
                logger.warning(
                    "Too few positive background values (%d) for MLE; "
                    "using default bg_rate=1.0, bg_concentration=1.0",
                    pos.numel(),
                )
        else:
            logger.warning(
                "No background column in metadata; "
                "using default bg_rate=1.0, bg_concentration=1.0"
            )


def inject_binning_labels(data_loader, cfg: dict) -> None:
    """Load binning label files and inject into the dataset's metadata.
    Called after data_loader.setup() to add group_label and
    profile_group_label to the dataset without mutating metadata.pt.
    Files are saved by prepare_per_bin_priors() as separate
    n_bins-suffixed .pt files.
    """

    loss_args = cfg.get("loss", {}).get("args", {})
    n_bins = int(loss_args.get("n_bins", 20))
    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])

    ref = data_loader.full_dataset.reference

    gl_path = _nbins_path("group_labels.pt", n_bins, data_dir)
    if gl_path.exists():
        ref["group_label"] = torch.load(gl_path, weights_only=True)
        logger.debug("Injected group_label from %s", gl_path.name)

    pgl_path = _nbins_path("profile_group_labels.pt", n_bins, data_dir)
    if pgl_path.exists():
        ref["profile_group_label"] = torch.load(pgl_path, weights_only=True)
        logger.debug("Injected profile_group_label from %s", pgl_path.name)


def _resolve_reference_path(data_dir: Path, cfg: dict) -> Path:
    """Find the metadata/reference .pt file from the config."""
    sfn = cfg["data_loader"]["args"].get("shoebox_file_names", {})
    ref_name = sfn.get("reference", "reference.pt")
    # Fall back to metadata.pt if reference.pt doesn't exist
    ref_path = data_dir / ref_name
    if not ref_path.exists():
        ref_path = data_dir / "metadata.pt"
    return ref_path


def _bin_by_resolution(
    d: Tensor,
    n_bins: int,
    min_per_bin: int = 50,
) -> tuple[Tensor, Tensor, int]:
    """Assign reflections to resolution bins via quantiles.

    If any bin has fewer than `min_per_bin` reflections, `n_bins` is
    reduced and the binning is retried until all bins are large enough
    (or n_bins reaches 1).

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


def _compute_wavelength_bin_edges(
    metadata_path: Path,
    n_lambda_bins: int,
) -> Tensor:
    """Quantile-based wavelength bin edges from metadata['wavelength'].

    Returns a 1-D tensor of length `n_lambda_bins + 1` such that consecutive
    edges define a half-open bin `[edges[i], edges[i+1])` (rightmost
    inclusive). Equal-quantile spacing -> roughly equal counts per bin -> the
    per-bin G_k posteriors get balanced gradient signal.
    """
    from integrator.io import load_metadata

    metadata = load_metadata(metadata_path)
    if "wavelength" not in metadata:
        raise KeyError(
            f"metadata.pt at {metadata_path} has no 'wavelength' column; "
            "this CLI step requires output from integrator.mksbox --laue."
        )
    wavelength = metadata["wavelength"].float()
    if wavelength.numel() == 0:
        raise ValueError("metadata['wavelength'] is empty")
    qs = torch.linspace(0.0, 1.0, n_lambda_bins + 1)
    edges = torch.quantile(wavelength, qs)
    # Guard: degenerate spectra (all-equal λ) would collapse all edges to one
    # value and break bucketize. Spread by a tiny epsilon if so.
    if edges[-1] <= edges[0]:
        raise ValueError(
            f"wavelength range is degenerate: [{float(edges[0]):.6f}, "
            f"{float(edges[-1]):.6f}]; cannot form {n_lambda_bins} bins"
        )
    return edges


def _fit_gamma_mle(
    x: Tensor,
    n_iter: int = 100,
) -> tuple[Tensor, Tensor]:
    """Fit Gamma distribution via Newton MLE on the profile log-likelihood.

    Args:
        x: Positive-valued samples (1-D).
        n_iter: Newton iterations.

    Returns:
        Tuple of (alpha, beta) -- MLE shape (concentration) and rate parameters.
    """
    xbar = x.mean()
    s = xbar.log() - x.log().mean()  # >= 0 by Jensen

    # Init from method of moments
    alpha = xbar**2 / x.var()

    for _ in range(n_iter):
        grad = alpha.log() - torch.digamma(alpha) - s
        hess = 1.0 / alpha - torch.polygamma(1, alpha)
        alpha = alpha - grad / hess
        alpha = alpha.clamp(min=1e-6)

    beta = alpha / xbar
    return alpha, beta


