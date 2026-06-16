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


def prepare_profile_basis(
    cfg: dict,
    *,
    force: bool = False,
    events_out: list[dict] | None = None,
) -> None:
    """Auto-generate a fixed Hermite profile basis .pt file if missing.

    Triggered when `surrogates.qp.args` references a basis file
    (`warmstart_basis_path` for `learned_basis_profile` or `basis_path`
    for `fixed_basis_profile`) that does not exist on disk. The basis is
    built from the spatial shape in `data_loader.args` and the Hermite
    knobs in `surrogates.qp.args`:

      - `hermite_max_order` (default 4): max polynomial order in x, y
      - `hermite_basis_sigma` (default 3.0): reference Gaussian width
      - `hermite_sigma_z` (default 1.0): Gaussian width along z (3D only)
      - `hermite_max_order_z` (default `min(1, D-1)`): max order in z

    Idempotent: skips when file already exists unless `force=True`.

    Args:
        cfg: Full YAML config dict.
        force: Regenerate even if file already exists.
        events_out: Optional list appended with a structured event dict
            describing the file action ("created").
    """
    qp_cfg = cfg.get("surrogates", {}).get("qp")
    if not isinstance(qp_cfg, dict):
        return
    name = qp_cfg.get("name")
    if name not in ("learned_basis_profile", "fixed_basis_profile"):
        return
    args = qp_cfg.get("args", {}) or {}

    if name == "learned_basis_profile":
        path_arg = args.get("warmstart_basis_path")
    else:
        path_arg = args.get("basis_path")
    if not isinstance(path_arg, str):
        return

    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])
    n_bins_val = cfg.get("loss", {}).get("args", {}).get("n_bins")
    if n_bins_val is not None:
        basis_path = _nbins_path(path_arg, int(n_bins_val), data_dir)
    else:
        p = Path(path_arg)
        basis_path = p if p.is_absolute() else data_dir / p

    if basis_path.exists() and not force:
        return

    max_order = int(args.get("hermite_max_order", 4))
    sigma_ref = float(args.get("hermite_basis_sigma", 3.0))
    sigma_z = float(args.get("hermite_sigma_z", 1.0))
    max_order_z_arg = args.get("hermite_max_order_z")
    max_order_z = int(max_order_z_arg) if max_order_z_arg is not None else None

    dl_args = cfg.get("data_loader", {}).get("args", {})
    D_dim = int(dl_args.get("D", dl_args.get("d", 1)))
    H_dim = int(dl_args.get("H", dl_args.get("h", 21)))
    W_dim = int(dl_args.get("W", dl_args.get("w", 21)))

    if D_dim > 1:
        W_basis, b, orders = _build_hermite_basis_3d(
            D_dim,
            H_dim,
            W_dim,
            max_order,
            sigma_ref,
            sigma_z,
            max_order_z=max_order_z,
        )
        max_order_z_used = (
            max_order_z if max_order_z is not None else min(1, D_dim - 1)
        )
    else:
        W_basis, b, orders = _build_hermite_basis_2d(
            H_dim, W_dim, max_order, sigma_ref
        )
        max_order_z_used = 0
    d = W_basis.shape[1]

    basis_data = {
        "W": W_basis,
        "b": b,
        "d": d,
        "orders": orders,
        "max_order": max_order,
        "max_order_z": max_order_z_used,
        "sigma_ref": sigma_ref,
        "sigma_z": sigma_z,
        "sigma_prior": 3.0,
        "basis_type": "hermite",
        "shape": (D_dim, H_dim, W_dim),
    }
    basis_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(basis_data, basis_path)
    logger.info(
        "Saved %s (Hermite basis: %dD, max_order=%d, max_order_z=%d, "
        "sigma_ref=%.2f, d=%d)",
        basis_path.name,
        3 if D_dim > 1 else 2,
        max_order,
        max_order_z_used,
        sigma_ref,
        d,
    )
    if events_out is not None:
        events_out.append(
            {
                "file": basis_path.name,
                "action": "created",
                "path": str(basis_path),
                "reason": "Hermite basis warmstart auto-generated",
                "max_order": max_order,
                "max_order_z": max_order_z_used,
                "sigma_ref": sigma_ref,
                "d": d,
            }
        )


def prepare_per_bin_priors(
    cfg: dict,
    *,
    n_bins: int = 0,
    force: bool = False,
    events_out: list[dict] | None = None,
) -> None:
    """Generate per-bin prior .pt files if the loss config requires them.

    Checks whether the loss config references per-bin files
    (s_squared_per_group).
    and generates any that are missing.

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

    # Determine which files are referenced and which are missing
    per_bin_keys = [
        "s_squared_per_group",
    ]

    needed = {}
    for key in per_bin_keys:
        if key not in loss_args:
            continue
        filename = loss_args[key]
        if isinstance(filename, str):
            path = _nbins_path(filename, n_bins, data_dir)
            if force or not path.exists():
                needed[key] = path

    # Check group_label consistency with n_bins even if all files exist
    need_group_labels = False
    gl_path = _nbins_path("group_labels.pt", n_bins, data_dir)
    if not gl_path.exists():
        need_group_labels = True
    else:
        existing_gl = torch.load(gl_path, weights_only=True)
        existing_n_bins = int(existing_gl.max().item()) + 1
        if existing_n_bins != n_bins:
            logger.warning(
                "group_labels_%d.pt has %d bins but config specifies n_bins=%d; "
                "re-binning and regenerating all per-bin files",
                n_bins,
                existing_n_bins,
                n_bins,
            )
            # Force regeneration of ALL referenced per-bin files
            for key in per_bin_keys:
                if key not in loss_args:
                    continue
                filename = loss_args[key]
                if isinstance(filename, str):
                    needed[key] = _nbins_path(filename, n_bins, data_dir)

    # Check if global background prior needs computing.
    # Fits Gamma MLE on all background values and saves (alpha, rate).
    bg_prior_path = data_dir / "bg_prior.pt"
    need_bg_prior = (
        "bg_rate" not in loss_args and "bg_concentration" not in loss_args
    ) and (force or not bg_prior_path.exists())

    if not needed and not need_group_labels and not need_bg_prior:
        return

    logger.info(
        "Generating per-bin prior files: %s",
        ", ".join(needed.keys()),
    )

    metadata = torch.load(
        _resolve_reference_path(data_dir, cfg), weights_only=False
    )
    d = metadata["d"]
    N = len(d)

    # Bin by resolution (n_bins may be reduced if bins are too small)
    group_labels, bin_edges, n_bins = _bin_by_resolution(d, n_bins)
    logger.info("Binned %d reflections into %d resolution shells", N, n_bins)

    # Save group_labels as a separate n_bins-suffixed file (never mutate metadata.pt)
    # Also set in-memory for downstream prior computation in this function.
    metadata["group_label"] = group_labels
    gl_path = _nbins_path("group_labels.pt", n_bins, data_dir)
    torch.save(group_labels, gl_path)
    logger.info("Saved %s (%d bins)", gl_path.name, n_bins)

    if "s_squared_per_group" in needed:
        s_squared = _compute_s_squared_per_group(d, group_labels, n_bins)
        torch.save(s_squared, needed["s_squared_per_group"])
        logger.info("Saved s_squared_per_group.pt")

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


def _compute_s_squared_per_group(
    d: Tensor,
    group_labels: Tensor,
    n_bins: int,
) -> Tensor:
    """Wilson resolution parameter: s_k^2 = 1 / (4 * mean_d_k^2)."""
    s_sq = torch.zeros(n_bins)
    for b in range(n_bins):
        sel = d[group_labels == b]
        if len(sel) > 0:
            mean_d = sel.mean()
            s_sq[b] = 1.0 / (4.0 * mean_d**2)
    return s_sq


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


def _hermite_polynomial(n_order: int, x: Tensor) -> Tensor:
    """Probabilist's Hermite polynomial H_n(x) by three-term recurrence."""
    if n_order == 0:
        return torch.ones_like(x)
    if n_order == 1:
        return x
    h_prev2 = torch.ones_like(x)
    h_prev1 = x
    for k in range(2, n_order + 1):
        h_curr = x * h_prev1 - (k - 1) * h_prev2
        h_prev2 = h_prev1
        h_prev1 = h_curr
    return h_curr


def _build_hermite_basis_2d(
    H: int = 21,
    W: int = 21,
    max_order: int = 4,
    sigma_ref: float = 3.0,
) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
    """2D Hermite function basis with half-Gaussian envelope.

    Each basis function: phi_{nx,ny}(x,y) = H_nx(x/s) * H_ny(y/s) * exp(-r^2/(4s^2))
    Orthogonal in unweighted L^2(R^2).  The (0,0) mode is excluded (absorbed by b).

    Returns:
        Tuple of (W, b, orders) where W is a (H*W, d) basis matrix, b is a
        (H*W,) bias (log of reference Gaussian profile), and orders is a
        list of (nx, ny) tuples.
    """
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float64),
        torch.arange(W, dtype=torch.float64),
        indexing="ij",
    )
    x_norm = (xx - cx) / sigma_ref
    y_norm = (yy - cy) / sigma_ref

    # Half-Gaussian envelope
    half_gaussian = torch.exp(-0.25 * (x_norm**2 + y_norm**2))

    # Reference profile (for bias b)
    full_gaussian = torch.exp(-0.5 * (x_norm**2 + y_norm**2))
    ref = full_gaussian / full_gaussian.sum()
    b = torch.log(ref.reshape(-1).clamp(min=1e-10)).float()

    basis_list = []
    orders = []
    for nx in range(max_order + 1):
        for ny in range(max_order + 1 - nx):
            if nx == 0 and ny == 0:
                continue
            phi = (
                _hermite_polynomial(nx, x_norm)
                * _hermite_polynomial(ny, y_norm)
                * half_gaussian
            )
            phi = phi / phi.norm()
            basis_list.append(phi.reshape(-1))
            orders.append((nx, ny))

    W_basis = torch.stack(basis_list, dim=1).float()
    return W_basis, b, orders


def _build_hermite_basis_3d(
    D: int,
    H: int = 21,
    W: int = 21,
    max_order: int = 4,
    sigma_ref: float = 3.0,
    sigma_z: float = 1.0,
    max_order_z: int | None = None,
) -> tuple[Tensor, Tensor, list[tuple[int, int, int]]]:
    """3D Hermite function basis with half-Gaussian envelope.

    Args:
        D: Number of frames (z dimension).
        H: Shoebox height.
        W: Shoebox width.
        max_order: Max polynomial order in x, y.
        sigma_ref: Reference Gaussian width in pixels (x, y).
        sigma_z: Reference Gaussian width along z/frame axis.
        max_order_z: Max polynomial order in z. Defaults to `min(1, D-1)`.

    Returns:
        Tuple of (W, b, orders) where W is a (D*H*W, d) basis matrix, b is a
        (D*H*W,) bias (log of reference 3D Gaussian profile), and orders is a
        list of (nx, ny, nz) tuples.
    """
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    cz = (D - 1) / 2.0

    zz, yy, xx = torch.meshgrid(
        torch.arange(D, dtype=torch.float64),
        torch.arange(H, dtype=torch.float64),
        torch.arange(W, dtype=torch.float64),
        indexing="ij",
    )
    x_norm = (xx - cx) / sigma_ref
    y_norm = (yy - cy) / sigma_ref
    z_norm = (zz - cz) / sigma_z

    # Half-Gaussian envelope in all 3 dims
    half_gaussian = torch.exp(-0.25 * (x_norm**2 + y_norm**2 + z_norm**2))

    # Reference 3D Gaussian profile
    full_gaussian = torch.exp(-0.5 * (x_norm**2 + y_norm**2 + z_norm**2))
    ref = full_gaussian / full_gaussian.sum()
    b = torch.log(ref.reshape(-1).clamp(min=1e-10)).float()

    if max_order_z is None:
        max_order_z = min(1, D - 1)

    basis_list = []
    orders = []
    for nz in range(max_order_z + 1):
        for nx in range(max_order + 1):
            for ny in range(max_order + 1 - nx):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                phi = (
                    _hermite_polynomial(nx, x_norm)
                    * _hermite_polynomial(ny, y_norm)
                    * _hermite_polynomial(nz, z_norm)
                    * half_gaussian
                )
                phi = phi / phi.norm()
                basis_list.append(phi.reshape(-1))
                orders.append((nx, ny, nz))

    W_basis = torch.stack(basis_list, dim=1).float()
    return W_basis, b, orders


