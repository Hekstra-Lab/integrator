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


def _gammaB_wants_mean_init(cfg: dict, sur_key: str) -> bool:
    """True if surrogates[sur_key] is gammaB and has no explicit mean_init.

    Used to decide whether we need to compute a dataset-level mean estimate
    from raw counts to seed the Gamma head's bias.
    """
    sur = cfg.get("surrogates", {}).get(sur_key, {})
    if not isinstance(sur, dict) or sur.get("name") != "gammaB":
        return False
    args = sur.get("args", {}) or {}
    return "mean_init" not in args


def _compute_qi_qbg_mean_init(
    counts,
    masks,
    D: int,
    H: int,
    W: int,
) -> dict:
    """Estimate sensible (qi, qbg) mean_init values directly from counts.

    Uses the same quiet-frame / border-pixel background subtraction as
    `_bg_subtract_signal` — fully independent of DIALS estimates.
    """
    import torch as _t

    signal = _bg_subtract_signal(counts, masks, D, H, W)
    intensity = signal.sum(dim=1)
    pos = intensity[intensity > 0]

    counts_clean = counts.float().clamp(min=0)
    masks_f = masks.float()
    counts_3d = (counts_clean * masks_f).reshape(counts.shape[0], D, H * W)
    masks_3d = masks_f.reshape(counts.shape[0], D, H * W)
    if D > 1:
        frame_sums = counts_3d.sum(dim=-1)
        frame_n = masks_3d.sum(dim=-1)
        min_idx = frame_sums.argmin(dim=-1)
        bg_tot = frame_sums.gather(1, min_idx.unsqueeze(-1)).squeeze(-1)
        bg_n = (
            frame_n.gather(1, min_idx.unsqueeze(-1)).squeeze(-1).clamp(min=1)
        )
        bg_rate = bg_tot / bg_n
    else:
        frame = counts_clean.reshape(counts.shape[0], H, W)
        border = _t.ones(H, W, dtype=_t.bool)
        border[2:-2, 2:-2] = False
        bg_rate = frame[:, border].mean(dim=-1)

    return {
        "qi_median": float(pos.median().item()) if pos.numel() > 0 else 1.0,
        "qi_mean": float(pos.mean().item()) if pos.numel() > 0 else 1.0,
        "qbg_median": float(bg_rate.median().item()),
        "qbg_mean": float(bg_rate.mean().item()),
        "n_reflections": int(counts.shape[0]),
        "n_positive_intensity": int(pos.numel()),
        "note": (
            "Computed from raw counts only (no DIALS). Intensity = sum of "
            "bg-subtracted counts per reflection; bg = per-pixel rate from "
            "quiet-frame estimator. Use qi_median/qbg_median as default "
            "mean_init for GammaB — more robust to heavy-tail peaks than "
            "the mean."
        ),
    }


def prepare_global_priors(
    cfg: dict,
    *,
    force: bool = False,
) -> None:
    """Generate global prior .pt files for the default Loss if needed.


    Args:
        cfg: Full YAML config dict.
        force: Regenerate even if file already exists.
    """
    loss_name = cfg.get("loss", {}).get("name", "")
    if loss_name != "default":
        return

    loss_args = cfg["loss"].get("args", {})
    pprf_cfg = loss_args.get("pprf_cfg")
    if pprf_cfg is None or pprf_cfg.get("name") != "dirichlet":
        return

    params = pprf_cfg.get("params", {})
    conc_value = params.get("concentration")
    if not isinstance(conc_value, str):
        # Scalar concentration, nothing to generate
        return

    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])
    conc_path = (
        Path(conc_value)
        if Path(conc_value).is_absolute()
        else data_dir / conc_value
    )

    if conc_path.exists() and not force:
        return

    logger.info("Generating global Dirichlet concentration: %s", conc_path)

    counts, masks, metadata = _load_raw_data(data_dir, cfg)

    dl_args = cfg.get("data_loader", {}).get("args", {})
    D_dim = int(dl_args.get("D", dl_args.get("d", 1)))
    H_dim = int(dl_args.get("H", dl_args.get("h", 21)))
    W_dim = int(dl_args.get("W", dl_args.get("w", 21)))

    # Single bin = global MOM across all reflections
    N = counts.shape[0]
    all_one_bin = torch.zeros(N, dtype=torch.long)
    concentration = _fit_dirichlet_per_group(
        counts, masks, all_one_bin, 1, D=D_dim, H=H_dim, W=W_dim
    )
    # Squeeze from (1, D*H*W) to (D*H*W,)
    concentration = concentration.squeeze(0)

    torch.save(concentration, conc_path)
    logger.info(
        "Saved global concentration.pt (MOM bg-sub, sum(alpha)=%.1f)",
        concentration.sum().item(),
    )


def prepare_per_bin_priors(
    cfg: dict,
    *,
    n_bins: int = 0,
    min_intensity: float = 0.01,
    force: bool = False,
    events_out: list[dict] | None = None,
) -> None:
    """Generate per-bin prior .pt files if the loss config requires them.

    Checks whether the loss config references per-bin files
    (s_squared_per_group, tau_per_group, i_concentration_per_group, etc.)
    and generates any that are missing.

    Idempotent: skips files that already exist unless force=True.

    Args:
        cfg: Full YAML config dict.  If `loss.args.n_bins` is set in the
            config, it is used as the number of resolution bins (unless
            overridden by the *n_bins* argument).
        n_bins: Number of resolution bins.  When <= 0 (default), reads from
            `cfg["loss"]["args"]["n_bins"]`, falling back to 20.
        min_intensity: Minimum intensity for tau estimation.
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

    # Read n_bins from config, fall back to 20
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

    # Auto-inject concentration file paths when pi_cfg requests gamma
    pi_cfg = loss_args.get("pi_cfg")
    if (
        isinstance(pi_cfg, dict)
        and pi_cfg.get("name") == "gamma"
        and "i_concentration_per_group" not in loss_args
    ):
        loss_args["i_concentration_per_group"] = "i_concentration_per_group.pt"
    # Determine which files are referenced and which are missing
    per_bin_keys = [
        "i_concentration_per_group",
        "s_squared_per_group",
        "tau_per_group",
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

    # Check if mean_init needs computing for any gammaB surrogate (qi/qbg)
    # that didn't set it explicitly. Cached to qi_qbg_mean_init.pt to avoid
    # recomputing on every run; any explicit YAML value takes precedence.
    init_path = data_dir / "qi_qbg_mean_init.pt"
    need_init = (
        _gammaB_wants_mean_init(cfg, "qi")
        or _gammaB_wants_mean_init(cfg, "qbg")
    ) and (force or not init_path.exists())

    # Check if global background prior needs computing.
    # Fits Gamma MLE on all background values and saves (alpha, rate).
    bg_prior_path = data_dir / "bg_prior.pt"
    need_bg_prior = (
        "bg_rate" not in loss_args and "bg_concentration" not in loss_args
    ) and (force or not bg_prior_path.exists())

    if (
        not needed
        and not need_group_labels
        and not need_init
        and not need_bg_prior
    ):
        return

    logger.info(
        "Generating per-bin prior files: %s",
        ", ".join(needed.keys()),
    )

    metadata = torch.load(
        _resolve_reference_path(data_dir, cfg), weights_only=False
    )
    counts, masks = _try_load_counts_masks(data_dir, cfg)

    def _require_counts(prior_name: str) -> tuple[Tensor, Tensor]:
        if counts is None or masks is None:
            raise FileNotFoundError(
                f"Computing '{prior_name}' requires per-pixel counts/masks, "
                f"but neither counts.npy nor counts.pt was found in "
                f"{data_dir}. "
                f"Either remove '{prior_name}' from the loss config, or "
                f"provide a precomputed file."
            )
        return counts, masks

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

    # Generate each missing file
    if "s_squared_per_group" in needed:
        s_squared = _compute_s_squared_per_group(d, group_labels, n_bins)
        torch.save(s_squared, needed["s_squared_per_group"])
        logger.info("Saved s_squared_per_group.pt")

    if "tau_per_group" in needed:
        tau_source = loss_args.get("tau_source", "dials")

        if tau_source == "crude":
            c, m = _require_counts("tau_per_group (crude)")
            crude_I = _crude_intensity_from_cfg(c, m, cfg)
            tau = _compute_tau_per_group(
                crude_I, group_labels, n_bins, min_intensity
            )
            torch.save(tau, needed["tau_per_group"])
            logger.info(
                "Saved tau_per_group.pt (from crude bg-subtraction, "
                "%d/%d reflections with I>0)",
                (crude_I > min_intensity).sum().item(),
                len(crude_I),
            )
        else:
            intensity = metadata.get(
                "intensity.prf.value",
                metadata.get("intensity.sum.value"),
            )
            if intensity is not None:
                tau = _compute_tau_per_group(
                    intensity, group_labels, n_bins, min_intensity
                )
                torch.save(tau, needed["tau_per_group"])
                logger.info("Saved tau_per_group.pt (from DIALS intensities)")
            else:
                logger.warning(
                    "No intensity column found and tau_source='dials'; "
                    "falling back to crude bg-subtraction"
                )
                c, m = _require_counts("tau_per_group (crude fallback)")
                crude_I = _crude_intensity_from_cfg(c, m, cfg)
                tau = _compute_tau_per_group(
                    crude_I, group_labels, n_bins, min_intensity
                )
                torch.save(tau, needed["tau_per_group"])
                logger.info("Saved tau_per_group.pt (crude fallback)")

    if "i_concentration_per_group" in needed:
        tau_source = loss_args.get("tau_source", "dials")
        intensity = _get_intensity_for_prior(
            tau_source, counts, masks, metadata, cfg
        )
        alpha = _fit_gamma_prior_per_group(
            intensity, group_labels, n_bins, min_intensity
        )
        torch.save(alpha, needed["i_concentration_per_group"])
        logger.info(
            "Saved i_concentration_per_group.pt (Gamma MLE alpha, source=%s)",
            tau_source,
        )

    # ── qi / qbg mean_init from raw counts ─────────────────────────────
    # Computes a dataset-level estimate of the typical intensity and bg
    # rate using the same bg-subtraction logic as _bg_subtract_signal, so
    # it's fully independent of DIALS. The factory injects these into
    # GammaB surrogate args at construction time (when mean_init isn't
    # explicitly set in the YAML), removing the "grow mu from 0.7 over
    # many epochs" early-training phase.
    if need_init:
        if counts is None or masks is None:
            logger.info(
                "qi_qbg_mean_init.pt requires counts/masks;  "
                "layout has none. Skipping — set explicit mean_init values "
                "in the YAML (or `mean_init: null` to opt out)."
            )
        else:
            dl_args = cfg.get("data_loader", {}).get("args", {})
            D_dim = int(dl_args.get("D", dl_args.get("d", 1)))
            H_dim = int(dl_args.get("H", dl_args.get("h", 21)))
            W_dim = int(dl_args.get("W", dl_args.get("w", 21)))
            init_stats = _compute_qi_qbg_mean_init(
                counts, masks, D_dim, H_dim, W_dim
            )
            torch.save(init_stats, init_path)
            logger.info(
                "Saved qi_qbg_mean_init.pt "
                "(qi_median=%.2f, qi_mean=%.2f, qbg_median=%.3f, qbg_mean=%.3f)",
                init_stats["qi_median"],
                init_stats["qi_mean"],
                init_stats["qbg_median"],
                init_stats["qbg_mean"],
            )
            if events_out is not None:
                events_out.append(
                    {
                        "file": "qi_qbg_mean_init.pt",
                        "action": "created",
                        "path": str(init_path),
                        "reason": "GammaB qi/qbg without explicit mean_init",
                        "qi_median": init_stats["qi_median"],
                        "qi_mean": init_stats["qi_mean"],
                        "qbg_median": init_stats["qbg_median"],
                        "qbg_mean": init_stats["qbg_mean"],
                    }
                )

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

    If the files don't exist (e.g. old data without per-bin priors),
    this is a no-op and existing labels in metadata.pt are used.
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


def _load_shoebox_array(path: Path) -> Tensor:
    """Load counts/masks from either .pt (legacy torch.save) or .npy
    (newer mksbox memmap output). Returns a torch.Tensor either way."""
    p = Path(path)
    if p.suffix == ".npy" or (
        not p.exists() and p.with_suffix(".npy").exists()
    ):
        # Prefer .npy if explicitly named or if a sibling .npy exists
        npy_path = p if p.suffix == ".npy" else p.with_suffix(".npy")
        import numpy as np

        return torch.from_numpy(np.asarray(np.load(npy_path)))
    return torch.load(p, weights_only=True)


def _load_raw_data(
    data_dir: Path,
    cfg: dict,
) -> tuple[Tensor, Tensor, dict]:
    """Load counts, masks, metadata from data_dir using config paths.

    Supports both .pt (legacy) and .npy (newer mksbox memmap) for the
    counts/masks arrays. Metadata stays .pt since it's a small dict."""
    sfn = cfg["data_loader"]["args"].get("shoebox_file_names", {})

    counts_name = sfn.get("counts", "counts.pt")
    masks_name = sfn.get("masks", "masks.pt")

    counts = _load_shoebox_array(data_dir / counts_name)
    masks = _load_shoebox_array(data_dir / masks_name)

    ref_path = _resolve_reference_path(data_dir, cfg)
    metadata = torch.load(ref_path, weights_only=False)

    return counts, masks, metadata


def _try_load_counts_masks(
    data_dir: Path,
    cfg: dict,
) -> tuple[Tensor | None, Tensor | None]:
    sfn = cfg["data_loader"]["args"].get("shoebox_file_names", {})
    counts_name = sfn.get("counts", "counts.pt")
    masks_name = sfn.get("masks", "masks.pt")
    counts_path = data_dir / counts_name
    masks_path = data_dir / masks_name

    # Match _load_shoebox_array's resolution: a sibling .npy is acceptable.
    def _exists(p: Path) -> bool:
        if p.exists():
            return True
        if p.suffix != ".npy" and p.with_suffix(".npy").exists():
            return True
        return False

    if not _exists(counts_path) or not _exists(masks_path):
        logger.info(
            "Counts/masks not found at %s / %s — skipping count-dependent "
            "prior computations. ",
            counts_path,
            masks_path,
        )
        return None, None

    return (
        _load_shoebox_array(counts_path),
        _load_shoebox_array(masks_path),
    )


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


def _crude_intensity_from_cfg(
    counts: Tensor, masks: Tensor, cfg: dict
) -> Tensor:
    """Read shoebox dimensions from config and compute crude intensity."""
    dl_args = cfg.get("data_loader", {}).get("args", {})
    n_frames = int(dl_args.get("d", 3))
    n_pixels = int(dl_args.get("h", 21)) * int(dl_args.get("w", 21))
    return _compute_crude_intensity(counts, masks, n_frames, n_pixels)


def _compute_crude_intensity(
    counts: Tensor,
    masks: Tensor,
    n_frames: int,
    n_pixels_per_frame: int,
) -> Tensor:
    """Estimate intensity from raw shoeboxes via quietest-frame bg subtraction.

    For each shoebox, the frame with the lowest total (masked) counts is
    treated as pure background.  The per-pixel background rate from that
    frame is subtracted from the total shoebox counts to give a crude
    intensity estimate.

    Args:
        counts: Raw shoebox counts, shape `(N, n_frames * n_pixels_per_frame)`.
        masks: Valid-pixel masks, same shape as *counts*.
        n_frames: Number of frames per shoebox (typically 3).
        n_pixels_per_frame: Pixels per frame (e.g. 21*21 = 441).

    Returns:
        Crude intensity estimate per reflection, shape `(N,)`.
        Can be negative for weak / noise-dominated reflections.
    """
    N = counts.shape[0]

    # Clamp dead pixels (count == -1) to 0
    counts_clean = counts.float().clamp(min=0)
    masks_f = masks.float()

    # Reshape to (N, n_frames, n_pixels_per_frame)
    counts_3d = (counts_clean * masks_f).reshape(
        N, n_frames, n_pixels_per_frame
    )
    masks_3d = masks_f.reshape(N, n_frames, n_pixels_per_frame)

    # Total masked counts per frame: (N, n_frames)
    frame_counts = counts_3d.sum(dim=-1)

    # Number of valid pixels per frame: (N, n_frames)
    frame_n_pixels = masks_3d.sum(dim=-1)

    # Quietest frame = frame with minimum total counts
    min_frame_idx = frame_counts.argmin(dim=-1)  # (N,)
    bg_frame_counts = frame_counts.gather(
        1, min_frame_idx.unsqueeze(-1)
    ).squeeze(-1)
    bg_frame_n_pixels = frame_n_pixels.gather(
        1, min_frame_idx.unsqueeze(-1)
    ).squeeze(-1)

    # Background rate per pixel from quietest frame
    bg_per_pixel = bg_frame_counts / bg_frame_n_pixels.clamp(min=1)

    # Total counts and total valid pixels across all frames
    total_counts = counts_3d.sum(dim=(1, 2))
    total_n_pixels = masks_3d.sum(dim=(1, 2))

    # Crude I = total_counts - n_valid_pixels * bg_per_pixel
    crude_I = total_counts - total_n_pixels * bg_per_pixel

    logger.info(
        "Crude intensity: min=%.1f, median=%.1f, max=%.1f, "
        "fraction negative=%.3f",
        crude_I.min().item(),
        crude_I.median().item(),
        crude_I.max().item(),
        (crude_I < 0).float().mean().item(),
    )

    return crude_I


def _compute_wavelength_bin_edges(
    metadata_path: Path,
    n_lambda_bins: int,
) -> Tensor:
    """Quantile-based wavelength bin edges from metadata['wavelength'].

    Returns a 1-D tensor of length `n_lambda_bins + 1` such that consecutive
    edges define a half-open bin `[edges[i], edges[i+1])` (rightmost
    inclusive). Equal-quantile spacing → roughly equal counts per bin → the
    per-bin G_k posteriors get balanced gradient signal.
    """
    metadata = torch.load(metadata_path, weights_only=True)
    if "wavelength" not in metadata:
        raise KeyError(
            f"metadata.pt at {metadata_path} has no 'wavelength' column; "
            "this CLI step requires output from refltorch.mksbox-laue."
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


def _compute_tau_per_group(
    intensity: Tensor,
    group_labels: Tensor,
    n_bins: int,
    min_intensity: float = 0.01,
) -> Tensor:
    """Exponential rate for intensity: tau_k = 1 / mean(I_k)."""
    tau = torch.zeros(n_bins)
    for b in range(n_bins):
        sel = intensity[group_labels == b]
        sel = sel[sel > min_intensity]
        if len(sel) > 0:
            tau[b] = 1.0 / sel.mean()
        else:
            tau[b] = 1.0
    return tau


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


def _fit_dirichlet_per_group(
    counts: Tensor,
    masks: Tensor,
    group_labels: Tensor,
    n_bins: int,
    D: int = 1,
    H: int = 21,
    W: int = 21,
) -> Tensor:
    """Fit Dirichlet concentration per bin via MOM on bg-subtracted profiles.

    For each bin:
      1. Subtract per-reflection crude background (quietest-frame estimate)
         to isolate the signal profile.
      2. Clamp and normalize bg-subtracted counts to proportions over all
         `D*H*W` pixels.
      3. Estimate Dirichlet precision kappa via method of moments.
      4. Return `concentration = kappa * p_bar`.

    Handles both 2D (`D=1`, stills/Laue) and 3D (`D>1`, rotation) cases.

    Args:
        counts: Raw shoebox counts, shape `(N, D*H*W)`.
        masks: Valid-pixel masks, same shape as *counts*.
        group_labels: Bin assignment per reflection, shape `(N,)`.
        n_bins: Number of resolution bins.
        D: Shoebox depth (frames).
        H: Shoebox height.
        W: Shoebox width.

    Returns:
        Dirichlet concentration, shape `(n_bins, D*H*W)`.
    """
    n_pixels = D * H * W
    n_pixels_per_frame = H * W
    N = counts.shape[0]

    counts_clean = counts.float().clamp(min=0)
    masks_f = masks.float()
    counts_masked = counts_clean * masks_f

    # ── Crude background subtraction (quietest frame) ─────────────────────
    counts_3d = counts_masked.reshape(N, D, n_pixels_per_frame)
    masks_3d = masks_f.reshape(N, D, n_pixels_per_frame)

    if D > 1:
        # Quietest frame = frame with minimum total counts
        frame_counts = counts_3d.sum(dim=-1)  # (N, D)
        frame_n_pixels = masks_3d.sum(dim=-1)  # (N, D)
        min_frame_idx = frame_counts.argmin(dim=-1)  # (N,)
        bg_frame_counts = frame_counts.gather(
            1, min_frame_idx.unsqueeze(-1)
        ).squeeze(-1)
        bg_frame_n_pixels = frame_n_pixels.gather(
            1, min_frame_idx.unsqueeze(-1)
        ).squeeze(-1)
        bg_per_pixel = bg_frame_counts / bg_frame_n_pixels.clamp(min=1)
    else:
        # Single frame: estimate bg from border pixels (outer 2-pixel ring)
        frame = counts_3d[:, 0, :]  # (N, H*W)
        frame_2d = frame.reshape(N, H, W)
        border_mask = torch.ones(H, W, dtype=torch.bool)
        border_mask[2:-2, 2:-2] = False
        border_vals = frame_2d[:, border_mask]  # (N, n_border)
        bg_per_pixel = border_vals.mean(dim=-1)  # (N,)

    # Subtract bg from all pixels: signal = counts - bg * mask
    signal = counts_masked - bg_per_pixel.unsqueeze(-1) * masks_f
    signal = signal.clamp(min=0)  # no negative signal

    concentration = torch.zeros(n_bins, n_pixels)

    for b in range(n_bins):
        sel = signal[group_labels == b]
        if len(sel) < 2:
            concentration[b] = 1e-6
            continue

        # Normalize bg-subtracted signal to proportions
        totals = sel.sum(dim=1, keepdim=True).clamp(min=1)
        sel_norm = sel / totals  # (n_refl, n_pixels)

        # Mean profile
        p_bar = sel_norm.mean(dim=0)  # (n_pixels,)

        # Estimate kappa (Dirichlet precision) via MOM
        var_p = sel_norm.var(dim=0)
        valid = p_bar > 1e-6
        if valid.sum() > 0:
            ratio = (p_bar[valid] * (1 - p_bar[valid])) / var_p[valid].clamp(
                min=1e-12
            ) - 1
            kappa = ratio.median().clamp(min=1.0)
        else:
            kappa = torch.tensor(1.0)

        concentration[b] = (kappa * p_bar).clamp(min=1e-6)

        logger.debug(
            "Bin %d: kappa=%.1f, sum(alpha)=%.1f, n_refl=%d",
            b,
            kappa.item(),
            concentration[b].sum().item(),
            len(sel),
        )

    return concentration


def _get_intensity_for_prior(
    tau_source: str,
    counts: Tensor | None,
    masks: Tensor | None,
    metadata: dict,
    cfg: dict,
) -> Tensor:
    """Return intensity tensor for prior fitting, respecting tau_source config."""
    if tau_source == "crude":
        if counts is None or masks is None:
            raise FileNotFoundError()
        return _crude_intensity_from_cfg(counts, masks, cfg)

    intensity = metadata.get(
        "intensity.prf.value",
        metadata.get("intensity.sum.value"),
    )
    if intensity is not None:
        return intensity

    if counts is None or masks is None:
        raise FileNotFoundError()

    logger.warning(
        "No intensity column found and tau_source='dials'; "
        "falling back to crude bg-subtraction"
    )
    return _crude_intensity_from_cfg(counts, masks, cfg)


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


def _fit_gamma_prior_per_group(
    intensity: Tensor,
    group_labels: Tensor,
    n_bins: int,
    min_intensity: float = 0.01,
) -> Tensor:
    """Fit Gamma MLE per resolution bin and return alpha (shape) per group.

    Args:
        intensity: Per-reflection intensity estimates, shape `(N,)`.
        group_labels: Bin assignment per reflection, shape `(N,)`.
        n_bins: Number of resolution bins.
        min_intensity: Minimum intensity threshold (exclude weak/negative
            reflections).

    Returns:
        Alpha (concentration) per bin, shape `(n_bins,)`.
    """
    alpha_per_group = torch.ones(n_bins)
    for b in range(n_bins):
        sel = intensity[group_labels == b]
        sel = sel[sel > min_intensity]
        if len(sel) < 10:
            # Too few reflections for reliable MLE; default to exponential
            alpha_per_group[b] = 1.0
            logger.warning(
                "Bin %d has only %d reflections with I > %.2f; "
                "defaulting alpha=1 (exponential)",
                b,
                len(sel),
                min_intensity,
            )
            continue
        alpha, _beta = _fit_gamma_mle(sel)
        alpha_per_group[b] = alpha.clamp(min=0.1)
        logger.debug(
            "Bin %d: Gamma MLE alpha=%.3f (n=%d)", b, alpha.item(), len(sel)
        )
    return alpha_per_group


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
) -> tuple[Tensor, Tensor, list[tuple[int, int, int]]]:
    """3D Hermite function basis (frame x spatial) with half-Gaussian envelope.

    Frame direction uses max order min(1, D-1) since D is typically small (3).
    Spatial directions use the full max_order.

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

    max_order_z = min(1, D - 1)  # at most linear in frame direction

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


def _bg_subtract_signal(
    counts: Tensor,
    masks: Tensor,
    D: int,
    H: int,
    W: int,
) -> Tensor:
    """Background-subtract raw shoebox counts.

    Reuses the same logic as _fit_dirichlet_per_group: quietest-frame
    for 3D, border-pixel average for 2D.

    Returns:
        Non-negative bg-subtracted counts, shape (N, D*H*W).
    """
    n_pixels_per_frame = H * W
    N = counts.shape[0]

    counts_clean = counts.float().clamp(min=0)
    masks_f = masks.float()
    counts_masked = counts_clean * masks_f

    counts_3d = counts_masked.reshape(N, D, n_pixels_per_frame)
    masks_3d = masks_f.reshape(N, D, n_pixels_per_frame)

    if D > 1:
        frame_counts = counts_3d.sum(dim=-1)
        frame_n_pixels = masks_3d.sum(dim=-1)
        min_frame_idx = frame_counts.argmin(dim=-1)
        bg_frame_counts = frame_counts.gather(
            1, min_frame_idx.unsqueeze(-1)
        ).squeeze(-1)
        bg_frame_n_pixels = frame_n_pixels.gather(
            1, min_frame_idx.unsqueeze(-1)
        ).squeeze(-1)
        bg_per_pixel = bg_frame_counts / bg_frame_n_pixels.clamp(min=1)
    else:
        frame = counts_3d[:, 0, :]
        frame_2d = frame.reshape(N, H, W)
        border_mask = torch.ones(H, W, dtype=torch.bool)
        border_mask[2:-2, 2:-2] = False
        border_vals = frame_2d[:, border_mask]
        bg_per_pixel = border_vals.mean(dim=-1)

    signal = counts_masked - bg_per_pixel.unsqueeze(-1) * masks_f
    return signal.clamp(min=0)
