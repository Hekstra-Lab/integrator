import logging
import math
from pathlib import Path

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _nbins_path(filename: str, n_bins: int, data_dir: Path) -> Path:
    """Resolve a prior filename with n_bins suffix: 'foo.pt' -> data_dir/'foo_30.pt'.

    This prevents concurrent runs with different n_bins from clobbering
    each other's prior files in the same data directory.
    """
    p = Path(filename)
    suffixed = f"{p.stem}_{n_bins}{p.suffix}"
    if p.is_absolute():
        return p.parent / suffixed
    return data_dir / suffixed


def prepare_global_priors(
    cfg: dict,
    *,
    force: bool = False,
) -> None:
    """Generate global prior .pt files for the default Loss if needed.

    When the default loss uses a Dirichlet profile prior
    (`pprf_cfg.name = "dirichlet"`) with a `.pt` file path as the
    concentration, this function generates that file using a global MOM
    estimate (bg-subtracted, across all reflections with no binning).

    Idempotent: skips if the file already exists unless force=True.

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
) -> None:
    """Generate per-bin prior .pt files if the loss config requires them.

    Checks whether the loss config references per-bin files
    (bg_rate_per_group, concentration_per_group, s_squared_per_group, etc.)
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
    """
    loss_name = cfg.get("loss", {}).get("name", "")
    if loss_name not in ("per_bin", "wilson"):
        return

    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])
    loss_args = cfg["loss"].get("args", {})

    # Read n_bins from config, fall back to 20
    if n_bins <= 0:
        n_bins = int(loss_args.get("n_bins", 20))

    # Auto-inject concentration file paths when pi_cfg/pbg_cfg request gamma
    pi_cfg = loss_args.get("pi_cfg")
    pbg_cfg = loss_args.get("pbg_cfg")
    if (
        isinstance(pi_cfg, dict)
        and pi_cfg.get("name") == "gamma"
        and "i_concentration_per_group" not in loss_args
    ):
        loss_args["i_concentration_per_group"] = "i_concentration_per_group.pt"
    if (
        isinstance(pbg_cfg, dict)
        and pbg_cfg.get("name") == "gamma"
        and "bg_concentration_per_group" not in loss_args
    ):
        loss_args["bg_concentration_per_group"] = (
            "bg_concentration_per_group.pt"
        )

    # Determine which files are referenced and which are missing
    per_bin_keys = [
        "bg_concentration_per_group",
        "bg_rate_per_group",
        "concentration_per_group",
        "i_concentration_per_group",
        "s_squared_per_group",
        "tau_per_group",
    ]

    # When pprf_quantile is set, the user provides a global concentration
    # file that should not get an n_bins suffix or be auto-generated.
    global_conc = "pprf_quantile" in loss_args

    needed = {}
    for key in per_bin_keys:
        if key not in loss_args:
            continue
        if key == "concentration_per_group" and global_conc:
            continue
        filename = loss_args[key]
        if isinstance(filename, str):
            path = _nbins_path(filename, n_bins, data_dir)
            if force or not path.exists():
                needed[key] = path

    # Check group_label consistency with n_bins even if all files exist
    rebinned = False
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
            rebinned = True

    # Check profile_group_label consistency with 2D binning config
    profile_binning = loss_args.get("profile_binning")
    need_2d_rebinning = False
    pgl_path = _nbins_path("profile_group_labels.pt", n_bins, data_dir)
    if profile_binning is not None:
        if not pgl_path.exists():
            need_2d_rebinning = True
        elif rebinned:
            need_2d_rebinning = True
        else:
            # Verify concentration file shape matches profile_group_label
            existing_pgl = torch.load(pgl_path, weights_only=True)
            n_expected = int(existing_pgl.max().item()) + 1
            conc_fn = loss_args.get("concentration_per_group")
            if isinstance(conc_fn, str):
                conc_path = _nbins_path(conc_fn, n_bins, data_dir)
                if conc_path.exists():
                    conc_on_disk = torch.load(conc_path, weights_only=True)
                    if conc_on_disk.shape[0] != n_expected:
                        logger.warning(
                            "concentration_per_group has %d bins but "
                            "profile_group_label expects %d; regenerating",
                            conc_on_disk.shape[0],
                            n_expected,
                        )
                        need_2d_rebinning = True

        if need_2d_rebinning and "concentration_per_group" in loss_args:
            fn = loss_args["concentration_per_group"]
            if isinstance(fn, str):
                needed["concentration_per_group"] = _nbins_path(
                    fn, n_bins, data_dir
                )

    # Check if profile_basis_per_bin needs generating
    basis_filename = loss_args.get("profile_basis_per_bin")
    need_basis = False
    if isinstance(basis_filename, str):
        basis_path = _nbins_path(basis_filename, n_bins, data_dir)
        if force or not basis_path.exists():
            need_basis = True

    # Check if empirical_profile_basis_per_bin needs generating
    emp_basis_filename = loss_args.get("empirical_profile_basis_per_bin")
    need_emp_basis = False
    if isinstance(emp_basis_filename, str):
        emp_basis_path = _nbins_path(emp_basis_filename, n_bins, data_dir)
        if force or not emp_basis_path.exists():
            need_emp_basis = True

    if (
        not needed
        and not need_2d_rebinning
        and not need_basis
        and not need_emp_basis
        and not need_group_labels
    ):
        return

    logger.info(
        "Generating per-bin prior files: %s",
        ", ".join(needed.keys()),
    )

    # Load raw data (reuse metadata if already loaded for consistency check)
    counts, masks, metadata = _load_raw_data(data_dir, cfg)

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
    if "bg_rate_per_group" in needed:
        bg_per_refl = _get_background_for_prior(
            loss_args.get("tau_source", "dials"), counts, masks, metadata, cfg
        )
        bg_rate = _compute_bg_rate_per_group(bg_per_refl, group_labels, n_bins)
        torch.save(bg_rate, needed["bg_rate_per_group"])
        logger.info("Saved bg_rate_per_group.pt")

    if "bg_concentration_per_group" in needed:
        bg_per_refl = _get_background_for_prior(
            loss_args.get("tau_source", "dials"), counts, masks, metadata, cfg
        )
        bg_alpha = _fit_gamma_prior_per_group(
            bg_per_refl, group_labels, n_bins, min_intensity=1e-6
        )
        torch.save(bg_alpha, needed["bg_concentration_per_group"])
        logger.info(
            "Saved bg_concentration_per_group.pt (Gamma MLE alpha, source=%s)",
            loss_args.get("tau_source", "dials"),
        )

    # 2D profile binning (resolution x azimuthal)
    # Run independently so profile_group_label is available for both
    # concentration_per_group (Dirichlet) and profile_basis_per_bin (latent).
    profile_binning = loss_args.get("profile_binning")
    pgl_path = _nbins_path("profile_group_labels.pt", n_bins, data_dir)
    if profile_binning is not None and (
        need_2d_rebinning or not pgl_path.exists()
    ):
        max_azi_bins = int(profile_binning.get("max_azi_bins", 16))
        min_per_bin = int(profile_binning.get("min_per_bin", 200))
        beam_center = profile_binning.get("beam_center")
        if beam_center is None:
            raise ValueError(
                "profile_binning.beam_center is required for 2D profile binning"
            )
        beam_center = (float(beam_center[0]), float(beam_center[1]))

        profile_group_labels, azi_per_shell, n_profile_bins = (
            _bin_2d_for_profiles(
                metadata,
                group_labels,
                n_bins,
                max_azi_bins,
                beam_center,
                min_per_bin=min_per_bin,
            )
        )

        # Save profile_group_labels as a separate n_bins-suffixed file
        # Also set in-memory for downstream prior computation in this function.
        metadata["profile_group_label"] = profile_group_labels
        pgl_path = _nbins_path("profile_group_labels.pt", n_bins, data_dir)
        torch.save(profile_group_labels, pgl_path)
        logger.info("Saved %s (%d 2D bins)", pgl_path.name, n_profile_bins)

        # Save diagnostic plot (non-fatal)
        try:
            _plot_profile_binning(
                metadata,
                group_labels,
                profile_group_labels,
                n_bins,
                azi_per_shell,
                max_azi_bins,
                beam_center,
                save_path=data_dir / "profile_binning.png",
            )
        except Exception as exc:
            logger.warning("Could not save profile binning plot: %s", exc)

    # If profile_group_labels file exists but wasn't just computed, load into metadata
    if "profile_group_label" not in metadata and pgl_path.exists():
        metadata["profile_group_label"] = torch.load(
            pgl_path, weights_only=True
        )

    if "concentration_per_group" in needed:
        dl_args = cfg.get("data_loader", {}).get("args", {})
        D_dim = int(dl_args.get("D", dl_args.get("d", 1)))
        H_dim = int(dl_args.get("H", dl_args.get("h", 21)))
        W_dim = int(dl_args.get("W", dl_args.get("w", 21)))

        if "profile_group_label" in metadata:
            profile_group_labels = metadata["profile_group_label"]
            n_profile_bins = int(profile_group_labels.max().item()) + 1
            concentration = _fit_dirichlet_per_group(
                counts,
                masks,
                profile_group_labels,
                n_profile_bins,
                D=D_dim,
                H=H_dim,
                W=W_dim,
            )
            binning_desc = f"2D profile bins ({n_profile_bins} bins)"
        else:
            concentration = _fit_dirichlet_per_group(
                counts,
                masks,
                group_labels,
                n_bins,
                D=D_dim,
                H=H_dim,
                W=W_dim,
            )
            binning_desc = f"{n_bins} resolution bins"

        torch.save(concentration, needed["concentration_per_group"])
        logger.info(
            "Saved concentration_per_group.pt (MOM bg-sub, %dD: %dx%dx%d, %s)",
            2 if D_dim == 1 else 3,
            D_dim,
            H_dim,
            W_dim,
            binning_desc,
        )

    if "s_squared_per_group" in needed:
        s_squared = _compute_s_squared_per_group(d, group_labels, n_bins)
        torch.save(s_squared, needed["s_squared_per_group"])
        logger.info("Saved s_squared_per_group.pt")

    if "tau_per_group" in needed:
        tau_source = loss_args.get("tau_source", "dials")

        if tau_source == "crude":
            crude_I = _crude_intensity_from_cfg(counts, masks, cfg)
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
                crude_I = _crude_intensity_from_cfg(counts, masks, cfg)
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

    # ── Profile basis generation ──────────────────────────────────────
    # Two independent config sources can request a Hermite/PCA basis file:
    #   1. loss.args.profile_basis_per_bin — loss consumes per-bin priors
    #      (mu_per_group / std_per_group) from the file
    #   2. surrogates.qp.args.warmstart_basis_path — learned_basis_profile
    #      warm-starts its decoder from W and b in the file
    # Collect all unique paths requested, regenerate any stale/missing.
    dl_args = cfg.get("data_loader", {}).get("args", {})
    D_dim = int(dl_args.get("D", dl_args.get("d", 1)))
    H_dim = int(dl_args.get("H", dl_args.get("h", 21)))
    W_dim = int(dl_args.get("W", dl_args.get("w", 21)))

    basis_type = str(loss_args.get("profile_basis_type", "hermite"))
    basis_d = int(loss_args.get("profile_basis_d", 14))
    basis_max_order = int(loss_args.get("profile_basis_max_order", 4))
    basis_sigma_ref = float(loss_args.get("profile_basis_sigma_ref", 3.0))
    basis_sigma_z = float(loss_args.get("profile_basis_sigma_z", 1.0))

    prf_labels = metadata.get("profile_group_label", group_labels)
    n_prf_bins = int(prf_labels.max().item()) + 1

    expected_prov = {
        "basis_type": basis_type,
        "D": D_dim,
        "H": H_dim,
        "W": W_dim,
        "max_order": basis_max_order,
        "sigma_ref": basis_sigma_ref,
        "sigma_z": basis_sigma_z,
        "n_bins": n_prf_bins,
    }

    basis_paths: set[Path] = set()
    loss_basis = loss_args.get("profile_basis_per_bin")
    if isinstance(loss_basis, str):
        basis_paths.add(_nbins_path(loss_basis, n_bins, data_dir))
    qp_cfg = cfg.get("surrogates", {}).get("qp", {})
    warmstart = (qp_cfg.get("args", {}) or {}).get("warmstart_basis_path")
    if isinstance(warmstart, str):
        basis_paths.add(_nbins_path(warmstart, n_bins, data_dir))

    for basis_path in basis_paths:
        need_regen = force or not basis_path.exists()
        if not need_regen:
            reason = _basis_provenance_mismatch_reason(basis_path, expected_prov)
            if reason is not None:
                logger.warning(
                    "Regenerating %s because provenance mismatch: %s",
                    basis_path.name, reason,
                )
                need_regen = True

        if not need_regen:
            continue

        basis_data = _fit_profile_basis_per_bin(
            counts,
            masks,
            prf_labels,
            n_prf_bins,
            basis_type=basis_type,
            D=D_dim,
            H=H_dim,
            W=W_dim,
            d=basis_d,
            max_order=basis_max_order,
            sigma_ref=basis_sigma_ref,
            sigma_z=basis_sigma_z,
        )
        torch.save(basis_data, basis_path)
        logger.info(
            "Saved %s (type=%s, d=%d, %d bins, sigma_ref=%.2f, sigma_z=%.2f)",
            basis_path.name,
            basis_type,
            basis_data["d"],
            n_prf_bins,
            basis_sigma_ref,
            basis_sigma_z,
        )

    # ── Empirical profile basis (per-bin empirical bias) ──────────────
    emp_basis_filename = loss_args.get("empirical_profile_basis_per_bin")
    if isinstance(emp_basis_filename, str):
        emp_basis_path = _nbins_path(emp_basis_filename, n_bins, data_dir)
        if force or not emp_basis_path.exists():
            dl_args = cfg.get("data_loader", {}).get("args", {})
            D_dim = int(dl_args.get("D", dl_args.get("d", 1)))
            H_dim = int(dl_args.get("H", dl_args.get("h", 21)))
            W_dim = int(dl_args.get("W", dl_args.get("w", 21)))

            basis_max_order = int(loss_args.get("profile_basis_max_order", 4))
            basis_sigma_ref = float(
                loss_args.get("profile_basis_sigma_ref", 3.0)
            )
            smooth_sigma = float(loss_args.get("profile_smooth_sigma", 0.0))

            prf_labels = metadata.get("profile_group_label", group_labels)
            n_prf_bins = int(prf_labels.max().item()) + 1

            emp_basis_data = _fit_empirical_profile_basis(
                counts,
                masks,
                prf_labels,
                n_prf_bins,
                D=D_dim,
                H=H_dim,
                W=W_dim,
                max_order=basis_max_order,
                sigma_ref=basis_sigma_ref,
                smooth_sigma=smooth_sigma,
            )
            torch.save(emp_basis_data, emp_basis_path)
            logger.info(
                "Saved %s (empirical bias, d=%d, %d bins)",
                emp_basis_path.name,
                emp_basis_data["d"],
                n_prf_bins,
            )


def _fit_empirical_profile_basis(
    counts: Tensor,
    masks: Tensor,
    group_labels: Tensor,
    n_bins: int,
    D: int = 1,
    H: int = 21,
    W: int = 21,
    d: int = 14,
    max_order: int = 4,
    sigma_ref: float = 3.0,
    smooth_sigma: float = 0.0,
) -> dict:
    """Build a profile basis with per-bin empirical biases.

    Like `_fit_profile_basis_per_bin` but replaces the single symmetric
    Gaussian bias with per-bin empirical biases computed from the mean
    bg-subtracted profile in each bin.  This means z=0 reproduces the
    empirical average profile for each bin; the model only learns
    per-reflection corrections.

    Args:
        counts: (N, D*H*W) raw data.
        masks: (N, D*H*W) raw data.
        group_labels: (N,) bin assignment per reflection.
        n_bins: Number of bins.
        D: Shoebox depth (frames).
        H: Shoebox height.
        W: Shoebox width.
        d: Unused (Hermite uses max_order).
        max_order: Max Hermite polynomial order.
        sigma_ref: Reference Gaussian width in pixels.
        smooth_sigma: If > 0, apply Gaussian smoothing to each mean profile
            before taking log.  Reduces noise in bins with few reflections.

    Returns:
        Dict ready for torch.save as empirical_profile_basis_per_bin.pt.
    """
    signal = _bg_subtract_signal(counts, masks, D, H, W)

    # Build shared Hermite basis
    if D > 1:
        W_basis, b_ref, orders = _build_hermite_basis_3d(
            D, H, W, max_order, sigma_ref
        )
    else:
        W_basis, b_ref, orders = _build_hermite_basis_2d(
            H, W, max_order, sigma_ref
        )
    d_actual = W_basis.shape[1]
    K = signal.shape[1]
    logger.info(
        "Empirical profile basis: %dD, max_order=%d, sigma_ref=%.1f, d=%d",
        3 if D > 1 else 2,
        max_order,
        sigma_ref,
        d_actual,
    )

    # Compute per-bin empirical biases from mean profiles
    totals = signal.sum(dim=1, keepdim=True).clamp(min=1)
    proportions = signal / totals
    log_profiles = torch.log(proportions.clamp(min=1e-8))

    b_per_group = torch.zeros(n_bins, K)
    for k in range(n_bins):
        mask_k = group_labels == k
        props_k = proportions[mask_k]
        if props_k.shape[0] >= 1:
            mean_k = props_k.mean(dim=0)

            # Remove noise floor: signal occupies a small fraction of
            # the shoebox, so a high quantile of pixel values estimates
            # the background level well
            floor = mean_k.quantile(0.75)
            mean_k = (mean_k - floor).clamp(min=0)

            # Optional Gaussian smoothing in pixel space
            if smooth_sigma > 0:
                mean_3d = mean_k.reshape(D, H, W)
                mean_3d = _gaussian_smooth_3d(mean_3d, smooth_sigma)
                mean_k = mean_3d.reshape(-1)

            mean_k = mean_k / mean_k.sum().clamp(min=1e-10)
            b_per_group[k] = torch.log(mean_k.clamp(min=1e-8))
        else:
            b_per_group[k] = b_ref  # fallback to symmetric Gaussian
    b_per_group = b_per_group.float()

    # Project each reflection using its bin's empirical bias
    b_selected = b_per_group[group_labels]  # (N, K)
    centered = log_profiles - b_selected
    h_all = centered @ W_basis  # (N, d)

    # Per-bin latent statistics
    mu_per_group = torch.zeros(n_bins, d_actual)
    std_per_group = torch.ones(n_bins, d_actual)

    for k in range(n_bins):
        h_k = h_all[group_labels == k]
        if h_k.shape[0] >= 2:
            mu_per_group[k] = h_k.mean(dim=0)
            std_per_group[k] = h_k.std(dim=0).clamp(min=0.1)
        elif h_k.shape[0] == 1:
            mu_per_group[k] = h_k[0]
        logger.debug(
            "Bin %d: n=%d, |mu|=%.2f, mean(std)=%.2f",
            k,
            h_k.shape[0],
            mu_per_group[k].norm().item(),
            std_per_group[k].mean().item(),
        )

    sigma_prior = max(float(h_all.std().item()), 1.0)

    result = {
        "W": W_basis,
        "b_per_group": b_per_group,
        "d": d_actual,
        "mu_per_group": mu_per_group,
        "std_per_group": std_per_group,
        "sigma_prior": sigma_prior,
        "basis_type": "empirical_per_bin",
        "orders": orders,
    }
    return result


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


def _load_raw_data(
    data_dir: Path,
    cfg: dict,
) -> tuple[Tensor, Tensor, dict]:
    """Load counts, masks, metadata from data_dir using config paths."""
    sfn = cfg["data_loader"]["args"].get("shoebox_file_names", {})

    counts_name = sfn.get("counts", "counts.pt")
    masks_name = sfn.get("masks", "masks.pt")

    counts = torch.load(data_dir / counts_name, weights_only=True)
    masks = torch.load(data_dir / masks_name, weights_only=True)

    ref_path = _resolve_reference_path(data_dir, cfg)
    metadata = torch.load(ref_path, weights_only=False)

    return counts, masks, metadata


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


def _bin_2d_for_profiles(
    metadata: dict,
    res_labels: Tensor,
    n_res_bins: int,
    max_azi_bins: int,
    beam_center: tuple[float, float],
    min_per_bin: int = 200,
) -> tuple[Tensor, list[int], int]:
    """Create adaptive 2D group labels (resolution x azimuthal) for profiles.

    For each resolution shell, the number of azimuthal sectors is reduced
    from *max_azi_bins* until every sector has at least *min_per_bin*
    reflections.  Shells with sparse detector coverage (e.g. outer shells
    where the rectangular detector doesn't fill all azimuthal angles)
    automatically get fewer sectors.

    Args:
        metadata: Must contain `xyzcal.px.0` (x) and `xyzcal.px.1` (y).
        res_labels: Resolution bin per reflection, shape `(N,)`.
        n_res_bins: Number of resolution bins.
        max_azi_bins: Maximum number of azimuthal sectors to try per shell.
        beam_center: Beam center on the detector in pixels `(x, y)`.
        min_per_bin: Minimum reflections per 2D bin.  Azimuthal sectors are
            reduced per shell until this threshold is met.

    Returns:
        Tuple of (profile_group_labels, azi_bins_per_shell, n_profile_bins)
        where profile_group_labels is a flat 2D bin index per reflection
        of shape `(N,)`, azi_bins_per_shell is the number of azimuthal
        bins actually used in each resolution shell, and n_profile_bins is
        the total number of 2D bins.
    """
    x_det = metadata["xyzcal.px.0"]
    y_det = metadata["xyzcal.px.1"]

    dx = x_det - beam_center[0]
    dy = y_det - beam_center[1]
    phi = torch.atan2(dy, dx)  # [-pi, pi]

    profile_group_labels = torch.zeros_like(res_labels)
    azi_bins_per_shell: list[int] = []
    offset = 0

    for rb in range(n_res_bins):
        shell_mask = res_labels == rb
        phi_shell = phi[shell_mask]
        n_in_shell = int(shell_mask.sum().item())

        # Try max_azi_bins, reduce until all sectors meet min_per_bin
        n_azi = max_azi_bins
        while n_azi > 1:
            azi_edges = torch.linspace(-math.pi, math.pi, n_azi + 1)
            azi_lab = torch.searchsorted(azi_edges[1:-1], phi_shell).long()
            occ = torch.bincount(azi_lab, minlength=n_azi)
            if occ.min().item() >= min_per_bin:
                break
            n_azi -= 1

        # Recompute with final n_azi (handles n_azi==1 case too)
        if n_azi > 1:
            azi_edges = torch.linspace(-math.pi, math.pi, n_azi + 1)
            azi_lab = torch.searchsorted(azi_edges[1:-1], phi_shell).long()
        else:
            azi_lab = torch.zeros(n_in_shell, dtype=torch.long)

        profile_group_labels[shell_mask] = offset + azi_lab
        azi_bins_per_shell.append(n_azi)
        offset += n_azi

    n_profile_bins = offset

    # Log summary
    occupancy = torch.bincount(profile_group_labels, minlength=n_profile_bins)
    unique_azi = sorted(set(azi_bins_per_shell))
    logger.info(
        "Adaptive 2D profile binning: %d res shells, max %d azi sectors, "
        "%d total bins (occupancy min=%d max=%d)",
        n_res_bins,
        max_azi_bins,
        n_profile_bins,
        occupancy.min().item(),
        occupancy.max().item(),
    )
    logger.info(
        "  Azi bins per shell: %s (unique values: %s)",
        azi_bins_per_shell,
        unique_azi,
    )

    return profile_group_labels, azi_bins_per_shell, n_profile_bins


def _plot_profile_binning(
    metadata: dict,
    res_labels: Tensor,
    profile_group_labels: Tensor,
    n_res_bins: int,
    azi_bins_per_shell: list[int],
    max_azi_bins: int,
    beam_center: tuple[float, float],
    save_path: Path,
) -> None:
    """Save a diagnostic plot of the adaptive 2D profile binning.

    Left: reflections colored by profile bin (flat index).
    Right: azi bins per resolution shell (bar chart).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.info("matplotlib not available; skipping binning diagram")
        return

    x = metadata["xyzcal.px.0"].numpy()
    y = metadata["xyzcal.px.1"].numpy()
    dx = x - beam_center[0]
    dy = y - beam_center[1]
    r = np.sqrt(dx**2 + dy**2)
    r_max = float(r.max()) * 1.05

    n_profile_bins = sum(azi_bins_per_shell)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Left: color by profile group label ──
    ax = axes[0]
    ax.scatter(
        dx,
        dy,
        c=profile_group_labels.numpy(),
        cmap="nipy_spectral",
        s=0.3,
        alpha=0.4,
        rasterized=True,
    )
    # Draw finest sector lines for reference
    azi_edges = np.linspace(-np.pi, np.pi, max_azi_bins + 1)
    for edge in azi_edges:
        ax.plot(
            [0, r_max * np.cos(edge)],
            [0, r_max * np.sin(edge)],
            "k-",
            linewidth=0.3,
            alpha=0.3,
        )
    ax.plot(0, 0, "r+", markersize=10, markeredgewidth=2)
    ax.set_aspect("equal")
    ax.set_title(f"Profile bins ({n_profile_bins} total)")
    ax.set_xlabel("dx (px from beam center)")
    ax.set_ylabel("dy (px from beam center)")

    # ── Center: color by resolution bin ──
    ax = axes[1]
    sc = ax.scatter(
        dx,
        dy,
        c=res_labels.numpy(),
        cmap="viridis",
        s=0.3,
        alpha=0.4,
        rasterized=True,
    )
    for edge in azi_edges:
        ax.plot(
            [0, r_max * np.cos(edge)],
            [0, r_max * np.sin(edge)],
            "k-",
            linewidth=0.3,
            alpha=0.3,
        )
    ax.plot(0, 0, "r+", markersize=10, markeredgewidth=2)
    ax.set_aspect("equal")
    ax.set_title(f"Resolution bins ({n_res_bins} shells)")
    ax.set_xlabel("dx (px from beam center)")
    fig.colorbar(sc, ax=ax, label="resolution bin")

    # ── Right: bar chart of azi bins per shell ──
    ax = axes[2]
    shells = np.arange(n_res_bins)
    ax.bar(shells, azi_bins_per_shell, color="steelblue", edgecolor="white")
    ax.axhline(
        max_azi_bins,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"max={max_azi_bins}",
    )
    ax.set_xlabel("Resolution shell")
    ax.set_ylabel("Azimuthal bins")
    ax.set_title("Adaptive azi bins per shell")
    ax.legend()

    fig.suptitle(
        f"Adaptive profile binning: {n_res_bins} res, max {max_azi_bins} azi, "
        f"{n_profile_bins} total bins\n"
        f"beam center = ({beam_center[0]:.1f}, {beam_center[1]:.1f}) px, "
        f"min_per_bin threshold applied",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved profile binning diagram: %s", save_path)


def _compute_bg_rate_per_group(
    bg_per_refl: Tensor,
    group_labels: Tensor,
    n_bins: int,
) -> Tensor:
    """Exponential rate for background: lambda_k = 1 / mean(bg_k).

    Works with any per-reflection background estimate (DIALS
    `background.mean` or crude quietest-frame estimate).
    """
    bg_rate = torch.zeros(n_bins)
    for b in range(n_bins):
        sel = bg_per_refl[group_labels == b]
        if len(sel) > 0:
            bg_rate[b] = 1.0 / sel.mean().clamp(min=1e-6)
    return bg_rate


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
    counts: Tensor,
    masks: Tensor,
    metadata: dict,
    cfg: dict,
) -> Tensor:
    """Return intensity tensor for prior fitting, respecting tau_source config."""
    if tau_source == "crude":
        return _crude_intensity_from_cfg(counts, masks, cfg)

    intensity = metadata.get(
        "intensity.prf.value",
        metadata.get("intensity.sum.value"),
    )
    if intensity is not None:
        return intensity

    logger.warning(
        "No intensity column found and tau_source='dials'; "
        "falling back to crude bg-subtraction"
    )
    return _crude_intensity_from_cfg(counts, masks, cfg)


def _get_background_for_prior(
    tau_source: str,
    counts: Tensor,
    masks: Tensor,
    metadata: dict,
    cfg: dict,
) -> Tensor:
    """Return per-reflection background estimate, respecting tau_source config."""
    if tau_source == "crude":
        return _crude_background_from_cfg(counts, masks, cfg)

    bg_mean = metadata.get("background.mean")
    if bg_mean is not None:
        return bg_mean

    logger.warning(
        "No background.mean column found and tau_source='dials'; "
        "falling back to crude bg estimation"
    )
    return _crude_background_from_cfg(counts, masks, cfg)


def _crude_background_from_cfg(
    counts: Tensor, masks: Tensor, cfg: dict
) -> Tensor:
    """Read shoebox dimensions from config and compute crude background."""
    dl_args = cfg.get("data_loader", {}).get("args", {})
    n_frames = int(dl_args.get("d", 3))
    n_pixels = int(dl_args.get("h", 21)) * int(dl_args.get("w", 21))
    return _compute_crude_background(counts, masks, n_frames, n_pixels)


def _compute_crude_background(
    counts: Tensor,
    masks: Tensor,
    n_frames: int,
    n_pixels_per_frame: int,
) -> Tensor:
    """Estimate per-pixel background rate from the quietest frame.

    For each shoebox, the frame with the lowest total (masked) counts is
    treated as pure background.  Returns the mean counts per pixel in that
    frame, analogous to DIALS `background.mean`.

    Args:
        counts: Raw shoebox counts, shape `(N, n_frames * n_pixels_per_frame)`.
        masks: Valid-pixel masks, same shape as *counts*.
        n_frames: Number of frames per shoebox (typically 3).
        n_pixels_per_frame: Pixels per frame (e.g. 21*21 = 441).

    Returns:
        Per-pixel background rate per reflection, shape `(N,)`.
    """
    N = counts.shape[0]

    counts_clean = counts.float().clamp(min=0)
    masks_f = masks.float()

    counts_3d = (counts_clean * masks_f).reshape(
        N, n_frames, n_pixels_per_frame
    )
    masks_3d = masks_f.reshape(N, n_frames, n_pixels_per_frame)

    frame_counts = counts_3d.sum(dim=-1)  # (N, n_frames)
    frame_n_pixels = masks_3d.sum(dim=-1)  # (N, n_frames)

    min_frame_idx = frame_counts.argmin(dim=-1)  # (N,)
    bg_frame_counts = frame_counts.gather(
        1, min_frame_idx.unsqueeze(-1)
    ).squeeze(-1)
    bg_frame_n_pixels = frame_n_pixels.gather(
        1, min_frame_idx.unsqueeze(-1)
    ).squeeze(-1)

    bg_per_pixel = bg_frame_counts / bg_frame_n_pixels.clamp(min=1)

    logger.info(
        "Crude background: min=%.2f, median=%.2f, max=%.2f",
        bg_per_pixel.min().item(),
        bg_per_pixel.median().item(),
        bg_per_pixel.max().item(),
    )

    return bg_per_pixel


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


#  Profile basis construction (Hermite + PCA) with per-bin priors


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


def _build_pca_basis(
    signal: Tensor,
    d: int = 8,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """PCA basis from bg-subtracted, normalized profiles.

    Args:
        signal: (N, K) bg-subtracted signal (already clamped >= 0).
        d: Number of principal components.

    Returns:
        Tuple of (W, b, explained_var) where W is a (K, d) basis matrix,
        b is a (K,) mean of log-profiles (bias), and explained_var is a
        (d,) fraction of variance explained per component.
    """
    # Normalize to proportions
    totals = signal.sum(dim=1, keepdim=True).clamp(min=1)
    proportions = signal / totals

    # Log-transform
    log_profiles = torch.log(proportions.clamp(min=eps))

    # Center
    b = log_profiles.mean(dim=0)  # (K,)
    centered = log_profiles - b

    # SVD
    _U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    # Top d components
    d_actual = min(d, S.shape[0])
    W_basis = Vh[:d_actual].T  # (K, d_actual)

    total_var = (S**2).sum()
    explained_var = (S[:d_actual] ** 2) / total_var

    return W_basis.float(), b.float(), explained_var.float()


def _gaussian_smooth_3d(vol: Tensor, sigma: float) -> Tensor:
    """Apply Gaussian smoothing to a (D, H, W) volume (or (1, H, W) for 2D).

    Uses separable 1D convolutions per axis. Kernel is truncated at 3*sigma.
    """
    import torch.nn.functional as F_smooth  # local to avoid top-level clash

    ksize = max(int(math.ceil(sigma * 3)) * 2 + 1, 3)
    half = ksize // 2
    x = torch.arange(ksize, dtype=vol.dtype, device=vol.device) - half
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Work in 5-D: (1, 1, D, H, W)
    v = vol.unsqueeze(0).unsqueeze(0)
    # Smooth along W (dim=-1)
    kw = kernel_1d.reshape(1, 1, 1, 1, -1)
    v = F_smooth.pad(v, (half, half, 0, 0, 0, 0), mode="reflect")
    v = F_smooth.conv3d(v, kw)
    # Smooth along H (dim=-2)
    kh = kernel_1d.reshape(1, 1, 1, -1, 1)
    v = F_smooth.pad(v, (0, 0, half, half, 0, 0), mode="reflect")
    v = F_smooth.conv3d(v, kh)
    # Smooth along D (dim=-3) only if D > kernel size
    if vol.shape[0] > ksize:
        kd = kernel_1d.reshape(1, 1, -1, 1, 1)
        v = F_smooth.pad(v, (0, 0, 0, 0, half, half), mode="reflect")
        v = F_smooth.conv3d(v, kd)
    elif vol.shape[0] > 1:
        # D is small, use a 3-tap kernel
        kd_small = torch.tensor(
            [0.25, 0.5, 0.25], dtype=vol.dtype, device=vol.device
        )
        kd_small = kd_small.reshape(1, 1, -1, 1, 1)
        v = F_smooth.pad(v, (0, 0, 0, 0, 1, 1), mode="reflect")
        v = F_smooth.conv3d(v, kd_small)
    return v.squeeze(0).squeeze(0).clamp(min=0)


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


def _basis_provenance_mismatch_reason(
    basis_path: Path, expected: dict
) -> str | None:
    """Load the basis file's provenance dict and diff against expected.

    Returns None when the cached file's generation parameters match the
    caller's expectations. Returns a human-readable description of the
    first mismatch otherwise, so the caller can log WHY regeneration was
    triggered. Also treats "no provenance key" as a mismatch — old-format
    files get rewritten with the new metadata schema.
    """
    try:
        cached = torch.load(basis_path, weights_only=False, map_location="cpu")
    except Exception as err:
        return f"failed to load cached file for validation: {err}"
    if not isinstance(cached, dict):
        return "cached file is not a dict"
    prov = cached.get("provenance")
    if prov is None:
        return "cached file predates provenance schema; rewriting with metadata"
    # Compare all expected keys; float fields use an epsilon to tolerate
    # float32 round-trip differences.
    for key, want in expected.items():
        got = prov.get(key)
        if got is None:
            return f"missing provenance key {key!r}"
        if isinstance(want, float):
            if abs(float(got) - want) > 1e-6:
                return f"{key}: cached={got}, config={want}"
        elif got != want:
            return f"{key}: cached={got!r}, config={want!r}"
    return None


def _fit_profile_basis_per_bin(
    counts: Tensor,
    masks: Tensor,
    group_labels: Tensor,
    n_bins: int,
    basis_type: str = "hermite",
    D: int = 1,
    H: int = 21,
    W: int = 21,
    d: int = 14,
    max_order: int = 4,
    sigma_ref: float = 3.0,
    sigma_z: float = 1.0,
) -> dict:
    """Build a fixed profile basis and compute per-bin latent priors.

    Args:
        counts: (N, D*H*W) raw data.
        masks: (N, D*H*W) raw data.
        group_labels: (N,) bin assignment per reflection.
        n_bins: Number of bins.
        basis_type: "hermite" or "pca".
        D: Shoebox depth (frames).
        H: Shoebox height.
        W: Shoebox width.
        d: Latent dimensionality (PCA only; Hermite uses max_order).
        max_order: Max Hermite polynomial order (Hermite only).
        sigma_ref: Reference Gaussian width in pixels (Hermite only).
        sigma_z: Reference Gaussian width along z/frame axis (Hermite 3D).

    Returns:
        Dict ready for torch.save as profile_basis_per_bin.pt. Includes
        provenance keys (max_order, sigma_ref, sigma_z, D, H, W, basis_type,
        n_bins) that downstream loaders validate against cfg before
        re-using the cached file.
    """
    signal = _bg_subtract_signal(counts, masks, D, H, W)

    if basis_type == "hermite":
        if D > 1:
            W_basis, b, orders = _build_hermite_basis_3d(
                D, H, W, max_order, sigma_ref, sigma_z
            )
        else:
            W_basis, b, orders = _build_hermite_basis_2d(
                H, W, max_order, sigma_ref
            )
        d_actual = W_basis.shape[1]
        explained_var = None
        logger.info(
            "Hermite basis: %dD, max_order=%d, sigma_ref=%.2f, sigma_z=%.2f, d=%d",
            3 if D > 1 else 2,
            max_order,
            sigma_ref,
            sigma_z,
            d_actual,
        )
    elif basis_type == "pca":
        W_basis, b, explained_var = _build_pca_basis(signal, d)
        d_actual = W_basis.shape[1]
        orders = None
        logger.info(
            "PCA basis: d=%d, explained variance=%.3f",
            d_actual,
            explained_var.sum().item(),
        )
    else:
        raise ValueError(f"Unknown profile basis type: {basis_type!r}")

    # Project all reflections into latent space
    # For Hermite: project bg-subtracted signal (as log-proportions) onto basis
    totals = signal.sum(dim=1, keepdim=True).clamp(min=1)
    proportions = signal / totals
    log_profiles = torch.log(proportions.clamp(min=1e-8))
    centered = log_profiles - b  # (N, K)
    h_all = centered @ W_basis  # (N, d)

    # Per-bin mean and std of latent codes
    mu_per_group = torch.zeros(n_bins, d_actual)
    std_per_group = torch.ones(n_bins, d_actual)

    for k in range(n_bins):
        mask_k = group_labels == k
        h_k = h_all[mask_k]
        if h_k.shape[0] >= 2:
            mu_per_group[k] = h_k.mean(dim=0)
            std_per_group[k] = h_k.std(dim=0).clamp(min=0.1)
        elif h_k.shape[0] == 1:
            mu_per_group[k] = h_k[0]
        logger.debug(
            "Bin %d: n=%d, |mu|=%.2f, mean(std)=%.2f",
            k,
            h_k.shape[0],
            mu_per_group[k].norm().item(),
            std_per_group[k].mean().item(),
        )

    # Global sigma_prior = std of all latent codes (for fallback)
    sigma_prior = float(h_all.std().item())
    sigma_prior = max(sigma_prior, 1.0)  # floor at 1.0

    result = {
        "W": W_basis,
        "b": b,
        "d": d_actual,
        "mu_per_group": mu_per_group,
        "std_per_group": std_per_group,
        "sigma_prior": sigma_prior,
        "basis_type": f"{basis_type}_per_bin",
        # Provenance: parameters used to generate this file. Downstream
        # loaders compare these to the YAML cfg to decide whether the
        # cached file is still valid or needs to be regenerated.
        "provenance": {
            "basis_type": basis_type,
            "D": int(D),
            "H": int(H),
            "W": int(W),
            "max_order": int(max_order),
            "sigma_ref": float(sigma_ref),
            "sigma_z": float(sigma_z),
            "n_bins": int(n_bins),
            "d": int(d_actual),
        },
    }
    if orders is not None:
        result["orders"] = orders
    if explained_var is not None:
        result["explained_var"] = explained_var

    return result
