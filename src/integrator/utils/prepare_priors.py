"""Auto-generate per-bin prior buffers for hierarchical loss functions.

Called automatically during training when the loss requires per-bin priors
(PerBinLoss, WilsonPerBinLoss) and the .pt files don't yet exist in the
data directory.

This avoids requiring users to run a separate preprocessing script.
"""

import logging
import math
from pathlib import Path

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


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

    Parameters
    ----------
    cfg : dict
        Full YAML config dict.  If ``loss.args.n_bins`` is set in the
        config, it is used as the number of resolution bins (unless
        overridden by the *n_bins* argument).
    n_bins : int
        Number of resolution bins.  When <= 0 (default), reads from
        ``cfg["loss"]["args"]["n_bins"]``, falling back to 20.
    min_intensity : float
        Minimum intensity for tau estimation.
    force : bool
        Regenerate even if files already exist.
    """
    loss_name = cfg.get("loss", {}).get("name", "")
    if loss_name not in ("per_bin", "wilson_per_bin"):
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
        loss_args["bg_concentration_per_group"] = "bg_concentration_per_group.pt"

    # Determine which files are referenced and which are missing
    per_bin_keys = [
        "bg_concentration_per_group",
        "bg_rate_per_group",
        "concentration_per_group",
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
            path = (
                Path(filename)
                if Path(filename).is_absolute()
                else data_dir / filename
            )
            if force or not path.exists():
                needed[key] = path

    # Check group_label consistency with n_bins even if all files exist
    rebinned = False
    ref_path = _resolve_reference_path(data_dir, cfg)
    metadata_on_disk = torch.load(ref_path, weights_only=False)
    if "group_label" in metadata_on_disk:
        existing_n_bins = int(metadata_on_disk["group_label"].max().item()) + 1
        if existing_n_bins != n_bins:
            logger.warning(
                "group_label has %d bins but config specifies n_bins=%d; "
                "re-binning and regenerating all per-bin files",
                existing_n_bins,
                n_bins,
            )
            # Force regeneration of ALL referenced per-bin files
            for key in per_bin_keys:
                if key not in loss_args:
                    continue
                filename = loss_args[key]
                if isinstance(filename, str):
                    path = (
                        Path(filename)
                        if Path(filename).is_absolute()
                        else data_dir / filename
                    )
                    needed[key] = path
            rebinned = True

    # Check profile_group_label consistency with 2D binning config
    profile_binning = loss_args.get("profile_binning")
    if profile_binning is not None:
        force_conc = False
        if "profile_group_label" not in metadata_on_disk:
            # 2D binning requested but profile_group_label missing
            force_conc = True
        elif rebinned:
            # Resolution bins changed — profile bins must be regenerated too
            force_conc = True

        if force_conc and "concentration_per_group" in loss_args:
            fn = loss_args["concentration_per_group"]
            if isinstance(fn, str):
                needed["concentration_per_group"] = (
                    Path(fn) if Path(fn).is_absolute() else data_dir / fn
                )

    if not needed:
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

    # Add or update group_label in metadata
    if "group_label" not in metadata or rebinned:
        metadata["group_label"] = group_labels
        meta_path = _resolve_reference_path(data_dir, cfg)
        torch.save(metadata, meta_path)
        if rebinned:
            logger.info("Updated 'group_label' in %s to %d bins", meta_path.name, n_bins)
        else:
            logger.info("Added 'group_label' to %s", meta_path.name)

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

    if "concentration_per_group" in needed:
        dl_args = cfg.get("data_loader", {}).get("args", {})
        D_dim = int(dl_args.get("D", dl_args.get("d", 1)))
        H_dim = int(dl_args.get("H", dl_args.get("h", 21)))
        W_dim = int(dl_args.get("W", dl_args.get("w", 21)))

        # Check for 2D profile binning config
        profile_binning = loss_args.get("profile_binning")
        if profile_binning is not None:
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
                    metadata, group_labels, n_bins, max_azi_bins,
                    beam_center, min_per_bin=min_per_bin,
                )
            )

            # Save profile_group_label in metadata
            metadata["profile_group_label"] = profile_group_labels
            meta_path = _resolve_reference_path(data_dir, cfg)
            torch.save(metadata, meta_path)
            logger.info(
                "Added 'profile_group_label' to %s (%d 2D bins)",
                meta_path.name,
                n_profile_bins,
            )

            # Save diagnostic plot
            _plot_profile_binning(
                metadata, group_labels, profile_group_labels,
                n_bins, azi_per_shell, max_azi_bins, beam_center,
                save_path=data_dir / "profile_binning.png",
            )

            concentration = _fit_dirichlet_per_group(
                counts, masks, profile_group_labels, n_profile_bins,
                D=D_dim, H=H_dim, W=W_dim,
            )
            binning_desc = (
                f"adaptive 2D {n_bins}res x max{max_azi_bins}azi = "
                f"{n_profile_bins} bins"
            )
        else:
            concentration = _fit_dirichlet_per_group(
                counts, masks, group_labels, n_bins,
                D=D_dim, H=H_dim, W=W_dim,
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

    Parameters
    ----------
    metadata : dict
        Must contain ``xyzcal.px.0`` (x) and ``xyzcal.px.1`` (y).
    res_labels : Tensor
        Resolution bin per reflection, shape ``(N,)``.
    n_res_bins : int
        Number of resolution bins.
    max_azi_bins : int
        Maximum number of azimuthal sectors to try per shell.
    beam_center : tuple[float, float]
        Beam center on the detector in pixels ``(x, y)``.
    min_per_bin : int
        Minimum reflections per 2D bin.  Azimuthal sectors are reduced
        per shell until this threshold is met.

    Returns
    -------
    profile_group_labels : Tensor
        Flat 2D bin index per reflection, shape ``(N,)``.
    azi_bins_per_shell : list[int]
        Number of azimuthal bins actually used in each resolution shell.
    n_profile_bins : int
        Total number of 2D bins (sum of azi_bins_per_shell).
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
        dx, dy,
        c=profile_group_labels.numpy(),
        cmap="nipy_spectral", s=0.3, alpha=0.4, rasterized=True,
    )
    # Draw finest sector lines for reference
    azi_edges = np.linspace(-np.pi, np.pi, max_azi_bins + 1)
    for edge in azi_edges:
        ax.plot(
            [0, r_max * np.cos(edge)],
            [0, r_max * np.sin(edge)],
            "k-", linewidth=0.3, alpha=0.3,
        )
    ax.plot(0, 0, "r+", markersize=10, markeredgewidth=2)
    ax.set_aspect("equal")
    ax.set_title(f"Profile bins ({n_profile_bins} total)")
    ax.set_xlabel("dx (px from beam center)")
    ax.set_ylabel("dy (px from beam center)")

    # ── Center: color by resolution bin ──
    ax = axes[1]
    sc = ax.scatter(
        dx, dy,
        c=res_labels.numpy(),
        cmap="viridis", s=0.3, alpha=0.4, rasterized=True,
    )
    for edge in azi_edges:
        ax.plot(
            [0, r_max * np.cos(edge)],
            [0, r_max * np.sin(edge)],
            "k-", linewidth=0.3, alpha=0.3,
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
    ax.axhline(max_azi_bins, color="red", linestyle="--", linewidth=1, label=f"max={max_azi_bins}")
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
    ``background.mean`` or crude quietest-frame estimate).
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

    Parameters
    ----------
    counts : Tensor
        Raw shoebox counts, shape ``(N, n_frames * n_pixels_per_frame)``.
    masks : Tensor
        Valid-pixel masks, same shape as *counts*.
    n_frames : int
        Number of frames per shoebox (typically 3).
    n_pixels_per_frame : int
        Pixels per frame (e.g. 21*21 = 441).

    Returns
    -------
    Tensor
        Crude intensity estimate per reflection, shape ``(N,)``.
        Can be negative for weak / noise-dominated reflections.
    """
    N = counts.shape[0]

    # Clamp dead pixels (count == -1) to 0
    counts_clean = counts.float().clamp(min=0)
    masks_f = masks.float()

    # Reshape to (N, n_frames, n_pixels_per_frame)
    counts_3d = (counts_clean * masks_f).reshape(N, n_frames, n_pixels_per_frame)
    masks_3d = masks_f.reshape(N, n_frames, n_pixels_per_frame)

    # Total masked counts per frame: (N, n_frames)
    frame_counts = counts_3d.sum(dim=-1)

    # Number of valid pixels per frame: (N, n_frames)
    frame_n_pixels = masks_3d.sum(dim=-1)

    # Quietest frame = frame with minimum total counts
    min_frame_idx = frame_counts.argmin(dim=-1)  # (N,)
    bg_frame_counts = frame_counts.gather(1, min_frame_idx.unsqueeze(-1)).squeeze(-1)
    bg_frame_n_pixels = frame_n_pixels.gather(1, min_frame_idx.unsqueeze(-1)).squeeze(-1)

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
         ``D*H*W`` pixels.
      3. Estimate Dirichlet precision kappa via method of moments.
      4. Return ``concentration = kappa * p_bar``.

    Handles both 2D (``D=1``, stills/Laue) and 3D (``D>1``, rotation) cases.

    Parameters
    ----------
    counts : Tensor
        Raw shoebox counts, shape ``(N, D*H*W)``.
    masks : Tensor
        Valid-pixel masks, same shape as *counts*.
    group_labels : Tensor
        Bin assignment per reflection, shape ``(N,)``.
    n_bins : int
        Number of resolution bins.
    D, H, W : int
        Shoebox dimensions (frames, height, width).

    Returns
    -------
    Tensor
        Dirichlet concentration, shape ``(n_bins, D*H*W)``.
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
            ratio = (
                (p_bar[valid] * (1 - p_bar[valid]))
                / var_p[valid].clamp(min=1e-12)
                - 1
            )
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
    frame — analogous to DIALS ``background.mean``.

    Parameters
    ----------
    counts : Tensor
        Raw shoebox counts, shape ``(N, n_frames * n_pixels_per_frame)``.
    masks : Tensor
        Valid-pixel masks, same shape as *counts*.
    n_frames : int
        Number of frames per shoebox (typically 3).
    n_pixels_per_frame : int
        Pixels per frame (e.g. 21*21 = 441).

    Returns
    -------
    Tensor
        Per-pixel background rate per reflection, shape ``(N,)``.
    """
    N = counts.shape[0]

    counts_clean = counts.float().clamp(min=0)
    masks_f = masks.float()

    counts_3d = (counts_clean * masks_f).reshape(N, n_frames, n_pixels_per_frame)
    masks_3d = masks_f.reshape(N, n_frames, n_pixels_per_frame)

    frame_counts = counts_3d.sum(dim=-1)  # (N, n_frames)
    frame_n_pixels = masks_3d.sum(dim=-1)  # (N, n_frames)

    min_frame_idx = frame_counts.argmin(dim=-1)  # (N,)
    bg_frame_counts = frame_counts.gather(1, min_frame_idx.unsqueeze(-1)).squeeze(-1)
    bg_frame_n_pixels = frame_n_pixels.gather(1, min_frame_idx.unsqueeze(-1)).squeeze(-1)

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

    Parameters
    ----------
    x : Tensor
        Positive-valued samples (1-D).
    n_iter : int
        Newton iterations.

    Returns
    -------
    alpha, beta : Tensor, Tensor
        MLE shape (concentration) and rate parameters.
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

    Parameters
    ----------
    intensity : Tensor
        Per-reflection intensity estimates, shape ``(N,)``.
    group_labels : Tensor
        Bin assignment per reflection, shape ``(N,)``.
    n_bins : int
        Number of resolution bins.
    min_intensity : float
        Minimum intensity threshold (exclude weak/negative reflections).

    Returns
    -------
    Tensor
        Alpha (concentration) per bin, shape ``(n_bins,)``.
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
                b, len(sel), min_intensity,
            )
            continue
        alpha, _beta = _fit_gamma_mle(sel)
        alpha_per_group[b] = alpha.clamp(min=0.1)
        logger.debug("Bin %d: Gamma MLE alpha=%.3f (n=%d)", b, alpha.item(), len(sel))
    return alpha_per_group
