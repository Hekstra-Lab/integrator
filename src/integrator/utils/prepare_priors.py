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


def _prepare_global_priors_default_loss(
    cfg: dict,
    *,
    force: bool = False,
) -> None:
    """Generate global prior .pt files for the default Loss if needed.

    Renamed from `prepare_global_priors` because a second function of that name
    at the bottom of this module shadowed it, so this body was unreachable. The
    two serve different losses -- this one gates on `loss.name == "default"`
    (`loss.py`, which is not in the REGISTRY and so cannot be selected by any
    config) while the live one gates on `"global_prior"`. Kept rather than
    deleted pending a decision on `loss.py` itself.


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


# the loader already lives with the data modules; no need for a second copy
from integrator.data_loaders.data_module import _load_shoebox_array

# --- raw-count helpers ----------------------------------------------------
# Ported with the scaling merge alongside the Hermite builders: the prior
# generators read counts/masks directly rather than trusting DIALS.

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


# --- Hermite basis construction -------------------------------------------
# Ported with the scaling merge: prepare_profile_basis calls these to write
# the peaked warm-start basis, without which the profile decoder starts near
# uniform and the intensity absorbs the background on Laue shoeboxes.

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
    """3D Hermite function basis (frame x spatial) with half-Gaussian envelope.

    Spatial directions (x, y) use `max_order`. The frame direction (z) uses
    `max_order_z` if provided; otherwise defaults to `min(1, D-1)` (linear
    tilt only) which is sufficient when D=3 — the (0,0,0) mode is absorbed
    by `b`, `nz=0` gives the symmetric rocking peak via the half-Gaussian
    envelope, and `nz=1` captures asymmetry. For D>3, pass a larger
    `max_order_z` to resolve higher-order z structure.

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
    max_order_z = (
        int(max_order_z_arg) if max_order_z_arg is not None else None
    )

    # The shoebox shape falls back to the dataset's own geometry, not to a
    # constant. This runs before resolve_config, so `data_loader.args` is still
    # raw YAML and usually carries no h/w; the previous default of 21 then
    # silently built a 441-component basis for a 625-pixel dataset, which the
    # surrogate's shape check rejects at construction.
    dl_args = cfg.get("data_loader", {}).get("args", {})
    geom = {}
    if dl_args.get("data_dir"):
        from integrator.io import read_dataset_spec

        geom = (
            read_dataset_spec(dl_args["data_dir"]) or {}
        ).get("geometry") or {}

    def _dim(*keys, default):
        for src in (dl_args, geom):
            for k in keys:
                if src.get(k) is not None:
                    return int(src[k])
        return default

    D_dim = _dim("D", "d", default=1)
    H_dim = _dim("H", "h", default=21)
    W_dim = _dim("W", "w", default=21)

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


def prepare_bg_prior(
    cfg: dict,
    *,
    force: bool = False,
    n_sample: int = 50000,
    events_out: list[dict] | None = None,
) -> None:
    """Auto-generate `bg_prior.pt` (global background Gamma prior) if missing.

    The Wilson losses default the background prior to mean 1.0 count/pixel,
    which is right for monochromatic data but far too low for the high Laue
    background -- the background then cannot claim the flat field and the
    intensity x (uniform/skirted) profile absorbs it, so `scale*I_h` collapses
    to ~constant (see scripts/diagnose_merge_collapse.py). This fits a Gamma to
    the empirical per-pixel background (robust per-shoebox median) and writes
    `bg_prior.pt`, which the loss factory auto-injects as `bg_rate` /
    `bg_concentration` when they are not set explicitly.

    Skipped when the loss is not a Wilson loss, when the config sets `bg_rate`
    or `bg_concentration` explicitly (those win), or when `bg_prior.pt` already
    exists (unless `force`). Idempotent.

    Args:
        cfg: Full YAML config dict.
        force: Regenerate even if `bg_prior.pt` already exists.
        n_sample: Reflections subsampled for the estimate; the global mean/var
            is stable well below the full set.
        events_out: Optional list appended with a structured file-action event.
    """
    loss_name = cfg.get("loss", {}).get("name", "")
    if loss_name not in ("monochromatic_wilson", "polychromatic_wilson"):
        return
    loss_args = cfg.get("loss", {}).get("args", {}) or {}
    if "bg_rate" in loss_args or "bg_concentration" in loss_args:
        return  # explicit config wins; nothing to generate

    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])
    bg_path = data_dir / "bg_prior.pt"
    # NOTE: existence is checked AFTER the empirical mean is computed below, so a
    # stale bg_prior.pt whose mean disagrees with the data is regenerated rather
    # than silently reused (a wrong bg prior pulls the background to the wrong
    # level and collapses the merge).

    counts, masks = _try_load_counts_masks(data_dir, cfg)
    if counts is None or masks is None:
        return  # no counts/masks on disk; leave the loss default

    counts = counts.reshape(counts.shape[0], -1).float()
    masks = masks.reshape(masks.shape[0], -1).float()
    n = counts.shape[0]
    if n > n_sample:
        idx = torch.randperm(n)[:n_sample]
        counts, masks = counts[idx], masks[idx]

    # Robust per-shoebox background = median over VALID pixels (the small Bragg
    # peak does not move the median in a mostly-background box). Masked-out
    # pixels are pushed to +inf so they sort to the end and the median index
    # (n_valid-1)//2 lands among the valid entries only.
    valid = masks > 0
    filled = torch.where(
        valid, counts.clamp(min=0), torch.full_like(counts, float("inf"))
    )
    n_valid = valid.sum(-1).clamp(min=1).long()
    sorted_c, _ = filled.sort(dim=-1)
    med_idx = ((n_valid - 1) // 2).unsqueeze(1)
    bg_box = sorted_c.gather(1, med_idx).squeeze(1)
    bg_box = bg_box[torch.isfinite(bg_box) & (bg_box > 0)]
    if bg_box.numel() < 100:
        logger.warning(
            "bg_prior: only %d valid shoeboxes; leaving the loss default",
            bg_box.numel(),
        )
        return

    mean_bg = float(bg_box.mean())
    var_bg = float(bg_box.var().clamp(min=1e-6))
    bg_rate = 1.0 / max(mean_bg, 1e-6)
    # Method-of-moments Gamma shape, clamped so the prior is informative but not
    # rigid (it must pull the background UP without freezing per-reflection
    # variation; concentration<=1 would be exponential and prefer bg->0).
    bg_concentration = float(min(max(mean_bg**2 / var_bg, 1.5), 50.0))

    # Self-heal: reuse an existing bg_prior.pt ONLY if its prior mean (1/bg_rate)
    # agrees with the empirical background to within 2x; otherwise it is stale or
    # was computed on different data and would silently mis-anchor the background.
    if bg_path.exists() and not force:
        try:
            saved = torch.load(
                bg_path, weights_only=False, map_location="cpu"
            )
            saved_mean = 1.0 / max(float(saved.get("bg_rate", 1.0)), 1e-12)
        except Exception:
            saved_mean = None
        if saved_mean is not None and 0.5 <= saved_mean / mean_bg <= 2.0:
            logger.info(
                "Reusing %s (prior mean %.2f/px agrees with empirical %.2f/px)",
                bg_path.name,
                saved_mean,
                mean_bg,
            )
            if events_out is not None:
                events_out.append(
                    {"file": bg_path.name, "action": "reused", "path": str(bg_path)}
                )
            return
        logger.warning(
            "%s has prior mean %s/px but the data's empirical background is "
            "%.2f/px -- regenerating (stale or computed on other data).",
            bg_path.name,
            f"{saved_mean:.2f}" if saved_mean is not None else "unreadable",
            mean_bg,
        )

    torch.save(
        {"bg_rate": bg_rate, "bg_concentration": bg_concentration}, bg_path
    )
    logger.info(
        "Saved %s (empirical per-pixel background mean=%.2f -> bg_rate=%.4f, "
        "bg_concentration=%.2f; prior mean=%.2f counts/pixel)",
        bg_path.name,
        mean_bg,
        bg_rate,
        bg_concentration,
        1.0 / bg_rate,
    )
    if events_out is not None:
        events_out.append(
            {
                "file": bg_path.name,
                "action": "created",
                "path": str(bg_path),
                "reason": f"empirical background mean={mean_bg:.2f}/pixel",
                "bg_rate": bg_rate,
                "bg_concentration": bg_concentration,
            }
        )


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
            # Says what actually happens, not what it used to claim. No
            # per-bin prior is written here, so whatever bg_prior.pt already
            # holds -- normally the border-pixel measurement from
            # prepare_bg_prior -- stays in force. The old wording asserted a
            # fallback to bg_rate=1.0, which is the documented cause of the
            # poly collapse, so a benign log line read as the failure itself.
            logger.info(
                "No background column in the metadata (%s), so no per-bin "
                "background prior was fitted; the existing bg_prior stands. "
                "Laue shoeboxes from mksbox carry no background column -- "
                "prepare_bg_prior measures it from border pixels instead.",
                "background.mean/background.sum.value",
            )


def prepare_global_priors(cfg: dict, *, force: bool = False) -> None:
    """Fit the global Gamma intensity/background priors for the global-prior loss.

    Unlike the Wilson prior, the global-prior baseline regularizes `q(I)` and
    `q(bg)` toward single resolution-independent `Gamma` distributions. This fits
    each by Gamma MLE on the dataset (`intensity.sum.value` and `background.mean`)
    and injects the `(concentration, rate)` pairs into the loss args so they are
    recorded in the resolved config. A single-bin `group_labels` file is also
    written so the data module always exposes `group_label` (the integrator reads
    it unconditionally, even though the global prior does not bin by resolution).

    Args:
        cfg: Full YAML config dict (mutated in place).
        force: Refit and overwrite even when prior args are already present.
    """
    if cfg.get("loss", {}).get("name") != "global_prior":
        return

    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])
    loss_args = cfg["loss"].setdefault("args", {})

    ref_path = _resolve_reference_path(data_dir, cfg)
    if data_path(ref_path) is None:
        logger.warning(
            "global_prior: no metadata at %s; using default priors", ref_path
        )
        return
    metadata = load_data(ref_path)

    # Single-bin group labels so metadata always carries `group_label`.
    gl_path = _nbins_path("group_labels.npy", 1, data_dir)
    if force or data_path(gl_path) is None:
        n = len(metadata["d"])
        p = save_data(torch.zeros(n, dtype=torch.long), gl_path)
        logger.info("Saved %s (single-bin labels for global prior)", p.name)

    # Global intensity Gamma prior.
    if force or (
        "i_concentration" not in loss_args and "i_rate" not in loss_args
    ):
        i_vals = metadata.get(
            "intensity.sum.value", metadata.get("intensity.prf.value")
        )
        a, r = _fit_global_gamma(i_vals, "intensity")
        if a is not None:
            loss_args.setdefault("i_concentration", a)
            loss_args.setdefault("i_rate", r)
            logger.info(
                "Global intensity Gamma MLE: concentration=%.4g rate=%.4g",
                a,
                r,
            )

    # Global background Gamma prior.
    if force or (
        "bg_concentration" not in loss_args and "bg_rate" not in loss_args
    ):
        bg_vals = metadata.get(
            "background.mean", metadata.get("background.sum.value")
        )
        a, r = _fit_global_gamma(bg_vals, "background")
        if a is not None:
            loss_args.setdefault("bg_concentration", a)
            loss_args.setdefault("bg_rate", r)
            logger.info(
                "Global background Gamma MLE: concentration=%.4g rate=%.4g",
                a,
                r,
            )


def _fit_global_gamma(
    vals: Tensor | None,
    label: str,
    min_samples: int = 10,
) -> tuple[float | None, float | None]:
    """Gamma MLE on the strictly-positive entries of `vals`."""
    if vals is None:
        logger.warning("global_prior: no %s column for MLE", label)
        return None, None
    pos = vals[vals > 0]
    if pos.numel() < min_samples:
        logger.warning(
            "global_prior: too few positive %s values (%d) for MLE",
            label,
            int(pos.numel()),
        )
        return None, None
    a, r = _fit_gamma_mle(pos.float())
    return float(a), float(r)


def inject_binning_labels(data_loader, cfg: dict) -> None:
    """Load binning label files and inject into the dataset's metadata."""

    loss_args = cfg.get("loss", {}).get("args", {})
    n_bins = int(loss_args.get("n_bins", 1))
    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])

    # The ragged data module has no `full_dataset.reference`; it carries its own
    # resolution-bin labels (group_labels_*.pt) inside the dataset, so there is
    # nothing to inject here.
    full_dataset = getattr(data_loader, "full_dataset", None)
    ref = getattr(full_dataset, "reference", None)
    if ref is None:
        return

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
    from ..io.metadata import load_metadata

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


def _compute_bg_rate_per_group(
    bg_mean: Tensor,
    group_labels: Tensor,
    n_bins: int,
    min_bg: float = 1e-6,
) -> Tensor:
    """Gamma rate for the background: `bg_rate_k = 1 / mean(bg_k)`.

    The background prior is built as `Gamma(alpha, alpha * bg_rate)`, whose mean
    is `1 / bg_rate`, so this returns the RECIPROCAL of the per-bin mean
    background rather than the mean itself -- the same convention as
    `_compute_tau_per_group` for the intensity. Writing the mean here instead
    would invert the prior and, in the Laue regime where shoeboxes are ~97%
    background, dominate the ELBO.
    """
    rate = torch.zeros(n_bins)
    for b in range(n_bins):
        sel = bg_mean[group_labels == b]
        sel = sel[sel > min_bg]
        rate[b] = 1.0 / sel.mean() if len(sel) > 0 else 1.0
    return rate


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
