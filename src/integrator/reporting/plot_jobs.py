"""Render every figure family from the dumps in a `figures/` directory.

The training callbacks and the post-hoc checkpoint replay both call these
functions, so a run's figures look the same regardless of which route
produced the underlying dumps.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .figure_data import load_basis, load_latents, load_tracked
from .figure_style import save_animation, save_figure

logger = logging.getLogger(__name__)

DEFAULT_FORMATS = ("png", "pdf")


def _resolve(fig_dir, out_dir) -> tuple[Path, Path]:
    fig_dir = Path(fig_dir)
    out_dir = Path(out_dir) if out_dir is not None else fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, out_dir


def render_tracked(
    fig_dir,
    out_dir=None,
    animate: bool = True,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    n_epochs: int = 6,
) -> list[Path]:
    """Trajectories, uncertainty, filmstrips, and the training movie."""
    from .figures_tracked import (
        animate_tracked,
        plot_tracked_filmstrip,
        plot_tracked_trajectories,
        plot_tracked_uncertainty,
    )

    fig_dir, out_dir = _resolve(fig_dir, out_dir)
    selection, scalars, arrays = load_tracked(fig_dir)
    written: list[Path] = []

    written += save_figure(
        plot_tracked_trajectories(scalars),
        out_dir,
        "fig_tracked_trajectories",
        formats,
    )
    written += save_figure(
        plot_tracked_uncertainty(scalars),
        out_dir,
        "fig_tracked_uncertainty",
        formats,
    )
    for field in ("rate", "profile", "residual"):
        written += save_figure(
            plot_tracked_filmstrip(
                selection, scalars, arrays, field=field, n_epochs=n_epochs
            ),
            out_dir,
            f"fig_tracked_filmstrip_{field}",
            formats,
        )
    if animate and len(arrays["epochs"]) > 1:
        fig, anim = animate_tracked(selection, arrays)
        written += save_animation(anim, out_dir, "anim_tracked", fps=3)
        import matplotlib.pyplot as plt

        plt.close(fig)
    return written


def render_basis(
    fig_dir,
    out_dir=None,
    animate: bool = True,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    n_epochs: int = 6,
) -> list[Path]:
    """Basis atlas, evolution filmstrip, convergence panel, and movie."""
    from .figures_basis import (
        animate_basis,
        plot_basis_atlas,
        plot_basis_convergence,
        plot_basis_filmstrip,
    )

    fig_dir, out_dir = _resolve(fig_dir, out_dir)
    snapshots, diagnostics = load_basis(fig_dir)
    shape = tuple(int(s) for s in snapshots["shape"])
    weight = snapshots["weights"][-1]
    bias = snapshots["biases"][-1]
    written: list[Path] = []

    for mode in ("weight", "effect"):
        written += save_figure(
            plot_basis_atlas(weight, bias, shape, mode=mode),
            out_dir,
            f"fig_basis_atlas_{mode}",
            formats,
        )
    if len(snapshots["epochs"]) > 1:
        written += save_figure(
            plot_basis_filmstrip(snapshots, n_epochs=n_epochs),
            out_dir,
            "fig_basis_filmstrip",
            formats,
        )
        written += save_figure(
            plot_basis_convergence(diagnostics),
            out_dir,
            "fig_basis_convergence",
            formats,
        )
        if animate:
            fig, anim = animate_basis(snapshots)
            written += save_animation(anim, out_dir, "anim_basis", fps=3)
            import matplotlib.pyplot as plt

            plt.close(fig)
    return written


def render_latent(
    fig_dir,
    out_dir=None,
    weight=None,
    bias=None,
    shape=None,
    n_clusters: int = 6,
    epoch: int | None = None,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> list[Path]:
    """PCA, clustering, covariate R², detector map, and latent usage."""
    from .figures_latent import (
        plot_detector_map,
        plot_latent_clusters,
        plot_latent_covariate_r2,
        plot_latent_pca,
        plot_latent_usage,
    )

    fig_dir, out_dir = _resolve(fig_dir, out_dir)
    frames = load_latents(fig_dir)
    if not frames:
        logger.warning("no latent dumps in %s", fig_dir)
        return []
    epoch = max(frames) if epoch is None else int(epoch)
    frame = frames[epoch]

    if weight is None and (fig_dir / "basis_snapshots.npz").exists():
        snapshots, _ = load_basis(fig_dir)
        weight = snapshots["weights"][-1]
        bias = snapshots["biases"][-1]
        shape = tuple(int(s) for s in snapshots["shape"])

    written: list[Path] = []
    written += save_figure(
        plot_latent_pca(frame), out_dir, "fig_latent_pca", formats
    )
    fig, labels = plot_latent_clusters(
        frame, weight=weight, bias=bias, shape=shape, k=n_clusters
    )
    written += save_figure(fig, out_dir, "fig_latent_clusters", formats)
    try:
        written += save_figure(
            plot_latent_covariate_r2(frame),
            out_dir,
            "fig_latent_covariate_r2",
            formats,
        )
    except ValueError as exc:
        logger.warning("covariate panel skipped: %s", exc)
    try:
        written += save_figure(
            plot_detector_map(frame, labels),
            out_dir,
            "fig_latent_detector_map",
            formats,
        )
    except ValueError as exc:
        logger.warning("detector map skipped: %s", exc)
    written += save_figure(
        plot_latent_usage(frame), out_dir, "fig_latent_usage", formats
    )
    return written


def render_explain(
    fig_dir,
    out_dir=None,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> list[Path]:
    """The audience-facing figures, built from the tracked shoeboxes."""
    from .figures_explain import (
        plot_decomposition,
        plot_posterior_gallery,
        plot_regime_shrinkage,
    )

    fig_dir, out_dir = _resolve(fig_dir, out_dir)
    selection, scalars, arrays = load_tracked(fig_dir)
    shape = tuple(int(s) for s in arrays["shape"])
    epochs = [int(e) for e in arrays["epochs"]]
    last = len(epochs) - 1
    written: list[Path] = []

    strong = [
        i for i, r in enumerate(selection["regime"]) if r == "strong"
    ] or [0]
    slot = strong[len(strong) // 2]
    row = scalars.filter(
        (scalars["epoch"] == epochs[last]) & (scalars["slot"] == slot)
    )
    written += save_figure(
        plot_decomposition(
            arrays["counts"][slot],
            arrays["rates"][last][slot],
            arrays["profiles"][last][slot],
            float(row["qi_mean"][0]),
            float(row["qbg_mean"][0]),
            shape,
        ),
        out_dir,
        "fig_explain_decomposition",
        formats,
    )
    written += save_figure(
        plot_posterior_gallery(scalars, arrays),
        out_dir,
        "fig_explain_posterior_gallery",
        formats,
    )
    written += save_figure(
        plot_regime_shrinkage(scalars),
        out_dir,
        "fig_explain_shrinkage",
        formats,
    )
    return written


def render_checks(
    fig_dir,
    out_dir=None,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> list[Path]:
    """Posterior predictive check and model-vs-DIALS, from `checks.npz`."""
    from .figures_checks import plot_model_vs_dials, plot_ppc_zscores

    fig_dir, out_dir = _resolve(fig_dir, out_dir)
    path = fig_dir / "checks.npz"
    if not path.exists():
        logger.warning("no checks.npz in %s", fig_dir)
        return []
    data = np.load(path)
    written = save_figure(
        plot_ppc_zscores(data["z"], data.get("rate")),
        out_dir,
        "fig_check_ppc",
        formats,
    )
    if "model_i" in data and "dials_i" in data:
        written += save_figure(
            plot_model_vs_dials(
                data["model_i"],
                data["dials_i"],
                data["dials_sigma"] if "dials_sigma" in data else None,
            ),
            out_dir,
            "fig_check_model_vs_dials",
            formats,
        )
    return written


def render_all(
    fig_dir,
    out_dir=None,
    animate: bool = True,
    n_clusters: int = 6,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> list[Path]:
    """Render whichever figure families have dumps present."""
    fig_dir, out_dir = _resolve(fig_dir, out_dir)
    written: list[Path] = []
    jobs = (
        ("tracked_arrays.npz", lambda: render_tracked(
            fig_dir, out_dir, animate=animate, formats=formats
        )),
        ("basis_snapshots.npz", lambda: render_basis(
            fig_dir, out_dir, animate=animate, formats=formats
        )),
        ("tracked_arrays.npz", lambda: render_explain(
            fig_dir, out_dir, formats=formats
        )),
        ("checks.npz", lambda: render_checks(
            fig_dir, out_dir, formats=formats
        )),
    )
    for required, job in jobs:
        if not (fig_dir / required).exists():
            continue
        try:
            written += job()
        except Exception as exc:  # noqa: BLE001 - one family must not stop the rest
            logger.warning("figure job failed: %s", exc, exc_info=True)
    if list(fig_dir.glob("latents_epoch_*.parquet")):
        try:
            written += render_latent(
                fig_dir, out_dir, n_clusters=n_clusters, formats=formats
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("latent figures failed: %s", exc, exc_info=True)
    return written
