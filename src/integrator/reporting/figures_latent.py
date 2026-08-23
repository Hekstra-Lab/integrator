"""Structure of the profile latent space: PCA, clustering, and what it encodes.

The profile surrogate maps each shoebox to a latent `h`, so the question
"does the model learn anything beyond intensity" is answerable directly:
project `h` and ask which physical covariates the projection tracks.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .figure_style import imshow_panel, middle_slice, paper_style

# Pretty labels for the covariates carried next to the latents.
COVARIATE_LABELS = {
    "log10_intensity": "log₁₀ I (DIALS)",
    "snr": "I/σ (DIALS)",
    "d": "resolution d (Å)",
    "detector_x": "detector x (px)",
    "detector_y": "detector y (px)",
    "detector_r": "detector radius (px)",
    "background_mean": "background (counts/px)",
    "log10_qi_mean": "log₁₀ I (model)",
    "profile_correlation": "DIALS profile CC",
    "partiality": "partiality",
}


def latent_matrix(frame: pl.DataFrame) -> np.ndarray:
    """Stack the `h0..hd` columns into an `(N, d)` array."""
    cols = sorted(
        (c for c in frame.columns if c.startswith("h") and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )
    if not cols:
        raise KeyError("no h0..hd latent columns in frame")
    return frame.select(cols).to_numpy().astype(float)


def scale_matrix(frame: pl.DataFrame) -> np.ndarray | None:
    """Stack the `s0..sd` posterior scale columns, when they were recorded."""
    cols = sorted(
        (c for c in frame.columns if c.startswith("s") and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )
    if not cols:
        return None
    return frame.select(cols).to_numpy().astype(float)


def pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plain SVD PCA.

    Returns:
        Tuple of scores `(N, d)`, components `(d, d)` as rows, and the
        explained variance ratio.
    """
    centered = matrix - matrix.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = u * s
    variance = s**2
    return scores, vt, variance / variance.sum()


def derive_covariates(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Physical covariates to color the latent projections by."""
    out: dict[str, np.ndarray] = {}

    def col(name):
        return frame[name].to_numpy().astype(float) if name in frame.columns else None

    intensity = col("intensity_prf_value")
    variance = col("intensity_prf_variance")
    if intensity is None:
        intensity = col("intensity_sum_value")
        variance = col("intensity_sum_variance")
    if intensity is not None:
        out["log10_intensity"] = np.log10(np.clip(intensity, 1.0, None))
        if variance is not None:
            sigma = np.sqrt(np.clip(variance, 1e-6, None))
            out["snr"] = np.clip(intensity / sigma, -5, 100)
    for src, dst in (
        ("d", "d"),
        ("xyzcal_px_0", "detector_x"),
        ("xyzcal_px_1", "detector_y"),
        ("background_mean", "background_mean"),
        ("profile_correlation", "profile_correlation"),
        ("partiality", "partiality"),
    ):
        values = col(src)
        if values is not None:
            out[dst] = values
    if "detector_x" in out and "detector_y" in out:
        x = out["detector_x"] - out["detector_x"].mean()
        y = out["detector_y"] - out["detector_y"].mean()
        out["detector_r"] = np.sqrt(x**2 + y**2)
    qi = col("qi_mean")
    if qi is not None:
        out["log10_qi_mean"] = np.log10(np.clip(qi, 1e-3, None))
    return out


def plot_latent_pca(frame: pl.DataFrame, max_panels: int = 6, point_size=2.0):
    """PC1/PC2 of the profile latent, colored by each physical covariate."""
    import matplotlib.pyplot as plt

    scores, _, ratio = pca(latent_matrix(frame))
    covariates = derive_covariates(frame)
    names = list(covariates)[:max_panels]

    ncols = min(3, max(len(names), 1) + 1)
    nrows = int(np.ceil((len(names) + 1) / ncols))
    with paper_style():
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(2.4 * ncols, 2.3 * nrows), squeeze=False
        )
        flat = axes.ravel()
        for ax, name in zip(flat, names, strict=False):
            values = covariates[name]
            lo, hi = np.nanpercentile(values, [2, 98])
            sc = ax.scatter(
                scores[:, 0],
                scores[:, 1],
                c=values,
                s=point_size,
                cmap="viridis",
                vmin=lo,
                vmax=hi,
                linewidths=0,
                alpha=0.7,
                rasterized=True,
            )
            ax.set_title(COVARIATE_LABELS.get(name, name), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        bar = flat[len(names)]
        bar.bar(
            np.arange(len(ratio)), 100 * ratio, color="#0173B2", width=0.7
        )
        bar.set_xlabel("component")
        bar.set_ylabel("variance explained (%)")
        bar.set_title("latent spectrum", fontsize=8)
        for ax in flat[len(names) + 1 :]:
            ax.axis("off")
        fig.suptitle(
            f"profile latent PCA  (PC1 {100 * ratio[0]:.0f}%, "
            f"PC2 {100 * ratio[1]:.0f}%)",
            fontsize=9,
        )
    return fig


def kmeans_labels(matrix: np.ndarray, k: int = 6, seed: int = 0):
    """k-means on the latent, returning `(labels, centroids)`."""
    from scipy.cluster.vq import kmeans2

    rng = np.random.default_rng(seed)
    centroids, labels = kmeans2(
        matrix, k, minit="++", seed=rng, missing="warn"
    )
    return labels, centroids


def plot_latent_clusters(
    frame: pl.DataFrame,
    weight: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    shape=None,
    k: int = 6,
):
    """Cluster the latent, then decode each centroid back into a profile.

    Decoding the centroid is what makes the clustering interpretable: a
    cluster is not a colored blob, it is a profile shape the model
    believes in.
    """
    import matplotlib.pyplot as plt

    matrix = latent_matrix(frame)
    labels, centroids = kmeans_labels(matrix, k=k)
    scores, components, ratio = pca(matrix)
    centered = centroids - matrix.mean(0, keepdims=True)
    centroid_scores = centered @ components.T

    can_decode = weight is not None and bias is not None and shape is not None
    ncols = 1 + (k if can_decode else 0)
    with paper_style():
        fig = plt.figure(figsize=(2.6 + 1.1 * (ncols - 1), 2.8))
        grid = fig.add_gridspec(
            2, ncols, width_ratios=[2.4] + [1.0] * (ncols - 1)
        )
        ax = fig.add_subplot(grid[:, 0])
        cmap = plt.get_cmap("tab10")
        for c in range(k):
            sel = labels == c
            ax.scatter(
                scores[sel, 0],
                scores[sel, 1],
                s=2.0,
                color=cmap(c % 10),
                alpha=0.6,
                linewidths=0,
                rasterized=True,
                label=f"{c} ({sel.mean() * 100:.0f}%)",
            )
            ax.annotate(
                str(c),
                centroid_scores[c, :2],
                fontsize=8,
                weight="bold",
                ha="center",
                va="center",
            )
        ax.set_xlabel(f"PC1 ({100 * ratio[0]:.0f}%)")
        ax.set_ylabel(f"PC2 ({100 * ratio[1]:.0f}%)")
        ax.set_title(f"k-means, k={k}", fontsize=8)
        ax.legend(markerscale=3, fontsize=6, loc="best")

        if can_decode:
            bias = np.asarray(bias, dtype=float)
            weight = np.asarray(weight, dtype=float)
            for c in range(k):
                logits = weight @ centroids[c] + bias
                profile = np.exp(logits - logits.max())
                profile = profile / profile.sum()
                top = fig.add_subplot(grid[0, c + 1])
                imshow_panel(top, middle_slice(profile, shape))
                top.set_title(f"{c}", fontsize=7, color=cmap(c % 10))
                below = fig.add_subplot(grid[1, c + 1])
                delta = profile - np.exp(bias - bias.max()) / np.exp(
                    bias - bias.max()
                ).sum()
                imshow_panel(below, middle_slice(delta, shape), symmetric=True)
                if c == 0:
                    top.set_ylabel("profile", fontsize=6)
                    below.set_ylabel("− mean", fontsize=6)
        fig.suptitle("latent clusters and their decoded profiles", fontsize=9)
    return fig, labels


def covariate_r2(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """R² of each covariate against each PC and against the full latent.

    Per-PC values are squared Pearson correlations; the `full` column is
    the R² of an ordinary least-squares fit on all latent dimensions,
    i.e. how much of that covariate the latent carries in total.
    """
    matrix = latent_matrix(frame)
    scores, _, _ = pca(matrix)
    covariates = derive_covariates(frame)
    design = np.column_stack([matrix, np.ones(len(matrix))])

    rows = []
    for name, values in covariates.items():
        good = np.isfinite(values)
        if good.sum() < 10 or np.std(values[good]) <= 0:
            continue
        y = values[good]
        row = {"covariate": name}
        for j in range(min(4, scores.shape[1])):
            r = np.corrcoef(scores[good, j], y)[0, 1]
            row[f"PC{j + 1}"] = float(r**2)
        beta, *_ = np.linalg.lstsq(design[good], y, rcond=None)
        resid = y - design[good] @ beta
        total = ((y - y.mean()) ** 2).sum()
        row["full"] = float(1 - (resid**2).sum() / total) if total > 0 else 0.0
        rows.append(row)
    return pl.DataFrame(rows), covariates


def plot_latent_covariate_r2(frame: pl.DataFrame):
    """Heatmap of how much of each covariate the latent explains."""
    import matplotlib.pyplot as plt

    table, _ = covariate_r2(frame)
    if table.is_empty():
        raise ValueError("no covariates available for the R² panel")
    value_cols = [c for c in table.columns if c != "covariate"]
    values = table.select(value_cols).to_numpy()
    labels = [
        COVARIATE_LABELS.get(name, name) for name in table["covariate"]
    ]

    with paper_style():
        fig, ax = plt.subplots(
            figsize=(0.7 * len(value_cols) + 2.6, 0.32 * len(labels) + 1.2)
        )
        im = ax.imshow(values, cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(value_cols)), value_cols)
        ax.set_yticks(range(len(labels)), labels)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if values[i, j] < 0.6 else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="R²")
        ax.set_title("what the profile latent encodes", fontsize=9)
    return fig


def plot_detector_map(frame: pl.DataFrame, labels: np.ndarray | None = None):
    """Reflections in detector coordinates, colored by cluster and by PC1."""
    import matplotlib.pyplot as plt

    covariates = derive_covariates(frame)
    if "detector_x" not in covariates:
        raise ValueError("frame has no xyzcal.px columns")
    x, y = covariates["detector_x"], covariates["detector_y"]
    scores, _, _ = pca(latent_matrix(frame))

    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
        if labels is not None:
            cmap = plt.get_cmap("tab10")
            for c in np.unique(labels):
                sel = labels == c
                axes[0].scatter(
                    x[sel], y[sel], s=1.5, color=cmap(int(c) % 10),
                    linewidths=0, alpha=0.7, rasterized=True, label=str(c),
                )
            axes[0].legend(markerscale=4, fontsize=6, ncols=2)
            axes[0].set_title("latent cluster", fontsize=8)
        else:
            axes[0].axis("off")
        lo, hi = np.nanpercentile(scores[:, 0], [2, 98])
        sc = axes[1].scatter(
            x, y, c=scores[:, 0], s=1.5, cmap="coolwarm", vmin=lo, vmax=hi,
            linewidths=0, alpha=0.8, rasterized=True,
        )
        fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.02, label="PC1")
        axes[1].set_title("PC1", fontsize=8)
        for ax in axes:
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
            ax.set_aspect("equal")
        fig.suptitle("latent structure on the detector", fontsize=9)
    return fig


def plot_latent_usage(frame: pl.DataFrame, prior_scale: float = 3.0):
    """Which latent dimensions carry information and which collapsed.

    A dimension whose posterior mean barely varies while its posterior
    scale sits at the prior is unused capacity: the encoder ignores it
    and the KL term pins it to the prior.
    """
    import matplotlib.pyplot as plt

    loc = latent_matrix(frame)
    scale = scale_matrix(frame)
    spread = loc.std(0)
    order = np.argsort(-spread)

    with paper_style():
        fig, ax = plt.subplots(figsize=(4.2, 2.4))
        idx = np.arange(len(spread))
        ax.bar(idx, spread[order], color="#0173B2", width=0.7, label="sd of q mean")
        if scale is not None:
            ax.plot(
                idx,
                scale.mean(0)[order],
                color="#DE8F05",
                marker="o",
                markersize=3,
                label="mean posterior sd",
            )
        ax.axhline(
            prior_scale, color="0.4", linestyle="--", linewidth=0.8,
            label="prior sd",
        )
        ax.set_xticks(idx, [f"h{i}" for i in order], fontsize=6)
        ax.set_ylabel("latent scale")
        ax.set_title("latent dimension usage", fontsize=9)
        ax.legend()
    return fig
