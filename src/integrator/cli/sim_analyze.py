"""
Post-training analysis CLI for simulated shoebox data.

Reads predictions, loss traces, and reference data from a completed training
run, computes error metrics binned by intensity, generates diagnostic plots,
and uploads everything to a W&B run.

Example use:
    integrator.sim_analyze -v --run-dir /path/to/run_dir
    integrator.sim_analyze --run-dir /path/to/run_dir --epochs 0 499 999
    integrator.sim_analyze --run-dir /path/to/run_dir --sbc --sbc-nsamples 5000
"""

import argparse
import logging

from integrator.cli.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-training analysis on simulated data"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to dir containing run_metadata.yaml",
    )
    parser.add_argument(
        "--wb-project",
        type=str,
        default=None,
        help="W&B project for analysis run (default: reuse training project)",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=None,
        help="Specific epochs to analyze (default: all available)",
    )
    parser.add_argument(
        "--sbc",
        action="store_true",
        help="Run simulation-based calibration",
    )
    parser.add_argument(
        "--sbc-nsamples",
        type=int,
        default=5000,
        help="Number of SBC simulations",
    )
    parser.add_argument(
        "--sbc-K",
        type=int,
        default=99,
        help="Number of posterior samples per SBC draw",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v = INFO, -vv = DEBUG)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_bins(edges: list) -> tuple[list[str], "pl.DataFrame"]:
    """Build bin labels and a base DataFrame for joining."""
    import polars as pl

    bin_labels = [f"{a} - {b}" for a, b in zip(edges[:-1], edges[1:])]
    bin_labels.insert(0, f"<{edges[0]}")
    bin_labels.append(f">{edges[-1]}")
    reversed_labels = list(reversed(bin_labels))

    base_df = pl.DataFrame(
        {
            "bin_labels": reversed_labels,
            "bin_id": list(range(len(reversed_labels))),
        },
        schema={"bin_labels": pl.Categorical, "bin_id": pl.Int32},
    )
    return bin_labels, base_df


def _compute_intensity_edges(I_true: "torch.Tensor") -> list:
    """Auto-compute intensity bin edges from the true intensity distribution."""
    import torch

    strong = I_true[I_true <= 250]
    n_bins = min(15, max(5, len(strong) // 200))
    edges = torch.quantile(strong, torch.linspace(0, 1, n_bins + 1)).tolist()[1:]
    return [round(e) for e in edges]


# ---------------------------------------------------------------------------
# Analysis routines
# ---------------------------------------------------------------------------

def compute_error_metrics(
    pred_lf: "pl.LazyFrame",
    intensity_edges: list,
    bin_labels: list[str],
    base_df: "pl.DataFrame",
) -> "pl.DataFrame":
    """Compute per-bin, per-epoch error metrics from predictions."""
    import polars as pl

    lf = pred_lf.with_columns(
        pl.col("intensity")
        .cut(breaks=intensity_edges, labels=bin_labels)
        .alias("bin_labels")
    )

    lf = lf.with_columns(
        signed_error=pl.col("qi_mean") - pl.col("intensity"),
        abs_error=(pl.col("qi_mean") - pl.col("intensity")).abs(),
        squared_error=(pl.col("qi_mean") - pl.col("intensity")).pow(2),
        rel_error=(pl.col("qi_mean") - pl.col("intensity")).abs()
        / pl.col("intensity").clip(lower_bound=1e-6),
    )

    agg = lf.group_by(["bin_labels", "epoch"]).agg(
        bias=pl.col("signed_error").mean(),
        mae=pl.col("abs_error").mean(),
        mse=pl.col("squared_error").mean(),
        rmse=pl.col("squared_error").mean().sqrt(),
        mre=pl.col("rel_error").mean(),
        median_ae=pl.col("abs_error").median(),
        median_re=pl.col("rel_error").median(),
        corr_i=pl.corr(pl.col("qi_mean"), pl.col("intensity")),
        corr_bg=pl.corr(pl.col("qbg_mean"), pl.col("background")),
        mean_qi_var=pl.col("qi_var").mean(),
        mean_qi_std=pl.col("qi_var").sqrt().mean(),
        true_std=pl.col("intensity").std(),
        pred_std=pl.col("qi_mean").std(),
        true_mean=pl.col("intensity").mean(),
        pred_mean=pl.col("qi_mean").mean(),
    )
    agg = agg.with_columns(
        var=pl.col("mse") - pl.col("bias").pow(2),
        dispersion_ratio=pl.col("pred_std") / pl.col("true_std"),
        pred_bias_ratio=pl.col("pred_mean") / pl.col("true_mean"),
    )

    return base_df.lazy().join(agg, on="bin_labels", how="left").collect().sort("bin_id")


def compute_loss_curves(
    wandb_dir: "Path",
) -> tuple["pl.DataFrame | None", "pl.DataFrame | None"]:
    """Load loss trace parquets and aggregate per-epoch means."""
    import polars as pl

    train_paths = sorted(wandb_dir.glob("**/loss_trace_train_*.parquet"))
    val_paths = sorted(wandb_dir.glob("**/loss_trace_val_*.parquet"))

    train_df = None
    if train_paths:
        train_df = (
            pl.scan_parquet([str(p) for p in train_paths])
            .group_by("epoch")
            .agg(
                train_loss=pl.col("loss").mean(),
                train_nll=pl.col("nll").mean(),
                train_kl=pl.col("kl").mean(),
            )
            .collect()
            .sort("epoch")
        )

    val_df = None
    if val_paths:
        val_df = (
            pl.scan_parquet([str(p) for p in val_paths])
            .group_by("epoch")
            .agg(
                val_loss=pl.col("loss").mean(),
                val_nll=pl.col("nll").mean(),
                val_kl=pl.col("kl").mean(),
            )
            .collect()
            .sort("epoch")
        )

    return train_df, val_df


def compute_profile_metrics(
    pred_paths: list,
    data_dir: "Path",
    concentration_path: str,
    intensity_edges: list,
    bin_labels: list[str],
    base_df: "pl.DataFrame",
) -> "pl.DataFrame | None":
    """Compute profile quality metrics per bin per epoch.

    Uses the normalized ``concentration_per_group`` as the reference profile
    for each resolution bin. Predictions must contain ``qp_mean``.
    """
    import polars as pl
    import torch

    from pathlib import Path

    conc_full_path = Path(concentration_path) if Path(concentration_path).is_absolute() else data_dir / concentration_path
    if not conc_full_path.exists():
        logger.warning("concentration file not found: %s", conc_full_path)
        return None

    conc = torch.load(conc_full_path, weights_only=True).float()  # (n_bins, 441)
    ref = torch.load(data_dir / "reference.pt", weights_only=False)

    # Normalize to get expected profiles per resolution bin
    ref_profiles = conc / conc.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    group_labels = torch.as_tensor(ref["group_label"]).long()
    refl_ids_ref = torch.as_tensor(ref["refl_ids"]).float()

    # Reference profile per reflection: (N_total, 441)
    ref_per_refl = ref_profiles[group_labels]

    # Map refl_id → row index for fast lookup
    refl_id_to_idx = {float(rid): i for i, rid in enumerate(refl_ids_ref.tolist())}

    # Check first file for qp_mean column
    first_schema = pl.read_parquet_schema(str(pred_paths[0]))
    if "qp_mean" not in first_schema:
        logger.info("Predictions do not contain qp_mean; skipping profile metrics")
        return None

    all_aggs: list[pl.DataFrame] = []

    for path in pred_paths:
        df = pl.read_parquet(
            str(path),
            columns=["qp_mean", "refl_ids", "intensity", "epoch"],
        )
        if len(df) == 0:
            continue

        epoch_val = int(df["epoch"][0])

        # Convert qp_mean list column to torch tensor
        qp = torch.tensor(df["qp_mean"].to_list(), dtype=torch.float32)
        rids = df["refl_ids"].to_numpy()

        # Look up reference profiles by refl_id
        indices = torch.tensor(
            [refl_id_to_idx.get(float(r), 0) for r in rids], dtype=torch.long
        )
        ref_p = ref_per_refl[indices]

        # Cosine similarity
        dot = (qp * ref_p).sum(dim=-1)
        norm_q = qp.norm(dim=-1).clamp(min=1e-12)
        norm_r = ref_p.norm(dim=-1).clamp(min=1e-12)
        cos_sim = dot / (norm_q * norm_r)

        # L2 error
        l2 = (qp - ref_p).pow(2).sum(dim=-1).sqrt()

        # Total variation
        tv = 0.5 * (qp - ref_p).abs().sum(dim=-1)

        refl_df = pl.DataFrame({
            "intensity": df["intensity"].to_numpy().astype("float32"),
            "cosine_similarity": cos_sim.numpy(),
            "l2_error": l2.numpy(),
            "tv": tv.numpy(),
        })
        refl_df = refl_df.with_columns(
            pl.col("intensity")
            .cut(breaks=intensity_edges, labels=bin_labels)
            .alias("bin_labels")
        )

        agg = refl_df.group_by("bin_labels").agg(
            mean_cos_sim=pl.col("cosine_similarity").mean(),
            mean_tv=pl.col("tv").mean(),
            mean_l2_error=pl.col("l2_error").mean(),
        ).with_columns(pl.lit(epoch_val).alias("epoch"))
        all_aggs.append(agg)

    if not all_aggs:
        return None

    prof_agg = pl.concat(all_aggs)
    return (
        base_df.lazy()
        .join(prof_agg.lazy(), on="bin_labels", how="left")
        .collect()
        .sort("bin_id")
    )


def compute_crlb_curves(
    data_dir: "Path",
    loss_args: dict,
    *,
    n_points: int = 10000,
    alpha_I: float = 5.0,
    alpha_bg: float = 5.0,
) -> "dict | None":
    """Compute CRLB and Laplace noise-to-signal curves.

    Uses the average (normalised) concentration profile across bins and the
    mean background rate.
    """
    import torch
    from pathlib import Path

    # Load concentration and bg_rate
    conc_path = loss_args.get("concentration_per_group")
    bg_path = loss_args.get("bg_rate_per_group")
    if conc_path is None or bg_path is None:
        return None

    conc_full = Path(conc_path) if Path(conc_path).is_absolute() else data_dir / conc_path
    bg_full = Path(bg_path) if Path(bg_path).is_absolute() else data_dir / bg_path

    if not conc_full.exists() or not bg_full.exists():
        return None

    conc = torch.load(conc_full, weights_only=True).float()  # (n_bins, 441)
    bg_rate = torch.load(bg_full, weights_only=True).float()  # (n_bins,)

    # Average profile across bins, then normalise
    p = conc.mean(dim=0)
    p = p / p.sum()

    # Mean background per pixel
    mean_bg = (1.0 / bg_rate).mean()

    x = torch.linspace(0.1, 1e4, n_points)

    # Fisher information matrix for Poisson model
    lam = x.view(-1, 1) * p.view(1, -1) + mean_bg  # (n_points, 441)
    F11 = (p.view(1, -1).pow(2) / lam).sum(-1)
    F12 = (p.view(1, -1) / lam).sum(-1)
    F22 = (1.0 / lam).sum(-1)
    det_F = (F11 * F22 - F12.pow(2)).clamp(min=1e-30)

    # CRLB: Var(I) = F^{-1}[0,0] = F22 / det(F)
    crlb_var = F22 / det_F
    crlb_ns = crlb_var.clamp(min=0).sqrt() / x

    # Laplace approximation: (F + P)^{-1}[0,0]
    P11 = (alpha_I - 1.0) / x.pow(2) if alpha_I > 1 else torch.zeros_like(x)
    P22_val = (alpha_bg - 1.0) / mean_bg**2 if alpha_bg > 1 else 0.0

    H11 = F11 + P11
    H12 = F12
    H22 = F22 + P22_val
    det_H = (H11 * H22 - H12.pow(2)).clamp(min=1e-30)
    laplace_var = H22 / det_H
    laplace_ns = laplace_var.clamp(min=0).sqrt() / x

    return {
        "x": x,
        "crlb_ns": crlb_ns,
        "laplace_ns": laplace_ns,
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_error_metric_evolution(
    error_df: "pl.DataFrame",
    metric: str,
    *,
    ylabel: str | None = None,
    yscale: str = "linear",
    ylim: tuple | None = None,
) -> "plt.Figure":
    """Plot a single error metric across bins, colored by epoch."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import polars as pl

    plot_df = error_df.select(["bin_id", "bin_labels", "epoch", metric]).filter(
        pl.col("epoch").is_not_null()
    )
    epochs = sorted(plot_df["epoch"].unique())
    if not epochs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=min(epochs), vmax=max(epochs))

    wide = plot_df.pivot(on="epoch", index="bin_id", values=metric).sort("bin_id")

    fig, ax = plt.subplots(1, figsize=(7, 4.5))
    for epoch in epochs:
        col = str(epoch)
        if col in wide.columns:
            ax.plot(wide["bin_id"], wide[col], color=cmap(norm(epoch)), alpha=0.8)

    ax.set_ylabel(ylabel or metric)
    ax.set_xlabel("intensity bin")
    if yscale != "linear":
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ticks = error_df.select(["bin_id", "bin_labels"]).unique().sort("bin_id")
    ax.set_xticks(ticks["bin_id"].to_list())
    ax.set_xticklabels(ticks["bin_labels"].to_list(), rotation=65, ha="right")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label="epoch")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_loss_curves(
    train_df: "pl.DataFrame | None",
    val_df: "pl.DataFrame | None",
    component: str = "loss",
) -> "plt.Figure":
    """Plot train/val loss component vs epoch."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    if train_df is not None and f"train_{component}" in train_df.columns:
        ax.plot(
            train_df["epoch"],
            train_df[f"train_{component}"],
            label=f"train {component}",
        )
    if val_df is not None and f"val_{component}" in val_df.columns:
        ax.plot(
            val_df["epoch"],
            val_df[f"val_{component}"],
            label=f"val {component}",
            linestyle="--",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel(component)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_scatter(
    pred_df: "pl.DataFrame",
    x_col: str,
    y_col: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    symlog: bool = False,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
) -> "plt.Figure":
    """Scatter plot of two columns with identity line."""
    import matplotlib.pyplot as plt
    import torch

    fig, ax = plt.subplots(figsize=(5, 5))
    x = pred_df[x_col].to_numpy()
    y = pred_df[y_col].to_numpy()

    ax.scatter(x, y, alpha=0.3, s=3, rasterized=True, color="black")

    # identity line
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    line = torch.linspace(float(lo), float(hi), 200)
    ax.plot(line, line, color="red", linewidth=1)

    if symlog:
        ax.set_xscale("symlog")
        ax.set_yscale("symlog")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_noise_to_signal(
    pred_df: "pl.DataFrame",
    *,
    crlb: "dict | None" = None,
) -> "plt.Figure":
    """Noise-to-signal ratio plot comparing model posterior to Poisson limit.

    If *crlb* is provided (from :func:`compute_crlb_curves`), CRLB and Laplace
    approximation bounds are overlaid.
    """
    import matplotlib.pyplot as plt
    import torch

    fig, ax = plt.subplots(figsize=(6, 5))

    qi_mean = torch.tensor(pred_df["qi_mean"].to_numpy())
    qi_var = torch.tensor(pred_df["qi_var"].to_numpy())

    x_model = qi_mean
    y_model = qi_var.sqrt() / qi_mean.clamp(min=1e-6)

    # Poisson limit
    x_ref = torch.linspace(0.1, float(qi_mean.max()), 1000)
    y_ref = x_ref.sqrt() / x_ref

    ax.plot(x_ref, y_ref, label="Poisson limit", linestyle="--", c="black")

    if crlb is not None:
        ax.plot(crlb["x"], crlb["crlb_ns"], label="CRLB", color="tab:orange")
        ax.plot(crlb["x"], crlb["laplace_ns"], label="Laplace", color="tab:green")

    ax.scatter(x_model, y_model, alpha=0.2, s=3, rasterized=True)
    ax.axhline(1, linestyle="dotted", color="red", label="signal = noise")

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(xmin=0.1)
    ax.set_xlabel("estimated intensity")
    ax.set_ylabel("noise-to-signal ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sbc_ranks(
    df_sbc: "pl.DataFrame",
    K: int = 99,
    min_count: int = 50,
) -> "plt.Figure":
    """SBC rank histogram faceted by intensity bin."""
    import matplotlib.pyplot as plt
    import polars as pl

    counts = df_sbc["bin_labels"].value_counts()
    valid_bins = (
        counts.filter(pl.col("count") >= min_count)
        .sort("bin_labels")["bin_labels"]
        .to_list()
    )

    n_bins = max(1, len(valid_bins))
    fig, axes = plt.subplots(1, n_bins, figsize=(3 * n_bins, 3), sharey=True)
    if n_bins == 1:
        axes = [axes]

    for ax, bin_label in zip(axes, valid_bins):
        r = df_sbc.filter(pl.col("bin_labels") == bin_label)["rank"].to_numpy()
        ax.hist(
            r,
            bins=20,
            range=(-0.5, K + 0.5),
            density=True,
            color="steelblue",
            alpha=0.7,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axhline(1 / (K + 1), color="red", linestyle="--", linewidth=1.0)
        ax.set_title(bin_label, fontsize=8)
        ax.set_xlabel("rank", fontsize=7)
        ax.text(0.05, 0.92, f"n={len(r)}", transform=ax.transAxes, fontsize=6)
        if ax is axes[0]:
            ax.set_ylabel("density", fontsize=7)

    fig.suptitle("SBC rank histograms", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SBC
# ---------------------------------------------------------------------------

def run_sbc(
    config: dict,
    wandb_dir: "Path",
    data_dir: "Path",
    *,
    nsamples: int = 5000,
    K: int = 99,
    intensity_edges: list,
    bin_labels: list[str],
    checkpoint_epoch: int | None = None,
) -> "pl.DataFrame":
    """Run simulation-based calibration and return rank DataFrame."""
    import polars as pl
    import torch

    from integrator.utils import construct_integrator

    # Load standardization stats
    stats_train = torch.load(data_dir / "stats_anscombe.pt", weights_only=False)

    # Use prior parameters from the config
    loss_args = config.get("loss", {}).get("args", {})

    # Intensity prior: Gamma(1, tau) where tau is per-group
    tau_path = loss_args.get("tau_per_group")
    if tau_path is not None:
        from pathlib import Path
        tau_p = Path(tau_path) if Path(tau_path).is_absolute() else data_dir / tau_path
        tau = torch.load(tau_p, weights_only=True)
    else:
        tau = torch.tensor([0.001])

    # Background prior: Gamma(1, bg_rate) per group
    bg_rate_path = loss_args.get("bg_rate_per_group")
    if bg_rate_path is not None:
        from pathlib import Path
        bg_p = Path(bg_rate_path) if Path(bg_rate_path).is_absolute() else data_dir / bg_rate_path
        bg_rate = torch.load(bg_p, weights_only=True)
    else:
        bg_rate = torch.tensor([1.0])

    # Concentration for profile prior
    conc_path = loss_args.get("concentration_per_group")
    if conc_path is not None:
        from pathlib import Path
        conc_p = Path(conc_path) if Path(conc_path).is_absolute() else data_dir / conc_path
        concentration = torch.load(conc_p, weights_only=True)
    else:
        concentration = torch.ones(441) * 1e-3

    # Use bin 0 priors for SBC (simplification: single-bin SBC)
    # If concentration is 2D (n_bins, 441), use first bin
    if concentration.dim() == 2:
        concentration = concentration[0]
    if tau.dim() >= 1:
        tau_val = tau[0]
    else:
        tau_val = tau

    if bg_rate.dim() >= 1:
        bg_rate_val = bg_rate[0]
    else:
        bg_rate_val = bg_rate

    H, W = 21, 21

    # Sample from priors
    pi = torch.distributions.Gamma(torch.ones(1), tau_val)
    pbg = torch.distributions.Gamma(torch.ones(1), bg_rate_val)
    pprf = torch.distributions.Dirichlet(concentration.clamp(min=1e-6))

    i_s = pi.sample([nsamples]).squeeze()
    bg_s = pbg.sample([nsamples]).squeeze()
    prf_s = pprf.sample([nsamples])

    # Generate shoeboxes
    rates = (
        i_s.view(nsamples, 1, 1) * prf_s.view(nsamples, H, W)
        + bg_s.view(nsamples, 1, 1)
    )
    shoeboxes = torch.poisson(rates).unsqueeze(1)  # (N, 1, H, W)

    # Standardize
    anscombe = 2 * (shoeboxes.float() + 0.375).sqrt()
    shoeboxes_std = (anscombe - stats_train[0]) / stats_train[1].sqrt()

    # Load model
    integrator = construct_integrator(config)

    # Find checkpoint
    ckpts = sorted(wandb_dir.glob("**/epoch*.ckpt"))
    if not ckpts:
        logger.warning("No checkpoints found for SBC")
        return pl.DataFrame()

    if checkpoint_epoch is not None:
        import re
        epoch_re = re.compile(r"epoch=(\d+)")
        target = None
        for c in ckpts:
            m = epoch_re.search(c.name)
            if m and int(m.group(1)) == checkpoint_epoch:
                target = c
                break
        if target is None:
            logger.warning("Checkpoint epoch %d not found, using last", checkpoint_epoch)
            target = ckpts[-1]
    else:
        target = ckpts[-1]

    logger.info("SBC using checkpoint: %s", target.name)
    ckpt_data = torch.load(target, weights_only=False, map_location="cpu")
    integrator.load_state_dict(ckpt_data["state_dict"])
    integrator.eval()

    # Build dummy inputs for the model's forward pass
    # counts = raw shoeboxes (flat), masks = all ones, metadata = minimal dict
    counts_flat = shoeboxes.squeeze(1).reshape(nsamples, -1)  # (N, H*W)
    masks_flat = torch.ones_like(counts_flat)
    dummy_meta = {
        "refl_ids": torch.arange(nsamples),
        "is_test": torch.ones(nsamples, dtype=torch.bool),
        "group_label": torch.zeros(nsamples, dtype=torch.long),
        "intensity": i_s,
        "background": bg_s,
    }

    # Run inference in batches using the model's forward pass
    batch_size = 512
    all_qi_mean = []
    all_qi_var = []

    with torch.no_grad():
        for start in range(0, nsamples, batch_size):
            end = min(start + batch_size, nsamples)
            batch_counts = counts_flat[start:end]
            batch_sbox = shoeboxes_std[start:end].squeeze(1).reshape(end - start, -1)
            batch_masks = masks_flat[start:end]
            batch_meta = {k: v[start:end] for k, v in dummy_meta.items()}

            outputs = integrator(batch_counts, batch_sbox, batch_masks, batch_meta)
            fwd = outputs["forward_out"]
            all_qi_mean.append(fwd["qi_mean"].cpu())
            all_qi_var.append(fwd["qi_var"].cpu())

    qi_mean = torch.cat(all_qi_mean)
    qi_var = torch.cat(all_qi_var)

    # Recover Gamma parameters: mean = conc/rate, var = conc/rate^2
    # => rate = mean/var, conc = mean^2/var
    qi_rate = qi_mean / qi_var.clamp(min=1e-12)
    qi_conc = qi_mean.pow(2) / qi_var.clamp(min=1e-12)

    # Compute SBC ranks
    post_samples = torch.distributions.Gamma(qi_conc, qi_rate).sample([K])
    ranks = (post_samples < i_s.unsqueeze(0)).sum(dim=0)

    # Build DataFrame
    df_sbc = pl.DataFrame({
        "rank": ranks.numpy().astype("int32"),
        "I_true": i_s.numpy().astype("float32"),
    })
    df_sbc = df_sbc.with_columns(
        pl.col("I_true")
        .cut(breaks=intensity_edges, labels=bin_labels)
        .alias("bin_labels")
    )
    return df_sbc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["text.usetex"] = False
    import polars as pl
    import torch
    import wandb
    import yaml

    from integrator.utils import load_config

    args = parse_args()
    setup_logging(args.verbose)

    # Summary lines accumulated for the text report
    summary_lines: list[str] = []

    def _log(msg: str):
        logger.info(msg)
        summary_lines.append(msg)

    _log("Starting analysis")

    # --- Load run metadata ---
    run_dir = Path(args.run_dir)
    meta = yaml.safe_load((run_dir / "run_metadata.yaml").read_text())
    config = load_config(meta["config"])
    wandb_info = meta["wandb"]

    log_dir = Path(wandb_info["log_dir"])
    wandb_dir = log_dir.parent
    data_dir = Path(config["data_loader"]["args"]["data_dir"])

    _log(f"W&B log dir: {log_dir}")
    _log(f"Data dir: {data_dir}")
    _log(f"Source run: {wandb_info.get('run_id', 'unknown')}")

    # --- Load reference data ---
    reference = torch.load(data_dir / "reference.pt", weights_only=False)
    I_true = torch.tensor(reference["intensity"])
    _log(f"Loaded {len(I_true)} reflections from reference")

    # --- Compute intensity bin edges ---
    intensity_edges = _compute_intensity_edges(I_true)
    bin_labels, base_df = _get_bins(intensity_edges)
    _log(f"Intensity edges ({len(intensity_edges)} bins): {intensity_edges}")

    # --- Initialize W&B analysis run ---
    wb_project = args.wb_project or wandb_info.get("project", "sim-analysis")
    wandb.init(
        project=wb_project,
        name=f"analysis-{wandb_info.get('run_id', 'unknown')}",
        tags=["analysis", "simulated"],
        config={
            "source_run_id": wandb_info.get("run_id"),
            "source_project": wandb_info.get("project"),
            "intensity_edges": intensity_edges,
        },
    )

    # --- Load predictions ---
    pred_dir = wandb_dir / "predictions"
    pred_paths = sorted(pred_dir.glob("**/preds_epoch_*.parquet"))

    if not pred_paths:
        _log(f"ERROR: No prediction parquet files found in {pred_dir}")
        wandb.finish()
        _write_summary(run_dir, summary_lines)
        return

    _log(f"Found {len(pred_paths)} prediction files")

    # Filter to requested epochs if specified
    if args.epochs is not None:
        import re
        epoch_re = re.compile(r"epoch_(\d+)")
        filtered = []
        epoch_set = set(args.epochs)
        for p in pred_paths:
            m = epoch_re.search(str(p))
            if m and int(m.group(1)) in epoch_set:
                filtered.append(p)
        pred_paths = filtered
        _log(f"Filtered to {len(pred_paths)} files for epochs {args.epochs}")

    pred_lf = pl.scan_parquet([str(p) for p in pred_paths])

    # Discover all available epochs
    all_epochs = sorted(pred_lf.select("epoch").unique().collect()["epoch"].to_list())
    _log(f"Epochs available: {len(all_epochs)} (min={min(all_epochs)}, max={max(all_epochs)})")

    last_epoch = max(all_epochs)

    # --- Compute error metrics (all epochs) ---
    _log("Computing error metrics...")
    error_df = compute_error_metrics(pred_lf, intensity_edges, bin_labels, base_df)

    # --- Per-epoch global metrics → wandb time series ---
    _log("Logging per-epoch global metrics...")
    for epoch in all_epochs:
        epoch_metrics = error_df.filter(pl.col("epoch") == epoch)
        if len(epoch_metrics) == 0:
            continue
        wandb.log({
            "epoch": epoch,
            "global/mae": epoch_metrics["mae"].mean(),
            "global/rmse": epoch_metrics["rmse"].mean(),
            "global/mre": epoch_metrics["mre"].mean(),
            "global/bias": epoch_metrics["bias"].mean(),
            "global/corr_i": epoch_metrics["corr_i"].mean(),
            "global/corr_bg": epoch_metrics["corr_bg"].mean(),
        })

    # --- Compute profile metrics (all epochs, file by file) ---
    loss_args = config.get("loss", {}).get("args", {})
    conc_path = loss_args.get("concentration_per_group")

    profile_df = None
    if conc_path is not None:
        _log("Computing profile metrics...")
        profile_df = compute_profile_metrics(
            pred_paths, data_dir, conc_path,
            intensity_edges, bin_labels, base_df,
        )
        if profile_df is not None:
            _log(f"  Profile metrics: {len(profile_df)} rows")
        else:
            _log("  Profile metrics: skipped (no qp_mean in predictions)")
    else:
        _log("No concentration_per_group in config; skipping profile metrics")

    # --- Compute CRLB / Laplace curves ---
    crlb_data = compute_crlb_curves(data_dir, loss_args)
    if crlb_data is not None:
        _log("Computed CRLB / Laplace bounds")
    else:
        _log("CRLB: skipped (missing concentration or bg_rate files)")

    # --- Compute loss curves ---
    _log("Computing loss curves...")
    train_loss_df, val_loss_df = compute_loss_curves(wandb_dir)

    if train_loss_df is not None:
        _log(f"  Train loss: {len(train_loss_df)} epochs")
    if val_loss_df is not None:
        _log(f"  Val loss: {len(val_loss_df)} epochs")

    # ===================================================================
    # Generate and log plots
    # ===================================================================

    n_plots_ok = 0
    n_plots_fail = 0

    def _try_plot(name: str, fn):
        nonlocal n_plots_ok, n_plots_fail
        try:
            fn()
            n_plots_ok += 1
        except Exception as e:
            n_plots_fail += 1
            logger.warning("Failed to plot %s: %s", name, e)
            summary_lines.append(f"  FAILED: {name} — {e}")

    # 1. Error metric evolution plots
    METRICS = {
        "mre": {"ylabel": "Mean Relative Error", "yscale": "linear", "ylim": (0, 10)},
        "mae": {"ylabel": "MAE", "yscale": "symlog"},
        "rmse": {"ylabel": "RMSE", "yscale": "log"},
        "bias": {"ylabel": "Bias", "yscale": "symlog"},
        "corr_i": {"ylabel": "Intensity Correlation", "yscale": "linear", "ylim": (0, 1)},
        "corr_bg": {"ylabel": "Background Correlation", "yscale": "linear", "ylim": (0.9, 1)},
        "var": {"ylabel": "Variance", "yscale": "log"},
    }

    for metric_name, metric_cfg in METRICS.items():
        def _plot_metric(mn=metric_name, mc=metric_cfg):
            fig = plot_error_metric_evolution(
                error_df, mn,
                ylabel=mc.get("ylabel"),
                yscale=mc.get("yscale", "linear"),
                ylim=mc.get("ylim"),
            )
            wandb.log({f"metrics/{mn}": wandb.Image(fig)})
            plt.close(fig)
        _try_plot(f"metrics/{metric_name}", _plot_metric)

    # 1b. Profile metric evolution plots
    if profile_df is not None:
        PROFILE_METRICS = {
            "mean_cos_sim": {"ylabel": "Cosine Similarity"},
            "mean_l2_error": {"ylabel": "L2 Norm"},
            "mean_tv": {"ylabel": "Total Variation"},
        }
        for pm_name, pm_cfg in PROFILE_METRICS.items():
            def _plot_prof(mn=pm_name, mc=pm_cfg):
                fig = plot_error_metric_evolution(
                    profile_df, mn, ylabel=mc["ylabel"],
                )
                wandb.log({f"profile/{mn}": wandb.Image(fig)})
                plt.close(fig)
            _try_plot(f"profile/{pm_name}", _plot_prof)

    # 2. Loss curves
    for component in ["loss", "nll", "kl"]:
        def _plot_loss(c=component):
            fig = plot_loss_curves(train_loss_df, val_loss_df, c)
            wandb.log({f"loss/{c}": wandb.Image(fig)})
            plt.close(fig)
        _try_plot(f"loss/{component}", _plot_loss)

    # 3. Loss gap
    if train_loss_df is not None and val_loss_df is not None:
        def _plot_gaps():
            gap_df = (
                val_loss_df.with_columns(pl.col("epoch").cast(pl.Int32))
                .sort("epoch")
                .join_asof(
                    train_loss_df.with_columns(pl.col("epoch").cast(pl.Int32)).sort("epoch"),
                    on="epoch",
                    strategy="nearest",
                    suffix="_train",
                )
                .with_columns(
                    loss_gap=pl.col("train_loss") - pl.col("val_loss"),
                    nll_gap=pl.col("train_nll") - pl.col("val_nll"),
                    kl_gap=pl.col("train_kl") - pl.col("val_kl"),
                )
                .sort("epoch")
            )
            for gap_name in ["loss_gap", "nll_gap", "kl_gap"]:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(gap_df["epoch"], gap_df[gap_name])
                ax.set_xlabel("epoch")
                ax.set_ylabel(gap_name.replace("_", " ").title())
                ax.axhline(0, linestyle="--", color="gray", alpha=0.5)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                wandb.log({f"loss/{gap_name}": wandb.Image(fig)})
                plt.close(fig)
        _try_plot("loss/gaps", _plot_gaps)

    # 4. Scatter plots for multiple epochs (first, middle, last)
    scatter_epochs = _pick_scatter_epochs(all_epochs)
    _log(f"Scatter plot epochs: {scatter_epochs}")

    for ep in scatter_epochs:
        epoch_lf = pred_lf.filter(pl.col("epoch") == ep)
        epoch_df = epoch_lf.select([
            "qi_mean", "qi_var", "qbg_mean", "intensity", "background",
        ]).collect()

        def _plot_scatter_i(df=epoch_df, e=ep):
            fig = plot_scatter(
                df, "qi_mean", "intensity",
                xlabel="estimated intensity", ylabel="true intensity",
                symlog=True, xlim=(0, 1e4),
            )
            fig.suptitle(f"epoch {e}", fontsize=9)
            wandb.log({f"scatter/I_vs_hatI_epoch{e}": wandb.Image(fig)})
            plt.close(fig)
        _try_plot(f"scatter/I_epoch{ep}", _plot_scatter_i)

        def _plot_scatter_bg(df=epoch_df, e=ep):
            fig = plot_scatter(
                df, "qbg_mean", "background",
                xlabel="estimated background", ylabel="true background",
            )
            fig.suptitle(f"epoch {e}", fontsize=9)
            wandb.log({f"scatter/bg_vs_hatbg_epoch{e}": wandb.Image(fig)})
            plt.close(fig)
        _try_plot(f"scatter/bg_epoch{ep}", _plot_scatter_bg)

        def _plot_ns(df=epoch_df, e=ep, cr=crlb_data):
            fig = plot_noise_to_signal(df, crlb=cr)
            fig.suptitle(f"epoch {e}", fontsize=9)
            wandb.log({f"scatter/noise_to_signal_epoch{e}": wandb.Image(fig)})
            plt.close(fig)
        _try_plot(f"scatter/noise_to_signal_epoch{ep}", _plot_ns)

    # 5. Final-epoch summary
    final_metrics = error_df.filter(pl.col("epoch") == last_epoch)
    if len(final_metrics) > 0:
        summary_dict = {
            "final/epoch": last_epoch,
            "final/mae": float(final_metrics["mae"].mean()),
            "final/rmse": float(final_metrics["rmse"].mean()),
            "final/mre": float(final_metrics["mre"].mean()),
            "final/bias": float(final_metrics["bias"].mean()),
            "final/corr_i": float(final_metrics["corr_i"].mean()),
            "final/corr_bg": float(final_metrics["corr_bg"].mean()),
        }
        wandb.log(summary_dict)

        _log("")
        _log("=== Final epoch summary (epoch %d) ===" % last_epoch)
        for k, v in summary_dict.items():
            _log(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # 6. SBC (optional)
    if args.sbc:
        _log(f"Running SBC with {args.sbc_nsamples} samples, K={args.sbc_K}...")
        try:
            df_sbc = run_sbc(
                config=config,
                wandb_dir=wandb_dir,
                data_dir=data_dir,
                nsamples=args.sbc_nsamples,
                K=args.sbc_K,
                intensity_edges=intensity_edges,
                bin_labels=bin_labels,
                checkpoint_epoch=last_epoch,
            )
            if len(df_sbc) > 0:
                fig = plot_sbc_ranks(df_sbc, K=args.sbc_K)
                wandb.log({"sbc/rank_histograms": wandb.Image(fig)})
                plt.close(fig)
                n_plots_ok += 1

                # Log rank uniformity p-value
                from scipy import stats as sp_stats
                ranks_np = df_sbc["rank"].to_numpy()
                _, p_value = sp_stats.kstest(
                    ranks_np, "uniform", args=(0, args.sbc_K + 1)
                )
                wandb.log({"sbc/ks_pvalue": p_value})
                _log(f"SBC KS p-value: {p_value:.4f}")
            else:
                _log("SBC returned empty DataFrame")
        except Exception as e:
            n_plots_fail += 1
            _log(f"SBC failed: {e}")

    _log("")
    _log(f"Plots: {n_plots_ok} succeeded, {n_plots_fail} failed")
    _log("Analysis complete!")

    # Write summary to run_dir
    _write_summary(run_dir, summary_lines)

    wandb.finish()


def _pick_scatter_epochs(all_epochs: list[int], max_scatter: int = 5) -> list[int]:
    """Pick a few representative epochs for scatter plots."""
    if len(all_epochs) <= max_scatter:
        return all_epochs
    # first, last, and evenly spaced in between
    indices = [0]
    step = (len(all_epochs) - 1) / (max_scatter - 1)
    for i in range(1, max_scatter - 1):
        indices.append(round(i * step))
    indices.append(len(all_epochs) - 1)
    return sorted(set(all_epochs[i] for i in indices))


def _write_summary(run_dir: "Path", lines: list[str]) -> None:
    """Write analysis summary text file to run_dir."""
    from datetime import datetime

    summary_path = run_dir / "analysis_summary.txt"
    header = f"Analysis run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary_path.write_text(header + "\n".join(lines) + "\n")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
