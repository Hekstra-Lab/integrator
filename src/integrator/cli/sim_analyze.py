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
) -> "plt.Figure":
    """Noise-to-signal ratio plot comparing model posterior to Poisson limit."""
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
    ax.scatter(x_model, y_model, alpha=0.2, s=3, rasterized=True)
    ax.axhline(1, linestyle="dotted", color="red", label=r"signal $=$ noise")

    ax.set_yscale("log")
    ax.set_xscale("log")
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

    from integrator.utils import construct_integrator, load_config

    # Load priors from reference data
    reference = torch.load(data_dir / "reference.pt", weights_only=False)
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

    # Run inference in batches
    batch_size = 512
    all_concentrations = []
    all_rates = []

    with torch.no_grad():
        for start in range(0, nsamples, batch_size):
            end = min(start + batch_size, nsamples)
            batch = shoeboxes_std[start:end]
            x_k = integrator.encoders["k"](batch)
            x_r = integrator.encoders["r"](batch)
            qi = integrator.surrogates["qi"](x_k, x_r)
            all_concentrations.append(qi.concentration.cpu())
            all_rates.append(qi.rate.cpu())

    qi_conc = torch.cat(all_concentrations)
    qi_rate = torch.cat(all_rates)

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
    import os
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

    logger.info("Starting analysis")

    # --- Load run metadata ---
    run_dir = Path(args.run_dir)
    meta = yaml.safe_load((run_dir / "run_metadata.yaml").read_text())
    config = load_config(meta["config"])
    wandb_info = meta["wandb"]

    log_dir = Path(wandb_info["log_dir"])
    wandb_dir = log_dir.parent
    data_dir = Path(config["data_loader"]["args"]["data_dir"])

    logger.info("W&B log dir: %s", log_dir)
    logger.info("Data dir: %s", data_dir)

    # --- Load reference data ---
    reference = torch.load(data_dir / "reference.pt", weights_only=False)
    I_true = torch.tensor(reference["intensity"])

    # --- Compute intensity bin edges ---
    intensity_edges = _compute_intensity_edges(I_true)
    bin_labels, base_df = _get_bins(intensity_edges)
    logger.info("Intensity edges: %s", intensity_edges)

    # --- Initialize W&B analysis run ---
    wb_project = args.wb_project or wandb_info.get("project", "sim-analysis")
    run = wandb.init(
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
        logger.error("No prediction parquet files found in %s", pred_dir)
        wandb.finish()
        return

    logger.info("Found %d prediction files", len(pred_paths))

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
        logger.info("Filtered to %d files for epochs %s", len(pred_paths), args.epochs)

    pred_lf = pl.scan_parquet([str(p) for p in pred_paths])

    # --- Compute error metrics ---
    logger.info("Computing error metrics...")
    error_df = compute_error_metrics(pred_lf, intensity_edges, bin_labels, base_df)

    # --- Compute loss curves ---
    logger.info("Computing loss curves...")
    train_loss_df, val_loss_df = compute_loss_curves(wandb_dir)

    # --- Generate and log plots ---

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
        try:
            fig = plot_error_metric_evolution(
                error_df,
                metric_name,
                ylabel=metric_cfg.get("ylabel"),
                yscale=metric_cfg.get("yscale", "linear"),
                ylim=metric_cfg.get("ylim"),
            )
            wandb.log({f"metrics/{metric_name}": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to plot %s: %s", metric_name, e)

    # 2. Loss curves
    for component in ["loss", "nll", "kl"]:
        try:
            fig = plot_loss_curves(train_loss_df, val_loss_df, component)
            wandb.log({f"loss/{component}": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to plot loss/%s: %s", component, e)

    # 3. Loss gap
    if train_loss_df is not None and val_loss_df is not None:
        try:
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
        except Exception as e:
            logger.warning("Failed to plot loss gaps: %s", e)

    # 4. Scatter plots for last available epoch
    all_epochs = pred_lf.select("epoch").unique().collect()["epoch"].to_list()
    last_epoch = max(all_epochs) if all_epochs else None

    if last_epoch is not None:
        epoch_lf = pred_lf.filter(pl.col("epoch") == last_epoch)
        epoch_df = epoch_lf.select([
            "qi_mean", "qi_var", "qbg_mean", "intensity", "background",
        ]).collect()

        # I vs hat I
        try:
            fig = plot_scatter(
                epoch_df, "qi_mean", "intensity",
                xlabel="estimated intensity", ylabel="true intensity",
                symlog=True, xlim=(0, 1e4),
            )
            fig.suptitle(f"epoch {last_epoch}", fontsize=9)
            wandb.log({f"scatter/I_vs_hatI_epoch{last_epoch}": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed I vs hat I scatter: %s", e)

        # bg vs hat bg
        try:
            fig = plot_scatter(
                epoch_df, "qbg_mean", "background",
                xlabel="estimated background", ylabel="true background",
            )
            fig.suptitle(f"epoch {last_epoch}", fontsize=9)
            wandb.log({f"scatter/bg_vs_hatbg_epoch{last_epoch}": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed bg vs hat bg scatter: %s", e)

        # Noise-to-signal
        try:
            fig = plot_noise_to_signal(epoch_df)
            fig.suptitle(f"epoch {last_epoch}", fontsize=9)
            wandb.log({f"scatter/noise_to_signal_epoch{last_epoch}": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed noise-to-signal plot: %s", e)

    # 5. Log summary table of final-epoch metrics
    if last_epoch is not None:
        final_metrics = error_df.filter(pl.col("epoch") == last_epoch)
        if len(final_metrics) > 0:
            summary = {
                "final_epoch": last_epoch,
                "global_mae": final_metrics["mae"].mean(),
                "global_rmse": final_metrics["rmse"].mean(),
                "global_mre": final_metrics["mre"].mean(),
                "global_corr_i": final_metrics["corr_i"].mean(),
                "global_corr_bg": final_metrics["corr_bg"].mean(),
            }
            wandb.log(summary)
            logger.info("Final-epoch summary: %s", summary)

    # 6. SBC (optional)
    if args.sbc:
        logger.info("Running SBC with %d samples, K=%d...", args.sbc_nsamples, args.sbc_K)
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

                # Log rank uniformity p-value
                from scipy import stats as sp_stats
                ranks_np = df_sbc["rank"].to_numpy()
                _, p_value = sp_stats.kstest(
                    ranks_np, "uniform", args=(0, args.sbc_K + 1)
                )
                wandb.log({"sbc/ks_pvalue": p_value})
                logger.info("SBC KS p-value: %.4f", p_value)
        except Exception as e:
            logger.warning("SBC failed: %s", e)

    wandb.finish()
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
