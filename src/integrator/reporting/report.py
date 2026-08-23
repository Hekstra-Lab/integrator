"""Build a self-contained HTML report from training run artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load_loss_traces(loss_trace_dir: Path) -> pd.DataFrame:
    """Read all loss_trace_*.parquet files and return a single DataFrame."""
    files = sorted(loss_trace_dir.glob("loss_trace_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No loss_trace_*.parquet files in {loss_trace_dir}"
        )
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if "split" not in df.columns:
            df["split"] = "train" if "train" in f.name else "val"
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _make_loss_figure(df: pd.DataFrame) -> go.Figure:
    """Per-epoch loss curves: total ELBO and its components."""
    components = ["loss", "nll", "kl"]
    labels = {"loss": "ELBO", "nll": "NLL", "kl": "KL"}
    colors = {"loss": "#1f77b4", "nll": "#ff7f0e", "kl": "#2ca02c"}

    fig = make_subplots(
        rows=1,
        cols=1,
        x_title="Epoch",
        y_title="Loss",
    )

    for split in ("train", "val"):
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        epoch_means = sub.groupby("epoch")[components].mean()

        dash = None if split == "train" else "dash"
        show_legend_prefix = "" if split == "train" else "val "

        for col in components:
            if col not in epoch_means.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=epoch_means.index,
                    y=epoch_means[col],
                    mode="lines",
                    name=f"{show_legend_prefix}{labels.get(col, col)}",
                    line=dict(color=colors.get(col), dash=dash),
                )
            )

    fig.update_layout(
        title="Loss Curves",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=450,
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Integrator Report</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    max-width: 960px;
    margin: 2rem auto;
    padding: 0 1rem;
    color: #222;
  }}
  h1 {{ font-size: 1.5rem; }}
  .plot-container {{ margin: 1.5rem 0; }}
</style>
</head>
<body>
<h1>Integrator Report</h1>
<div class="plot-container">
{loss_plot}
</div>
</body>
</html>
"""


def build_report(loss_trace_dir: Path) -> str:
    """Return a self-contained HTML string with the training report."""
    df = _load_loss_traces(loss_trace_dir)
    fig = _make_loss_figure(df)
    loss_plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    return _HTML_TEMPLATE.format(loss_plot=loss_plot_html)
