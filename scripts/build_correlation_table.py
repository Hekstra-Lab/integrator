"""Build a per-epoch table of training-time metrics + downstream metrics,
and compute correlations.

Parsing conventions match `/Users/luis/refltorch/scripts/dials_output/
{process_out.py, compare_models.py}` (cchalf/ccanom asterisk stripping,
phenix Start/Final R-work regex, peaks.csv columns seqid/residue/peakz).

Training metrics are loaded via the wandb API — same pattern as refltorch
`wandb.Api().run(...).history()`. `epoch` is already a logged column
(from base_integrator `_log_loss`), so no step/epoch arithmetic is needed.

Inputs:
  --run-dir:    wandb run directory containing:
                  predictions/epoch_XXXX/dials/merged.html
                  predictions/epoch_XXXX/dials/phenix_out/refine_001.log
                  predictions/epoch_XXXX/dials/phenix_out/peaks.csv
  --wandb-run:  e.g. "laldama/wilson_loss/8s91or4i". Runs wandb.Api to
                pull the full history (all logged scalars).
  --wandb-csv:  alternate path: one or more wandb CSV exports.

Outputs (--out-dir):
  per_epoch.csv     one row per epoch, all columns
  pearson.csv       cross-correlation: rows = train metrics, cols = downstream
  spearman.csv      same, Spearman
  top_pairs.csv     train × downstream pairs ranked by |Spearman|
  top_scatter.png   scatter of top 6 pairs, labeled by epoch

Usage (wandb API):
  uv run python scripts/build_correlation_table.py \\
      --run-dir /n/.../run-20260418_175258-8s91or4i \\
      --wandb-run laldama/<project>/8s91or4i \\
      --out-dir ./correlations

Or with CSV exports:
  ... --wandb-csv export1.csv export2.csv ...
"""

import argparse
import re
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MERGED_KEYS = (
    "resolution",
    "n_refls",
    "n_unique",
    "multiplicity",
    "completeness",
    "mean_i",
    "meani_sigi",
    "rmerge",
    "rmeas",
    "rpim",
    "ranom",
    "cchalf",
    "ccanom",
)


def parse_merged_html(path: Path) -> dict:
    """Extract per-shell stats + overall row from DIALS merged.html."""
    tbls = pd.read_html(path)
    tbl = tbls[1]  # refltorch uses df[1]
    if len(tbl.columns) != len(MERGED_KEYS):
        raise RuntimeError(
            f"{path} has {len(tbl.columns)} cols, expected {len(MERGED_KEYS)}"
        )
    tbl.columns = list(MERGED_KEYS)
    # Strip asterisks on cchalf/ccanom (DIALS marks significant bins)
    for col in ("cchalf", "ccanom"):
        tbl[col] = tbl[col].apply(lambda v: float(str(v).strip("*")))
    # Last row is "Overall" — keep as scalar metrics
    overall = tbl.iloc[-1]
    out = {
        "cchalf_overall": float(overall["cchalf"]),
        "ccanom_overall": float(overall["ccanom"]),
        "rmeas_overall": float(overall["rmeas"]),
        "rpim_overall": float(overall["rpim"]),
        "ranom_overall": float(overall["ranom"]),
        "meani_sigi_overall": float(overall["meani_sigi"]),
        "completeness_overall": float(overall["completeness"]),
        "multiplicity_overall": float(overall["multiplicity"]),
    }
    # Also: high-res shell (first data row) ccanom — often the signal bottleneck
    inner = tbl.iloc[0]
    out["cchalf_inner"] = float(inner["cchalf"])
    out["ccanom_inner"] = float(inner["ccanom"])
    return out


def parse_phenix_log(path: Path) -> dict:
    """Return {r_work_start, r_free_start, r_work_final, r_free_final}."""
    text = path.read_text().splitlines()
    out: dict[str, float] = {}
    start_re = re.compile(r"Start R-work")
    final_re = re.compile(r"Final R-work")
    float_re = re.compile(r"\d\.\d+")
    for line in text:
        if start_re.search(line):
            fl = float_re.findall(line)
            if len(fl) >= 2:
                out["r_work_start"] = float(fl[0])
                out["r_free_start"] = float(fl[1])
        if final_re.search(line):
            fl = float_re.findall(line)
            if len(fl) >= 2:
                out["r_work_final"] = float(fl[0])
                out["r_free_final"] = float(fl[1])
    return out


def parse_peaks_csv(path: Path) -> dict:
    """Reduce peaks.csv to scalar summary statistics."""
    df = pd.read_csv(path)[["seqid", "residue", "peakz"]]
    df = df.sort_values("seqid")  # type: ignore[call-overload]
    return {
        "peakz_total": float(df["peakz"].sum()),
        "peakz_mean": float(df["peakz"].mean()),
        "peakz_max": float(df["peakz"].max()),
        "peakz_min": float(df["peakz"].min()),
        "peakz_n_atoms": int(len(df)),
    }


def parse_epoch_dir(epoch_dir: Path) -> dict | None:
    m = re.search(r"epoch_(\d+)", epoch_dir.name)
    if not m:
        return None
    epoch = int(m.group(1))

    row: dict = {"epoch": epoch}
    merged = epoch_dir / "dials" / "merged.html"
    phenix = epoch_dir / "dials" / "phenix_out" / "refine_001.log"
    peaks = epoch_dir / "dials" / "phenix_out" / "peaks.csv"

    if merged.is_file():
        try:
            row.update(parse_merged_html(merged))
        except Exception as e:
            print(f"[{epoch_dir.name}] merged.html parse failed: {e}")

    if phenix.is_file():
        try:
            row.update(parse_phenix_log(phenix))
        except Exception as e:
            print(f"[{epoch_dir.name}] phenix log parse failed: {e}")

    if peaks.is_file():
        try:
            row.update(parse_peaks_csv(peaks))
        except Exception as e:
            print(f"[{epoch_dir.name}] peaks.csv parse failed: {e}")

    return row if len(row) > 1 else None


def _to_per_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a step-indexed wandb history to one row per epoch.

    Expects a DataFrame containing a logged `epoch` column (as
    base_integrator's _log_loss emits via self.log('epoch', ...)).
    Numeric columns are reduced with .last() per epoch (end-of-epoch
    value), mirroring refltorch's drop_nulls + per-epoch selection.
    """
    if "epoch" not in df.columns:
        raise RuntimeError(
            "wandb history has no 'epoch' column — verify _log_loss is "
            "logging epoch. Falling back to CSV path requires it too."
        )
    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(int)
    # Drop non-numeric/bookkeeping columns that break .last() semantics
    drop_like = (
        "trainer/",
        "_runtime",
        "_timestamp",
        "_step",
        "_wandb",
        "system/",
        "parameters/",
    )
    keep = [
        c
        for c in df.columns
        if c == "epoch"
        or not any(c.startswith(p) or p in c for p in drop_like)
    ]
    df = df[keep]
    per_epoch = df.groupby("epoch", as_index=False).last()
    return per_epoch


def load_wandb_api(run_path: str) -> pd.DataFrame:
    """Pull full history from wandb.Api() — same pattern as refltorch.

    `run_path` is the standard wandb identifier "entity/project/run_id".
    """
    import wandb  # lazy import so --wandb-csv doesn't require wandb

    hist = wandb.Api().run(run_path).history(samples=100_000)
    return _to_per_epoch(pd.DataFrame(hist))


def load_wandb_csvs(paths: list[Path]) -> pd.DataFrame:
    """Merge one or more wandb CSV exports → per-epoch DataFrame.

    Requires the exports to include an `epoch` column. If only the
    bookkeeping `__MIN`/`__MAX`/` - _step` variants are present, they
    are stripped. 'RUN - metric' column names are de-prefixed.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        # Drop noise columns
        keep = [
            c
            for c in df.columns
            if not c.endswith("__MIN")
            and not c.endswith("__MAX")
            and not c.endswith(" - _step")
            and not c.endswith(" - _step__MIN")
            and not c.endswith(" - _step__MAX")
        ]
        df = df[keep]
        # Strip 'RUN - ' prefix from column labels
        rename_map = {
            c: (c.split(" - ", 1)[1] if " - " in c else c) for c in df.columns
        }
        df.columns = [rename_map[c] for c in df.columns]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged: pd.DataFrame = frames[0]
    for f in frames[1:]:
        on = (
            "trainer/global_step"
            if "trainer/global_step" in merged.columns
            else ("epoch" if "epoch" in merged.columns else None)
        )
        if on is None:
            raise RuntimeError(
                "No common key (trainer/global_step or epoch) in CSVs"
            )
        merged = merged.merge(f, on=on, how="outer")
    return _to_per_epoch(merged)


def align_on_epoch(
    downstream: pd.DataFrame, training: pd.DataFrame
) -> pd.DataFrame:
    if training.empty:
        return downstream
    # Downstream has a few epochs (every 5); training has more.
    # Outer join on epoch, then keep only rows with at least one downstream col.
    merged = downstream.merge(training, on="epoch", how="outer")
    merged = merged.sort_values("epoch").reset_index(drop=True)
    return merged


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Heuristic split: anything named like a downstream key stays in the
    downstream bucket; everything else is training-time."""
    downstream_tokens = (
        "cchalf",
        "ccanom",
        "rmeas",
        "rpim",
        "ranom",
        "meani_sigi",
        "completeness",
        "multiplicity",
        "r_work",
        "r_free",
        "peakz",
    )
    ds = [
        c
        for c in df.columns
        if c != "epoch" and any(t in c for t in downstream_tokens)
    ]
    tr = [c for c in df.columns if c != "epoch" and c not in ds]
    return tr, ds


def compute_corr(
    df: pd.DataFrame,
    tr_cols: list[str],
    ds_cols: list[str],
    method: Literal["pearson", "spearman"],
) -> pd.DataFrame:
    """Cross-correlation: rows = training metric, cols = downstream metric."""
    sub = df.dropna(subset=tr_cols + ds_cols, how="all")
    out = pd.DataFrame(
        np.full((len(tr_cols), len(ds_cols)), np.nan),
        index=pd.Index(tr_cols),
        columns=pd.Index(ds_cols),
    )
    for t in tr_cols:
        for d in ds_cols:
            pair = sub[[t, d]].dropna()
            if len(pair) < 3:
                continue
            tt = cast(pd.Series, pair[t])
            dd = cast(pd.Series, pair[d])
            out.loc[t, d] = tt.corr(dd, method=method)
    return out


def plot_top_scatter(
    df: pd.DataFrame, top_pairs: pd.DataFrame, out_path: Path, n: int = 6
):
    top = top_pairs.head(n)
    ncols = 3
    nrows = (len(top) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.2))
    axes = np.atleast_2d(axes).flatten()
    for i, (_, row) in enumerate(top.iterrows()):
        t, d = row["train_metric"], row["downstream_metric"]
        ax = axes[i]
        pair = df[[t, d, "epoch"]].dropna()
        ax.scatter(pair[t], pair[d], c=pair["epoch"], cmap="viridis", s=40)
        for _, pt in pair.iterrows():
            ax.annotate(
                f"{int(pt['epoch'])}",
                (pt[t], pt[d]),
                fontsize=6,
                alpha=0.7,
            )
        ax.set_xlabel(t, fontsize=8)
        ax.set_ylabel(d, fontsize=8)
        ax.set_title(
            f"r_P={row['pearson']:+.3f}  r_S={row['spearman']:+.3f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)
    for j in range(len(top), len(axes)):
        axes[j].set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="wandb run dir; the script looks for {run_dir}/predictions/epoch_* "
        "unless --predictions-dir is given.",
    )
    p.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Override: directory directly containing epoch_* subdirs.",
    )
    p.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        help="wandb 'entity/project/run_id' — pulls history via wandb.Api().",
    )
    p.add_argument(
        "--wandb-csv",
        type=Path,
        nargs="*",
        default=[],
        help="Alternate: wandb CSV exports (must include an 'epoch' column).",
    )
    p.add_argument("--out-dir", type=Path, default=Path("correlations"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    preds_dir = args.predictions_dir or (args.run_dir / "predictions")
    epoch_dirs = sorted(preds_dir.glob("epoch_*"))
    if not epoch_dirs:
        raise SystemExit(
            f"No epoch_* directories found under {preds_dir}. "
            f"Pass --predictions-dir pointing directly at the directory "
            f"that contains the epoch_XXXX folders."
        )
    rows = [r for r in (parse_epoch_dir(d) for d in epoch_dirs) if r]
    if not rows:
        raise SystemExit(
            f"Found {len(epoch_dirs)} epoch_* dirs under {preds_dir}, but "
            f"none contained parseable merged.html / refine_001.log / "
            f"peaks.csv. Check the directory layout."
        )
    downstream = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)  # type: ignore[call-overload]
    print(
        f"Downstream rows: {len(downstream)}, cols: {list(downstream.columns)}"
    )

    if args.wandb_run:
        training = load_wandb_api(args.wandb_run)
    elif args.wandb_csv:
        training = load_wandb_csvs(list(args.wandb_csv))
    else:
        training = pd.DataFrame()
    if not training.empty:
        print(
            f"Training rows: {len(training)}, cols: {list(training.columns)}"
        )
    else:
        print("No wandb CSVs provided — writing downstream-only table.")

    merged = align_on_epoch(downstream, training)
    merged.to_csv(args.out_dir / "per_epoch.csv", index=False)
    print(f"Wrote {args.out_dir / 'per_epoch.csv'}  ({len(merged)} rows)")

    if training.empty:
        return

    tr_cols, ds_cols = split_columns(merged)
    print(f"training cols: {tr_cols}")
    print(f"downstream cols: {ds_cols}")

    pear = compute_corr(merged, tr_cols, ds_cols, method="pearson")
    spea = compute_corr(merged, tr_cols, ds_cols, method="spearman")
    pear.to_csv(args.out_dir / "pearson.csv")
    spea.to_csv(args.out_dir / "spearman.csv")

    # Top pairs ranked by |spearman| (robust to outliers)
    pairs = []
    for t in tr_cols:
        for d in ds_cols:
            rp = pear.loc[t, d]
            rs = spea.loc[t, d]
            if pd.isna(rp) and pd.isna(rs):
                continue
            pairs.append(
                {
                    "train_metric": t,
                    "downstream_metric": d,
                    "pearson": rp,
                    "spearman": rs,
                    "abs_spearman": abs(rs) if not pd.isna(rs) else 0.0,
                }
            )
    ranked = (
        pd.DataFrame(pairs)
        .sort_values("abs_spearman", ascending=False)
        .reset_index(drop=True)
    )
    ranked.to_csv(args.out_dir / "top_pairs.csv", index=False)
    print("Wrote top_pairs.csv (first 10):")
    print(ranked.head(10).to_string(index=False))

    plot_top_scatter(merged, ranked, args.out_dir / "top_scatter.png")
    print("Wrote top_scatter.png")


if __name__ == "__main__":
    main()
