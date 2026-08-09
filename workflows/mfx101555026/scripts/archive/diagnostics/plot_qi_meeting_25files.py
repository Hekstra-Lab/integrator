#!/usr/bin/env python3
"""
Minimal meeting plots for Gamma, LogNormal, and FoldedNormal qi comparison.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUNS_ROOT = Path(
    "/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/"
    "mfx101555026_cctbx/runs"
)

RUNS = {
    "Gamma": RUNS_ROOT / "mfx_gamma_qi_25files",
    "LogNormal": RUNS_ROOT / "mfx_lognormal_qi_25files_mc_kl",
    "FoldedNormal": RUNS_ROOT / "mfx_foldednormal_qi_25files_mc_kl",
}

OUT_DIR = RUNS_ROOT / "diagnostics_qi_meeting_25files"


def find_prediction_file(run_dir: Path) -> Path:
    files = list(run_dir.rglob("test_preds_all.parquet"))
    if files:
        return sorted(files)[-1]

    files = list(run_dir.rglob("pred.parquet"))
    if files:
        return sorted(files)[-1]

    raise FileNotFoundError(f"No prediction parquet found under {run_dir}")


def load_all() -> dict[str, pd.DataFrame]:
    dfs = {}

    for name, run_dir in RUNS.items():
        path = find_prediction_file(run_dir)
        print(f"{name}: {path}")
        dfs[name] = pd.read_parquet(path)

    return dfs


def plot_qi_mean_hist(dfs: dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(8, 5))

    for name, df in dfs.items():
        x = pd.to_numeric(df["qi_mean"], errors="coerce").dropna()
        x = x[x > 0]
        plt.hist(x, bins=80, alpha=0.5, label=name)

    plt.xscale("log")
    plt.xlabel("qi_mean")
    plt.ylabel("count")
    plt.title("Predicted intensity mean: Gamma vs LogNormal vs FoldedNormal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "meeting_01_qi_mean_hist_logx.png", dpi=200)
    plt.close()


def plot_qi_var_hist(dfs: dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(8, 5))

    for name, df in dfs.items():
        x = pd.to_numeric(df["qi_var"], errors="coerce").dropna()
        x = x[x > 0]
        plt.hist(x, bins=80, alpha=0.5, label=name)

    plt.xscale("log")
    plt.xlabel("qi_var")
    plt.ylabel("count")
    plt.title("Predicted intensity variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "meeting_02_qi_var_hist_logx.png", dpi=200)
    plt.close()


def plot_qi_vs_integrated_intensity(dfs: dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(7, 6))

    for name, df in dfs.items():
        x = pd.to_numeric(df["intensity.sum.value"], errors="coerce")
        y = pd.to_numeric(df["qi_mean"], errors="coerce")

        mask = x.notna() & y.notna() & (y > 0)
        plt.scatter(x[mask], y[mask], s=4, alpha=0.25, label=name)

    plt.yscale("log")
    plt.xlabel("integrated intensity.sum.value")
    plt.ylabel("qi_mean")
    plt.title("qi_mean vs integrated intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "meeting_03_qi_mean_vs_integrated_intensity.png", dpi=200)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dfs = load_all()

    plot_qi_mean_hist(dfs)
    plot_qi_var_hist(dfs)
    plot_qi_vs_integrated_intensity(dfs)

    print(f"\nWrote meeting plots to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()