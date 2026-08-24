"""Aggregate the NB dispersion screen and plot held-out fit vs dispersion r.

Reads each arm's per-epoch `loss_history.csv` (written by `LossCurveLogger` to
`<run_dir>/plots/`), builds a summary table, and plots the held-out
reconstruction NLL and total ELBO against the fixed dispersion r.

The screen arms are laid out one directory per arm under `--runs-dir` (the OUT
of `dispersion_screen.slurm`):
    nb_screen/poisson/      r -> inf   (baseline)
    nb_screen/nb_r0p5/ ...  fixed r
    nb_screen/nb_learned/   r learned

Held-out NLL is directly comparable across Poisson and Negative Binomial: it is
the negative log-likelihood of the held-out counts, normalizing constants
included, so lower = better predictive fit. Poisson is drawn as a horizontal
reference line (r -> inf) and the learned-r arm as a separate marker (its
converged r is read from the checkpoint when available).

Usage:
    uv run python scripts/ablation_likelihood/plot_dispersion_screen.py \
        --runs-dir /path/to/nb_screen --out /path/to/nb_screen
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

_POISSON = "poisson"
_LEARNED = "nb_learned"
_DISPERSION_FLOOR = 1e-3  # matches CountLikelihood default nb_dispersion_floor


def _tag_to_r(tag: str) -> float | None:
    """Map an arm directory name to its dispersion r.

    `poisson` -> inf, `nb_rXpY` -> X.Y, `nb_learned` -> None (read elsewhere).
    """
    if tag == _POISSON:
        return math.inf
    if tag == _LEARNED:
        return None
    if tag.startswith("nb_r"):
        return float(tag[len("nb_r") :].replace("p", "."))
    return None


def _find_loss_history(arm_dir: Path) -> Path | None:
    """Locate loss_history.csv under an arm dir (handles W&B and local layouts)."""
    direct = arm_dir / "plots" / "loss_history.csv"
    if direct.exists():
        return direct
    hits = sorted(arm_dir.glob("**/loss_history.csv"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def _learned_r_from_checkpoint(arm_dir: Path) -> float | None:
    """Best-effort read of the converged dispersion from a checkpoint.

    softplus(raw_dispersion) + floor, averaged if per-bin. Returns None if torch
    or the checkpoint is unavailable.
    """
    ckpts = sorted(arm_dir.glob("**/checkpoints/*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        return None
    try:
        import torch
        import torch.nn.functional as F

        sd = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        state = sd.get("state_dict", sd)
        key = next((k for k in state if k.endswith("raw_dispersion")), None)
        if key is None:
            return None
        r = F.softplus(state[key].float()) + _DISPERSION_FLOOR
        return float(r.mean())
    except Exception:
        return None


def _summarize_arm(tag: str, arm_dir: Path) -> dict | None:
    """Return one summary row (best/final held-out metrics) for an arm."""
    csv = _find_loss_history(arm_dir)
    if csv is None:
        return None
    df = pl.read_csv(csv)
    if "val_loss" not in df.columns or df.height == 0:
        return None
    # validation may run every N epochs, so the last row can lack a val value;
    # take the last epoch that actually has one.
    val_rows = df.filter(pl.col("val_loss").is_not_null()).sort("epoch")
    final = (val_rows if val_rows.height else df.sort("epoch")).tail(1).to_dicts()[0]
    best_val_loss = df.select(pl.col("val_loss").min()).item()
    best_val_nll = (
        df.select(pl.col("val_nll").min()).item() if "val_nll" in df.columns else None
    )
    r = _tag_to_r(tag)
    if tag == _LEARNED:
        r = _learned_r_from_checkpoint(arm_dir)
    return {
        "tag": tag,
        "r": r,
        "kind": _POISSON if tag == _POISSON else (_LEARNED if tag == _LEARNED else "nb_fixed"),
        "epochs": int(final["epoch"]) + 1,
        "final_val_loss": final.get("val_loss"),
        "best_val_loss": best_val_loss,
        "final_val_nll": final.get("val_nll"),
        "best_val_nll": best_val_nll,
        "final_val_kl_i": final.get("val_kl_i"),
    }


def _plot(summary: pl.DataFrame, out_dir: Path) -> Path:
    """Two panels: held-out NLL and total ELBO vs fixed dispersion r."""
    fixed = summary.filter(pl.col("kind") == "nb_fixed").sort("r")
    pois = summary.filter(pl.col("kind") == _POISSON)
    learned = summary.filter(pl.col("kind") == _LEARNED)

    panels = [
        ("best_val_nll", "held-out NLL (reconstruction)"),
        ("best_val_loss", "held-out ELBO (total loss)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=120)
    for ax, (col, ylabel) in zip(axes, panels):
        if fixed.height and fixed[col].null_count() < fixed.height:
            ax.plot(fixed["r"], fixed[col], "o-", color="C0", label="NB, fixed r", zorder=3)
        if pois.height and pois[col][0] is not None:
            ax.axhline(pois[col][0], ls="--", color="C3", label="Poisson (r=∞)")
        if learned.height and learned[col][0] is not None:
            lr = learned["r"][0]
            if lr is not None:
                ax.plot([lr], [learned[col][0]], "*", ms=13, color="C2",
                        label=f"NB, learned (r={lr:.2g})", zorder=4)
            else:
                ax.axhline(learned[col][0], ls=":", color="C2", label="NB, learned")
        ax.set_xscale("log")
        ax.set_xlabel("dispersion  r")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
    fig.suptitle("Negative-Binomial dispersion screen (lower is better)")
    fig.tight_layout()
    out = out_dir / "dispersion_screen.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, help="OUT dir of the screen (one subdir per arm).")
    parser.add_argument("--out", default=None, help="Where to write outputs (default: --runs-dir).")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out) if args.out else runs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for arm_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        row = _summarize_arm(arm_dir.name, arm_dir)
        if row is None:
            print(f"  skip {arm_dir.name}: no loss_history.csv yet")
            continue
        rows.append(row)

    if not rows:
        raise SystemExit(f"No completed arms found under {runs_dir}")

    summary = pl.DataFrame(rows).sort(["kind", "r"])
    summary_path = out_dir / "dispersion_screen_summary.csv"
    summary.write_csv(summary_path)
    fig_path = _plot(summary, out_dir)

    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(summary)
    print(f"\nwrote {summary_path}\nwrote {fig_path}")


if __name__ == "__main__":
    main()
