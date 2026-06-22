"""Model-vs-DIALS plots for a finished merging run.

Standalone post-processor. Makes the four comparison plots, pulling each plot's
inputs from where the eval / prediction wrote them:

    intensity scatter   model qi_mean   vs DIALS intensity.prf.value  (per-obs,
    background scatter   model qbg_mean  vs DIALS background.mean       log/linear
                         <- predictions parquet (integrator.predict)
    R-factors vs epoch   r_work / r_free per checkpoint
                         <- ckpt_eval/epoch*/result.json
    anomalous peakz      per-residue peakz over epoch, with the DIALS reference
      vs epoch           <- ckpt_eval/epoch*/<variant>/peaks.csv (rs.find_peaks)
                            + a reference peaks.csv (--ref-peaks)

Decoupled from the submission / worker scripts so plots can be (re)made without
rerunning phenix. Usage:

    python plot_ckpt_eval.py --run-dir RUN_DIR --ref-peaks reference_peaks.csv
    python plot_ckpt_eval.py --ckpt-eval-dir /path/ckpt_eval
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

import merging_plots as mp

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger(__name__)


def resolve_ckpt_eval(run_dir: Path, override: Path | None) -> Path:
    """Where the per-checkpoint workers wrote (netscratch for a W&B run).

    override > eval config out_root > run_paths output_root/ckpt_eval > run_dir.
    """
    if override:
        return Path(override)
    cfg_file = run_dir / "merging_eval_cfg.yaml"
    if cfg_file.exists():
        c = yaml.safe_load(cfg_file.read_text()) or {}
        if c.get("out_root"):
            return Path(c["out_root"])
    meta_path = run_dir / "run_paths.yaml"
    if meta_path.exists():
        m = yaml.safe_load(meta_path.read_text()) or {}
        if m.get("output_root"):
            return Path(m["output_root"]) / "ckpt_eval"
    return run_dir / "ckpt_eval"


def load_results(ckpt_eval: Path) -> tuple[list[dict], list[dict]]:
    """Load every epoch*/result.json -> (rows per (epoch, variant), raw dicts)."""
    rows, results = [], []
    for rj in sorted(ckpt_eval.glob("epoch*/result.json")):
        try:
            res = json.loads(rj.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip %s: %s", rj, exc)
            continue
        res["_dir"] = str(rj.parent)
        results.append(res)
        for vname, v in (res.get("variants") or {}).items():
            rows.append(
                {
                    "epoch": res.get("epoch"),
                    "variant": vname,
                    "r_work": v.get("r_work"),
                    "r_free": v.get("r_free"),
                    "top_anom_peak": v.get("top_anom_peak"),
                    "n_anom_peaks": v.get("n_anom_peaks"),
                }
            )
    rows.sort(key=lambda r: (r["variant"] or "", r["epoch"] or -1))
    return rows, results


_MONO = {"intensity": ("intensity.prf.value", "qi_mean"),
         "background": ("background.mean", "qbg_mean")}


def _find_pred_parquet(ckpt_eval: Path) -> Path | None:
    """The latest-epoch per-obs predictions parquet, under output_root/predictions."""
    pred_dir = ckpt_eval.parent / "predictions"
    if not pred_dir.exists():
        return None
    # integrator.predict writes epoch_NNNN/{pred.parquet | preds_epoch_*}.
    cands = sorted(pred_dir.glob("**/pred.parquet")) + sorted(
        pred_dir.glob("**/preds_epoch_*")
    )
    return cands[-1] if cands else None


def _scatter_plots(ckpt_eval: Path, pred_path: Path) -> list[Path]:
    """Per-obs model-vs-DIALS intensity (log) + background (linear) scatters."""
    import pandas as pd

    df = pd.read_parquet(pred_path)
    if len(df) > 50000:
        df = df.sample(50000, random_state=0)
    saved = []
    specs = [
        ("intensity", "intensity_scatter.png", True, "model qi_mean",
         "DIALS intensity.prf.value"),
        ("background", "background_scatter.png", False, "model qbg_mean",
         "DIALS background.mean"),
    ]
    for key, fname, log, xlab, ylab in specs:
        dials_col, model_col = _MONO[key]
        if dials_col not in df.columns or model_col not in df.columns:
            logger.warning("scatter %s skipped: missing %s/%s", key, dials_col,
                           model_col)
            continue
        fig, _ = mp.plot_scatter_identity(
            df[model_col].to_numpy(), df[dials_col].to_numpy(),
            xlabel=xlab, ylabel=ylab, title=f"{key}: model vs DIALS", log=log,
        )
        saved.append(mp.save_figure(fig, ckpt_eval / fname))
    return saved


def _peak_rows(ckpt_eval: Path) -> list[dict]:
    """Per-(epoch, variant, seqid) anomalous peakz from each find_peaks CSV."""
    import pandas as pd

    rows = []
    for pc in sorted(ckpt_eval.glob("epoch*/*/peaks.csv")):
        epoch = _epoch_of(pc.parent.parent.name)
        variant = pc.parent.name
        try:
            df = pd.read_csv(pc)
        except Exception:  # noqa: BLE001
            continue
        if "seqid" not in df.columns or "peakz" not in df.columns:
            continue
        for _, r in df.iterrows():
            rows.append({"epoch": epoch, "variant": variant,
                         "seqid": int(r["seqid"]), "peakz": float(r["peakz"])})
    return rows


def _epoch_of(name: str) -> int:
    import re
    m = re.search(r"epoch0*(\d+)", name)
    return int(m.group(1)) if m else -1


def _anomalous_plots(ckpt_eval: Path, ref_peaks: str | None) -> list[Path]:
    """Per-residue anomalous peakz over epoch (one PNG per site), DIALS ref line."""
    import pandas as pd

    peaks = _peak_rows(ckpt_eval)
    if not peaks:
        logger.warning("no per-epoch peaks.csv found; skipping anomalous plots")
        return []
    ref = {}
    if ref_peaks and Path(ref_peaks).exists():
        rdf = pd.read_csv(ref_peaks)
        if {"seqid", "peakz"} <= set(rdf.columns):
            ref = {int(s): float(z) for s, z in zip(rdf["seqid"], rdf["peakz"])}

    # Sites to plot: the reference sites if given, else those seen in the model.
    seqids = sorted(ref) if ref else sorted({p["seqid"] for p in peaks})
    variants = sorted({p["variant"] for p in peaks})
    pal = mp._palette(variants)
    saved = []
    for s in seqids:
        series = []
        for v in variants:
            pts = sorted(
                (p["epoch"], p["peakz"]) for p in peaks
                if p["seqid"] == s and p["variant"] == v
            )
            if pts:
                series.append(
                    (v, [a for a, _ in pts], [b for _, b in pts], pal[v])
                )
        if not series:
            continue
        fig, _ = mp.plot_metric_over_epoch(
            series, ref_value=ref.get(s), ref_label="DIALS",
            y_label="anomalous peakz (sigma)",
            title=f"Anomalous peak at residue {s} vs checkpoint",
        )
        saved.append(mp.save_figure(fig, ckpt_eval / f"anom_peakz_{s}.png"))
    return saved


def make_all_plots(
    ckpt_eval: Path, *, ref_peaks: str | None = None,
    pred_path: Path | None = None,
) -> list[Path]:
    """Write the four model-vs-DIALS plots into `ckpt_eval`; returns the paths."""
    rows, results = load_results(ckpt_eval)
    if not rows:
        raise FileNotFoundError(f"no result.json under {ckpt_eval}")
    logger.info("Loaded %d (epoch, variant) results", len(rows))

    saved = []
    # (3) Refinement R-work/R-free vs epoch.
    fig, _ = mp.plot_r_values_vs_epoch(
        rows, title="Refinement R-factors vs checkpoint"
    )
    saved.append(mp.save_figure(fig, ckpt_eval / "r_factors_vs_epoch.png"))

    # (4) Per-residue anomalous peakz over epoch vs the DIALS reference.
    saved += _anomalous_plots(ckpt_eval, ref_peaks)

    # (1, 2) Per-obs model-vs-DIALS intensity + background scatters.
    pred_path = pred_path or _find_pred_parquet(ckpt_eval)
    if pred_path and Path(pred_path).exists():
        try:
            saved += _scatter_plots(ckpt_eval, Path(pred_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("scatter plots skipped: %s", exc)
    else:
        logger.warning(
            "no predictions parquet (run integrator.predict); "
            "skipping intensity/background scatters"
        )
    return saved


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot a finished ckpt_eval directory (refinement / "
        "anomalous / scatter)."
    )
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument(
        "--ckpt-eval-dir",
        type=Path,
        default=None,
        help="The ckpt_eval directory directly (else resolved from --run-dir).",
    )
    p.add_argument(
        "--ref-peaks",
        type=str,
        default=None,
        help="Reference peaks.csv (seqid, peakz) for the DIALS anomalous "
        "reference line on the per-residue peak plots.",
    )
    p.add_argument(
        "--pred-parquet",
        type=str,
        default=None,
        help="Per-obs predictions parquet for the scatters (else auto-found "
        "under <output_root>/predictions).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.run_dir and not args.ckpt_eval_dir:
        raise SystemExit("pass --run-dir or --ckpt-eval-dir")
    if args.ckpt_eval_dir:
        ckpt_eval = Path(args.ckpt_eval_dir).resolve()
    else:
        ckpt_eval = resolve_ckpt_eval(args.run_dir.resolve(), None)
    if not ckpt_eval.exists():
        raise FileNotFoundError(f"no ckpt_eval at {ckpt_eval}")

    saved = make_all_plots(
        ckpt_eval, ref_peaks=args.ref_peaks, pred_path=args.pred_parquet
    )
    print("\n".join(["Wrote:"] + [f"  {p}" for p in saved]))


if __name__ == "__main__":
    main()
