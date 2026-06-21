import argparse
import csv
import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger(__name__)


def _load_results(out_root: Path) -> list[dict]:
    """Flatten every epoch*/result.json into one row per (epoch, variant)."""
    rows = []
    for rj in sorted(out_root.glob("epoch*/result.json")):
        try:
            res = json.loads(rj.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip %s: %s", rj, exc)
            continue
        for vname, v in (res.get("variants") or {}).items():
            rows.append(
                {
                    "epoch": res.get("epoch"),
                    "variant": vname,
                    "n_hkl_seen": res.get("n_hkl_seen"),
                    "r_work": v.get("r_work"),
                    "r_free": v.get("r_free"),
                    "top_anom_peak": v.get("top_anom_peak"),
                    "n_anom_peaks": v.get("n_anom_peaks"),
                    "phenix_ok": v.get("phenix_ok"),
                }
            )
    return sorted(rows, key=lambda r: (r["variant"] or "", r["epoch"] or -1))


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _best(rows, key, prefer_min):
    """The (epoch, variant, value) with the best finite `key`."""
    cand = [
        r
        for r in rows
        if isinstance(r.get(key), (int, float)) and r[key] == r[key]  # not NaN
    ]
    if not cand:
        return None
    return (
        min(cand, key=lambda r: r[key])
        if prefer_min
        else max(cand, key=lambda r: r[key])
    )


def _plot(rows: list[dict], out_root: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("matplotlib unavailable; skipping plots")
        return []

    variants = sorted({r["variant"] for r in rows if r["variant"]})

    def series(variant, key):
        pts = [
            (r["epoch"], r[key])
            for r in rows
            if r["variant"] == variant
            and isinstance(r.get(key), (int, float))
            and r[key] == r[key]
        ]
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]

    saved = []
    # R-factors vs epoch
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant in variants:
        for key, ls in (("r_work", "-"), ("r_free", "--")):
            xs, ys = series(variant, key)
            if xs:
                ax.plot(xs, ys, ls, marker="o", ms=3, label=f"{variant} {key}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("R-factor")
    ax.set_title("Refinement R-factors vs checkpoint")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_root / "r_factors_vs_epoch.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)

    # Top anomalous peak vs epoch
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant in variants:
        xs, ys = series(variant, "top_anom_peak")
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, label=variant)
    ax.set_xlabel("epoch")
    ax.set_ylabel("top anomalous peak (sigma)")
    ax.set_title("Top anomalous peak vs checkpoint")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_root / "anom_peak_vs_epoch.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)
    return saved


def _maybe_log_wandb(run_dir: Path, csv_path: Path, pngs: list[Path]) -> None:
    """Resume the run's W&B run (from run_paths.yaml) and log the artifacts."""
    meta_path = run_dir / "run_paths.yaml"
    if not meta_path.exists():
        return
    meta = yaml.safe_load(meta_path.read_text()) or {}
    wb = meta.get("wandb")
    if not wb:
        return
    try:
        import wandb

        run = wandb.init(
            project=wb.get("project"),
            id=wb.get("run_id"),
            entity=wb.get("entity"),
            resume="allow",
        )
        run.log({Path(p).stem: wandb.Image(str(p)) for p in pngs})
        art = wandb.Artifact("ckpt_eval", type="evaluation")
        art.add_file(str(csv_path))
        run.log_artifact(art)
        run.finish()
        logger.info("Logged eval to W&B run %s", wb.get("run_id"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("W&B logging skipped: %s", exc)


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate per-checkpoint phenix results across epochs."
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Per-checkpoint output root (default: <run_dir>/ckpt_eval).",
    )
    p.add_argument("--no-wandb", action="store_true", help="Don't log to W&B.")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_root = args.out_root or (run_dir / "ckpt_eval")
    if not out_root.exists():
        raise FileNotFoundError(f"no eval outputs at {out_root}")

    rows = _load_results(out_root)
    if not rows:
        raise FileNotFoundError(f"no result.json under {out_root}")
    logger.info("Aggregated %d (epoch, variant) results", len(rows))

    csv_path = out_root / "summary.csv"
    _write_csv(rows, csv_path)
    pngs = _plot(rows, out_root)

    best_rfree = _best(rows, "r_free", prefer_min=True)
    best_peak = _best(rows, "top_anom_peak", prefer_min=False)
    lines = ["", "=" * 60, f"Checkpoint evaluation: {run_dir.name}"]
    if best_rfree:
        lines.append(
            f"  best R-free : {best_rfree['r_free']} "
            f"(epoch {best_rfree['epoch']}, {best_rfree['variant']})"
        )
    if best_peak:
        lines.append(
            f"  best anom peak: {best_peak['top_anom_peak']:.2f} sigma "
            f"(epoch {best_peak['epoch']}, {best_peak['variant']})"
        )
    lines += [f"  csv  : {csv_path}"] + [f"  plot : {p}" for p in pngs]
    lines.append("=" * 60)
    print("\n".join(lines))

    if not args.no_wandb:
        _maybe_log_wandb(run_dir, csv_path, pngs)


if __name__ == "__main__":
    main()
