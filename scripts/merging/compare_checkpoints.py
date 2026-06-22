import argparse
import csv
import logging
from pathlib import Path

import yaml

# Plotting + ckpt_eval loading live in the standalone plotter; this script just
# adds the summary.csv + W&B logging on top.
from plot_ckpt_eval import (
    _resolve_pdb,
    load_results,
    make_all_plots,
    resolve_ckpt_eval,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger(__name__)


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
        import os
        import shutil

        import wandb

        # Stage wandb in ONE fixed netscratch dir under the run's output_root and
        # clear it first, so reruns reuse the same folder instead of dropping a
        # new run-<timestamp>-<id> in the submitting directory each time. Resume
        # pulls run state from the server, so clearing local staging is safe; the
        # same run_id keeps it the same W&B run.
        out = meta.get("output_root") or str(run_dir)
        eval_wb = Path(out) / "eval_wandb"
        shutil.rmtree(eval_wb / "wandb", ignore_errors=True)
        eval_wb.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(eval_wb)

        run = wandb.init(
            project=wb.get("project"),
            id=wb.get("run_id"),
            entity=wb.get("entity"),
            resume="allow",
            dir=str(eval_wb),
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
    p.add_argument(
        "--anom-atom-sel",
        type=str,
        default="",
        help="gemmi selection (e.g. '[S]') to also make the per-site "
        "anomalous-peak plot (needs the eval config's pdb).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_root = resolve_ckpt_eval(run_dir, args.out_root)
    if not out_root.exists():
        raise FileNotFoundError(f"no eval outputs at {out_root}")

    rows, _ = load_results(out_root)
    if not rows:
        raise FileNotFoundError(f"no result.json under {out_root}")
    logger.info("Aggregated %d (epoch, variant) results", len(rows))

    csv_path = out_root / "summary.csv"
    _write_csv(rows, csv_path)
    # Plotting lives in the standalone plot_ckpt_eval; pass pdb (from the eval
    # config) + the atom selection so the per-site anomalous plot is made too.
    pdb = _resolve_pdb(run_dir, None)
    pngs = make_all_plots(out_root, pdb=pdb, anom_sel=args.anom_atom_sel)

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
