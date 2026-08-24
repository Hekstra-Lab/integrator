"""Extract a W&B run's logged metrics from its LOCAL files (no server needed).

W&B writes the full history to the run's local `.wandb` file and the final
values to `wandb-summary.json`, independently of whether the web dashboard
renders. This reads them directly, dumps a tidy CSV, and (if matplotlib is
present) saves PNG plots of the metrics you care about.

Usage:
    # point at the wandb run dir, the .wandb file, or a lightning run dir
    python scripts/extract_wandb_metrics.py /n/.../wandb/run-<ts>-<id>/
    python scripts/extract_wandb_metrics.py <run_dir> --filter scale consistency phi loss

Outputs (next to the .wandb file): metrics.csv, and metrics_<key>.png plots.

If the data IS on the server (dashboard just not rendering), the alternative is:
    python -c "import wandb,pandas; wandb.Api().run('ENTITY/PROJECT/RUNID').history(samples=1000000).to_csv('h.csv')"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _find_wandb_file(path: Path) -> Path:
    """Resolve a .wandb file from a file / wandb-run dir / lightning run dir."""
    if path.is_file() and path.suffix == ".wandb":
        return path
    cands = sorted(path.glob("**/run-*.wandb"))
    if not cands:
        cands = sorted(path.glob("**/*.wandb"))
    if not cands:
        raise FileNotFoundError(f"No .wandb file found under {path}")
    if len(cands) > 1:
        logger.warning("Multiple .wandb files; using newest: %s", cands[-1])
    return cands[-1]


def read_summary(wandb_file: Path) -> dict:
    """Final value of each metric, from wandb-summary.json (always reliable)."""
    for p in (
        wandb_file.parent / "files" / "wandb-summary.json",
        wandb_file.parent / "wandb-summary.json",
    ):
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception as e:  # noqa: BLE001
                logger.warning("summary read failed (%s): %s", p, e)
    return {}


def read_history(wandb_file: Path) -> list[dict]:
    """Full per-step history by scanning the local .wandb datastore."""
    from wandb.sdk.internal.datastore import DataStore

    pb = None
    for modpath in (
        "wandb.proto.wandb_internal_pb2",
        "wandb.proto.v5.wandb_internal_pb2",
        "wandb.proto.v4.wandb_internal_pb2",
        "wandb.proto.v3.wandb_internal_pb2",
    ):
        try:
            pb = __import__(modpath, fromlist=["Record"])
            break
        except Exception:  # noqa: BLE001
            continue
    if pb is None:
        raise ImportError("Could not import wandb_internal_pb2 (version mismatch)")

    ds = DataStore()
    ds.open_for_scan(str(wandb_file))
    rows: list[dict] = []
    while True:
        try:
            data = ds.scan_data()
        except Exception:  # noqa: BLE001 - truncated/last record
            break
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if len(rec.history.item):
            row: dict = {}
            for it in rec.history.item:
                # modern wandb uses repeated `nested_key`; older uses `key`.
                key = ".".join(it.nested_key) if it.nested_key else it.key
                try:
                    row[key] = json.loads(it.value_json)
                except Exception:  # noqa: BLE001
                    row[key] = it.value_json
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Extract W&B metrics from local files.")
    ap.add_argument("path", type=Path, help="wandb run dir / .wandb file / run dir")
    ap.add_argument(
        "--filter", nargs="*", default=["scale", "consistency", "phi", "loss"],
        help="substrings of metric names to plot (default: scale/consistency/phi/loss)",
    )
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    wandb_file = _find_wandb_file(args.path.resolve())
    out_dir = wandb_file.parent
    logger.info("Reading %s", wandb_file)

    summary = read_summary(wandb_file)
    if summary:
        print("\n=== final metric values (wandb-summary.json) ===")
        for k in sorted(summary):
            if k.startswith("_"):
                continue
            v = summary[k]
            if isinstance(v, (int, float)):
                print(f"  {k:40s} {v:.6g}")

    try:
        rows = read_history(wandb_file)
    except Exception as e:  # noqa: BLE001
        logger.error(
            "Could not parse .wandb history (%s). The summary above still has "
            "final values; for full curves use the wandb.Api() one-liner in the "
            "module docstring if the run is on the server.", e,
        )
        return
    if not rows:
        logger.warning("No history records found in %s", wandb_file)
        return

    import pandas as pd

    df = pd.DataFrame(rows)
    x = "_step" if "_step" in df else ("epoch" if "epoch" in df else None)
    if x:
        df = df.sort_values(x)
    csv = out_dir / "metrics.csv"
    df.to_csv(csv, index=False)
    logger.info(
        "Wrote %s (%d rows, %d columns)", csv, len(df), df.shape[1]
    )

    metric_cols = [
        c for c in df.columns
        if not c.startswith("_") and df[c].dtype.kind in "fi" and c != x
    ]
    hits = [
        c for c in metric_cols
        if not args.filter or any(f.lower() in c.lower() for f in args.filter)
    ]
    print(f"\n=== metrics matching {args.filter or 'ALL'} ({len(hits)}) ===")
    for c in sorted(hits):
        s = df[c].dropna()
        if len(s):
            print(f"  {c:40s} first={s.iloc[0]:.4g}  last={s.iloc[-1]:.4g}")

    if args.no_plot:
        return
    try:
        import math

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        logger.info("matplotlib not available; CSV written, skipping plots.")
        return

    # One at-a-glance dashboard: every logged metric in a grid. This is the
    # "is the model working" view -- works straight off the local .wandb file,
    # no server/sync needed. Re-run during training to refresh.
    overview = sorted(metric_cols, key=lambda c: c.lower())
    if overview:
        ncol = min(4, len(overview))
        nrow = math.ceil(len(overview) / ncol)
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow), squeeze=False
        )
        for ax, c in zip(axes.flat, overview):
            s = df[[x, c]].dropna() if x else df[[c]].dropna()
            if len(s) < 2:
                ax.set_visible(False)
                continue
            ax.plot(s[x] if x else range(len(s)), s[c], lw=1.0)
            ax.set_title(c, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.margins(x=0)
        for ax in axes.flat[len(overview):]:
            ax.set_visible(False)
        fig.supxlabel(x or "row", fontsize=9)
        fig.tight_layout()
        ov = out_dir / "metrics_overview.png"
        fig.savefig(ov, dpi=110)
        plt.close(fig)
        logger.info("Saved overview of %d metrics -> %s", len(overview), ov)

    for c in hits:
        s = df[[x, c]].dropna() if x else df[[c]].dropna()
        if len(s) < 2:
            continue
        plt.figure(figsize=(6, 4))
        plt.plot(s[x] if x else range(len(s)), s[c])
        plt.xlabel(x or "row")
        plt.ylabel(c)
        plt.title(c)
        plt.tight_layout()
        png = out_dir / f"metric_{c.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(png, dpi=110)
        plt.close()
    logger.info("Saved %d individual metric plots to %s", len(hits), out_dir)


if __name__ == "__main__":
    main()
