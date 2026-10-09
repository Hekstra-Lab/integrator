"""Generate an HTML report for a training run.

Usage:
    integrator.report --run-dir /path/to/run
"""

import argparse
import logging
from pathlib import Path

from .utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an HTML report for a training run"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory (contains run_paths.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output HTML path (default: <run-dir>/report.html)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v = INFO, -vv = DEBUG)",
    )
    return parser.parse_args()


def main():
    import yaml

    from integrator.reporting import build_report

    args = parse_args()
    setup_logging(args.verbose)

    run_dir = Path(args.run_dir)
    meta = yaml.safe_load((run_dir / "run_paths.yaml").read_text())
    log_dir = Path(meta["wandb"]["log_dir"])

    loss_trace_dir = log_dir / "loss_traces"
    if not loss_trace_dir.is_dir():
        logger.error("No loss_traces directory found at %s", loss_trace_dir)
        raise SystemExit(1)

    out_path = Path(args.output) if args.output else run_dir / "report.html"

    html = build_report(loss_trace_dir)
    out_path.write_text(html)
    logger.info("Report written to %s", out_path)
    print(f"Report: {out_path}")
