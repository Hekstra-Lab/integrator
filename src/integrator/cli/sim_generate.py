"""CLI for generating simulated shoebox datasets.

Usage::

    integrator.sim_generate --config sim.yaml --output /path/to/data --seed 42 -v
"""

import argparse
import logging
from pathlib import Path

import yaml

from integrator.cli.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate simulated shoebox datasets"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to simulation config YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for .pt files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v = INFO, -vv = DEBUG)",
    )
    return parser.parse_args()


def main() -> None:
    from integrator.simulate import save_dataset, simulate
    from integrator.simulate.priors import (
        fit_priors_from_experimental,
        make_config_priors,
    )

    args = parse_args()
    setup_logging(args.verbose)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 1. Build per-bin priors
    prior_cfg = cfg["priors"]
    if prior_cfg.get("mode", "config") == "experimental":
        data_dir = Path(prior_cfg["data_dir"])
        integrator_cfg = prior_cfg.get("integrator_config", {})
        # Ensure data_dir is set in the nested config
        integrator_cfg.setdefault("data_loader", {}).setdefault("args", {})[
            "data_dir"
        ] = str(data_dir)
        priors = fit_priors_from_experimental(
            data_dir=data_dir,
            cfg=integrator_cfg,
            n_bins=prior_cfg.get("n_bins", 20),
        )
    else:
        priors = make_config_priors(prior_cfg)

    n_bins = priors["n_bins"]
    logger.info("Using %d resolution bins", n_bins)

    # 2. Simulate
    sim = simulate(
        n_per_bin=cfg.get("n_per_bin", 10_000),
        n_bins=n_bins,
        tau=priors["tau"],
        bg_rate=priors["bg_rate"],
        H=cfg.get("H", 21),
        W=cfg.get("W", 21),
        n_frames=cfg.get("n_frames", 3),
        profile_kwargs=cfg.get("profile", {}),
        seed=args.seed,
    )

    # 3. Save
    save_dataset(
        sim=sim,
        tau=priors["tau"],
        bg_rate=priors["bg_rate"],
        save_dir=Path(args.output),
        s_squared=priors.get("s_squared"),
        concentration=priors.get("concentration"),
        K_true=priors.get("K_true"),
        B_true=priors.get("B_true"),
        test_frac=cfg.get("test_frac", 0.05),
    )

    logger.info("Done. Output in %s", args.output)


if __name__ == "__main__":
    main()
