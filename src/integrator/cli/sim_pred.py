"""
Prediction CLI for simulated shoebox data.

Example use:
    integrator.sim_pred -v --run-dir /path/to/run_dir
"""

import argparse
import logging
import re

from integrator.cli.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict from checkpoints (simulated data)"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to dir containing run_metadata.yaml",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for prediction",
    )
    parser.add_argument(
        "--save-preds-as",
        type=str,
        default="parquet",
        help="Filetype to save predictions as (h5, pt, parquet)",
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
    from pathlib import Path

    import torch
    import yaml

    from integrator.callbacks import BatchPredWriter
    from integrator.utils import (
        construct_data_loader,
        construct_integrator,
        construct_trainer,
        load_config,
    )

    # Get args
    args = parse_args()

    # setup logger
    setup_logging(args.verbose)

    logger.info("Run directory: %s", args.run_dir)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("Starting Predictions (simulated data)")

    # run dir
    run_dir = Path(args.run_dir)

    # Reading in run_metadata.yaml
    meta = yaml.safe_load((run_dir / "run_metadata.yaml").read_text())
    config = load_config(meta["config"])
    wandb_info = meta["wandb"]

    # Override batch size if provided
    if args.batch_size is not None:
        config["data_loader"]["args"]["batch_size"] = args.batch_size

    # writing prediction directories
    log_dir = Path(wandb_info["log_dir"])
    wandb_dir = log_dir.parent
    pred_dir = wandb_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    # list of .ckpt files
    checkpoints = sorted(log_dir.glob("**/epoch*.ckpt"))
    logger.info("Found %d checkpoints", len(checkpoints))

    # load data
    data_loader = construct_data_loader(config)
    data_loader.setup()

    epoch_re = re.compile(r"epoch=(\d+)")
    for ckpt in checkpoints:
        # Finding checkpoint epoch
        m = epoch_re.search(ckpt.name)
        if not m:
            raise ValueError(f"Could not parse epoch from {ckpt.name}")
        epoch = int(m.group(1))

        # logger
        logger.info("Processing checkpoint: %s", ckpt.name)
        logger.debug("Checkpoint path: %s", ckpt)
        logger.info("Epoch: %d", epoch)

        # Writing epoch prediction dir
        ckpt_dir = pred_dir / f"epoch_{epoch:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Construct callbacks
        pred_writer = BatchPredWriter(
            output_dir=ckpt_dir,
            write_interval="batch",
            epoch=epoch,
        )

        # Construct trainer
        trainer = construct_trainer(
            config,
            callbacks=[pred_writer],
            logger=None,
        )

        ckpt_ = torch.load(ckpt.as_posix())
        integrator = construct_integrator(config)
        integrator.load_state_dict(ckpt_["state_dict"])

        # Use gpu if available
        if torch.cuda.is_available():
            integrator.to(torch.device("cuda"))

        integrator.eval()
        trainer.predict(
            integrator,
            return_predictions=False,
            dataloaders=data_loader.predict_dataloader(),
        )

    logger.info("Prediction complete!")


if __name__ == "__main__":
    main()
