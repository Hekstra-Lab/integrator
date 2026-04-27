"""
# Example use
integrator.pred -v \
        --run-dir \
"""

import argparse
import logging
import re

from integrator.cli.utils.io import write_refl_from_preds
from integrator.cli.utils.logger import setup_logging
from integrator.cli.utils.mtz_writer import write_mtz_from_preds

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict from a set of pytorch.ckpt files"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=False,
        help="Path to dir containing config.yaml file",
    )
    parser.add_argument(
        "--write-refl",
        action="store_true",
        help="Write predictions as a .refl file",
    )
    parser.add_argument(
        "--write-mtz",
        type=str,
        help="Write predictions as an .mtz file; for polychromatic data only",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Integer value specifying the size of each training batch",
    )
    parser.add_argument(
        "--save-preds-as",
        type=str,
        default="parquet",
        help="Filetype to save predictions as. Support for .h5, .pt, and .parquet",
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

    torch.set_float32_matmul_precision("high")

    import yaml

    from integrator.callbacks import BatchPredWriter
    from integrator.utils import (
        construct_data_loader,
        construct_integrator,
        construct_trainer,
        inject_binning_labels,
        load_config,
    )

    # Get args
    args = parse_args()

    # setup logger
    setup_logging(args.verbose)

    logger.info("Run directory: %s", args.run_dir)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("Starting Predictions")

    # run dir
    run_dir = Path(args.run_dir)

    # Reading in pred.yaml
    meta = yaml.safe_load((run_dir / "run_metadata.yaml").read_text())
    config = load_config(meta["config"])
    wandb_info = meta["wandb"]
    slurm_info = meta["slurm"]

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
    inject_binning_labels(data_loader, config)

    # path to input refl file (only needed for --write-refl)
    refl_file = config.get("output", {}).get("refl_file")
    if args.write_refl and not refl_file:
        raise ValueError(
            "--write-refl requires 'output.refl_file' in the YAML config"
        )

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

        # Skip prediction if outputs already exist (resume support),
        # but still run post-processing (write-refl, write-mtz) below.
        has_preds = any(ckpt_dir.glob("preds_epoch_*"))
        if has_preds:
            logger.info(
                "Predictions for epoch %d already exist — skipping inference",
                epoch,
            )
        else:
            callbacks = []
            pred_writer = BatchPredWriter(
                output_dir=ckpt_dir,
                write_interval="batch",
                epoch=epoch,
            )
            callbacks.append(pred_writer)

            trainer = construct_trainer(
                config,
                callbacks=callbacks,
                logger=None,
            )

            ckpt_ = torch.load(ckpt.as_posix())
            integrator = construct_integrator(config, skip_warmstart=True)
            integrator.load_state_dict(ckpt_["state_dict"])

            if torch.cuda.is_available():
                integrator.to(torch.device("cuda"))

            integrator.eval()
            trainer.predict(
                integrator,
                return_predictions=False,
                dataloaders=data_loader.predict_dataloader(),
            )

        if args.write_refl:
            logger.info("Writing .refl output for epoch %d", epoch)
            write_refl_from_preds(
                ckpt_dir=ckpt_dir,
                refl_file=refl_file,
                epoch=epoch,
                config=config,
                filetype=args.save_preds_as,
            )

        if args.write_mtz:
            from integrator.cli.utils.io import get_pred_files

            logger.info("Writing .mtz output for epoch %d", epoch)
            pred_data = get_pred_files(
                ckpt_dir=ckpt_dir, filetype=args.save_preds_as
            )
            data_dir = Path(config["data_loader"]["args"]["data_dir"])
            write_mtz_from_preds(
                pred_data=pred_data,
                metadata_path=data_dir / "metadata.pt",
                crystal_yaml_path=data_dir / "crystal.yaml",
                out_path=ckpt_dir / args.write_mtz,
            )

    logger.info("Prediction complete!")

    try:
        import polars as pl
    except ImportError:
        logger.warning(
            "polars not installed — skipping test_preds_all.parquet"
            " aggregation."
        )
    else:
        parquet_glob = str(pred_dir / "*" / "*.parquet")
        from glob import glob as _glob

        if not _glob(parquet_glob):
            logger.info(
                "No parquet files under %s — skipping test-set aggregation"
                " (use --save-preds-as parquet if you want it).",
                pred_dir,
            )
        else:
            lf = pl.scan_parquet(parquet_glob, include_file_paths="src")
            schema = lf.collect_schema()
            if "is_test" not in schema:
                logger.info(
                    "is_test not in predictions — skipping test-set aggregation"
                )
            else:
                out_path = pred_dir / "test_preds_all.parquet"
                logger.info("Aggregating test predictions → %s", out_path)
                (
                    lf.filter(pl.col("is_test") == 1.0)
                    .with_columns(
                        pl.col("src")
                        .str.extract(r"epoch_(\d+)", 1)
                        .cast(pl.Int32)
                        .alias("epoch")
                    )
                    .drop("src")
                    .sink_parquet(out_path)
                )
                logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
