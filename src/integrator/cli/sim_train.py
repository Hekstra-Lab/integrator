import argparse
import logging
from copy import deepcopy
from pathlib import Path

import yaml

from .utils.io import _apply_cli_overrides
from .utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train model on simulated shoebox data"
    )

    # --- Required ---
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--wb-project",
        type=str,
        required=True,
        help="Name of the W&B project to save to",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory (saves config copy and run metadata)",
    )

    # --- Paths ---
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override data_dir in the YAML config",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/n/netscratch/hekstra_lab/Lab/laldama/lightning_logs/",
        help="Path to store local W&B logs",
    )

    # --- Training ---
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        help="Gradient clipping value",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["16", "32"],
        help="Training precision",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["cpu", "gpu", "auto"],
        help="Accelerator type",
    )
    parser.add_argument(
        "--devices",
        type=int,
        help="Number of devices",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        help="Run validation every N epochs",
    )

    # --- Model ---
    parser.add_argument(
        "--integrator-name",
        type=str,
        help="Name of the integrator module to use",
    )
    parser.add_argument(
        "--qbg",
        type=str,
        help="Name of the background surrogate module",
    )
    parser.add_argument(
        "--qi",
        type=str,
        help="Name of the intensity surrogate module",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        help="Number of Monte Carlo samples for KL estimation",
    )

    # --- Loss weights ---
    parser.add_argument(
        "--pprf-weight",
        type=float,
        help="Profile KL weight",
    )
    parser.add_argument(
        "--pbg-weight",
        type=float,
        help="Background KL weight",
    )
    parser.add_argument(
        "--pi-weight",
        type=float,
        help="Intensity KL weight",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        help="Number of resolution bins for per-bin priors",
    )

    # --- Data ---
    parser.add_argument(
        "--val-split",
        type=float,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        help="Use a subset of the data (for debugging)",
    )

    # --- Misc ---
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v = INFO, -vv = DEBUG)",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Optional W&B tags for run identification",
    )
    return parser.parse_args()


def main():
    import os

    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    from integrator.callbacks import (
        EpochMetricRecorder,
        LossTraceRecorder,
        assign_labels,
    )
    from integrator.utils import (
        construct_data_loader,
        construct_integrator,
        construct_trainer,
        load_config,
        prepare_per_bin_priors,
        save_run_artifacts,
    )

    args = parse_args()
    setup_logging(args.verbose)

    torch.set_float32_matmul_precision("medium")

    # load configuration file
    cfg = load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args=args)

    # load data
    logger.info("Starting Training (simulated data)")

    data_loader = construct_data_loader(cfg)
    data_loader.setup()

    # Auto-generate per-bin prior files if needed by the loss
    prepare_per_bin_priors(cfg)

    # Tags for identification
    tags = [
        cfg["integrator"]["name"],
        cfg["integrator"]["args"]["data_dim"],
    ]
    if args.tags:
        tags.extend(args.tags)

    # load wandb logger
    wb_logger = WandbLogger(
        project=args.wb_project,
        save_dir=args.save_dir,
        tags=tags,
    )

    # Logging directory
    logdir = Path(wb_logger.experiment.dir)
    run_dir = Path(args.run_dir)

    logger.info(f"Logging directory: {logdir.as_posix()}")
    logger.info(f"Run directory: {run_dir}")

    # Write a copy of the config.yaml file
    config_copy = run_dir / "config_copy.yaml"
    cfg_json = deepcopy(cfg)

    with open(config_copy, "w") as f:
        yaml.safe_dump(cfg_json, f, sort_keys=False)

    # log hyperparameters
    wb_logger.log_hyperparams(cfg_json)

    # Run metadata
    metadata = {
        "config": config_copy.as_posix(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "wandb": {
            "project": args.wb_project,
            "run_id": wb_logger.experiment.id,
            "entity": wb_logger.experiment.entity,
            "log_dir": wb_logger.experiment.dir,
        },
    }

    wb_logger.log_hyperparams(metadata)

    m_fname = run_dir / "run_metadata.yaml"
    (m_fname).write_text(yaml.safe_dump(metadata))

    logger.info(f"Saved run_metadata: {m_fname}")

    # assign validation/train labels to each shoebox
    assign_labels(dataset=data_loader, save_dir=logdir.as_posix())

    # create integrator
    integrator = construct_integrator(cfg)

    # Set dataset_size for losses that need it (e.g. HierarchicalShoeboxLoss)
    # Must be done here because sim_train passes raw dataloaders, not the
    # data module, so trainer.datamodule is None during model.setup().
    if hasattr(integrator.loss, "dataset_size"):
        integrator.loss.dataset_size = len(data_loader.train_dataset)

    # save prior artifacts (rescaled concentration, param counts, etc.)
    save_run_artifacts(integrator, cfg, logdir)

    # Callbacks — simulated-appropriate keys (no DIALS metadata)
    keys = [
        "refl_ids",
        "is_test",
        "qi_mean",
        "qi_var",
        "qbg_mean",
        "qbg_var",
        "intensity",
        "background",
    ]
    train_metric_dir = logdir / "train_metrics"
    val_metric_dir = logdir / "val_metrics"

    train_epoch_recorder = EpochMetricRecorder(
        out_dir=train_metric_dir,
        keys=keys,
        split="train",
        every_n_epochs=1,
        max_rows_per_epoch=200_000,
    )

    val_epoch_recorder = EpochMetricRecorder(
        out_dir=val_metric_dir,
        keys=keys,
        split="val",
    )

    loss_trace_recorder = LossTraceRecorder(
        out_dir=logdir / "loss_traces",
    )

    # to save checkpoints
    ckpt_dir = logdir / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:04d}",
        every_n_epochs=1,
        save_top_k=-1,
        save_last="link",
    )
    logger.info(f"Checkpoints saved to: {Path(ckpt_dir).as_posix()}")

    # PyTorch-Lightning Trainer
    trainer = construct_trainer(
        cfg,
        callbacks=[
            val_epoch_recorder,
            train_epoch_recorder,
            loss_trace_recorder,
            checkpoint_callback,
        ],
        logger=wb_logger,
    )

    # Fit the model
    trainer.fit(
        integrator,
        train_dataloaders=data_loader.train_dataloader(),
        val_dataloaders=data_loader.val_dataloader(),
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
