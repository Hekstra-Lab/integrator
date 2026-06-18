"""
CLI to train the integratio nmodel

Example usage:

integrator.train \
    --config 
"""

import argparse
import logging
from copy import deepcopy
from pathlib import Path

import yaml

from .utils.config_overrides import _apply_cli_overrides
from .utils.logger import setup_logging

logger = logging.getLogger(__name__)


def _default_run_name() -> str:
    """Auto run-dir name: sortable timestamp + short random id."""
    from datetime import datetime
    from uuid import uuid4

    return f"run_{datetime.now():%Y%m%d-%H%M%S}_{uuid4().hex[:4]}"


def parse_args():
    parser = argparse.ArgumentParser(
        prog="integrator.train", description="Train integrator model"
    )

    # Required
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )

    # Run location: a run dir is created under --log-dir with an auto name,
    # unless --run-dir gives an exact path.
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="Parent directory holding run dirs (default: ./runs)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the run dir (default: <timestamp>_<id>)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Exact run directory; overrides --log-dir/--run-name",
    )

    # Paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data_dir in the YAML config",
    )

    # W&B (optional)
    parser.add_argument(
        "--wb-project",
        type=str,
        default=None,
        help="W&B project name. Enables W&B logging when set.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="lightning_logs",
        help="W&B output root (typically on scratch); only used with "
        "--wb-project. W&B writes <save-dir>/wandb/run-<id>/.",
    )
    parser.add_argument(
        "--wandb-resume-id",
        type=str,
        default=None,
        help="W&B run ID to resume logging into (uses 'must' resume mode)",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Optional W&B tags for run identification",
    )

    # Training
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

    # Model
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

    # Loss weights
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

    # Data
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

    # Resume
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (e.g. last.ckpt)",
    )

    # Misc
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Log model-vs-DIALS intensity/background scatters (off by default)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v = INFO, -vv = DEBUG)",
    )
    return parser.parse_args()


def _make_wandb_logger(args, tags):
    """Return a WandbLogger if --wb-project is set and wandb is importable, else None."""
    if args.wb_project is None:
        return None
    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError:
        logger.warning(
            "wandb not installed; using local logging. "
            "Install with: pip install wandb"
        )
        return None

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    wb_kwargs = dict(
        project=args.wb_project, save_dir=args.save_dir, tags=tags
    )
    if args.wandb_resume_id:
        wb_kwargs["id"] = args.wandb_resume_id
        wb_kwargs["resume"] = "must"
    return WandbLogger(**wb_kwargs)


def _log_git_info(pl_logger):
    """Capture git SHA and dirty state into the logger (W&B only)."""
    import subprocess

    if not hasattr(pl_logger, "experiment") or not hasattr(
        pl_logger.experiment, "config"
    ):
        return

    _start = Path(__file__).resolve()
    _repo_root = next(
        (p for p in [_start, *_start.parents] if (p / ".git").exists()),
        None,
    )
    if _repo_root is None:
        return
    try:
        _sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root,
            text=True,
        ).strip()
        _dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=_repo_root,
                text=True,
            ).strip()
        )
        pl_logger.experiment.config.update(
            {"git_sha": _sha, "git_dirty": _dirty},
            allow_val_change=True,
        )
    except Exception as _exc:
        logger.warning("git capture failed: %s", _exc)


def main():
    import os

    import torch
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    from integrator.callbacks import (
        EpochMetricRecorder,
        LossCurveLogger,
        LossTraceRecorder,
        PredictionScatterLogger,
        WilsonParamLogger,
        assign_labels,
    )
    from integrator.configs import CheckpointConfig, EarlyStopConfig
    from integrator.utils import (
        apply_dataset_defaults,
        construct_data_loader,
        construct_integrator,
        construct_trainer,
        inject_binning_labels,
        load_config,
        prepare_per_bin_priors,
        resolve_config,
        save_run_artifacts,
    )
    from integrator.utils.factory_utils import _collect_resolved_paths

    torch.set_float32_matmul_precision("high")

    args = parse_args()
    setup_logging(args.verbose)

    cfg = load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args=args)
    cfg = apply_dataset_defaults(cfg)
    cfg = resolve_config(cfg)

    logger.info("Starting Training")

    # Auto-generate prior distribution files if needed by the loss
    prior_events: list[dict] = []
    prepare_per_bin_priors(cfg, events_out=prior_events)

    for event in prior_events:
        action = event["action"]
        if action == "reused":
            logger.info(f"Prior file reused: {event['file']}")
        else:
            logger.info(
                f"Prior file {action}: {event['file']} — {event['reason']}"
            )

    data_loader = construct_data_loader(cfg)
    data_loader.setup()
    inject_binning_labels(data_loader, cfg)

    tags = [
        cfg["integrator"]["name"],
        cfg["integrator"]["args"]["data_dim"],
        cfg["surrogates"]["qi"]["name"],
    ]

    pl_logger = _make_wandb_logger(args, tags)
    is_wandb = pl_logger is not None
    if is_wandb:
        _log_git_info(pl_logger)
        default_name = pl_logger.experiment.name or pl_logger.experiment.id
    else:
        default_name = _default_run_name()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(args.log_dir) / (args.run_name or default_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    # W&B: heavy output lives in the wandb run dir; else under run_dir.
    output_root = (
        Path(pl_logger.experiment.dir).parent if is_wandb else run_dir
    )

    # Standard W&B layout: files/ holds checkpoints + metrics + traces, with
    # plots/ and predictions/ as siblings.
    logdir = output_root / "files"
    plots_dir = output_root / "plots"
    predictions_dir = output_root / "predictions"
    logdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run directory (handle): {run_dir}")
    logger.info(f"Output root (files/plots/predictions): {output_root}")

    config_copy = run_dir / "config_log.yaml"
    cfg_json = deepcopy(cfg)
    with open(config_copy, "w") as f:
        yaml.safe_dump(cfg_json, f, sort_keys=False)

    if is_wandb:
        pl_logger.log_hyperparams(cfg_json)

    resolved_paths = _collect_resolved_paths(cfg)

    metadata = {
        "config": config_copy.as_posix(),
        "log_dir": logdir.as_posix(),
        "output_root": output_root.as_posix(),
        "predictions_dir": predictions_dir.as_posix(),
        "plots_dir": plots_dir.as_posix(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "resolved_paths": resolved_paths,
        "prior_events": prior_events,
    }
    if is_wandb:
        metadata["wandb"] = {
            "project": args.wb_project,
            "run_id": pl_logger.experiment.id,
            "entity": pl_logger.experiment.entity,
            "log_dir": logdir.as_posix(),
        }
        pl_logger.log_hyperparams(metadata)

    m_fname = run_dir / "run_metadata.yaml"
    m_fname.write_text(yaml.safe_dump(metadata))
    logger.info(f"Saved run_metadata: {m_fname}")

    assign_labels(dataset=data_loader, save_dir=logdir.as_posix())

    integrator = construct_integrator(cfg)
    save_run_artifacts(integrator, cfg, logdir)

    logger.info(f"data_dir: {resolved_paths.get('data_dir')}")
    logger.info(f"n_bins: {resolved_paths.get('n_bins')}")
    for section in ("data_loader", "surrogates", "loss"):
        entries = resolved_paths.get(section, {})
        if not entries:
            continue
        logger.info(f"Resolved {section} paths:")
        for k, info in entries.items():
            status = "" if info.get("exists") else "  [MISSING]"
            logger.info(f"  {k} -> {info['path']}{status}")

    keys = [
        "refl_ids",
        "qi_mean",
        "qi_var",
        "qbg_mean",
        "background.mean",
        "intensity.prf.value",
        "intensity.prf.variance",
        "xyzcal.px.0",
        "xyzcal.px.1",
        "xyzcal.px.2",
        "d",
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

    early_stop_cb = None
    es_raw = cfg.get("early_stop")
    es = EarlyStopConfig(**es_raw) if es_raw else None
    if es is not None:
        early_stop_cb = EarlyStopping(
            monitor=es.monitor,
            mode=es.mode,
            patience=es.patience,
            min_delta=es.min_delta,
            strict=es.strict,
            verbose=True,
        )
        logger.info(
            "EarlyStopping: monitor=%s mode=%s patience=%d min_delta=%.4f",
            es.monitor,
            es.mode,
            es.patience,
            es.min_delta,
        )

    ckpt = CheckpointConfig(**(cfg.get("checkpoint") or {}))
    default_top_k = 1 if early_stop_cb else -1
    save_top_k = (
        ckpt.save_top_k if ckpt.save_top_k is not None else default_top_k
    )
    ckpt_monitor = ckpt.monitor or (es.monitor if es else None)
    ckpt_mode = ckpt.mode or (es.mode if es else "min")
    ckpt_dir = logdir / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:04d}",
        every_n_epochs=ckpt.every_n_epochs,
        save_top_k=save_top_k,
        save_last="link",
        monitor=ckpt_monitor if save_top_k > 0 else None,
        mode=ckpt_mode if save_top_k > 0 else "min",
    )
    logger.info(
        "Checkpoints: dir=%s save_top_k=%d every_n_epochs=%d monitor=%s",
        ckpt_dir.as_posix(),
        save_top_k,
        ckpt.every_n_epochs,
        ckpt_monitor,
    )

    callbacks = [
        val_epoch_recorder,
        train_epoch_recorder,
        loss_trace_recorder,
        LossCurveLogger(out_dir=plots_dir),
        WilsonParamLogger(out_dir=plots_dir),
        checkpoint_callback,
    ]
    if args.scatter:
        callbacks.append(PredictionScatterLogger(out_dir=plots_dir))
    if early_stop_cb is not None:
        callbacks.append(early_stop_cb)

    trainer = construct_trainer(
        cfg,
        callbacks=callbacks,
        logger=pl_logger if is_wandb else False,
    )

    trainer.fit(
        integrator,
        train_dataloaders=data_loader.train_dataloader(),
        val_dataloaders=data_loader.val_dataloader(),
        ckpt_path=args.ckpt_path,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
