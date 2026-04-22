import argparse
import logging
import re
from copy import deepcopy
from pathlib import Path

import yaml

from .utils.io import _apply_cli_overrides
from .utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train integrator model")

    # Required
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

    # Paths
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

    # Misc
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

    torch.set_float32_matmul_precision("high")

    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    from integrator.callbacks import (
        EpochMetricRecorder,
        LossTraceRecorder,
        Plotter,
        PlotterLD,
        assign_labels,
    )
    from integrator.utils import (
        construct_data_loader,
        construct_integrator,
        construct_trainer,
        inject_binning_labels,
        load_config,
        prepare_global_priors,
        prepare_per_bin_priors,
        save_run_artifacts,
    )
    from integrator.utils.factory_utils import _collect_resolved_paths

    # parse args
    args = parse_args()

    # logger
    setup_logging(args.verbose)

    # to use gpu
    torch.set_float32_matmul_precision("medium")

    # load configuration file
    cfg = load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args=args)

    # load data
    logger.info("Starting Training")

    # Auto-generate prior files if needed by the loss
    prior_events: list[dict] = []
    prepare_per_bin_priors(cfg, events_out=prior_events)
    prepare_global_priors(cfg)

    for event in prior_events:
        action = event["action"]
        if action == "reused":
            logger.info(f"Prior file reused: {event['file']}")
        else:
            logger.info(
                f"Prior file {action}: {event['file']} — {event['reason']}"
            )

    # load data
    data_loader = construct_data_loader(cfg)
    data_loader.setup()
    inject_binning_labels(data_loader, cfg)

    # Tags for identification
    tags = [
        cfg["integrator"]["name"],
        cfg["integrator"]["args"]["data_dim"],
        cfg["surrogates"]["qi"]["name"],
    ]

    # load wandb logger
    wb_logger = WandbLogger(
        project=args.wb_project,
        save_dir=args.save_dir,
        tags=tags,
    )

    # Explicit git capture; gets integrator git location;
    # NOTE: for development only
    import subprocess

    _start = Path(__file__).resolve()
    _repo_root = next(
        (p for p in [_start, *_start.parents] if (p / ".git").exists()),
        None,
    )
    if _repo_root is not None:
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
            wb_logger.experiment.config.update(
                {"git_sha": _sha, "git_dirty": _dirty},
                allow_val_change=True,
            )
        except Exception as _exc:
            logger.warning("git capture failed: %s", _exc)

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

    # Resolve every file the factory will load, with absolute paths
    resolved_paths = _collect_resolved_paths(cfg)

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
        "resolved_paths": resolved_paths,
        "prior_events": prior_events,
    }

    wb_logger.log_hyperparams(metadata)

    m_fname = run_dir / "run_metadata.yaml"
    (m_fname).write_text(yaml.safe_dump(metadata))

    logger.info(f"Saved run_metadata: {m_fname}")

    # assign validation/train labels to each shoebox
    assign_labels(dataset=data_loader, save_dir=logdir.as_posix())

    # create integrator
    integrator = construct_integrator(cfg)

    # save prior artifacts (rescaled concentration, param counts, etc.)
    save_run_artifacts(integrator, cfg, logdir)

    # Echo resolved file paths (same data that's also in run_metadata.yaml)
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

    # Callbacks
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

    data_dim = cfg["integrator"]["args"]["data_dim"]
    if data_dim == "3d":
        plotter = Plotter(n_profiles=10)
    elif data_dim == "2d":
        plotter = PlotterLD(
            n_profiles=10,
            plot_every_n_epochs=1,
            d=cfg["integrator"]["args"]["d"],
            h=cfg["integrator"]["args"]["h"],
            w=cfg["integrator"]["args"]["w"],
        )
    else:
        raise ValueError(
            f"Specified shoebox data dimension is incompatible: data_dim={data_dim}"
        )

    loss_trace_recorder = LossTraceRecorder(
        out_dir=logdir / "loss_traces",
    )

    # Early-stopping callback (optional).
    early_stop_cb = None
    es_cfg = cfg.get("early_stop")
    if es_cfg:
        early_stop_cb = EarlyStopping(
            monitor=es_cfg["monitor"],
            mode=es_cfg.get("mode", "min"),
            patience=int(es_cfg.get("patience", 3)),
            min_delta=float(es_cfg.get("min_delta", 0.0)),
            strict=bool(es_cfg.get("strict", True)),
            verbose=True,
        )
        logger.info(
            "EarlyStopping: monitor=%s mode=%s patience=%d min_delta=%.4f",
            es_cfg["monitor"],
            es_cfg.get("mode", "min"),
            int(es_cfg.get("patience", 3)),
            float(es_cfg.get("min_delta", 0.0)),
        )

    # to save checkpoints. When early-stop is active, default to
    # save_top_k=1 (pick the best according to the monitored metric)
    ckpt_cfg = cfg.get("checkpoint", {}) or {}
    default_top_k = 1 if early_stop_cb else -1
    save_top_k = int(ckpt_cfg.get("save_top_k", default_top_k))
    ckpt_monitor = ckpt_cfg.get(
        "monitor",
        es_cfg["monitor"] if es_cfg else None,
    )
    ckpt_mode = ckpt_cfg.get(
        "mode",
        es_cfg.get("mode", "min") if es_cfg else "min",
    )
    ckpt_dir = logdir / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:04d}",
        every_n_epochs=1,
        save_top_k=save_top_k,
        save_last="link",
        monitor=ckpt_monitor if save_top_k > 0 else None,
        mode=ckpt_mode if save_top_k > 0 else "min",
    )
    logger.info(
        "Checkpoints: dir=%s save_top_k=%d monitor=%s",
        ckpt_dir.as_posix(),
        save_top_k,
        ckpt_monitor,
    )

    callbacks = [
        val_epoch_recorder,
        train_epoch_recorder,
        loss_trace_recorder,
        checkpoint_callback,
    ]
    if early_stop_cb is not None:
        callbacks.append(early_stop_cb)

    # PyTorch-Lightning Trainer
    trainer = construct_trainer(
        cfg,
        callbacks=callbacks,
        logger=wb_logger,
    )

    # Fit the model
    trainer.fit(
        integrator,
        train_dataloaders=data_loader.train_dataloader(),
        val_dataloaders=data_loader.val_dataloader(),
    )

    logger.info("Traning complete!")


def write_mtz_files():
    pred_dir.glob("**/preds.pt")
    pattern = re.compile(r"epoch_\d")
    for p in pred_dir.iterdir():
        if re.search(pattern, p.as_posix()):
            groups = re.findall(pattern, p.as_posix())
            epoch = groups[0]
            pred_file = list(p.glob("preds.pt"))[0].as_posix()
            mtz_path = Path(p.as_posix() + "/preds.mtz").as_posix()

            mtz_writer(pred_path=pred_file, file_name=mtz_path)
            ds = rs.read_mtz(mtz_path)

            # remove refls with SIGI==0.0
            n_filtered = 0

            #
            mask = ds["SIGI"] == 0.0
            n_filtered = mask.sum()
            ds = ds[~mask]

            # Include all centrics in friedel plus
            plus = ds.hkl_to_asu()["M/ISYM"].to_numpy() % 2 == 1
            centrics = ds.label_centrics().CENTRIC.to_numpy()
            plus |= centrics
            ds[plus].write_mtz(
                Path(p.as_posix() + "/friedel_plus.mtz").as_posix()
            )
            ds[~plus].write_mtz(
                Path(p.as_posix() + "/friedel_minus.mtz").as_posix()
            )

            # report for file
            log_path = p.as_posix() + "/filter_log.txt"

            with open(log_path, "w") as f:
                f.write(f"file: {p.as_posix()}\n")
                f.write(f"refls with sigi0: {n_filtered}\n")


if __name__ == "__main__":
    main()
