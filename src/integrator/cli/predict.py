import argparse
import logging
import re

from integrator.cli.utils.logger import setup_logging
from integrator.io import write_mtz_from_preds, write_refl_from_preds

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="integrator.predict",
        description="Predict from a set of pytorch.ckpt files",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=False,
        help="Run dir with run_paths.yaml (config + checkpoints + predictions)",
    )
    # explicit overrides: use these to run without a --run-dir, or to override
    # individual pieces of one
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML (overrides --run-dir's config)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="A .ckpt file or a directory of them (overrides --run-dir's checkpoints)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output dir for predictions (default: run-dir predictions/, else <ckpt>/predictions)",
    )
    parser.add_argument(
        "--write-refl",
        action="store_true",
        help="Write predictions as a .refl file",
    )
    parser.add_argument(
        "--write-mtz",
        action="store_true",
        help="Write predictions as an .mtz file; for polychromatic data only",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Integer value specifying the size of each training batch",
    )
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="Print the predict_keys available from this data/model, then exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v = INFO, -vv = DEBUG)",
    )

    return parser.parse_args()


def _resolve_sources(args):
    """Resolve (config, checkpoints, pred_dir) from --run-dir and/or the explicit
    --config / --ckpt / --out-dir overrides."""
    from pathlib import Path

    import yaml

    from integrator.utils import apply_dataset_defaults, load_config

    meta: dict = {}
    if args.run_dir:
        meta = yaml.safe_load(
            (Path(args.run_dir) / "run_paths.yaml").read_text()
        )

    config_path = args.config or meta.get("config")
    if not config_path:
        raise SystemExit("integrator.predict: provide --config or --run-dir")
    config = apply_dataset_defaults(load_config(config_path))

    def _log_dir() -> Path:
        return Path(meta.get("log_dir") or meta["wandb"]["log_dir"])

    if args.ckpt:
        p = Path(args.ckpt)
        if p.is_dir():
            checkpoints = [
                c for c in sorted(p.glob("**/*.ckpt")) if c.name != "last.ckpt"
            ]
        elif p.exists():
            checkpoints = [p]
        else:
            raise SystemExit(f"integrator.predict: --ckpt not found: {p}")
    elif meta:
        checkpoints = sorted(_log_dir().glob("**/epoch*.ckpt"))
    else:
        raise SystemExit("integrator.predict: provide --ckpt or --run-dir")
    if not checkpoints:
        raise SystemExit("integrator.predict: no checkpoints found")

    if args.out_dir:
        pred_dir = Path(args.out_dir)
    elif meta:
        pred_dir = Path(
            meta.get("predictions_dir") or _log_dir().parent / "predictions"
        )
    else:
        pred_dir = checkpoints[0].parent / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    return config, checkpoints, pred_dir


def main():
    from pathlib import Path

    import torch

    torch.set_float32_matmul_precision("high")

    from integrator.callbacks import BatchPredWriter
    from integrator.utils import (
        construct_data_loader,
        construct_integrator,
        construct_trainer,
        inject_binning_labels,
    )

    args = parse_args()

    setup_logging(args.verbose)

    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("Starting Predictions")

    config, checkpoints, pred_dir = _resolve_sources(args)
    logger.info(
        "Found %d checkpoint(s); predictions -> %s", len(checkpoints), pred_dir
    )

    data_loader = construct_data_loader(config)
    data_loader.setup()
    inject_binning_labels(data_loader, config)

    if args.list_keys:
        integrator = construct_integrator(config)
        integrator.eval()
        batch = next(iter(data_loader.predict_dataloader()))
        with torch.no_grad():
            out = integrator(*batch)
        keys = sorted(out["forward_out"].keys())
        print(f"Available predict_keys ({len(keys)}):")
        for k in keys:
            print(f"  {k}")
        return

    # path to input refl file (only needed for --write-refl)
    refl_file = config.get("output", {}).get("refl_file")
    if args.write_refl and not refl_file:
        raise ValueError(
            "--write-refl requires 'output.refl_file' in the YAML config"
        )

    epoch_re = re.compile(r"epoch=(\d+)")
    for ckpt in checkpoints:
        m = epoch_re.search(ckpt.name)
        epoch = int(m.group(1)) if m else 0
        ckpt_dir = pred_dir / (f"epoch_{epoch:04d}" if m else ckpt.stem)

        logger.info("Processing checkpoint: %s", ckpt.name)
        logger.debug("Checkpoint path: %s", ckpt)

        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Skip prediction if outputs already exist, but still
        # run post-processing (write-refl, write-mtz) below.
        has_preds = (
            any(ckpt_dir.glob("preds_epoch_*"))
            or (ckpt_dir / "pred.parquet").exists()
        )
        if has_preds:
            logger.info(
                "Predictions for epoch %d already exist: skipping inference",
                epoch,
            )
        else:
            integrator = construct_integrator(config)
            integrator.load_state_dict(
                torch.load(ckpt.as_posix())["state_dict"]
            )
            if torch.cuda.is_available():
                integrator.to(torch.device("cuda"))
            integrator.eval()

            # qp_mean is a large per-pixel vector:
            # shard to manage memory;
            # otherwise everything fits in one pred.parquet
            partition = "qp_mean" in integrator.predict_keys
            pred_writer = BatchPredWriter(
                output_dir=ckpt_dir,
                write_interval="batch",
                epoch=epoch,
                partition=partition,
            )
            trainer = construct_trainer(
                config,
                callbacks=[pred_writer],
                logger=False,
            )
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
                filetype="parquet",
            )

        if args.write_mtz:
            from integrator.io import get_pred_files

            logger.info("Writing .mtz output for epoch %d", epoch)
            pred_data = get_pred_files(ckpt_dir=ckpt_dir, filetype="parquet")
            data_dir = Path(config["data_loader"]["args"]["data_dir"])
            write_mtz_from_preds(
                pred_data=pred_data,
                metadata_path=data_dir / "metadata.npy",
                data_dir=data_dir,
                out_path=ckpt_dir / f"preds_epoch_{epoch:04d}.mtz",
            )

    logger.info("Prediction complete!")

    try:
        import polars as pl
    except ImportError:
        logger.warning(
            "polars not installed: skipping test_preds_all.parquet"
            " aggregation."
        )
    else:
        parquet_glob = str(pred_dir / "*" / "*.parquet")
        from glob import glob as _glob

        if not _glob(parquet_glob):
            logger.info(
                "No parquet files under %s: skipping test-set aggregation"
                " (use --save-preds-as parquet if you want it).",
                pred_dir,
            )
        else:
            lf = pl.scan_parquet(parquet_glob, include_file_paths="src")
            schema = lf.collect_schema()
            if "is_test" not in schema:
                logger.info(
                    "is_test not in predictions: skipping test-set aggregation"
                )
            else:
                out_path = pred_dir / "test_preds_all.parquet"
                logger.info("Aggregating test predictions -> %s", out_path)
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
