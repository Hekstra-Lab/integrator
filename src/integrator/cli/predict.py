import argparse
import logging
import re
from pathlib import Path

from integrator.cli.utils.logger import setup_logging
from integrator.io import write_mtz_from_preds, write_refl_from_preds
from integrator.utils import resolve_source_data_dir

logger = logging.getLogger(__name__)


def _resolve_and_check_metadata(config: dict) -> Path:
    """Resolve source_data_dir and verify metadata.npy exists.

    Used by --mfx-writeback to obtain the metadata.npy path automatically
    from manifest.yaml (chunked_rotation_data) or from data_loader.args.data_dir
    (rotation_data).

    Raises:
        FileNotFoundError: metadata.npy not found under the resolved directory,
                           with the expected path included in the message.
        ValueError / KeyError: propagated from resolve_source_data_dir when the
                               config or manifest is malformed.
    """
    source_data_dir = resolve_source_data_dir(config)
    metadata_path = source_data_dir / "metadata.npy"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.npy not found at expected path:\n"
            f"  {metadata_path}\n"
            "Verify that source_data_dir in manifest.yaml points to the correct "
            "original shoebox dataset directory, or check that the file has not "
            "been moved or deleted."
        )
    return metadata_path


def _resolve_and_check_mtz_sources(config: dict) -> Path:
    """Resolve source_data_dir and verify metadata.npy + dataset.yaml both exist.

    Used by --write-mtz.  MTZ export requires both metadata.npy (wavelength data)
    and dataset.yaml (crystal geometry).

    Raises:
        FileNotFoundError: either required file not found under source_data_dir,
                           with the expected path included in the message.
        ValueError / KeyError: propagated from resolve_source_data_dir when the
                               config or manifest is malformed.
    """
    source_data_dir = resolve_source_data_dir(config)
    for fname in ("metadata.npy", "dataset.yaml"):
        fpath = source_data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"--write-mtz requires {fname} under source_data_dir but it was "
                f"not found:\n"
                f"  {fpath}\n"
                "Verify that source_data_dir in manifest.yaml points to the correct "
                "original shoebox dataset directory, or check that the file has not "
                "been moved or deleted."
            )
    return source_data_dir


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
    # MFX write-back extension (Thao): keep Luis's --write-refl path,
    # but allow writing predictions into many original MFX .refl files.
    parser.add_argument(
        "--mfx-writeback",
        action="store_true",
        help=(
            "Use MFX many-file .refl write-back instead of the default "
            "single-source .refl write-back. Requires --write-refl and "
            "--original-refl-dir."
        ),
    )
    parser.add_argument(
        "--original-refl-dir",
        type=str,
        default=None,
        help=(
            "Original MFX/cctbx out folder containing "
            "idx-data_*_integrated.refl/.expt files. Used only with "
            "--write-refl --mfx-writeback."
        ),
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

    # Path to input refl file for Luis's original single-file write-back.
    # MFX write-back extension (Thao): the many-file MFX path uses
    # --original-refl-dir + metadata.npy instead, so it does not require
    # output.refl_file in the YAML config.
    refl_file = config.get("output", {}).get("refl_file")
    if args.write_refl and not args.mfx_writeback and not refl_file:
        raise ValueError(
            "--write-refl requires 'output.refl_file' in the YAML config"
        )
    if args.mfx_writeback:
        if not args.write_refl:
            raise ValueError("--mfx-writeback must be used with --write-refl")
        if args.original_refl_dir is None:
            raise ValueError("--mfx-writeback requires --original-refl-dir")

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

            if args.mfx_writeback:
                # MFX write-back: resolve and validate the original shoebox data
                # directory automatically from manifest.yaml (chunked) or from
                # data_loader.args.data_dir (rotation_data).
                # --original-refl-dir remains separate: it points to the
                # original .refl/.expt files, which may differ from source_data_dir.
                from integrator.io.pred_io import write_mfx_refl_from_preds

                metadata_path = _resolve_and_check_metadata(config)
                write_mfx_refl_from_preds(
                    ckpt_dir=ckpt_dir,
                    metadata_path=metadata_path,
                    original_refl_dir=Path(args.original_refl_dir),
                    out_dir=pred_dir / "mfx_refl_writeback",
                    filetype="parquet",
                )
            else:
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
            source_data_dir = _resolve_and_check_mtz_sources(config)
            write_mtz_from_preds(
                pred_data=pred_data,
                metadata_path=source_data_dir / "metadata.npy",
                data_dir=source_data_dir,
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
