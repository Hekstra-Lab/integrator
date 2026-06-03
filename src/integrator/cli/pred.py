import argparse
import logging
import re

from integrator.cli.utils.io import write_refl_from_preds
from integrator.cli.utils.logger import setup_logging
from integrator.cli.utils.mtz_writer import write_mtz_from_preds

logger = logging.getLogger(__name__)


def _match_profile_basis_dim(config: dict, state_dict: dict) -> None:
    """Pin the learned_basis_profile latent_dim to match the checkpoint.

    The surrogate's basis dim is set by the warmstart basis file at train time.
    At predict time `skip_warmstart` drops that file, so without an explicit
    `latent_dim` the dim defaults (e.g. 8) and `load_state_dict` fails with a
    size mismatch. The checkpoint's `surrogates.qp.decoder.weight` is
    (output_dim, latent_dim), so we read latent_dim from it and inject it.

    Only `latent_dim` is touched: `input_dim` (= encoder output width) and
    `output_dim` come from the config and must stay honest, so a genuine
    config/encoder mismatch still surfaces as a clear load error.
    """
    qp = config.get("surrogates", {}).get("qp")
    if not isinstance(qp, dict) or qp.get("name") != "learned_basis_profile":
        return
    w = state_dict.get("surrogates.qp.decoder.weight")
    if w is None:
        return
    latent_dim = int(w.shape[1])
    qp.setdefault("args", {})["latent_dim"] = latent_dim
    logger.info("Set qp latent_dim=%d to match checkpoint", latent_dim)


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
        "--write-merged-mtz",
        type=str,
        default=None,
        help="Write merged MTZ from scaling model checkpoint (e.g. merged.mtz)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Integer value specifying the size of each training batch",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Process only the checkpoint for this epoch (single-checkpoint mode)",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="Process only the highest-epoch checkpoint (single-checkpoint mode)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Process only this explicit .ckpt path (overrides --epoch/--last and "
        "the run_metadata checkpoint search)",
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

    # writing prediction directories
    log_dir = Path(wandb_info["log_dir"])
    wandb_dir = log_dir.parent
    pred_dir = wandb_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    epoch_re = re.compile(r"epoch=(\d+)")

    # list of .ckpt files (optionally narrowed to a single checkpoint)
    if args.ckpt:
        checkpoints = [Path(args.ckpt)]
        if not checkpoints[0].exists():
            raise FileNotFoundError(f"--ckpt not found: {args.ckpt}")
    else:
        checkpoints = sorted(log_dir.glob("**/epoch*.ckpt"))
        if args.epoch is not None:
            checkpoints = [
                c
                for c in checkpoints
                if (m := epoch_re.search(c.name)) and int(m.group(1)) == args.epoch
            ]
            if not checkpoints:
                raise ValueError(
                    f"No checkpoint with epoch={args.epoch} found under {log_dir}"
                )
        elif args.last:
            parsed = [
                (int(m.group(1)), c)
                for c in checkpoints
                if (m := epoch_re.search(c.name))
            ]
            if not parsed:
                raise ValueError(f"No epoch*.ckpt found under {log_dir}")
            checkpoints = [max(parsed, key=lambda t: t[0])[1]]
    logger.info("Found %d checkpoint(s) to process", len(checkpoints))

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

        # Skip inference if only --write-merged-mtz is requested
        # (it reads from the checkpoint directly, not from predictions)
        needs_inference = args.write_refl or args.write_mtz
        only_merged = args.write_merged_mtz and not needs_inference

        has_preds = any(ckpt_dir.glob("preds_epoch_*"))
        if only_merged or has_preds:
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
            # The learned_basis_profile surrogate is sized by the warmstart
            # basis at train time, but skip_warmstart drops that file here, so
            # its dim would silently fall back to the default. Pin it from the
            # checkpoint (its own source of truth) to avoid a size mismatch.
            _match_profile_basis_dim(config, ckpt_["state_dict"])
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

        if args.write_merged_mtz:
            from integrator.cli.utils.merged_mtz_writer import (
                write_merged_mtz_from_checkpoint,
            )

            data_dir = Path(config["data_loader"]["args"]["data_dir"])
            ref_name = (
                config["data_loader"]["args"]
                .get("shoebox_file_names", {})
                .get("reference", "metadata.pt")
            )
            crystal_yaml = data_dir / "crystal.yaml"
            out_path = ckpt_dir / args.write_merged_mtz
            if not out_path.exists():
                logger.info(
                    "Writing merged .mtz for epoch %d", epoch
                )
                write_merged_mtz_from_checkpoint(
                    checkpoint_path=ckpt,
                    metadata_path=data_dir / ref_name,
                    crystal_yaml_path=crystal_yaml,
                    out_path=out_path,
                )
            else:
                logger.info(
                    "Merged .mtz for epoch %d already exists — skipping",
                    epoch,
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
