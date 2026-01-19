"""
# Example use
integrator.pred -v \
        --run-dir \
        --data-dim \
"""

import argparse
import logging
import re

from integrator.cli.utils.io import write_refl_from_preds
from integrator.cli.utils.logger import setup_logging

logger = logging.getLogger(__name__)

# def _get_pred_writer(config):
#     from integrator.callbacks import EpochPredWriter,BatchPredWriter
#
#     return
#


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict from a set of pytorch.ckpt files"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=False,
        help="Path to dir containing pred.yaml file",
    )
    parser.add_argument(
        "--data-dim",
        type=str,
        default="3d",
        help="String denoting shoebox dimensions; e.g. 2d or 3d",
    )
    parser.add_argument(
        "--write-refl",
        action="store_true",
        help="Write predictions as a .refl file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Integer value specifying the size of each training batch",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v = INFO, -vv = DEBUG)",
    )

    # TODO: Add mtz writer
    # parser.add_argument(
    #     "--write-as-mtz",
    #     action="store_true",
    #     help="Write predictions as a .mtz file",
    # )

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

    # path to input refl file
    refl_file = config["output"]["refl_file"]

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
        callbacks = []
        pred_writer = BatchPredWriter(
            output_dir=ckpt_dir,
            write_interval="batch",
        )
        callbacks.append(pred_writer)

        # Construct trainer
        trainer = construct_trainer(
            config,
            callbacks=callbacks,
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

        if args.write_refl:
            logger.info("Writing .refl output for epoch %d", epoch)
            write_refl_from_preds(
                ckpt_dir=ckpt_dir,
                refl_file=refl_file,
                epoch=epoch,
                config=config,
            )

    logger.info("Prediction complete!")


if __name__ == "__main__":
    main()
    #
    # import argparse
    # import re
    # from pathlib import Path
    #
    # import torch
    # import yaml
    #
    # from integrator.callbacks import PredWriter
    # from integrator.utils import (
    #     construct_data_loader,
    #     construct_integrator,
    #     construct_trainer,
    #     load_config,
    # )
    #
    # # Path to run-dir
    # run_dir = Path("/Users/luis/integrator/temp_run_dir/")
    #
    # # Reading in pred.yaml
    # meta = yaml.safe_load((run_dir / "run_metadata.yaml").read_text())
    # config = load_config(meta["config"])
    # wandb_info = meta["wandb"]
    # slurm_info = meta["slurm"]
    #
    # # writing prediction directories
    # log_dir = Path(wandb_info["log_dir"])
    # wandb_dir = log_dir.parent
    # pred_dir = wandb_dir / "predictions"
    # pred_dir.mkdir(exist_ok=True)
    #
    # # list of .ckpt files
    # checkpoints = sorted(log_dir.glob("**/epoch*.ckpt"))
    #
    # # load data
    # data_loader = construct_data_loader(config)
    # data_loader.setup()
    #
    # refl_file = config["output"]["refl_file"]
    #
    # epoch_re = re.compile(r"epoch=(\d+)")
    # for ckpt in checkpoints:
    #     # Finding checkpoint epoch
    #     m = epoch_re.search(ckpt.name)
    #     if not m:
    #         raise ValueError(f"Could not parse epoch from {ckpt.name}")
    #     epoch = int(m.group(1))
    #
    #     # Writing epoch prediction dir
    #     ckpt_dir = pred_dir / f"epoch_{epoch:04d}"
    #     ckpt_dir.mkdir(parents=True, exist_ok=True)
    #
    #     # Construct callbacks
    #     callbacks = []
    #     pred_writer = PredWriter(
    #         output_dir=ckpt_dir,
    #         write_interval="epoch",
    #     )
    #     callbacks.append(pred_writer)
    #
    #     # Construct trainer
    #     trainer = construct_trainer(
    #         config,
    #         callbacks=callbacks,
    #         logger=None,
    #     )
    #
    #     ckpt_ = torch.load(ckpt.as_posix())
    #     integrator = construct_integrator(config)
    #     integrator.load_state_dict(ckpt_["state_dict"])
    #
    #     # Use gpu if available
    #     if torch.cuda.is_available():
    #         integrator.to(torch.device("cuda"))
    #
    #     integrator.eval()
    #
    #     data = trainer.predict(
    #         integrator,
    #         return_predictions=True,
    #         dataloaders=data_loader.predict_dataloader(),
    #     )
    #
    #     write_refl_from_preds(
    #         ckpt_dir=ckpt_dir,
    #         refl_file=refl_file,
    #         epoch=epoch,
    #     )
    #
    #     # %%
    #     list(ckpt_dir.glob("preds*"))
