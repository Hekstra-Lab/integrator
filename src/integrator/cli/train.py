import argparse
import logging
import re
from copy import deepcopy
from pathlib import Path

import yaml

from .utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train W&B model")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="The size of a train batch",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to directory cotaining TensorDatasets",
    )
    parser.add_argument(
        "--wb-project",
        type=str,
        help="Name of the W&B project to save to",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/n/netscratch/hekstra_lab/Lab/laldama/lightning_logs/",
        help="Path to store local W&B logs",
    )

    parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to run directory; located where integrator.train is called",
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
    import os

    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    from integrator.callbacks import (
        LogFano,
        Plotter,
        PlotterLD,
        assign_labels,
    )
    from integrator.utils import (
        construct_data_loader,
        construct_integrator,
        construct_trainer,
        load_config,
    )

    args = parse_args()
    setup_logging(args.verbose)

    torch.set_float32_matmul_precision("medium")

    # load configuration file
    cfg = load_config(args.config)

    # cfg = apply_cli_overrides(
    #     cfg,
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     data_path=args.data_path,
    # )

    # load data

    logger.info("Starting Training")

    data_loader = construct_data_loader(cfg)
    data_loader.setup()

    # load wandb logger
    wb_logger = WandbLogger(
        project=args.wb_project,
        save_dir=args.save_dir,
    )

    # Logging directory
    logdir = Path(wb_logger.experiment.dir)
    run_dir = Path(args.run_dir)

    # Write a copy of the config.yaml file
    config_copy = logdir / "config_copy.yaml"
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

    (run_dir / "run_metadata.yaml").write_text(yaml.safe_dump(metadata))

    # assign validation/train labels to each shoebox
    assign_labels(dataset=data_loader, save_dir=logdir.as_posix())

    # create integrator
    integrator = construct_integrator(cfg)

    # Callbacks
    data_dim = cfg["integrator"]["args"]["data_dim"]
    if data_dim == "3d":
        plotter = Plotter(n_profiles=10)
    elif data_dim == "2d":
        plotter = PlotterLD(
            n_profiles=10,
            plot_every_n_epochs=1,
            d=cfg["wb_logger"]["d"],
            h=cfg["wb_logger"]["h"],
            w=cfg["wb_logger"]["w"],
        )
    else:
        raise ValueError(
            f"Specified shoebox data dimension is incompatible: data_dim={data_dim}"
        )

    fano_wb_logger = LogFano()

    # to save checkpoints
    ckpt_dir = logdir / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:04d}-{val_loss:.2f}",
        every_n_epochs=1,
        save_top_k=-1,
        save_last="link",
    )

    # PyTorch-Lightning Trainer
    trainer = construct_trainer(
        cfg,
        callbacks=[
            fano_wb_logger,
            checkpoint_callback,
            plotter,
        ],
        logger=wb_logger,
    )

    # Fit the model
    trainer.fit(
        integrator,
        train_dataloaders=data_loader.train_dataloader(),
        val_dataloaders=data_loader.val_dataloader(),
    )

    print("Traning complete!")

    # clean_from_memory(
    #     pred_writer, pred_writer, pred_writer, checkpoint_callback
    # )
    #
    # # prediction
    # logdir = Path(logdir)
    # pred_dir = logdir.parent / "predictions"
    # pred_dir.mkdir(exist_ok=True)
    #
    # checkpoints = list(logdir.glob("**/*.ckpt"))
    #
    # pattern = re.compile(r"epoch=\d+")
    # for ckpt in checkpoints:
    #     if re.search(pattern, ckpt.as_posix()):
    #         groups = re.findall(pattern, ckpt.as_posix())
    #         epoch = groups[0].replace("=", "_")
    #         out_dir = pred_dir.parent.as_posix() + f"/predictions/{epoch}"
    #         Path(out_dir).mkdir(exist_ok=True)
    #
    #         pred_writer = PredWriter(
    #             output_dir=out_dir,
    #             write_interval="epoch",
    #         )
    #         trainer = construct_trainer(
    #             cfg,
    #             callbacks=[pred_writer],
    #             logger=None,
    #         )
    #         ckpt_ = torch.load(ckpt.as_posix())
    #         integrator = construct_integrator(cfg)
    #         integrator.load_state_dict(ckpt_["state_dict"])
    #         if torch.cuda.is_available():
    #             integrator.to(torch.device("cuda"))
    #         integrator.eval()
    #         trainer.predict(
    #             integrator,
    #             return_predictions=False,
    #             dataloaders=data_loader.predict_dataloader(),
    #         )
    #


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
            ds[plus].write_mtz(Path(p.as_posix() + "/friedel_plus.mtz").as_posix())
            ds[~plus].write_mtz(Path(p.as_posix() + "/friedel_minus.mtz").as_posix())

            # report for file
            log_path = p.as_posix() + "/filter_log.txt"

            with open(log_path, "w") as f:
                f.write(f"file: {p.as_posix()}\n")
                f.write(f"refls with sigi0: {n_filtered}\n")


if __name__ == "__main__":
    main()

#     import os
#     from pathlib import Path
#
#     import torch
#     from pytorch_lightning.callbacks import ModelCheckpoint
#     from pytorch_lightning.loggers import WandbLogger
#
#     from integrator.callbacks import (
#         LogFano,
#         Plotter,
#         PlotterLD,
#         assign_labels,
#     )
#     from integrator.utils import (
#         construct_data_loader,
#         construct_integrator,
#         construct_trainer,
#         load_config,
#     )
#
#     torch.set_float32_matmul_precision("medium")
#
#     # load configuration file
#     path = "/Users/luis/temp/data/test_yaml.yaml"
#     cfg = load_config(path)
#
#     # cfg = apply_cli_overrides(
#     #     cfg,
#     #     epochs=args.epochs,
#     #     batch_size=args.batch_size,
#     #     data_path=args.data_path,
#     # )
#
#     # load data
#     data_loader = construct_data_loader(cfg)
#     data_loader.setup()
#
#     # load wandb logger
#     logger = WandbLogger(
#         project="local-default",
#         save_dir="/Users/luis/temp/wandb/",
#     )
#
#     # get logging directory
#     logdir = Path(logger.experiment.dir)
#
#     run_dir = Path("/Users/luis/integrator/temp_run_dir/")
#
#     config_copy = logdir / "config_copy.yaml"
#     cfg_json = deepcopy(cfg)
#
#     logger.log_hyperparams(cfg_json)
#
#     with open(config_copy, "w") as f:
#         yaml.safe_dump(cfg_json, f, sort_keys=False)
#
#     metadata = {
#         "config": config_copy.as_posix(),
#         "slurm": {
#             "job_id": os.environ.get("SLURM_JOB_ID"),
#         },
#         "wandb": {
#             "project": "local-default",
#             "run_id": logger.experiment.id,
#             "entity": logger.experiment.entity,
#             "log_dir": logger.experiment.dir,
#         },
#     }
#
#     (run_dir / "run_metadata.yaml").write_text(yaml.safe_dump(metadata))
#
#     # assign validation/train labels to each shoebox
#     assign_labels(dataset=data_loader, save_dir=logdir.as_posix())
#
#     # %%
#     # create integrator
#     integrator = construct_integrator(cfg)
#
#     # to generate plots
#     data_dim = cfg["integrator"]["args"]["data_dim"]
#     if data_dim == "3d":
#         plotter = Plotter(n_profiles=10)
#     elif data_dim == "2d":
#         plotter = PlotterLD(
#             n_profiles=10,
#             plot_every_n_epochs=1,
#             d=cfg["logger"]["d"],
#             h=cfg["logger"]["h"],
#             w=cfg["logger"]["w"],
#         )
#     else:
#         raise ValueError(
#             f"Specified shoebox data dimension is incompatible: data_dim={data_dim}"
#         )
#
#     fano_logger = LogFano()
#
#     ckpt_dir = logdir / "checkpoints"
#
#     # to save checkpoints
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=ckpt_dir,
#         filename="{epoch:04d}-{val_loss:.2f}",
#         every_n_epochs=1,
#         save_top_k=-1,
#         save_last="link",
#     )
#
#     # to train integrator
#     trainer = construct_trainer(
#         cfg,
#         callbacks=[
#             fano_logger,
#             pred_writer,
#             checkpoint_callback,
#             plotter,
#         ],
#         logger=logger,
#     )
#
#     # Fit the model
#     trainer.fit(
#         integrator,
#         train_dataloaders=data_loader.train_dataloader(),
#         val_dataloaders=data_loader.val_dataloader(),
#     )
#
#     print("Traning complete!")
#
#     # %%
#     counts, sbox, masks, metadata = next(iter(data_loader.train_dataloader()))
#     out = integrator.forward(counts, sbox, masks, metadata)
#     forward_out = out["forward_out"]
#
#     forward_out.keys()
#
# integrator.logger.experiment.dir
