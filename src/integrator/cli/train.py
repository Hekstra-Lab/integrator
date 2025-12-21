import argparse
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_cli_overrides(
    cfg: dict,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    data_path: Path | None = None,
) -> dict:
    base = cfg  # plain dict
    updates: dict[str, Any] = {}
    if epochs is not None:
        updates.setdefault("trainer", {}).setdefault("args", {})[
            "max_epochs"
        ] = epochs
    if batch_size is not None:
        updates.setdefault("data_loader", {}).setdefault("args", {})[
            "batch_size"
        ] = batch_size
    if data_path is not None:
        updates.setdefault("data_loader", {}).setdefault("args", {})[
            "data_dir"
        ] = str(data_path)

    merged = _deep_merge(base, updates)
    return merged


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
    return parser.parse_args()


def main():
    args = parse_args()

    import reciprocalspaceship as rs
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    from integrator.callbacks import (
        LogFano,
        Plotter,
        PlotterLD,
        PredWriter,
        assign_labels,
    )
    from integrator.utils import (
        clean_from_memory,
        create_data_loader,
        create_integrator,
        create_trainer,
        load_config,
        mtz_writer,
    )

    torch.set_float32_matmul_precision("medium")

    # load configuration file
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(
        cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_path=args.data_path,
    )

    # load data
    data = create_data_loader(cfg)

    # load wandb logger
    logger = WandbLogger(
        project=args.wb_project,
        save_dir=args.save_dir,
    )

    # get logging directory
    logdir = logger.experiment.dir

    path = Path("wandb_log_dir.txt")
    path.write_text(logdir)

    config_out = Path(logdir) / "config_copy.yaml"
    cfg_json = deepcopy(cfg)

    logger.log_hyperparams(cfg_json)

    with open(config_out, "w") as f:
        yaml.safe_dump(cfg_json, f, sort_keys=False)

    # assign validation/train labels to each shoebox
    assign_labels(dataset=data, save_dir=logdir)

    # create integrator
    integrator = create_integrator(cfg)

    # create prediction writer
    pred_writer = PredWriter(
        output_dir=None,
        write_interval=cfg["trainer"]["args"]["callbacks"]["pred_writer"][
            "write_interval"
        ],
    )

    # to generate plots

    if cfg["integrator"]["args"]["data_dim"] == "3d":
        plotter = Plotter(n_profiles=10)
    elif cfg["integrator"]["args"]["data_dim"] == "2d":
        plotter = PlotterLD(
            n_profiles=10,
            plot_every_n_epochs=1,
            d=cfg["logger"]["d"],
            h=cfg["logger"]["h"],
            w=cfg["logger"]["w"],
        )
    else:
        print("Incorrect data_dim value")

    fano_logger = LogFano()

    # to save checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.experiment.dir
        + "/checkpoints",  # when using wandb logger
        filename="{epoch}-{val_loss:.2f}",
        every_n_epochs=1,
        save_top_k=-1,
        save_last="link",
    )

    # to train integrator
    trainer = create_trainer(
        cfg,
        callbacks=[
            fano_logger,
            pred_writer,
            checkpoint_callback,
            plotter,
        ],
        logger=logger,
    )

    # Fit the model
    trainer.fit(
        integrator,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )

    # logdir: "/path/to/lightning_logs/wandb/run*/files/"

    cfg["trainer"]["args"]["logger"] = False

    clean_from_memory(
        pred_writer, pred_writer, pred_writer, checkpoint_callback
    )

    # prediction
    logdir = Path(logdir)
    pred_dir = logdir.parent / "predictions"
    pred_dir.mkdir(exist_ok=True)

    checkpoints = list(logdir.glob("**/*.ckpt"))

    pattern = re.compile(r"epoch=\d+")
    for ckpt in checkpoints:
        if re.search(pattern, ckpt.as_posix()):
            groups = re.findall(pattern, ckpt.as_posix())
            epoch = groups[0].replace("=", "_")
            out_dir = pred_dir.parent.as_posix() + f"/predictions/{epoch}"
            Path(out_dir).mkdir(exist_ok=True)
            pred_writer = PredWriter(
                output_dir=out_dir,
                write_interval="epoch",
            )
            trainer = create_trainer(
                cfg,
                callbacks=[pred_writer],
                logger=None,
            )
            ckpt_ = torch.load(ckpt.as_posix())
            integrator = create_integrator(cfg)
            integrator.load_state_dict(ckpt_["state_dict"])
            if torch.cuda.is_available():
                integrator.to(torch.device("cuda"))
            integrator.eval()
            trainer.predict(
                integrator,
                return_predictions=False,
                dataloaders=data.predict_dataloader(),
            )

    # Write output mtz files
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

            #           # switch to typer arg
            # if args.filter_sigi:
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


# import re
# from copy import deepcopy
# from pathlib import Path
# from typing import Annotated, Any
#
# import typer
# import yaml
#
# app = typer.Typer()
#
#
# def _deep_merge(a: dict, b: dict) -> dict:
#     out = deepcopy(a)
#     for k, v in b.items():
#         if isinstance(v, dict) and isinstance(out.get(k), dict):
#             out[k] = _deep_merge(out[k], v)
#         else:
#             out[k] = v
#     return out
#
#
# def apply_cli_overrides(
#     cfg: dict,
#     *,
#     epochs: int | None = None,
#     batch_size: int | None = None,
#     data_path: Path | None = None,
# ) -> dict:
#     base = cfg  # plain dict
#     updates: dict[str, Any] = {}
#     if epochs is not None:
#         updates.setdefault("trainer", {}).setdefault("args", {})[
#             "max_epochs"
#         ] = epochs
#     if batch_size is not None:
#         updates.setdefault("data_loader", {}).setdefault("args", {})[
#             "batch_size"
#         ] = batch_size
#     if data_path is not None:
#         updates.setdefault("data_loader", {}).setdefault("args", {})[
#             "data_dir"
#         ] = str(data_path)
#
#     merged = _deep_merge(base, updates)
#     return merged
#
#
# @app.command()
# def train(
#     config: Annotated[
#         Path,
#         typer.Argument(help="Path to YAML config file"),
#     ],
#     epochs: Annotated[
#         int | None,
#         typer.Argument(
#             help="Number of epochs to train for",
#         ),
#     ] = 2,
#     batch_size: Annotated[
#         int,
#         typer.Argument(help="The size of a train batch"),
#     ] = 64,
#     data_path: Annotated[
#         Path | None,
#         typer.Argument(help="Override data path in config.yaml file"),
#     ] = None,
#     wandb_project: Annotated[
#         str, typer.Option(help="The name of W&B project")
#     ] = "default",
#     save_dir: Annotated[
#         str,
#         typer.Option(help="Path to logging directory"),
#     ] = "/n/netscratch/hekstra_lab/Lab/laldama/lightning_logs/",
# ):
#     import reciprocalspaceship as rs
#     import torch
#     from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
#     from pytorch_lightning.loggers import WandbLogger
#
#     from integrator.callbacks import (
#         LogFano,
#         Plotter,
#         PlotterLD,
#         PredWriter,
#         assign_labels,
#     )
#     from integrator.utils import (
#         clean_from_memory,
#         create_data_loader,
#         create_integrator,
#         create_trainer,
#         load_config,
#         mtz_writer,
#     )
#
#     torch.set_float32_matmul_precision("medium")
#
#     # load configuration file
#     cfg = load_config(config)
#     cfg = apply_cli_overrides(
#         cfg,
#         epochs=epochs,
#         batch_size=batch_size,
#         data_path=data_path,
#     )
#
#     # load data
#     data = create_data_loader(cfg)
#
#     # load wandb logger
#     logger = WandbLogger(
#         project=wandb_project,
#         save_dir=save_dir,
#     )
#
#     # get logging directory
#     logdir = logger.experiment.dir
#
#     path = Path("wandb_log_dir.txt")
#     path.write_text(logdir)
#
#     config_out = Path(logdir) / "config_copy.yaml"
#     cfg_json = deepcopy(cfg)
#
#     logger.log_hyperparams(cfg_json)
#
#     with open(config_out, "w") as f:
#         yaml.safe_dump(cfg_json, f, sort_keys=False)
#
#     # assign validation/train labels to each shoebox
#     assign_labels(dataset=data, save_dir=logdir)
#
#     # create integrator
#     integrator = create_integrator(cfg)
#
#     # create prediction writer
#     pred_writer = PredWriter(
#         output_dir=None,
#         write_interval=cfg["trainer"]["args"]["callbacks"]["pred_writer"][
#             "write_interval"
#         ],
#     )
#
#     # to generate plots
#
#     if cfg["integrator"]["args"]["data_dim"] == "3d":
#         plotter = Plotter(n_profiles=10)
#     elif cfg["integrator"]["args"]["data_dim"] == "2d":
#         plotter = PlotterLD(
#             n_profiles=10,
#             plot_every_n_epochs=1,
#             d=cfg["logger"]["d"],
#             h=cfg["logger"]["h"],
#             w=cfg["logger"]["w"],
#         )
#     else:
#         print("Incorrect data_dim value")
#
#     fano_logger = LogFano()
#
#     # to save checkpoints
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=logger.experiment.dir
#         + "/checkpoints",  # when using wandb logger
#         filename="{epoch}-{val_loss:.2f}",
#         every_n_epochs=1,
#         save_top_k=-1,
#         save_last="link",
#     )
#
#     # to train integrator
#     trainer = create_trainer(
#         cfg,
#         callbacks=[
#             fano_logger,
#             pred_writer,
#             checkpoint_callback,
#             plotter,
#             RichProgressBar(),
#         ],
#         logger=logger,
#     )
#
#     # Fit the model
#     trainer.fit(
#         integrator,
#         train_dataloaders=data.train_dataloader(),
#         val_dataloaders=data.val_dataloader(),
#     )
#
#     # logdir: "/path/to/lightning_logs/wandb/run*/files/"
#
#     cfg["trainer"]["args"]["logger"] = False
#
#     clean_from_memory(
#         pred_writer, pred_writer, pred_writer, checkpoint_callback
#     )
#
#     # prediction
#     logdir = Path(logdir)
#     pred_dir = logdir.parent / "predictions"
#     pred_dir.mkdir(exist_ok=True)
#
#     checkpoints = list(logdir.glob("**/*.ckpt"))
#
#     pattern = re.compile(r"epoch=\d+")
#     for ckpt in checkpoints:
#         if re.search(pattern, ckpt.as_posix()):
#             groups = re.findall(pattern, ckpt.as_posix())
#             epoch = groups[0].replace("=", "_")
#             out_dir = pred_dir.parent.as_posix() + f"/predictions/{epoch}"
#             Path(out_dir).mkdir(exist_ok=True)
#             pred_writer = PredWriter(
#                 output_dir=out_dir, write_interval="max_epochs"
#             )
#             trainer = create_trainer(
#                 cfg,
#                 callbacks=[pred_writer],
#                 logger=None,
#             )
#             ckpt_ = torch.load(ckpt.as_posix())
#             integrator = create_integrator(cfg)
#             integrator.load_state_dict(ckpt_["state_dict"])
#             if torch.cuda.is_available():
#                 integrator.to(torch.device("cuda"))
#             integrator.eval()
#             trainer.predict(
#                 integrator,
#                 return_predictions=False,
#                 dataloaders=data.predict_dataloader(),
#             )
#
#     # Write output mtz files
#     pred_dir.glob("**/preds.pt")
#
#     pattern = re.compile(r"epoch_\d")
#     for p in pred_dir.iterdir():
#         if re.search(pattern, p.as_posix()):
#             groups = re.findall(pattern, p.as_posix())
#             epoch = groups[0]
#             pred_file = list(p.glob("preds.pt"))[0].as_posix()
#             mtz_path = Path(p.as_posix() + "/preds.mtz").as_posix()
#
#             mtz_writer(pred_path=pred_file, file_name=mtz_path)
#             ds = rs.read_mtz(mtz_path)
#
#             # remove refls with SIGI==0.0
#             n_filtered = 0
#
#             #           # switch to typer arg
#             # if args.filter_sigi:
#             #
#             mask = ds["SIGI"] == 0.0
#             n_filtered = mask.sum()
#             ds = ds[~mask]
#
#             # Include all centrics in friedel plus
#             plus = ds.hkl_to_asu()["M/ISYM"].to_numpy() % 2 == 1
#             centrics = ds.label_centrics().CENTRIC.to_numpy()
#             plus |= centrics
#             ds[plus].write_mtz(
#                 Path(p.as_posix() + "/friedel_plus.mtz").as_posix()
#             )
#             ds[~plus].write_mtz(
#                 Path(p.as_posix() + "/friedel_minus.mtz").as_posix()
#             )
#
#             # report for file
#             log_path = p.as_posix() + "/filter_log.txt"
#             with open(log_path, "w") as f:
#                 f.write(f"file: {p.as_posix()}\n")
#                 f.write(f"refls with sigi0: {n_filtered}\n")
#
#
# # %%
# if __name__ == "__main__":
#     pass
