import re
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any

import reciprocalspaceship as rs
import typer
import yaml

from integrator.config.schema import Cfg
from integrator.utils import clean_from_memory, mtz_writer

app = typer.Typer()


def _deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_cli_overrides(
    cfg: Cfg,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    data_path: Path | None = None,
) -> Cfg:
    base = cfg.model_dump()  # plain dict
    updates: dict[str, Any] = {}
    if epochs is not None:
        updates.setdefault("trainer", {}).setdefault("args", {})[
            "max_epochs"
        ] = epochs
    if batch_size is not None:
        updates.setdefault("data_loader", {}).setdefault("params", {})[
            "batch_size"
        ] = batch_size
    if data_path is not None:
        updates.setdefault("data_loader", {}).setdefault("params", {})[
            "data_dir"
        ] = str(data_path)

    merged = _deep_merge(base, updates)
    return Cfg.model_validate(merged)


@app.command()
def train(
    config: Annotated[Path, typer.Option(help="Path to YAML config file")],
    epochs: Annotated[
        int | None, typer.Option(help="Number of epochs to train for")
    ] = None,
    batch_size: Annotated[
        int | None, typer.Option(help="The size of a train batch")
    ] = None,
    data_path: Annotated[
        Path | None,
        typer.Option(help="Override data path in config.yaml file"),
    ] = None,
):
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
    from pytorch_lightning.loggers import WandbLogger

    from integrator.callbacks import (
        Plotter,
        PlotterLD,
        PredWriter,
        assign_labels,
    )
    from integrator.utils import (
        create_data_loader,
        create_integrator,
        create_trainer,
        load_config,
    )

    torch.set_float32_matmul_precision("medium")

    # load configuration file
    cfg = load_config(config)
    cfg = apply_cli_overrides(
        cfg,
        epochs=epochs,
        batch_size=batch_size,
        data_path=data_path,
    )

    # load data
    data = create_data_loader(cfg.dict())

    # load wandb logger
    logger = WandbLogger(
        project="integrator_updated",
        save_dir="/n/netscratch/hekstra_lab/Lab/laldama/lightning_logs/",
    )

    # get logging directory
    logdir = logger.experiment.dir

    config_out = Path(logdir) / "config_copy.yaml"
    copy = cfg.model_dump(mode="json")

    logger.log_hyperparams(cfg.model_dump())

    with open(config_out, "w") as f:
        yaml.safe_dump(copy, f, sort_keys=False)

    # assign validation/train labels to each shoebox
    assign_labels(dataset=data, save_dir=logdir)

    # create integrator
    integrator = create_integrator(cfg.dict())

    # create prediction writer
    pred_writer = PredWriter(
        output_dir=None,
        write_interval=cfg.dict()["trainer"]["args"]["callbacks"][
            "pred_writer"
        ]["write_interval"],
    )

    # to generate plots

    if cfg.dict()["integrator"]["args"]["data_dim"] == "3d":
        plotter = Plotter(n_profiles=10)
    elif cfg.dict()["integrator"]["args"]["data_dim"] == "2d":
        plotter = PlotterLD(
            n_profiles=10,
            plot_every_n_epochs=1,
            d=cfg.model_dump()["logger"]["d"],
            h=cfg.model_dump()["logger"]["h"],
            w=cfg.model_dump()["logger"]["w"],
        )
    else:
        print("Incorrect data_dim value")

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
        cfg.dict(),
        callbacks=[
            pred_writer,
            checkpoint_callback,
            plotter,
            RichProgressBar(),
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
    integrator.train_df.write_csv(logdir + "avg_train_metrics.csv")
    integrator.val_df.write_csv(logdir + "avg_val_metrics.csv")

    cfg.model_dump()["trainer"]["args"]["logger"] = False

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
                output_dir=out_dir, write_interval="epoch"
            )
            trainer = create_trainer(
                cfg.model_dump(),
                callbacks=[pred_writer],
                logger=None,
            )
            ckpt_ = torch.load(ckpt.as_posix())
            integrator = create_integrator(cfg.model_dump())
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


# -
if __name__ == "__main__":
    pass
