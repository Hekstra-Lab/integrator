fsrom copy import deepcopy
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from integrator.config.schema import Cfg
from integrator.utils import clean_from_memory

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
    from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
    from pytorch_lightning.loggers import WandbLogger

    from integrator.callbacks import Plotter, PredWriter, assign_labels
    from integrator.utils import (
        create_data_loader,
        create_integrator,
        create_trainer,
        load_config,
    )

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

    logger.log_hyperparams(cfg.model_dump())

    # get logging directory
    logdir = logger.experiment.dir

    config_out = Path(logdir) / "config_copy.yaml"
    copy = cfg.model_dump(mode="json")

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
    plotter = Plotter(n_profiles=10)

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

    integrator.train_df.write_csv(logdir + "avg_train_metrics.csv")
    integrator.val_df.write_csv(logdir + "avg_val_metrics.csv")
    # path = Path(logdir) / "checkpoints/epoch*.ckpt"

    cfg.model_dump()["trainer"]["args"]["logger"] = False

    clean_from_memory(
        pred_writer, pred_writer, pred_writer, checkpoint_callback
    )

    # pred_integrator = create_integrator(cfg.model_dump())
