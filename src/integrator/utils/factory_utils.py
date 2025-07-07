import argparse
import gc
import glob
import re
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml

from integrator.callbacks import PredWriter
from integrator.registry import ARGUMENT_RESOLVER, REGISTRY


def create_module(module_type, module_name, **kwargs):
    """"""
    try:
        module_class = REGISTRY[module_type][module_name]
        return module_class(**kwargs)
    except KeyError as e:
        raise ValueError(
            f"Unknown {module_type}: {module_name}. Available options: {list(REGISTRY[module_type].keys())}"
        ) from e


def create_integrator(config, checkpoint=None):
    modules = dict()
    integrator_class = REGISTRY["integrator"][config["integrator"]["name"]]

    for component in config["components"].items():
        if component[0].find("encoder") == -1:
            module_type = component[0]
        else:
            module_type = "shoebox_encoders"

        modules[component[0]] = create_module(
            module_type, component[1]["name"], **component[1]["args"]
        )

    if checkpoint is not None:
        integrator = integrator_class.load_from_checkpoint(
            checkpoint, **modules, **config["integrator"]["args"]
        )
        return integrator
    else:
        integrator = integrator_class(**modules, **config["integrator"]["args"])
        return integrator


def create_argument(module_type, argument_name, argument_value):
    try:
        arg = ARGUMENT_RESOLVER[module_type][argument_name][argument_value]
        return arg
    except KeyError as e:
        raise ValueError(
            f"Unknown {module_type}: {argument_name}. Available options: {list(ARGUMENT_RESOLVER[module_type].keys())}"
        ) from e


# %%
def load_config(config_path):
    """utility function to load a yaml config file"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_data_loader(config):
    data_loader_name = config["data_loader"]["name"]
    data_loader_class = REGISTRY["data_loader"][data_loader_name]

    if data_loader_name in {"default", "shoebox_data_module"}:
        data_module = data_loader_class(
            data_dir=config["data_loader"]["params"]["data_dir"],
            batch_size=config["data_loader"]["params"]["batch_size"],
            val_split=config["data_loader"]["params"]["val_split"],
            test_split=config["data_loader"]["params"]["test_split"],
            num_workers=config["data_loader"]["params"]["num_workers"],
            include_test=config["data_loader"]["params"]["include_test"],
            subset_size=config["data_loader"]["params"]["subset_size"],
            cutoff=config["data_loader"]["params"]["cutoff"],
            use_metadata=config["data_loader"]["params"]["use_metadata"],
            shoebox_file_names=config["data_loader"]["params"]["shoebox_file_names"],
        )
        data_module.setup()
        return data_module
    else:
        raise ValueError(f"Unknown data loader name: {data_loader_name}")


# TODO: Tod
def create_trainer(
    config,
    data_module,
    callbacks=None,
    logger=None,
):
    trainer = pl.Trainer(
        max_epochs=config["trainer"]["params"]["max_epochs"],
        accelerator=create_argument(
            "trainer", "accelerator", config["trainer"]["params"]["accelerator"]
        )(),
        devices=config["trainer"]["params"]["devices"],
        logger=logger,
        precision=config["trainer"]["params"]["precision"],
        check_val_every_n_epoch=config["trainer"]["params"]["check_val_every_n_epoch"],
        log_every_n_steps=config["trainer"]["params"]["log_every_n_steps"],
        deterministic=config["trainer"]["params"]["deterministic"],
        # callbacks=config["trainer"]["params"]["callbacks"],
        callbacks=callbacks,
        enable_checkpointing=config["trainer"]["params"]["enable_checkpointing"],
    )
    trainer.datamodule = data_module
    return trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for Integration Model")
    parser.add_argument(
        "--config",
        type=str,
        default="./src/integrator/configs/config.yaml",
        help="Path to the config.yaml file",
    )
    parser.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="Optional ID for the current run",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=250,
        help="Batch size for training",
    )
    return parser.parse_args()


def override_config(args, config):
    # Override config options from command line
    if args.batch_size:
        config["data_loader"]["params"]["batch_size"] = args.batch_size
    if args.epochs:
        config["trainer"]["params"]["max_epochs"] = args.epochs


def clean_from_memory(trainer, pred_writer, pred_integrator, checkpoint_callback=None):
    del trainer
    del pred_writer
    del pred_integrator
    if checkpoint_callback is not None:
        del checkpoint_callback
    torch.cuda.empty_cache()
    gc.collect()


def predict_from_checkpoints(config, trainer, pred_integrator, data, version_dir, path):
    for ckpt in glob.glob(path):
        epoch = re.search(r"epoch=(\d+)", ckpt).group(0)
        epoch = epoch.replace("=", "_")
        ckpt_dir = version_dir + "/predictions/" + epoch
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

        # prediction writer for current checkpoint
        pred_writer = PredWriter(
            output_dir=ckpt_dir,
            write_interval=config["trainer"]["params"]["callbacks"]["pred_writer"][
                "write_interval"
            ],
        )

        trainer.callbacks = [pred_writer]
        print(f"checkpoint:{ckpt}")

        checkpoint = torch.load(
            ckpt,
            weights_only=False,
        )

        pred_integrator.load_state_dict(checkpoint["state_dict"])

        if torch.cuda.is_available():
            pred_integrator.to(torch.device("cuda"))
        pred_integrator.eval()

        print("created integrator from checkpoint")
        print("running trainer.predict")

        trainer.predict(
            pred_integrator,
            return_predictions=False,
            dataloaders=data.predict_dataloader(),
        )

        del pred_writer
        torch.cuda.empty_cache()
        gc.collect()


# assign train/val labels
if __name__ == "__main__":
    from integrator.utils.factory_utils import create_integrator, load_config
    from src.utils import ROOT_DIR

    config = load_config(ROOT_DIR + "/integrator/config/config2.yaml")

    integrator = create_integrator(config)
