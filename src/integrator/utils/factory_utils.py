from integrator.registry import REGISTRY, ARGUMENT_RESOLVER
import argparse
import pytorch_lightning as pl
import yaml


def create_module(module_type, module_name, **kwargs):
    """"""
    try:
        module_class = REGISTRY[module_type][module_name]
        return module_class(**kwargs)
    except KeyError as e:
        raise ValueError(
            f"Unknown {module_type}: {module_name}. Available options: {list(REGISTRY[module_type].keys())}"
        ) from e


def create_argument(module_type, argument_name, argument_value):
    try:
        arg = ARGUMENT_RESOLVER[module_type][argument_name][argument_value]
        return arg
    except KeyError as e:
        raise ValueError(
            f"Unknown {module_type}: {argument_name}. Available options: {list(ARGUMENT_RESOLVER[module_type].keys())}"
        ) from e


def load_config(config_path):
    """utility function to load a yaml config file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_integrator(config):
    integrator_name = config["integrator"]["name"]
    integrator_class = REGISTRY["integrator"][integrator_name]

    # Base modules
    encoder = create_module(
        "encoder",
        config["components"]["encoder"]["name"],
        **config["components"]["encoder"]["params"],
    )
    profile = create_module(
        "profile",
        config["components"]["profile"]["name"],
        **config["components"]["profile"]["params"],
    )
    decoder = create_module(
        "decoder",
        config["components"]["decoder"]["name"],
        **config["components"]["decoder"]["params"],
    )

    background_distribution = create_module(
        "q_bg",
        config["components"]["q_bg"]["name"],
        **config["components"]["q_bg"]["params"],
    )

    intensity_distribution = create_module(
        "q_I",
        config["components"]["q_I"]["name"],
        **config["components"]["q_I"]["params"],
    )

    if integrator_name == "integrator1":
        fc_encoder = create_module(
            "encoder",
            config["integrator"]["encoder"]["name"],
            **config["integrator"]["encoder"]["params"],
        )
        integrator = integrator_class(
            cnn_encoder=encoder,
            fc_encoder=fc_encoder,
            q_bg=background_distribution,
            q_I=intensity_distribution,
            profile_model=profile,
            dmodel=64,
            mc_samples=100,
            learning_rate=0.0001,
            profile_threshold=0.005,
        )
        return integrator
    else:
        raise ValueError(f"Unknown integrator name: {integrator_name}")


def create_data_loader(config):
    data_loader_name = config["data_loader"]["name"]
    data_loader_class = REGISTRY["data_loader"][data_loader_name]

    if data_loader_name == "default":
        data_module = data_loader_class(
            data_dir=config["data_loader"]["params"]["data_dir"],
            batch_size=config["data_loader"]["params"]["batch_size"],
            val_split=config["data_loader"]["params"]["val_split"],
            test_split=config["data_loader"]["params"]["test_split"],
            num_workers=config["data_loader"]["params"]["num_workers"],
            include_test=config["data_loader"]["params"]["include_test"],
            subset_size=config["data_loader"]["params"]["subset_size"],
            cutoff=config["data_loader"]["params"]["cutoff"],
            shoebox_features=config["data_loader"]["params"]["shoebox_features"],
            shoebox_file_names=config["data_loader"]["params"]["shoebox_file_names"],
        )
        data_module.setup()
        return data_module
    else:
        raise ValueError(f"Unknown data loader name: {data_loader_name}")


def create_trainer(config, data_module):
    trainer = pl.Trainer(
        max_epochs=config["trainer"]["params"]["max_epochs"],
        accelerator=create_argument(
            "trainer", "accelerator", config["trainer"]["params"]["accelerator"]
        )(),
        devices=config["trainer"]["params"]["devices"],
        logger=config["trainer"]["params"]["logger"],
        precision=config["trainer"]["params"]["precision"],
        check_val_every_n_epoch=config["trainer"]["params"]["check_val_every_n_epoch"],
        log_every_n_steps=config["trainer"]["params"]["log_every_n_steps"],
        deterministic=config["trainer"]["params"]["deterministic"],
        callbacks=config["trainer"]["params"]["callbacks"],
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
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    return parser.parse_args()


# %%
