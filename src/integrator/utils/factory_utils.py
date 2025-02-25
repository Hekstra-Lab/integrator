from integrator.registry import REGISTRY, ARGUMENT_RESOLVER
from copy import deepcopy
import torch
import gc
import argparse
import pytorch_lightning as pl
import yaml
import glob
import re
from integrator.callbacks import PredWriter
from pathlib import Path


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


def create_prior(dist_name, dist_params):
    if dist_name == "gamma":
        concentration = torch.tensor(dist_params["concentration"])
        rate = torch.tensor(dist_params["rate"])
        return torch.distributions.gamma.Gamma(concentration, rate)

    if dist_name == "log_normal":
        loc = torch.tensor(dist_params["loc"])
        scale = torch.tensor(dist_params["scale"])
        return torch.distributions.log_normal.LogNormal(loc, scale)

    if dist_name == "beta":
        concentration1 = torch.tensor(dist_params["concentration1"])
        concentration0 = torch.tensor(dist_params["concentration0"])
        return torch.distributions.beta.Beta(concentration1, concentration0)

    if dist_name == "laplace":
        loc = torch.tensor(dist_params["loc"])
        scale = torch.tensor(dist_params["scale"])
        return torch.distributions.laplace.Laplace(loc, scale)

    elif dist_name == "dirichlet":
        # For a simple case where all dimensions use the same concentration
        if "concentration" in dist_params:
            concentration = torch.ones(3 * 21 * 21) * dist_params["concentration"]
        # For the case where a specific concentration vector is provided
        elif "concentration_vector" in dist_params:
            concentration = torch.tensor(dist_params["concentration_vector"])
        else:
            raise ValueError(
                f"Missing concentration parameters for Dirichlet distribution"
            )

        return torch.distributions.dirichlet.Dirichlet(concentration)

    else:
        raise ValueError(f"Unknown distribution name: {dist_name}")


def create_loss(config):
    # Create a DEEP COPY of the loss parameters
    param_dict = deepcopy(config["components"]["loss"]["params"])

    bg_pairing = param_dict.pop("bg_pairing", "gamma_gamma")
    I_pairing = param_dict.pop("I_pairing", "gamma_gamma")
    p_pairing = param_dict.pop("p_pairing", "dirichlet_dirichlet")

    # Build p_bg from the copied params
    p_bg = create_prior(param_dict["p_bg"]["name"], param_dict["p_bg"]["params"])

    # Build p_I from the copied params
    p_I = create_prior(param_dict["p_I"]["name"], param_dict["p_I"]["params"])

    # Build p_p prior distribution if needed
    if (
        "p_p" in param_dict
        and param_dict["p_p"] is not None
        and param_dict["p_p"]["name"] is not None
    ):
        p_p = create_prior(param_dict["p_p"]["name"], param_dict["p_p"]["params"])
        param_dict["p_p"] = p_p
    else:
        param_dict["p_p"] = None  # Modify the COPY, not the original config

    param_dict["p_bg"] = p_bg
    param_dict["p_I"] = p_I

    param_dict["bg_pairing"] = bg_pairing
    param_dict["I_pairing"] = I_pairing
    param_dict["p_pairing"] = p_pairing

    return create_module("loss", config["components"]["loss"]["name"], **param_dict)


def load_config(config_path):
    """utility function to load a yaml config file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_components(config):
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

    loss = create_loss(config)
    return (
        encoder,
        profile,
        decoder,
        background_distribution,
        intensity_distribution,
        loss,
    )


def create_integrator(config):
    integrator_name = config["integrator"]["name"]
    integrator_class = REGISTRY["integrator"][integrator_name]
    (
        encoder,
        profile,
        decoder,
        background_distribution,
        intensity_distribution,
        loss,
    ) = create_components(config)

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
            dmodel=config["global"]["dmodel"],
            loss=loss,
            mc_samples=config["integrator"]["mc_samples"],
            learning_rate=config["integrator"]["learning_rate"],
            profile_threshold=config["integrator"]["profile_threshold"],
        )
        return integrator
    else:
        raise ValueError(f"Unknown integrator name: {integrator_name}")


def create_integrator_from_checkpoint(config, checkpoint_path):
    integrator_name = config["integrator"]["name"]
    integrator_class = REGISTRY["integrator"][integrator_name]
    (
        encoder,
        profile,
        decoder,
        background_distribution,
        intensity_distribution,
        loss,
    ) = create_components(config)

    if integrator_name == "integrator1":
        fc_encoder = create_module(
            "encoder",
            config["integrator"]["encoder"]["name"],
            **config["integrator"]["encoder"]["params"],
        )
        integrator = integrator_class.load_from_checkpoint(
            checkpoint_path,
            cnn_encoder=encoder,
            fc_encoder=fc_encoder,
            q_bg=background_distribution,
            q_I=intensity_distribution,
            profile_model=profile,
            dmodel=64,
            loss=loss,
            mc_samples=100,
            learning_rate=0.0001,
            profile_threshold=0.005,
            map_location="cpu",
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

        checkpoint = torch.load(ckpt, map_location="cpu")
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
