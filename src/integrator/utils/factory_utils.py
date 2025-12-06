import gc
import glob
import re
from importlib.resources import as_file
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import LightningModule

from integrator.callbacks import PredWriter
from integrator.model.integrators import (
    EncoderModules,
    IntegratorHyperParameters,
    SurrogateModules,
)
from integrator.model.loss import LossConfig, PriorConfig
from integrator.registry import ARGUMENT_RESOLVER, REGISTRY


def create_module(module_type: str, module_name: str, **kwargs):
    try:
        cls = REGISTRY[module_type][module_name]
        return cls(**kwargs)
    except KeyError as e:
        raise ValueError(
            f"Unknown {module_type}: {module_name}. Available: {list(REGISTRY[module_type].keys())}"
        ) from e


def _build_modules(components: dict) -> dict:
    modules: dict[str, object] = {}

    enc_field = components.get("encoders", [])
    if isinstance(enc_field, dict):
        enc_iter = [{k: v} for k, v in enc_field.items()]
    elif isinstance(enc_field, list):
        enc_iter = enc_field
    elif enc_field in (None, []):
        enc_iter = []
    else:
        raise TypeError("components.encoders must be a list or dict")

    for enc_dict in enc_iter:
        for enc_key, sub in enc_dict.items():  # (name, sub-config)
            modules[enc_key] = create_module(
                "encoders", sub["name"], **(sub.get("args"))
            )

    for k, v in components.items():
        if k == "encoders":
            continue
        modules[k] = create_module(k, v["name"], **(v.get("args") or {}))

    return modules


def _build_encoders(config: dict) -> dict:
    encoders = {}
    enc_field = config.get("encoders", [])
    if isinstance(enc_field, dict):
        enc_iter = [{k: v} for k, v in enc_field.items()]
    elif isinstance(enc_field, list):
        enc_iter = enc_field
    elif enc_field in (None, []):
        enc_iter = []
    else:
        raise TypeError("components.encoders must be a list or dict")

    for enc_dict in enc_iter:
        for enc_key, sub in enc_dict.items():  # (name, sub-config)
            encoders[enc_key] = create_module(
                "encoders", sub["name"], **(sub.get("args"))
            )
    return encoders


def _build_surrogates(config: dict) -> dict:
    surrogates = {}
    surr_dict = config.get("surrogates", [])

    for k, v in surr_dict.items():
        surrogates[k] = create_module(
            k,
            v.get("name"),
            **(v.get("args")),
        )

    return surrogates


def build_prior(prior_cfg: dict | None) -> PriorConfig | None:
    if prior_cfg is None:
        return None
    return PriorConfig(**prior_cfg)


def create_integrator(
    config: dict,
    checkpoint: str | None = None,
) -> LightningModule:
    # load integrator class
    integrator_cls = REGISTRY["integrator"][config["integrator"]["name"]]
    modules = {}

    # build encoders
    encoders = _build_encoders(config)
    encoders = EncoderModules(**encoders)

    # build surrogates
    surrogates = _build_surrogates(config)
    surrogates = SurrogateModules(**surrogates)

    # build loss
    priors_config = config["loss"]["priors"]
    pi = build_prior(priors_config.get("intensity"))
    pprf = build_prior(priors_config.get("profile"))
    pbg = build_prior(priors_config.get("background"))

    # loss module
    loss_cfg = LossConfig(
        pprf=pprf,
        pbg=pbg,
        pi=pi,
        mc_smpls=config["loss"]["mc_smpls"],
        eps=config["loss"]["eps"],
    )
    loss_name = config["loss"]["name"]
    loss_cls = REGISTRY["loss"][loss_name](cfg=loss_cfg)
    modules["loss"] = loss_cls

    # integrator hyperparameters
    cfg = IntegratorHyperParameters(**config["integrator"]["args"])

    # integrator arguments
    args = {
        "loss": loss_cls,
        "encoders": encoders,
        "surrogates": surrogates,
        "cfg": cfg,
    }

    # load from checkpoint
    if checkpoint is not None:
        return integrator_cls.load_from_checkpoint(checkpoint, **args)
    return integrator_cls(**args)


def create_argument(module_type, argument_name, argument_value):
    try:
        arg = ARGUMENT_RESOLVER[module_type][argument_name][argument_value]
        return arg
    except KeyError as e:
        raise ValueError(
            f"Unknown {module_type}: {argument_name}. Available options: {list(ARGUMENT_RESOLVER[module_type].keys())}"
        ) from e


def create_data_loader(config):
    data_loader_name = config["data_loader"]["name"]
    data_loader_class = REGISTRY["data_loader"][data_loader_name]

    if data_loader_name in {
        "default",
        "shoebox_data_module",
        "shoebox_data_module_2d",
    }:
        data_module = data_loader_class(
            data_dir=config["data_loader"]["args"]["data_dir"],
            batch_size=config["data_loader"]["args"]["batch_size"],
            val_split=config["data_loader"]["args"]["val_split"],
            test_split=config["data_loader"]["args"]["test_split"],
            num_workers=config["data_loader"]["args"]["num_workers"],
            include_test=config["data_loader"]["args"]["include_test"],
            subset_size=config["data_loader"]["args"]["subset_size"],
            cutoff=config["data_loader"]["args"]["cutoff"],
            use_metadata=config["data_loader"]["args"]["use_metadata"],
            shoebox_file_names=config["data_loader"]["args"][
                "shoebox_file_names"
            ],
            H=config["data_loader"]["args"]["H"],
            W=config["data_loader"]["args"]["W"],
            anscombe=config["data_loader"]["args"]["anscombe"],
        )
        data_module.setup()
        return data_module
    else:
        raise ValueError(f"Unknown data loader name: {data_loader_name}")


def create_trainer(config, callbacks=None, logger=None):
    return pl.Trainer(
        max_epochs=config["trainer"]["args"]["max_epochs"],
        accelerator=create_argument(
            "trainer", "accelerator", config["trainer"]["args"]["accelerator"]
        )(),
        devices=config["trainer"]["args"]["devices"],
        logger=logger,
        precision=config["trainer"]["args"]["precision"],
        check_val_every_n_epoch=config["trainer"]["args"][
            "check_val_every_n_epoch"
        ],
        log_every_n_steps=config["trainer"]["args"]["log_every_n_steps"],
        deterministic=config["trainer"]["args"]["deterministic"],
        callbacks=callbacks,
        enable_checkpointing=config["trainer"]["args"]["enable_checkpointing"],
    )


def override_config(args, config):
    # Override config options from command line
    if args.batch_size:
        config["data_loader"]["args"]["batch_size"] = args.batch_size
    if args.epochs:
        config["trainer"]["args"]["max_epochs"] = args.epochs


def clean_from_memory(
    trainer, pred_writer, pred_integrator, checkpoint_callback=None
):
    del trainer
    del pred_writer
    del pred_integrator
    if checkpoint_callback is not None:
        del checkpoint_callback
    torch.cuda.empty_cache()
    gc.collect()


def predict_from_checkpoints(
    config, trainer, pred_integrator, data, version_dir, path
):
    for ckpt in glob.glob(path):
        match = re.search(r"epoch=(\d+)", ckpt)
        if match is None:
            continue
        epoch = match.group(1)
        epoch = epoch.replace("=", "_")
        ckpt_dir = version_dir + "/predictions/" + epoch
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

        # prediction writer for current checkpoint
        pred_writer = PredWriter(
            output_dir=ckpt_dir,
            write_interval=config["trainer"]["args"]["callbacks"][
                "pred_writer"
            ]["write_interval"],
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


def load_config(resource: str | Path) -> dict:
    """resource is a Traversable from get_configs()."""

    if isinstance(resource, str):
        resource = Path(resource)

    with as_file(resource) as p:
        with open(Path(p), encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    return raw


# assign train/val labels
if __name__ == "__main__":
    from integrator.utils import load_config
    from utils import CONFIGS

    cfg = list(CONFIGS.glob("*"))[0]
    cfg = load_config(cfg)

    for k, v in cfg["surrogates"].items():
        print(k, v)

    _build_encoders(cfg)

    # Laue model config
    config2d = load_config(CONFIGS["config2d"])
    create_integrator(config2d.dict())

    # Two encoder config
    config3d = load_config(CONFIGS["config3d"])
    create_integrator(config3d.dict())

    updates = {
        "trainer": {"args": {"max_epochs": 100}},
    }

    updates = dict()
    updates.setdefault("trainer", {}).setdefault("args", {})["max_epochs"] = (
        100
    )

    config3d.dict()["trainer"]

    updates.setdefault("trainer", {}).setdefault("args", {})["max_epochs"] = (
        100
    )

    config3d.model_copy(update=updates).dict()["trainer"]
