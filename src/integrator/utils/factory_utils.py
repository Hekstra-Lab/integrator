import gc
import glob
import re
from dataclasses import asdict
from importlib.resources import as_file
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

from integrator import configs
from integrator.callbacks import PredWriter
from integrator.configs import shallow_dict
from integrator.model.integrators.base_integrator import BaseIntegrator
from integrator.registry import REGISTRY

PRIOR_PARAMS = {
    "gamma": configs.GammaParams,
    "dirichlet": configs.DirichletParams,
    "exponential": configs.ExponentialParams,
    "half_cauchy": configs.HalfCauchyParams,
    "log_normal": configs.LogNormalParams,
}

TUPLE_FIELDS = {
    "input_shape",
    "conv1_kernel_size",
    "conv1_padding",
    "pool_kernel_size",
    "pool_stride",
    "conv2_kernel_size",
    "conv2_padding",
    "conv3_kernel_size",
    "conv3_padding",
    "shape",
    "sbox_shape",
}


def _get_integrator_cls(name: str) -> type[BaseIntegrator]:
    return REGISTRY["integrator"][name]


def _get_encoder_cls(name: str) -> nn.Module:
    return REGISTRY["encoders"][name]


def _get_loss_cls(name: str) -> nn.Module:
    return REGISTRY["loss"][name]


def _get_dataloader_cls(name: str) -> nn.Module:
    return REGISTRY["data_loader"][name]


def _normalize_tuples(d: dict) -> dict:
    out = dict(d)
    for k in TUPLE_FIELDS:
        if k in out:
            v = out[k]
            if isinstance(v, list):
                out[k] = tuple(v)
            elif not isinstance(v, tuple):
                raise TypeError(f"{k} must be list or tuple, got {type(v)}")
    return out


def _get_loss_args(
    cfg: dict,
    prior_configs: dict,
) -> configs.LossArgs:
    loss_cfg = dict(cfg["loss"])
    loss_args = _normalize_tuples(loss_cfg["args"])

    args_cls = configs.LossArgs(
        mc_samples=loss_args["mc_samples"],
        eps=loss_args["eps"],
        pbg_cfg=prior_configs["pbg_cfg"],
        pi_cfg=prior_configs["pi_cfg"],
        pprf_cfg=prior_configs["pprf_cfg"],
    )

    return args_cls


def _get_surrogate_modules(
    cfg: dict,
) -> dict[str, nn.Module]:
    qp_args = configs.DirichletArgs(**cfg["surrogates"]["qp"]["args"])
    qp_cls = REGISTRY["surrogates"][cfg["surrogates"]["qp"]["name"]]
    qp = qp_cls(**asdict(qp_args))

    qbg_args = configs.SurrogateArgs(**cfg["surrogates"]["qbg"]["args"])
    qbg_cls = REGISTRY["surrogates"][cfg["surrogates"]["qbg"]["name"]]
    qbg = qbg_cls(**asdict(qbg_args))

    qi_args = configs.SurrogateArgs(**cfg["surrogates"]["qi"]["args"])
    qi_cls = REGISTRY["surrogates"][cfg["surrogates"]["qi"]["name"]]
    qi = qi_cls(**asdict(qi_args))

    return {
        "qp": qp,
        "qbg": qbg,
        "qi": qi,
    }


def _get_prior_cfgs(
    cfg: dict,
    pprf_cfg: str = "pprf_cfg",
    pbg_cfg: str = "pbg_cfg",
    pi_cfg: str = "pi_cfg",
) -> dict[str, configs.PriorConfig | None]:
    # Building loss class
    priors = {}
    prior_cfgs = dict(cfg)
    for p in (pprf_cfg, pbg_cfg, pi_cfg):
        if p in prior_cfgs["loss"]["args"]:
            p_dict = prior_cfgs["loss"]["args"][p]
            p_name = p_dict["name"]
            p_params = PRIOR_PARAMS[p_name](**p_dict["params"])
            p_prior_cfgs = configs.PriorConfig(
                name=p_name,
                params=p_params,
                weight=p_dict["weight"],
            )
            priors[p] = p_prior_cfgs
        else:
            priors[p] = None
    return priors


def _get_encoder_modules(
    cfg: dict,
) -> dict[str, nn.Module]:
    cfg_ = dict(cfg)
    model = cfg_["integrator"]["name"]
    required_encoders = REGISTRY["integrator"][model].REQUIRED_ENCODERS

    # Verify config has the correct number of encoders for specified model
    if len(required_encoders) != len(cfg_["encoders"]):
        raise ValueError(
            f"""
            Integration model '{model}' requires {len(required_encoders)}
            encoders, but {len(cfg_["encoders"])} were passed
            """
        )
    # loading encoder arguments in corresponding dataclasses
    encoders = {}
    for items, encoder_cfg in zip(
        required_encoders.items(), cfg_["encoders"], strict=False
    ):
        name, args = items
        encoder_args = args(**encoder_cfg["args"])
        encoders[name] = REGISTRY["encoders"][encoder_cfg["name"]](
            **asdict(encoder_args)
        )
    return encoders


def _get_loss_module(
    cfg: dict,
) -> nn.Module:
    # get loss cls
    loss_cls = _get_loss_cls(cfg["loss"]["name"])
    # construct priors
    prior_configs = _get_prior_cfgs(cfg)
    # construct loss args
    loss_args = _get_loss_args(
        cfg=cfg,
        prior_configs=prior_configs,
    )
    return loss_cls(**shallow_dict(loss_args))


def construct_integrator(
    cfg: dict,
) -> BaseIntegrator:
    # integrator class
    integrator_cls = _get_integrator_cls(cfg["integrator"]["name"])

    # get integrator components
    integrator_args = configs.IntegratorCfg(**cfg["integrator"]["args"])
    encoders = _get_encoder_modules(cfg)
    surrogates = _get_surrogate_modules(cfg)
    loss = _get_loss_module(cfg)

    return integrator_cls(
        cfg=integrator_args,
        encoders=encoders,
        surrogates=surrogates,
        loss=loss,
    )


def construct_data_loader(cfg):
    dl_cls = _get_dataloader_cls(cfg["data_loader"]["name"])
    dl_args = configs.DataLoaderArgs(**cfg["data_loader"]["args"])

    data_dir = Path(dl_args.data_dir)

    return dl_cls(
        data_dir=data_dir.as_posix(),
        batch_size=dl_args.batch_size,
        val_split=dl_args.val_split,
        test_split=dl_args.test_split,
        num_workers=dl_args.num_workers,
        include_test=dl_args.include_test,
        subset_size=dl_args.subset_size,
        cutoff=dl_args.cutoff,
        use_metadata=dl_args.use_metadata,
        shoebox_file_names=dl_args.shoebox_file_names,
        D=dl_args.D,
        H=dl_args.H,
        W=dl_args.W,
        anscombe=dl_args.anscombe,
    )


def construct_trainer(
    cfg: dict,
    logger: Logger | None = None,
    callbacks: list[Callback] | Callback | None = None,
) -> pl.Trainer:
    tr_cfg = configs.TrainerConfig(**cfg["trainer"])

    return pl.Trainer(
        max_epochs=tr_cfg.max_epochs,
        accelerator=tr_cfg.accelerator,
        devices=tr_cfg.devices,
        logger=logger,
        precision=tr_cfg.precision,
        check_val_every_n_epoch=tr_cfg.check_val_every_n_epoch,
        log_every_n_steps=tr_cfg.log_every_n_steps,
        deterministic=tr_cfg.deterministic,
        enable_checkpointing=tr_cfg.enable_checkpointing,
        callbacks=callbacks,
        enable_progress_bar=False,
    )


def override_config(args, config):
    # Override config options from command line
    if args.batch_size:
        config["data_loader"]["args"]["batch_size"] = args.batch_size
    if args.epochs:
        config["trainer"]["args"]["max_epochs"] = args.epochs


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
            write_interval=config["trainer"]["args"]["callbacks"]["pred_writer"][
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


def load_config(resource: str | Path) -> dict:
    """resource is a Traversable from get_configs()."""

    if isinstance(resource, str):
        resource = Path(resource)

    with as_file(resource) as p:
        with open(Path(p), encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    return raw


def dump_yaml_config(
    cfg: configs.YAMLConfig,
    path: str,
):
    with open(path, "w") as f:
        yaml.safe_dump(
            asdict(cfg),
            f,
            sort_keys=False,
            default_flow_style=False,
        )
