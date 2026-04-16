import os
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
from integrator.configs import shallow_dict
from integrator.model.integrators.base_integrator import BaseIntegrator
from integrator.registry import REGISTRY

PRIOR_PARAMS = {
    "gamma": configs.GammaParams,
    "dirichlet": configs.DirichletParams,
    "exponential": configs.ExponentialParams,
    "half_cauchy": configs.HalfCauchyParams,
    "log_normal": configs.LogNormalParams,
    "gaussian_profile": configs.GaussianProfilePriorParams,
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


def _resolve_data_path(
    path: str, data_dir: str, n_bins: int | None = None
) -> str:
    """Resolve a relative path against data_dir, optionally inserting n_bins suffix."""
    if os.path.isabs(path) or path.startswith("~"):
        return path
    if n_bins is not None:
        p = Path(path)
        path = f"{p.stem}_{n_bins}{p.suffix}"
    return os.path.join(data_dir, path)


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
    """Construct all surrogate distribution modules from config.

    Iterates over all keys in `cfg["surrogates"]` so that any combination
    of surrogates is supported (e.g., the standard `qp`/`qi`/`qbg` trio).  The raw
    args dict from the YAML is passed directly to each class constructor,
    avoiding the need for a separate `SurrogateArgs` dataclass per class.

    For `fixed_basis_profile` (and similar), a relative `basis_path` is resolved
    against `data_loader.args.data_dir` so the YAML only needs the filename.
    """
    surrogates = {}
    data_dir = cfg.get("data_loader", {}).get("args", {}).get("data_dir", "")
    n_bins = cfg.get("loss", {}).get("args", {}).get("n_bins")

    # Models with >2 encoders have separate k/r heads for qi/qbg
    integrator_cls = REGISTRY["integrator"].get(
        cfg.get("integrator", {}).get("name", "")
    )
    separate_inputs = (
        integrator_cls is not None
        and len(integrator_cls.REQUIRED_ENCODERS) > 2
    )

    for key, surrogate_cfg in cfg["surrogates"].items():
        surrogate_cls = REGISTRY["surrogates"][surrogate_cfg["name"]]
        args = dict(surrogate_cfg["args"])
        if (
            surrogate_cfg["name"]
            in (
                "fixed_basis_profile",
                "learned_basis_profile",

                "per_bin_profile",
                "empirical_profile_surrogate",
                # Legacy aliases
                "logistic_normal_surrogate",
                "per_bin_logistic_normal",
                "linear_profile_surrogate",
            )
            and "basis_path" in args
        ):
            bp = args["basis_path"]
            if isinstance(bp, str):
                args["basis_path"] = _resolve_data_path(bp, data_dir, n_bins)
        # Auto-set separate_inputs for two-param surrogates (qi, qbg)
        if key in ("qi", "qbg") and "separate_inputs" not in args:
            args["separate_inputs"] = separate_inputs
        surrogates[key] = surrogate_cls(**args)
    return surrogates


def _get_prior_cfgs(
    cfg: dict,
    pprf_cfg: str = "pprf_cfg",
    pbg_cfg: str = "pbg_cfg",
    pi_cfg: str = "pi_cfg",
) -> dict[str, configs.PriorConfig | None]:
    # Building loss class
    priors = {}
    prior_cfgs = dict(cfg)
    data_dir = cfg.get("data_loader", {}).get("args", {}).get("data_dir", "")
    for p in (pprf_cfg, pbg_cfg, pi_cfg):
        p_dict = prior_cfgs["loss"]["args"].get(p)
        if p_dict is not None:
            p_name = p_dict["name"]
            p_params_dict = dict(p_dict.get("params") or {})
            # Resolve relative concentration paths for Dirichlet
            if p_name == "dirichlet" and "concentration" in p_params_dict:
                conc = p_params_dict["concentration"]
                if (
                    isinstance(conc, str)
                    and not os.path.isabs(conc)
                    and not conc.startswith("~")
                ):
                    p_params_dict["concentration"] = os.path.join(
                        data_dir, conc
                    )
            p_params = PRIOR_PARAMS[p_name](**p_params_dict)
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
    kwargs = shallow_dict(loss_args)

    # Forward extra keys from loss.args for custom loss classes
    standard_keys = {
        "mc_samples",
        "eps",
        "pprf_cfg",
        "pbg_cfg",
        "pi_cfg",
        "n_bins",
        "profile_binning",
        "profile_basis_type",
        "profile_basis_d",
        "profile_basis_max_order",
        "profile_basis_sigma_ref",
        "profile_smooth_sigma",
    }
    for k, v in cfg["loss"]["args"].items():
        if k not in standard_keys:
            kwargs[k] = v

    # Auto-compute pprf_n_pixels for scalar concentration_per_group
    conc_val = kwargs.get("concentration_per_group")
    if isinstance(conc_val, (int, float)) and "pprf_n_pixels" not in kwargs:
        d = cfg["integrator"]["args"].get("d", 3)
        h = cfg["integrator"]["args"].get("h", 21)
        w = cfg["integrator"]["args"].get("w", 21)
        kwargs["pprf_n_pixels"] = d * h * w

    # Map empirical_profile_basis_per_bin -> profile_basis_per_bin
    if "empirical_profile_basis_per_bin" in kwargs and "profile_basis_per_bin" not in kwargs:
        kwargs["profile_basis_per_bin"] = kwargs.pop("empirical_profile_basis_per_bin")
    elif "empirical_profile_basis_per_bin" in kwargs:
        kwargs.pop("empirical_profile_basis_per_bin")

    # Resolve relative .pt paths for custom loss buffers
    # Include n_bins in filename to prevent concurrent runs from clobbering files
    data_dir = cfg.get("data_loader", {}).get("args", {}).get("data_dir", "")
    n_bins = cfg.get("loss", {}).get("args", {}).get("n_bins")
    # When pprf_quantile is set, the user provides a global concentration
    # file that should not get an n_bins suffix.
    global_conc = "pprf_quantile" in kwargs
    for pt_key in (
        "tau_per_group",
        "bg_rate_per_group",
        "concentration_per_group",
        "s_squared_per_group",
        "i_concentration_per_group",
        "bg_concentration_per_group",
        "profile_basis_per_bin",
        "empirical_profile_basis_per_bin",
    ):
        if pt_key in kwargs and isinstance(kwargs[pt_key], str):
            # Global concentration doesn't get n_bins suffix
            skip_nbins = (
                pt_key == "concentration_per_group" and global_conc
            )
            nbins = None if skip_nbins else n_bins
            kwargs[pt_key] = _resolve_data_path(
                kwargs[pt_key], data_dir, nbins
            )

    return loss_cls(**kwargs)


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

    trainer_kwargs = dict(
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
        enable_progress_bar=True,
    )

    if tr_cfg.gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = tr_cfg.gradient_clip_val
    if tr_cfg.gradient_clip_algorithm is not None:
        trainer_kwargs["gradient_clip_algorithm"] = (
            tr_cfg.gradient_clip_algorithm
        )

    return pl.Trainer(**trainer_kwargs)


def save_run_artifacts(
    integrator: BaseIntegrator,
    cfg: dict,
    logdir: Path,
) -> None:
    """Save model metadata and prior artifacts to the wandb log directory.

    Saves:
        - prior_concentration.pt: the rescaled Dirichlet concentration vector
          actually used during training (after load/amplify/normalize)
        - run_artifacts.yaml: prior configs, model param counts, loss settings
    """
    artifacts_dir = Path(logdir) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    loss_module = integrator.loss
    artifacts = {}

    # Dirichlet prior concentration (rescaled)
    pprf_params = getattr(loss_module, "pprf_params", None)
    if pprf_params is not None and "concentration" in pprf_params:
        conc = pprf_params["concentration"]
        torch.save(conc, artifacts_dir / "prior_concentration.pt")
        artifacts["prior_concentration"] = {
            "n_elements": int(conc.numel()),
            "sum": float(conc.sum()),
            "min": float(conc.min()),
            "max": float(conc.max()),
            "mean": float(conc.mean()),
        }

    # Prior configs
    prior_summary = {}
    for attr, label in [
        ("pprf_cfg", "profile"),
        ("pi_cfg", "intensity"),
        ("pbg_cfg", "background"),
    ]:
        prior_cfg = getattr(loss_module, attr, None)
        if prior_cfg is not None:
            entry = {
                "name": prior_cfg.name,
                "weight": float(prior_cfg.weight),
            }
            params = prior_cfg.params
            for field in params.__dataclass_fields__:
                v = getattr(params, field)
                if isinstance(v, (int, float, str, tuple, list)):
                    entry[field] = v
            prior_summary[label] = entry
    if prior_summary:
        artifacts["priors"] = prior_summary

    # Loss settings
    artifacts["loss"] = {
        "name": cfg["loss"]["name"],
        "mc_samples": getattr(loss_module, "mc_samples", None),
        "eps": getattr(loss_module, "eps", None),
    }

    # Model parameter counts
    param_counts = {}
    for name, module in integrator.named_children():
        if isinstance(module, nn.ModuleDict):
            for sub_name, sub_module in module.items():
                n = sum(p.numel() for p in sub_module.parameters())
                param_counts[f"{name}.{sub_name}"] = n
        else:
            n = sum(p.numel() for p in module.parameters())
            param_counts[name] = n
    param_counts["total"] = sum(p.numel() for p in integrator.parameters())
    artifacts["param_counts"] = param_counts

    # Write summary YAML
    with open(artifacts_dir / "run_artifacts.yaml", "w") as f:
        yaml.safe_dump(artifacts, f, sort_keys=False, default_flow_style=False)



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
