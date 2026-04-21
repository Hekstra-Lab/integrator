import os
from dataclasses import asdict
from importlib.resources import as_file
from pathlib import Path
from typing import Any

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


def _require(cfg: dict, *path: str) -> Any:
    """Fetch `cfg[path[0]][path[1]]...`, raising a clear KeyError if any step is missing."""
    node: Any = cfg
    traversed: list[str] = []
    for key in path:
        if not isinstance(node, dict) or key not in node:
            trail = ".".join(traversed) or "<root>"
            raise KeyError(
                f"Config missing required key '{key}' under '{trail}'"
            )
        node = node[key]
        traversed.append(key)
    return node


def _get_data_dir(cfg: dict) -> str:
    return str(_require(cfg, "data_loader", "args", "data_dir"))


def _get_n_bins(cfg: dict) -> int | None:
    # n_bins is genuinely optional (some losses don't use it).
    loss_args = _require(cfg, "loss", "args")
    return loss_args.get("n_bins")


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
    skip_warmstart: bool = False,
) -> dict[str, nn.Module]:
    """Construct all surrogate distribution modules from config.

    Iterates over all keys in `cfg["surrogates"]` so that any combination
    of surrogates is supported.

    When `skip_warmstart=True`, `warmstart_basis_path` is stripped from
    qp args before instantiation — useful at prediction time when the
    trained weights will be restored via `load_state_dict` anyway, so
    warmstart from a basis file is redundant and a potential source of
    bugs (missing/stale basis files, shape mismatches).

    For `fixed_basis_profile` (and similar), a relative `basis_path` is resolved
    against `data_loader.args.data_dir` so the YAML only needs the filename.
    """
    surrogates = {}
    data_dir = _get_data_dir(cfg)
    n_bins = _get_n_bins(cfg)

    # Models with >2 encoders have separate k/r heads for qi/qbg
    integrator_name = _require(cfg, "integrator", "name")
    integrator_cls = REGISTRY["integrator"][integrator_name]
    separate_inputs = len(integrator_cls.REQUIRED_ENCODERS) > 2

    # Load GammaB mean_init stats once (if available). Written by
    # prepare_per_bin_priors from raw counts (no DIALS). Used below to
    # inject mean_init into gammaB qi/qbg surrogates that didn't set it.
    init_stats_path = Path(data_dir) / "qi_qbg_mean_init.pt"
    init_stats: dict | None = None
    if init_stats_path.is_file():
        try:
            init_stats = torch.load(
                init_stats_path,
                weights_only=False,
                map_location="cpu",
            )
        except Exception:
            init_stats = None

    for key, surrogate_cfg in cfg["surrogates"].items():
        surrogate_cls = REGISTRY["surrogates"][surrogate_cfg["name"]]
        args = dict(surrogate_cfg["args"])

        # GammaB: inject mean_init from the cached counts-derived stats
        # if the user didn't set it explicitly in the YAML. Defaults use
        # the median (robust to heavy-tail peaks) — user can override by
        # setting mean_init explicitly.
        if (
            surrogate_cfg["name"] == "gammaB"
            and "mean_init" not in args
            and init_stats is not None
            and key in ("qi", "qbg")
        ):
            stat_key = "qi_median" if key == "qi" else "qbg_median"
            if stat_key in init_stats:
                args["mean_init"] = float(init_stats[stat_key])
        if (
            surrogate_cfg["name"]
            in (
                "fixed_basis_profile",
                "learned_basis_profile",
            )
            and "basis_path" in args
        ):
            bp = args["basis_path"]
            if isinstance(bp, str):
                args["basis_path"] = _resolve_data_path(bp, data_dir, n_bins)
        # warmstart_basis_path for learned_basis_profile: same data_dir +
        # n_bins suffix handling as basis_path above.
        if surrogate_cfg["name"] == "learned_basis_profile" and isinstance(
            args.get("warmstart_basis_path"), str
        ):
            if skip_warmstart:
                args.pop("warmstart_basis_path")
            else:
                args["warmstart_basis_path"] = _resolve_data_path(
                    args["warmstart_basis_path"],
                    data_dir,
                    n_bins,
                )
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
    data_dir = _get_data_dir(cfg)
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
    # Keys consumed by the prior-prep pipeline or globally, not by
    # individual loss classes. Filter them out before forwarding
    # loss.args kwargs to the loss constructor.
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
        "profile_basis_sigma_z",
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
    if (
        "empirical_profile_basis_per_bin" in kwargs
        and "profile_basis_per_bin" not in kwargs
    ):
        kwargs["profile_basis_per_bin"] = kwargs.pop(
            "empirical_profile_basis_per_bin"
        )
    elif "empirical_profile_basis_per_bin" in kwargs:
        kwargs.pop("empirical_profile_basis_per_bin")

    # Resolve relative .pt paths for custom loss buffers
    # Include n_bins in filename to prevent concurrent runs from clobbering files
    data_dir = _get_data_dir(cfg)
    n_bins = _get_n_bins(cfg)
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
            skip_nbins = pt_key == "concentration_per_group" and global_conc
            nbins = None if skip_nbins else n_bins
            kwargs[pt_key] = _resolve_data_path(
                kwargs[pt_key], data_dir, nbins
            )

    return loss_cls(**kwargs)


def _check_name(category: str, name: str) -> None:
    options = REGISTRY[category]
    if name not in options:
        valid = ", ".join(sorted(options))
        raise ValueError(
            f"Unknown {category} name '{name}'. Valid options: {valid}"
        )


def _validate_registry_names(cfg: dict) -> None:
    """Validate all REGISTRY name references in cfg before construction.

    Produces a single, clear error listing valid options instead of a deep
    KeyError raised mid-construction.
    """
    _check_name("integrator", _require(cfg, "integrator", "name"))
    _check_name("loss", _require(cfg, "loss", "name"))
    _check_name("data_loader", _require(cfg, "data_loader", "name"))

    encoders = _require(cfg, "encoders")
    if not isinstance(encoders, list):
        raise TypeError(
            f"cfg['encoders'] must be a list, got {type(encoders).__name__}"
        )
    for i, enc in enumerate(encoders):
        if not isinstance(enc, dict) or "name" not in enc:
            raise ValueError(
                f"cfg['encoders'][{i}] must be a dict with a 'name' key"
            )
        _check_name("encoders", enc["name"])

    surrogates = _require(cfg, "surrogates")
    if not isinstance(surrogates, dict):
        raise TypeError(
            f"cfg['surrogates'] must be a dict, got {type(surrogates).__name__}"
        )
    for key, sur in surrogates.items():
        if not isinstance(sur, dict) or "name" not in sur:
            raise ValueError(
                f"cfg['surrogates'][{key!r}] must be a dict with a 'name' key"
            )
        _check_name("surrogates", sur["name"])


def construct_integrator(
    cfg: dict,
    skip_warmstart: bool = False,
) -> BaseIntegrator:
    """Build the integrator + its components from a YAML config.

    `skip_warmstart=True` strips `warmstart_basis_path` from qp's args
    so no basis file is read at construction time. Intended for the
    predict CLI, where `load_state_dict` will restore trained weights
    right after construction anyway and warmstarts are pure overhead
    (and a failure mode if the basis file is missing or has been
    regenerated since training).
    """
    _validate_registry_names(cfg)

    # integrator class
    integrator_cls = _get_integrator_cls(cfg["integrator"]["name"])

    # get integrator components
    integrator_args = configs.IntegratorCfg(**cfg["integrator"]["args"])
    encoders = _get_encoder_modules(cfg)
    surrogates = _get_surrogate_modules(cfg, skip_warmstart=skip_warmstart)
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


def _resolved_path_info(path: str) -> dict:
    """Return absolute path, existence, and size for a resolved file path."""
    p = Path(path).resolve()
    entry: dict = {"path": str(p)}
    try:
        st = p.stat()
        entry["exists"] = True
        entry["size_bytes"] = st.st_size
    except (FileNotFoundError, PermissionError):
        entry["exists"] = False
    return entry


def _collect_resolved_paths(cfg: dict) -> dict:
    """Collect the absolute paths of every file the factory actually loads.

    Mirrors the resolution logic in `_get_surrogate_modules` and
    `_get_loss_module` so the saved report matches what training actually
    read from disk, including any `_{n_bins}` suffixes and `data_dir`
    joins that the YAML's relative paths hide.
    """
    report: dict = {"data_dir": _get_data_dir(cfg), "n_bins": _get_n_bins(cfg)}
    data_dir = report["data_dir"]
    n_bins = report["n_bins"]

    # Data loader: shoebox_file_names joined against data_dir
    dl_args = cfg.get("data_loader", {}).get("args", {})
    shoebox_files = dl_args.get("shoebox_file_names", {})
    dl_paths: dict = {}
    if isinstance(shoebox_files, dict):
        for k, fname in shoebox_files.items():
            if k == "data_dir":
                # `shoebox_file_names.data_dir` is a directory override,
                # not a filename; skip.
                continue
            if isinstance(fname, str):
                dl_paths[k] = _resolved_path_info(
                    _resolve_data_path(fname, data_dir, None)
                )
    if dl_paths:
        report["data_loader"] = dl_paths

    # Surrogates: resolve any of the two basis-like path kwargs. Both get
    # the n_bins suffix rewrite via _resolve_data_path.
    surr_paths: dict = {}
    for key, surrogate_cfg in cfg.get("surrogates", {}).items():
        args = surrogate_cfg.get("args", {}) or {}
        name = surrogate_cfg.get("name")
        if name not in ("fixed_basis_profile", "learned_basis_profile"):
            continue
        for arg_key in ("basis_path", "warmstart_basis_path"):
            v = args.get(arg_key)
            if isinstance(v, str):
                surr_paths[f"{key}.{arg_key}"] = _resolved_path_info(
                    _resolve_data_path(v, data_dir, n_bins)
                )
    if surr_paths:
        report["surrogates"] = surr_paths

    # Loss buffers (bg_rate_per_group, tau_per_group, etc.)
    loss_args = cfg.get("loss", {}).get("args", {}) or {}
    loss_paths: dict = {}
    global_conc = "pprf_quantile" in loss_args
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
        v = loss_args.get(pt_key)
        if isinstance(v, str):
            skip_nbins = pt_key == "concentration_per_group" and global_conc
            nbins = None if skip_nbins else n_bins
            loss_paths[pt_key] = _resolved_path_info(
                _resolve_data_path(v, data_dir, nbins)
            )

    # Dirichlet prior concentration file (resolved in _get_prior_cfgs)
    for p_key in ("pprf_cfg", "pbg_cfg", "pi_cfg"):
        p_dict = loss_args.get(p_key)
        if not isinstance(p_dict, dict):
            continue
        if p_dict.get("name") != "dirichlet":
            continue
        params = p_dict.get("params") or {}
        conc = params.get("concentration")
        if (
            isinstance(conc, str)
            and not os.path.isabs(conc)
            and not conc.startswith("~")
        ):
            loss_paths[f"{p_key}.concentration"] = _resolved_path_info(
                os.path.join(data_dir, conc)
            )
    if loss_paths:
        report["loss"] = loss_paths

    return report


def save_run_artifacts(
    integrator: BaseIntegrator,
    cfg: dict,
    logdir: Path,
) -> None:
    """Save model metadata and prior artifacts to the wandb log directory.

    Saves:
        - prior_concentration.pt: the rescaled Dirichlet concentration vector
          used during training
        - run_artifacts.yaml: prior configs, model param counts, loss settings,
          and every file the factory actually loaded (post path resolution)
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

    # Resolved absolute paths for every file the factory loaded, with
    # existence + size. Lets you audit exactly which _{n_bins}-suffixed
    # and data_dir-joined files were used, even if the YAML only names them
    # relatively.
    artifacts["resolved_paths"] = _collect_resolved_paths(cfg)

    # Write summary YAML
    with open(artifacts_dir / "run_artifacts.yaml", "w") as f:
        yaml.safe_dump(artifacts, f, sort_keys=False, default_flow_style=False)


def load_config(resource: str | Path) -> dict:
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
