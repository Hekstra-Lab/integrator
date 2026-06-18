import os
from dataclasses import asdict
from importlib.resources import as_file
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch.nn as nn
import yaml
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

from integrator import configs
from integrator.configs import shallow_dict
from integrator.model.integrators.base_integrator import BaseIntegrator
from integrator.registry import REGISTRY

# User-supplied YAML args override these.
# Only data_dim is required.
_ENCODER_PRESETS: dict[tuple[str, str], dict] = {
    ("profile_encoder", "3d"): dict(
        in_channels=1,
        encoder_out=64,
        input_shape=(3, 21, 21),
        conv1_out_channels=64,
        conv1_kernel_size=(1, 3, 3),
        conv1_padding=(0, 1, 1),
        norm1_num_groups=4,
        pool_kernel_size=(1, 2, 2),
        pool_stride=(1, 2, 2),
        conv2_out_channels=128,
        conv2_kernel_size=(3, 3, 3),
        conv2_padding=(0, 0, 0),
        norm2_num_groups=4,
    ),
    ("profile_encoder", "2d"): dict(
        in_channels=1,
        encoder_out=32,
        input_shape=(21, 21),
        conv1_out_channels=32,
        conv1_kernel_size=(3, 3),
        conv1_padding=(1, 1),
        norm1_num_groups=4,
        pool_kernel_size=(2, 2),
        pool_stride=(2, 2),
        conv2_out_channels=64,
        conv2_kernel_size=(3, 3),
        conv2_padding=(0, 0),
        norm2_num_groups=4,
    ),
    ("intensity_encoder", "3d"): dict(
        in_channels=1,
        encoder_out=64,
        conv1_out_channels=64,
        conv1_kernel_size=(3, 3, 3),
        conv1_padding=(1, 1, 1),
        norm1_num_groups=4,
        pool_kernel_size=(1, 2, 2),
        pool_stride=(1, 2, 2),
        conv2_out_channels=128,
        conv2_kernel_size=(3, 3, 3),
        conv2_padding=(0, 0, 0),
        norm2_num_groups=4,
        conv3_out_channels=256,
        conv3_kernel_size=(3, 3, 3),
        conv3_padding=(1, 1, 1),
        norm3_num_groups=8,
    ),
    ("intensity_encoder", "2d"): dict(
        in_channels=1,
        encoder_out=32,
        conv1_out_channels=32,
        conv1_kernel_size=(3, 3),
        conv1_padding=(1, 1),
        norm1_num_groups=4,
        pool_kernel_size=(2, 2),
        pool_stride=(2, 2),
        conv2_out_channels=64,
        conv2_kernel_size=(3, 3),
        conv2_padding=(0, 0),
        norm2_num_groups=4,
        conv3_out_channels=128,
        conv3_kernel_size=(3, 3),
        conv3_padding=(1, 1),
        norm3_num_groups=8,
    ),
}

_MODE_DEFAULTS: dict[str, dict[str, str]] = {
    "monochromatic": {
        "loss": "monochromatic_wilson",
        "data_loader": "rotation_data",
    },
    "polychromatic": {
        "loss": "polychromatic_wilson",
        "data_loader": "polychromatic_data",
    },
}

# fail at import if a mode preset points at an unregistered component
for _m, _preset in _MODE_DEFAULTS.items():
    for _cat, _name in _preset.items():
        if _name not in REGISTRY[_cat]:
            raise ValueError(
                f"mode {_m!r}: {_cat} {_name!r} is not in REGISTRY[{_cat!r}]"
            )

# reverse lookup: a loss name implies its mode (configs may set loss.name directly)
_MODE_BY_LOSS = {p["loss"]: m for m, p in _MODE_DEFAULTS.items()}

# Minimal per-mode prediction columns
_DEFAULT_PREDICT_KEYS: dict[str, list[str]] = {
    "monochromatic": [
        "refl_ids",
        "is_test",
        "qi_mean",
        "qi_var",
        "qbg_mean",
        "qbg_var",
        "intensity.prf.value",
        "intensity.prf.variance",
        "intensity.sum.value",
        "intensity.sum.variance",
        "background.mean",
        "d",
        "H",
        "K",
        "L",
    ],
    "polychromatic": [
        "refl_ids",
        "is_test",
        "qi_mean",
        "qi_var",
        "qbg_mean",
        "qbg_var",
        "intensity.sum.value",
        "intensity.sum.variance",
        "background.sum.value",
        "background.sum.variance",
        "d",
        "wavelength",
        "H",
        "K",
        "L",
    ],
}


def _resolve_predict_keys(cfg: dict) -> None:
    """Expand predict_keys ('default' -> mode minimal set) + append additional_predict_keys."""
    iargs = cfg.get("integrator", {}).get("args")
    if not isinstance(iargs, dict):
        return
    extra = iargs.pop("additional_predict_keys", None) or []
    pk = iargs.get("predict_keys", "default")
    if pk == "default":
        loss_name = cfg.get("loss", {}).get("name")
        mode = cfg.get("mode") or _MODE_BY_LOSS.get(loss_name)
        base = _DEFAULT_PREDICT_KEYS.get(mode)
        if base is None:
            # unknown mode: leave "default" for the integrator's own fallback
            if not extra:
                return
            base = []
        pk = list(base)
    else:
        pk = list(pk)
    seen: set[str] = set()
    merged: list[str] = []
    for k in pk + list(extra):
        if k not in seen:
            seen.add(k)
            merged.append(k)
    iargs["predict_keys"] = merged


def _apply_encoder_preset(name: str, args: dict) -> dict:
    """Merge encoder preset defaults under user-supplied args."""
    data_dim = args.get("data_dim")
    if data_dim is None:
        return args
    key = (name, data_dim)
    preset = _ENCODER_PRESETS.get(key)
    if preset is None:
        return args
    merged = {**preset, **args}
    return merged


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
    # n_bins is optional
    loss_args = _require(cfg, "loss", "args")
    return loss_args.get("n_bins")


def _get_integrator_cls(name: str) -> type[BaseIntegrator]:
    return REGISTRY["integrator"][name]


def _get_loss_cls(name: str) -> nn.Module:
    return REGISTRY["loss"][name]


def _get_dataloader_cls(name: str) -> nn.Module:
    return REGISTRY["data_loader"][name]


def _get_loss_args(
    cfg: dict,
    prior_configs: dict,
) -> configs.LossArgs:
    return configs.LossArgs(
        pbg_cfg=prior_configs["pbg_cfg"],
        pi_cfg=prior_configs["pi_cfg"],
        pprf_cfg=prior_configs["pprf_cfg"],
    )


def _get_surrogate_modules(cfg: dict) -> dict[str, nn.Module]:
    """Construct all surrogate distribution modules from config."""
    surrogates = {}

    for key, surrogate_cfg in cfg["surrogates"].items():
        surrogate_cls = REGISTRY["surrogates"][surrogate_cfg["name"]]
        args = dict(surrogate_cfg["args"])
        surrogates[key] = surrogate_cls(**args)
    return surrogates


def _get_prior_cfgs(
    cfg: dict,
    pprf_cfg: str = "pprf_cfg",
    pbg_cfg: str = "pbg_cfg",
    pi_cfg: str = "pi_cfg",
) -> dict[str, configs.PriorConfig | None]:
    priors = {}
    data_dir = _get_data_dir(cfg)
    for p in (pprf_cfg, pbg_cfg, pi_cfg):
        p_dict = cfg["loss"]["args"].get(p)
        if p_dict is None:
            priors[p] = None
            continue
        p_name = p_dict["name"]
        params_cls, _ = REGISTRY["priors"][p_name]
        p_params_dict = _resolve_prior_paths(
            p_name, dict(p_dict.get("params") or {}), data_dir
        )
        priors[p] = configs.PriorConfig(
            name=p_name,
            params=params_cls(**p_params_dict),
            weight=p_dict["weight"],
        )
    return priors


def _resolve_prior_paths(name: str, params: dict, data_dir: str) -> dict:
    """Resolve a prior's path-valued args (per the registry) against data_dir."""
    _, path_fields = REGISTRY["priors"][name]
    for field in path_fields:
        v = params.get(field)
        if (
            isinstance(v, str)
            and not os.path.isabs(v)
            and not v.startswith("~")
        ):
            params[field] = os.path.join(data_dir, v)
    return params


def _get_encoder_modules(
    cfg: dict,
) -> dict[str, nn.Module]:
    cfg_ = dict(cfg)
    model = cfg_["integrator"]["name"]
    required_encoders = REGISTRY["integrator"][model].REQUIRED_ENCODERS

    if len(required_encoders) != len(cfg_["encoders"]):
        raise ValueError(
            f"""
            Integration model '{model}' requires {len(required_encoders)}
            encoders, but {len(cfg_["encoders"])} were passed
            """
        )
    encoders = {}
    for (slot, (enc_name, args_cls)), encoder_cfg in zip(
        required_encoders.items(), cfg_["encoders"], strict=True
    ):
        got = encoder_cfg.get("name")
        if got != enc_name:
            raise ValueError(
                f"Integrator '{model}' slot '{slot}' expects encoder "
                f"'{enc_name}', got '{got}'. Check the order of `encoders:`."
            )
        merged = _apply_encoder_preset(enc_name, encoder_cfg.get("args") or {})
        encoder_args = args_cls(**merged)
        encoders[slot] = REGISTRY["encoders"][enc_name](**asdict(encoder_args))
    return encoders


def _get_loss_module(
    cfg: dict,
) -> nn.Module:
    loss_cls = _get_loss_cls(cfg["loss"]["name"])
    prior_configs = _get_prior_cfgs(cfg)
    loss_args = _get_loss_args(
        cfg=cfg,
        prior_configs=prior_configs,
    )
    kwargs = shallow_dict(loss_args)

    for k, v in cfg["loss"]["args"].items():
        if k not in kwargs:
            kwargs[k] = v

    data_dir = _get_data_dir(cfg)

    if "wavelength_bin_edges" in kwargs and isinstance(
        kwargs["wavelength_bin_edges"], str
    ):
        kwargs["wavelength_bin_edges"] = _resolve_data_path(
            kwargs["wavelength_bin_edges"], data_dir, None
        )

    if "spectrum_init_from" in kwargs and isinstance(
        kwargs["spectrum_init_from"], str
    ):
        kwargs["spectrum_init_from"] = _resolve_data_path(
            kwargs["spectrum_init_from"], data_dir, None
        )

    # Inject the empirical background prior
    if "bg_rate" not in kwargs or "bg_concentration" not in kwargs:
        from integrator.io import data_path, load_data
        from integrator.utils.prepare_priors import _nbins_path

        n_bins = int(kwargs.get("n_bins", 1))
        bg_path = _nbins_path("bg_prior.npy", n_bins, Path(data_dir))
        if data_path(bg_path) is not None:
            bg_prior = load_data(bg_path)
            # .tolist() -> float for a 0-d scalar fit, list for per-bin arrays
            kwargs.setdefault("bg_rate", bg_prior["bg_rate"].tolist())
            kwargs.setdefault(
                "bg_concentration", bg_prior["bg_concentration"].tolist()
            )

    valid_keys = _valid_loss_keys(loss_cls)
    unknown = set(kwargs) - valid_keys
    if unknown:
        raise ValueError(
            f"Unknown loss arg(s) for {cfg['loss']['name']}: {sorted(unknown)}. "
            f"Valid args: {sorted(valid_keys)}."
        )

    return loss_cls(**kwargs)


def _valid_loss_keys(loss_cls: type) -> set[str]:
    """All explicit `__init__` arg names across a loss class's MRO."""
    import inspect

    valid: set[str] = set()
    for klass in loss_cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for name, p in inspect.signature(init).parameters.items():
            if name != "self" and p.kind not in (
                p.VAR_KEYWORD,
                p.VAR_POSITIONAL,
            ):
                valid.add(name)
    return valid


def _check_name(category: str, name: str) -> None:
    options = REGISTRY[category]
    if name not in options:
        valid = ", ".join(sorted(options))
        raise ValueError(
            f"Unknown {category} name '{name}'. Valid options: {valid}"
        )


def _validate_registry_names(cfg: dict) -> None:
    """Validate all REGISTRY name references in cfg before construction."""
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
    arg_errors: list[str] = []
    for key, sur in surrogates.items():
        if not isinstance(sur, dict) or "name" not in sur:
            raise ValueError(
                f"cfg['surrogates'][{key!r}] must be a dict with a 'name' key"
            )
        _check_name("surrogates", sur["name"])
        accepted = _constructor_arg_names(REGISTRY["surrogates"][sur["name"]])
        unknown = set(sur.get("args") or {}) - accepted
        if unknown:
            arg_errors.append(
                f"  surrogate {key!r} ({sur['name']}): unknown args "
                f"{sorted(unknown)}; valid: {sorted(accepted)}"
            )
    if arg_errors:
        raise ValueError("Invalid surrogate args:\n" + "\n".join(arg_errors))


def construct_integrator(
    cfg: dict,
) -> BaseIntegrator:
    """Build the integrator + its components from a YAML config."""
    cfg = resolve_config(cfg)
    _validate_registry_names(cfg)

    integrator_cls = _get_integrator_cls(cfg["integrator"]["name"])

    integrator_args = configs.IntegratorCfg(**cfg["integrator"]["args"])
    optimizer_cfg = configs.OptimizerConfig(**cfg.get("optimizer", {}))
    encoders = _get_encoder_modules(cfg)
    surrogates = _get_surrogate_modules(cfg)
    loss = _get_loss_module(cfg)

    return integrator_cls(
        cfg=integrator_args,
        encoders=encoders,
        surrogates=surrogates,
        loss=loss,
        optimizer=optimizer_cfg,
    )


def construct_data_loader(cfg):
    cfg = resolve_config(cfg)
    dl_cls = _get_dataloader_cls(cfg["data_loader"]["name"])
    return dl_cls(**cfg["data_loader"]["args"])


def construct_trainer(
    cfg: dict,
    logger: Logger | bool | None = False,
    callbacks: list[Callback] | Callback | None = None,
) -> pl.Trainer:
    # trainer section is optional; TrainerConfig defaults cover an empty one
    tr_cfg = configs.TrainerConfig(**(cfg.get("trainer") or {}))

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
    "Return absolute path, existence, and size for a resolved file path."
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
    "Collect the absolute paths of every file the factory loads."
    report: dict = {"data_dir": _get_data_dir(cfg), "n_bins": _get_n_bins(cfg)}
    data_dir = report["data_dir"]

    # Data loader: shoebox_file_names joined against data_dir
    dl_args = cfg.get("data_loader", {}).get("args", {})
    shoebox_files = dl_args.get("shoebox_file_names", {})
    dl_paths: dict = {}
    if isinstance(shoebox_files, dict):
        for k, fname in shoebox_files.items():
            if k == "data_dir":
                continue
            if isinstance(fname, str):
                dl_paths[k] = _resolved_path_info(
                    _resolve_data_path(fname, data_dir, None)
                )
    if dl_paths:
        report["data_loader"] = dl_paths

    loss_args = cfg.get("loss", {}).get("args", {}) or {}
    loss_paths: dict = {}

    for p_key in ("pprf_cfg", "pbg_cfg", "pi_cfg"):
        p_dict = loss_args.get(p_key)
        if not isinstance(p_dict, dict):
            continue
        spec = REGISTRY["priors"].get(p_dict.get("name"))
        if not spec:
            continue
        params = p_dict.get("params") or {}
        for field in spec[1]:
            v = params.get(field)
            if (
                isinstance(v, str)
                and not os.path.isabs(v)
                and not v.startswith("~")
            ):
                loss_paths[f"{p_key}.{field}"] = _resolved_path_info(
                    os.path.join(data_dir, v)
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
        - run_artifacts.yaml: prior configs, model param counts, loss settings,
          and every file the factory actually loaded (post path resolution)
    """
    artifacts_dir = Path(logdir) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    loss_module = integrator.loss
    artifacts = {}

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

    artifacts["loss"] = {
        "name": cfg["loss"]["name"],
    }

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

    artifacts["resolved_paths"] = _collect_resolved_paths(cfg)

    with open(artifacts_dir / "run_artifacts.yaml", "w") as f:
        yaml.safe_dump(artifacts, f, sort_keys=False, default_flow_style=False)


def load_config(resource: str | Path) -> dict:
    if isinstance(resource, str):
        resource = Path(resource)

    with as_file(resource) as p:
        with open(Path(p), encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    return raw


def apply_dataset_defaults(cfg: dict) -> dict:
    """Fill dataset-derived values from <data_dir>/dataset.yaml where unset in cfg."""
    from integrator.io import data_dim_for, read_dataset_spec

    dl_args = cfg.get("data_loader", {}).get("args", {})
    data_dir = dl_args.get("data_dir")
    if not data_dir:
        return cfg
    spec = read_dataset_spec(data_dir)
    if not spec:
        return cfg

    geom = spec.get("geometry", {})
    d, h, w = geom.get("d"), geom.get("h"), geom.get("w")
    if d is None or h is None or w is None:
        return cfg
    data_dim = geom.get("data_dim") or data_dim_for(d)

    def fill(target: dict, key, value):
        if value is not None and target.get(key) is None:
            target[key] = value

    iargs = cfg.setdefault("integrator", {}).setdefault("args", {})
    fill(iargs, "data_dim", data_dim)
    fill(iargs, "d", d)
    fill(iargs, "h", h)
    fill(iargs, "w", w)

    input_shape = [h, w] if data_dim == "2d" else [d, h, w]
    for enc in cfg.get("encoders", []) or []:
        eargs = enc.setdefault("args", {})
        fill(eargs, "data_dim", data_dim)
        if enc.get("name") == "profile_encoder":
            fill(eargs, "input_shape", input_shape)

    # derive `transform` from the dataset's anscombe flag
    if dl_args.get("transform") is None:
        dl_args["transform"] = (
            "anscombe" if spec.get("anscombe") else "standardization"
        )
    files = spec.get("files", {})
    if files and dl_args.get("shoebox_file_names") is None:
        dl_args["shoebox_file_names"] = {
            "data_dir": data_dir,
            "counts": files.get("counts", "counts.npy"),
            "masks": files.get("masks", "masks.npy"),
            "reference": files.get("reference", "metadata.npy"),
            "standardized_counts": None,
        }

    refl_file = spec.get("refl_file")
    if refl_file:
        fill(cfg.setdefault("output", {}), "refl_file", refl_file)

    return cfg


def _apply_mode_defaults(cfg: dict) -> None:
    """Fill loss/data_loader registry keys implied by a top-level `mode`, if set."""
    mode = cfg.get("mode")
    if mode is None:
        return
    preset = _MODE_DEFAULTS.get(mode)
    if preset is None:
        valid = ", ".join(sorted(_MODE_DEFAULTS))
        raise ValueError(f"Unknown mode {mode!r}. Valid options: {valid}")

    loss = cfg.setdefault("loss", {})
    loss.setdefault("name", preset["loss"])
    loss.setdefault("args", {})

    cfg.setdefault("data_loader", {}).setdefault("name", preset["data_loader"])


def _inject_poly_metadata(cfg: dict) -> None:
    """Fill the poly loss beam_center + lambda range from <data_dir>/dataset.yaml."""
    if cfg.get("loss", {}).get("name") != "polychromatic_wilson":
        return
    from integrator.io import read_dataset_spec

    data_dir = cfg.get("data_loader", {}).get("args", {}).get("data_dir")
    if not data_dir:
        return
    crystal = (read_dataset_spec(data_dir) or {}).get("crystal", {})
    loss_args = cfg.setdefault("loss", {}).setdefault("args", {})
    if crystal.get("beam_center_px") is not None:
        loss_args.setdefault("beam_center", list(crystal["beam_center_px"]))
    for key in ("lambda_min", "lambda_max"):
        if crystal.get(key) is not None:
            loss_args.setdefault(key, crystal[key])


def _default_encoders(
    integrator_cls: type[BaseIntegrator],
    data_dim: str | None,
    encoder_out: int,
    input_shape: list[int] | None,
) -> list[dict]:
    """Synthesize the `encoders:` list from an integrator's REQUIRED_ENCODERS."""
    encoders: list[dict] = []
    for enc_name, args_cls in integrator_cls.REQUIRED_ENCODERS.values():
        enc_args: dict[str, Any] = {
            "data_dim": data_dim,
            "encoder_out": encoder_out,
        }
        # inject input_shape only for encoders that declare it
        if (
            "input_shape" in args_cls.__dataclass_fields__
            and input_shape is not None
        ):
            enc_args["input_shape"] = input_shape
        encoders.append({"name": enc_name, "args": enc_args})
    return encoders


def _merge_surrogates(defaults: dict, user: dict | None) -> dict:
    """Merge a (possibly partial) `surrogates:` section over the integrator defaults."""
    user = user or {}
    merged: dict[str, dict] = {}
    keys = list(defaults) + [k for k in user if k not in defaults]
    for key in keys:
        d = defaults.get(key, {})
        u = user.get(key, {})
        u_args = dict(u.get("args") or {})
        if "name" in u and u["name"] != d.get("name"):
            merged[key] = {"name": u["name"], "args": u_args}
        else:
            merged[key] = {
                "name": u.get("name", d.get("name")),
                "args": {**dict(d.get("args") or {}), **u_args},
            }
    return merged


def _constructor_arg_names(target) -> set[str]:
    import inspect

    explicit = getattr(target, "arg_names", None)
    if explicit is not None:
        return set(explicit)
    params = inspect.signature(target).parameters
    return {
        n
        for n, p in params.items()
        if n != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }


def _inject_runtime_surrogate_args(
    cfg: dict,
    encoder_out: int,
    profile_n: int | None,
    sbox_shape: list[int] | None,
) -> None:
    """Inject runtime-derived args into each surrogate that declares them."""
    runtime = {
        "input_dim": encoder_out,
        "in_features": encoder_out,
        "output_dim": profile_n,
        "sbox_shape": sbox_shape,
    }
    for sur in (cfg.get("surrogates") or {}).values():
        if not isinstance(sur, dict):
            continue
        target = REGISTRY["surrogates"].get(sur.get("name"))
        if target is None:
            continue
        accepted = _constructor_arg_names(target)
        args = sur.setdefault("args", {})
        for arg, val in runtime.items():
            if arg in accepted and val is not None:
                args.setdefault(arg, val)


def resolve_config(cfg: dict) -> dict:
    """Expand a config to a fully-specified one, filling everything inferable."""
    cfg = apply_dataset_defaults(cfg)
    _apply_mode_defaults(cfg)
    _inject_poly_metadata(cfg)
    _resolve_predict_keys(cfg)

    iargs = cfg.get("integrator", {}).get("args", {})
    encoder_out = iargs.get("encoder_out", 64)
    data_dim = iargs.get("data_dim")
    d, h, w = iargs.get("d"), iargs.get("h"), iargs.get("w")
    input_shape = None
    profile_n = None
    if None not in (d, h, w):
        input_shape = [h, w] if data_dim == "2d" else [d, h, w]
        profile_n = h * w if data_dim == "2d" else d * h * w
    sbox_shape = [d, h, w] if None not in (d, h, w) else None

    integrator_name = cfg.get("integrator", {}).get("name")
    integrator_cls = REGISTRY["integrator"].get(integrator_name)
    if integrator_cls is not None:
        if not cfg.get("encoders"):
            cfg["encoders"] = _default_encoders(
                integrator_cls, data_dim, encoder_out, input_shape
            )
        cfg["surrogates"] = _merge_surrogates(
            integrator_cls.DEFAULT_SURROGATES, cfg.get("surrogates")
        )

    for enc in cfg.get("encoders", []) or []:
        if isinstance(enc, dict):
            enc.setdefault("args", {}).setdefault("encoder_out", encoder_out)
    _inject_runtime_surrogate_args(cfg, encoder_out, profile_n, sbox_shape)

    return cfg
