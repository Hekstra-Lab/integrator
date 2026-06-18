"""Apply command-line overrides onto a loaded YAML config dict."""

from copy import deepcopy

# arg name -> (config path, optional value transform)
ARG_OVERRIDES = {
    "max_epochs": (("trainer", "max_epochs"), None),
    "gradient_clip_val": (("trainer", "gradient_clip_val"), None),
    "precision": (("trainer", "precision"), None),
    "accelerator": (("trainer", "accelerator"), None),
    "devices": (("trainer", "devices"), None),
    "check_val_every_n_epoch": (("trainer", "check_val_every_n_epoch"), None),
    "batch_size": (("data_loader", "args", "batch_size"), None),
    "data_dir": (("data_loader", "args", "data_dir"), str),
    "num_workers": (("data_loader", "args", "num_workers"), None),
    "val_split": (("data_loader", "args", "val_split"), None),
    "subset_size": (("data_loader", "args", "subset_size"), None),
    "lr": (("optimizer", "lr"), None),
    "weight_decay": (("optimizer", "weight_decay"), None),
    "mc_samples": (("integrator", "args", "mc_samples"), None),
    "integrator_name": (("integrator", "name"), None),
    "qi": (("surrogates", "qi", "name"), None),
    "qbg": (("surrogates", "qbg", "name"), None),
    "pprf_weight": (("loss", "args", "pprf_weight"), None),
    "pbg_weight": (("loss", "args", "pbg_weight"), None),
    "pi_weight": (("loss", "args", "pi_weight"), None),
    "n_bins": (("loss", "args", "n_bins"), None),
}


def _deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_nested(d: dict, path: tuple, value) -> None:
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


def _apply_cli_overrides(cfg: dict, *, args) -> dict:
    updates: dict = {}
    for arg, (path, transform) in ARG_OVERRIDES.items():
        v = getattr(args, arg, None)
        if v is None:
            continue
        _set_nested(updates, path, transform(v) if transform else v)
    return _deep_merge(dict(cfg), updates)
