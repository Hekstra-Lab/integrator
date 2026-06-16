"""Apply command-line overrides onto a loaded YAML config dict."""

from copy import deepcopy


def _deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_cli_overrides(
    cfg: dict,
    *,
    args,
) -> dict:
    base = dict(cfg)
    updates = {}

    def _trainer(k, v):
        updates.setdefault("trainer", {})[k] = v

    def _dl_args(k, v):
        updates.setdefault("data_loader", {}).setdefault("args", {})[k] = v

    def _integrator_args(k, v):
        updates.setdefault("integrator", {}).setdefault("args", {})[k] = v

    def _loss_args(k, v):
        updates.setdefault("loss", {}).setdefault("args", {})[k] = v

    if getattr(args, "max_epochs", None) is not None:
        _trainer("max_epochs", args.max_epochs)
    if getattr(args, "gradient_clip_val", None) is not None:
        _trainer("gradient_clip_val", args.gradient_clip_val)
    if getattr(args, "precision", None) is not None:
        _trainer("precision", args.precision)
    if getattr(args, "accelerator", None) is not None:
        _trainer("accelerator", args.accelerator)
    if getattr(args, "devices", None) is not None:
        _trainer("devices", args.devices)
    if getattr(args, "check_val_every_n_epoch", None) is not None:
        _trainer("check_val_every_n_epoch", args.check_val_every_n_epoch)

    if getattr(args, "batch_size", None) is not None:
        _dl_args("batch_size", args.batch_size)
    if getattr(args, "data_path", None) is not None:
        _dl_args("data_dir", str(args.data_path))
    if getattr(args, "num_workers", None) is not None:
        _dl_args("num_workers", args.num_workers)
    if getattr(args, "val_split", None) is not None:
        _dl_args("val_split", args.val_split)
    if getattr(args, "subset_size", None) is not None:
        _dl_args("subset_size", args.subset_size)

    if getattr(args, "integrator_name", None) is not None:
        updates.setdefault("integrator", {})["name"] = args.integrator_name
    if getattr(args, "lr", None) is not None:
        _integrator_args("lr", args.lr)
    if getattr(args, "weight_decay", None) is not None:
        _integrator_args("weight_decay", args.weight_decay)
    if getattr(args, "mc_samples", None) is not None:
        _integrator_args("mc_samples", args.mc_samples)

    if getattr(args, "qi", None) is not None:
        updates.setdefault("surrogates", {}).setdefault("qi", {})["name"] = (
            args.qi
        )
    if getattr(args, "qbg", None) is not None:
        updates.setdefault("surrogates", {}).setdefault("qbg", {})["name"] = (
            args.qbg
        )

    if getattr(args, "pprf_weight", None) is not None:
        _loss_args("pprf_weight", args.pprf_weight)
    if getattr(args, "pbg_weight", None) is not None:
        _loss_args("pbg_weight", args.pbg_weight)
    if getattr(args, "pi_weight", None) is not None:
        _loss_args("pi_weight", args.pi_weight)
    if getattr(args, "n_bins", None) is not None:
        _loss_args("n_bins", args.n_bins)

    merged = _deep_merge(base, updates)
    return merged
