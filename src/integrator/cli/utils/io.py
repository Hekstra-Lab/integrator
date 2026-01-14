from copy import deepcopy
from pathlib import Path
from typing import Any

from integrator.utils import load_config


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
    epochs: int | None = None,
    batch_size: int | None = None,
    data_path: Path | None = None,
) -> dict:
    base = cfg  # plain dict
    updates: dict[str, Any] = {}
    if epochs is not None:
        updates.setdefault("trainer", {}).setdefault("args", {})[
            "max_epochs"
        ] = epochs
    if batch_size is not None:
        updates.setdefault("data_loader", {}).setdefault("args", {})[
            "batch_size"
        ] = batch_size
    if data_path is not None:
        updates.setdefault("data_loader", {}).setdefault("args", {})[
            "data_dir"
        ] = str(data_path)

    merged = _deep_merge(base, updates)
    return merged


def write_refl_from_preds(
    ckpt_dir,
    refl_file,
    config: dict,
    epoch: int,
):
    import pandas as pd
    import reciprocalspaceship as rs
    import torch

    from integrator.utils.refl_utils import (
        DEFAULT_REFL_COLS,
        unstack_preds,
        write_refl_from_ds,
    )

    pred_file = list(ckpt_dir.glob("preds.pt"))[0]
    data = torch.load(pred_file, weights_only=False)
    fname = ckpt_dir / f"preds_epoch_{epoch:04d}.refl"

    ds = rs.io.read_dials_stills(refl_file, extra_cols=DEFAULT_REFL_COLS)
    unstacked_preds = unstack_preds(data)

    id_filter = ds["refl_ids"].isin(unstacked_preds["refl_ids"])
    ds_filtered = ds[id_filter].sort_values(by="refl_ids")

    pred_df = pd.DataFrame(unstacked_preds).sort_values(by="refl_ids")

    # Overwriting columns
    ds_filtered["intensity.prf.value"] = pred_df["qi_mean"]
    ds_filtered["intensity.prf.variance"] = pred_df["qi_var"]
    ds_filtered["intensity.sum.value"] = pred_df["qi_mean"]
    ds_filtered["intensity.sum.variance"] = pred_df["qi_var"]
    ds_filtered["background.mean"] = pred_df["qbg_mean"]

    # Getting identifiers
    identifiers_path = (
        Path(config["global_vars"]["data_dir"]) / "identifiers.yaml"
    )

    if not identifiers_path.exists():
        raise RuntimeError(f"Missing identifiers.yaml at {identifiers_path}")

    identifiers = load_config(identifiers_path)

    write_refl_from_ds(ds_filtered, fname, identifiers=identifiers)
