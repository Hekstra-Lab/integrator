from copy import deepcopy
from pathlib import Path
from typing import Literal

import h5py
import pandas as pd
import polars as pl
import reciprocalspaceship as rs
import torch

from integrator.utils import load_config
from integrator.utils.refl_utils import (
    DEFAULT_REFL_COLS,
    unstack_preds,
    write_refl_from_ds,
)


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

    if args.max_epochs is not None:
        updates.setdefault("trainer", {})["max_epochs"] = args.epochs
    if args.batch_size is not None:
        updates.setdefault("data_loader", {}).setdefault("args", {})[
            "batch_size"
        ] = args.batch_size
    if args.data_path is not None:
        updates.setdefault("data_loader", {}).setdefault("args", {})[
            "data_dir"
        ] = str(args.data_path)
    if args.qi is not None:
        updates.setdefault("surrogates", {}).setdefault("qi", {})["name"] = (
            args.qi
        )
    if args.qbg is not None:
        updates.setdefault("surrogates", {}).setdefault("qbg", {})["name"] = (
            args.qbg
        )
    if args.integrator_name is not None:
        updates.setdefault("integrator", {})["name"] = args.integrator_name

    merged = _deep_merge(base, updates)
    return merged


# WARNING: Will use an arbitrary file if both .h5 and pt files are in ckpt_dir
def get_pred_files(
    ckpt_dir: Path,
    filetype: Literal[
        "h5",
        "pt",
        "parquet",
    ],
):
    data = None
    if filetype == "pt":
        pred_files = list(ckpt_dir.glob("preds_epoch_*.pt"))
        if not pred_files:
            raise RuntimeError(f"No prediction files found in {ckpt_dir}")
        pred_file = pred_files[0]
        data = torch.load(pred_file, weights_only=False)
        data = unstack_preds(data)
    elif filetype == "h5":
        pred_files = list(ckpt_dir.glob("preds_epoch_*.h5"))
        if not pred_files:
            raise RuntimeError(f"No prediction files found in {ckpt_dir}")
        pred_file = pred_files[0]
        with h5py.File(pred_file, "r") as f:
            data = {
                "refl_ids": f["refl_ids"][:],
                "qi_mean": f["qi_mean"][:],
                "qi_var": f["qi_var"][:],
                "qbg_mean": f["qbg_mean"][:],
            }
    elif filetype == "parquet":
        pred_files = list(ckpt_dir.glob("preds_epoch_*.parquet"))
        lf = pl.scan_parquet(pred_files)
        refl_ids = lf.select("refl_ids").collect().to_numpy().ravel()
        qi_mean = lf.select("qi_mean").collect().to_numpy().ravel()
        qi_var = lf.select("qi_var").collect().to_numpy().ravel()
        qbg_mean = lf.select("qbg_mean").collect().to_numpy().ravel()

        data = {
            "refl_ids": refl_ids,
            "qi_mean": qi_mean,
            "qi_var": qi_var,
            "qbg_mean": qbg_mean,
        }
    else:
        raise ValueError(f"Unsupported filetype: {filetype}")

    if data is None:
        raise ValueError(f"Unsupported prediction file type: {suffix}")

    return data


def write_refl_from_preds(
    ckpt_dir,
    refl_file,
    config: dict,
    epoch: int,
    filetype: Literal["h5", "pt", "parquet"],
):
    # REPLACE WITH FUNCTION
    data = get_pred_files(ckpt_dir=ckpt_dir, filetype=filetype)

    # filename of output .refl file
    fname = ckpt_dir / f"preds_epoch_{epoch:04d}.refl"

    # Read .refl file with rs
    ds = rs.io.read_dials_stills(refl_file, extra_cols=DEFAULT_REFL_COLS)

    id_filter = ds["refl_ids"].isin(data["refl_ids"])
    ds_filtered = ds[id_filter].sort_values(by="refl_ids")
    pred_df = pd.DataFrame(data).sort_values(by="refl_ids")

    ds_filtered = ds_filtered.sort_values("refl_ids").reset_index(drop=True)
    pred_df = pred_df.sort_values("refl_ids").reset_index(drop=True)

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
