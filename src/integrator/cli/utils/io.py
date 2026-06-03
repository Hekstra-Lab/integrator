import logging
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
    BOOL_COLS,
    DEFAULT_REFL_COLS,
    unstack_preds,
    write_refl_from_ds,
)

logger = logging.getLogger(__name__)


def _read_reference_refl(refl_file):
    """Read a DIALS .refl, tolerating columns rs cannot parse as extra_cols.

    Some rs versions fail to read 1-byte `bool` columns (e.g. `entering`) via
    `extra_cols` ("buffer size must be a multiple of element size"). These
    columns are not needed for the integrator's output, so we degrade
    gracefully: try the full set, then drop bool columns, then (last resort)
    probe and drop any individually-unreadable column.
    """
    attempts = [
        list(DEFAULT_REFL_COLS),
        [c for c in DEFAULT_REFL_COLS if c not in BOOL_COLS],
    ]
    for cols in attempts:
        try:
            return rs.io.read_dials_stills(refl_file, extra_cols=cols)
        except Exception as e:  # noqa: BLE001 - rs raises bare ValueError
            logger.warning(
                "read_dials_stills failed with %d extra_cols (%s); retrying "
                "with fewer columns",
                len(cols),
                e,
            )
    good = []
    for c in DEFAULT_REFL_COLS:
        try:
            rs.io.read_dials_stills(refl_file, extra_cols=[c])
            good.append(c)
        except Exception:  # noqa: BLE001
            logger.warning("Dropping unreadable refl column: %s", c)
    return rs.io.read_dials_stills(refl_file, extra_cols=good)


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

    # Training
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

    # Data loader
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

    # Integrator
    if getattr(args, "integrator_name", None) is not None:
        updates.setdefault("integrator", {})["name"] = args.integrator_name
    if getattr(args, "lr", None) is not None:
        _integrator_args("lr", args.lr)
    if getattr(args, "weight_decay", None) is not None:
        _integrator_args("weight_decay", args.weight_decay)
    if getattr(args, "mc_samples", None) is not None:
        _integrator_args("mc_samples", args.mc_samples)

    # Surrogates
    if getattr(args, "qi", None) is not None:
        updates.setdefault("surrogates", {}).setdefault("qi", {})["name"] = (
            args.qi
        )
    if getattr(args, "qbg", None) is not None:
        updates.setdefault("surrogates", {}).setdefault("qbg", {})["name"] = (
            args.qbg
        )

    # Loss weights
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
        available = set(lf.collect_schema().names())
        # Required scalar columns + optional ones (calibrated exact posterior,
        # coset flag) loaded only when present. Array columns like qp_mean are
        # intentionally excluded (downstream uses pd.DataFrame on this dict).
        wanted = [
            "refl_ids",
            "qi_mean",
            "qi_var",
            "qbg_mean",
            "qi_exact_mean",
            "qi_exact_var",
            "qi_exact_std",
            "is_coset",
        ]
        cols = [c for c in wanted if c in available]
        sel = lf.select(cols).collect()
        data = {c: sel[c].to_numpy().ravel() for c in cols}
    else:
        raise ValueError(f"Unsupported filetype: {filetype}")

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

    # Exclude coset reflections — they have no meaningful intensity predictions
    if "is_coset" in data:
        is_coset = data["is_coset"]
        if hasattr(is_coset, "astype"):
            is_coset = is_coset.astype(bool)
        lattice_mask = ~is_coset
        data = {k: v[lattice_mask] for k, v in data.items()}

    # filename of output .refl file
    fname = ckpt_dir / f"preds_epoch_{epoch:04d}.refl"

    # Read .refl file with rs
    ds = rs.io.read_dials_stills(refl_file, extra_cols=DEFAULT_REFL_COLS)

    id_filter = ds["refl_ids"].isin(data["refl_ids"])
    ds_filtered = ds[id_filter].sort_values(by="refl_ids")
    pred_df = pd.DataFrame(data).sort_values(by="refl_ids")

    ds_filtered = ds_filtered.sort_values("refl_ids").reset_index(drop=True)
    pred_df = pred_df.sort_values("refl_ids").reset_index(drop=True)

    # Choose the intensity/uncertainty source
    if "qi_exact_mean" in pred_df and "qi_exact_var" in pred_df:
        i_value = pred_df["qi_exact_mean"]
        i_variance = pred_df["qi_exact_var"]
        logger.info(
            "Writing .refl intensities from the calibrated exact posterior "
            "(qi_exact_mean / qi_exact_var)"
        )
    else:
        i_value = pred_df["qi_mean"]
        i_variance = pred_df["qi_var"]
        logger.info(
            "Writing .refl intensities from the mean-field posterior "
            "(qi_mean / qi_var); qi_exact_* not found in predictions"
        )

    # Overwriting columns
    ds_filtered["intensity.prf.value"] = i_value
    ds_filtered["intensity.prf.variance"] = i_variance
    ds_filtered["intensity.sum.value"] = i_value
    ds_filtered["intensity.sum.variance"] = i_variance
    ds_filtered["background.mean"] = pred_df["qbg_mean"]

    # Getting identifiers
    identifiers_path = (
        Path(config["data_loader"]["args"]["data_dir"]) / "identifiers.yaml"
    )

    if not identifiers_path.exists():
        raise RuntimeError(f"Missing identifiers.yaml at {identifiers_path}")

    identifiers = load_config(identifiers_path)

    write_refl_from_ds(ds_filtered, fname, identifiers=identifiers)
