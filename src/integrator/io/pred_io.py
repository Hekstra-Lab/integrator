"""Load integrator predictions from disk and write them back to a DIALS `.refl`."""

from pathlib import Path
from typing import Literal

import h5py
import polars as pl
import torch

from .refl_io import unstack_preds, write_refl_with_predictions


# WARNING: Will use an arbitrary file if both .h5 and pt files are in ckpt_dir
def get_pred_files(
    ckpt_dir: Path,
    filetype: Literal["h5", "pt", "parquet"],
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

    return data


def write_refl_from_preds(
    ckpt_dir,
    refl_file,
    epoch: int,
    filetype: Literal["h5", "pt", "parquet"],
):
    """Write per-epoch predictions back into a copy of the source `.refl`."""
    data = get_pred_files(ckpt_dir=ckpt_dir, filetype=filetype)

    # Exclude coset reflections — they have no meaningful intensity predictions
    if "is_coset" in data:
        is_coset = data["is_coset"]
        if hasattr(is_coset, "astype"):
            is_coset = is_coset.astype(bool)
        lattice_mask = ~is_coset
        data = {k: v[lattice_mask] for k, v in data.items()}

    fname = ckpt_dir / f"preds_epoch_{epoch:04d}.refl"
    pred_df = pl.DataFrame(data).sort("refl_ids")

    write_refl_with_predictions(
        refl_file=refl_file,
        out_file=fname,
        refl_ids=pred_df["refl_ids"].to_numpy(),
        i_value=pred_df["qi_mean"].to_numpy(),
        i_variance=pred_df["qi_var"].to_numpy(),
        bg_mean=pred_df["qbg_mean"].to_numpy(),
    )
    return fname
