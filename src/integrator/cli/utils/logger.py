import logging
import sys


def setup_logging(verbosity: int = 0):
    """
    verbosity = 0 -> WARNING
    verbosity = 1 -> INFO
    verbosity >= 2 -> DEBUG
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def write_refl_from_preds(
    ckpt_dir,
    refl_file,
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

    write_refl_from_ds(ds_filtered, fname)
