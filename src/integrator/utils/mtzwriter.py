from pathlib import Path

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch

# TODO: pass optional spacegroup and unitcell parameters
# Currently setting default spacegroup and unitcell parameters


def mtz_writer(
    pred_path: str | Path,
    file_name: str | Path,
    filter_by_batch: int | None = None,
) -> None:
    preds = torch.load(
        pred_path,
        weights_only=False,
    )
    data = rs.DataSet(
        {
            "H": np.hstack(preds["h"]).astype(np.int32),
            "K": np.hstack(preds["k"]).astype(np.int32),
            "L": np.hstack(preds["l"]).astype(np.int32),
            "BATCH": np.hstack(preds["batch"]) + 1,
            "I": np.hstack(preds["intensity_mean"]),
            "SIGI": np.hstack(preds["intensity_var"]) ** 0.5,
            "xcal": np.hstack(preds["x_c"]),
            "ycal": np.hstack(preds["y_c"]),
            "wavelength": np.hstack(preds["wavelength"]),
            "BG": np.hstack(preds["qbg_mean"]),
            "SIGBG": np.hstack(preds["qbg_var"]) ** 0.5,
            "PARTIAL": np.zeros(len(np.hstack(preds["batch"])), dtype=bool),
        },
        cell=gemmi.UnitCell(78.312, 78.312, 37.234, 90.0, 90.0, 90.0),
        spacegroup=gemmi.SpaceGroup("P 43 21 2"),
    ).infer_mtz_dtypes()

    if filter_by_batch is not None:
        data = data[data["BATCH"] < filter]

    data.write_mtz(file_name)
