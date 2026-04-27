"""Write model predictions to MTZ format for downstream scaling with Careless.

Produces an MTZ file matching the column layout of laue-dials integrate.py
so that Careless can consume it directly:
    H, K, L, BATCH, I, SIGI, xcal, ycal, wavelength, BG, SIGBG
"""

import logging
from pathlib import Path

import gemmi
import numpy as np
import torch
import yaml

import reciprocalspaceship as rs

logger = logging.getLogger(__name__)


def write_mtz_from_preds(
    pred_data: dict,
    metadata_path: Path,
    crystal_yaml_path: Path,
    out_path: Path,
):
    """Build and write an MTZ file from integrator predictions.

    Args:
        pred_data: dict with keys refl_ids, qi_mean, qi_var, qbg_mean.
        metadata_path: path to metadata.pt (has H,K,L, wavelength, xyzcal, etc.)
        crystal_yaml_path: path to crystal.yaml (has cell + spacegroup).
        out_path: where to write the .mtz file.
    """
    meta = torch.load(metadata_path, weights_only=True, map_location="cpu")
    crystal = yaml.safe_load(crystal_yaml_path.read_text())

    refl_ids = pred_data["refl_ids"]
    if isinstance(refl_ids, torch.Tensor):
        refl_ids = refl_ids.numpy()
    refl_ids = refl_ids.astype(int)

    idx = np.argsort(refl_ids)
    refl_ids = refl_ids[idx]

    def _get(key):
        v = pred_data[key]
        if isinstance(v, torch.Tensor):
            v = v.numpy()
        return v[idx]

    qi_mean = _get("qi_mean")
    qi_var = _get("qi_var")
    qbg_mean = _get("qbg_mean")

    def _meta(key):
        return meta[key].numpy()[refl_ids]

    H = _meta("H").astype(np.int32)
    K = _meta("K").astype(np.int32)
    L = _meta("L").astype(np.int32)
    wavelength = _meta("wavelength").astype(np.float64)
    xcal = _meta("xyzcal.px.0").astype(np.float64)
    ycal = _meta("xyzcal.px.1").astype(np.float64)

    if "image_num" in meta:
        batch = _meta("image_num").astype(np.int32) + 1
    else:
        batch = np.ones(len(refl_ids), dtype=np.int32)
        logger.warning("no image_num in metadata; BATCH set to 1 for all rows")

    cell_params = crystal["cell"]
    cell = gemmi.UnitCell(*cell_params)

    sg_str = crystal.get("space_group", crystal.get("space_group_symbol", "P1"))
    sg_str = sg_str.split("(")[0].strip()
    spacegroup = gemmi.SpaceGroup(sg_str)

    sigi = np.sqrt(np.clip(qi_var, 0, None))
    sigbg = np.sqrt(np.clip(qbg_mean, 0, None))

    ds = rs.DataSet(
        {
            "H": H,
            "K": K,
            "L": L,
            "BATCH": batch,
            "I": qi_mean.astype(np.float64),
            "SIGI": sigi.astype(np.float64),
            "xcal": xcal,
            "ycal": ycal,
            "wavelength": wavelength,
            "BG": qbg_mean.astype(np.float64),
            "SIGBG": sigbg.astype(np.float64),
        },
        cell=cell,
        spacegroup=spacegroup,
    ).infer_mtz_dtypes()

    mask = ds["SIGI"] > 0
    n_filtered = (~mask).sum()
    if n_filtered > 0:
        logger.info("filtered %d reflections with SIGI==0", n_filtered)
        ds = ds[mask]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.write_mtz(str(out_path), skip_problem_mtztypes=True)
    logger.info("wrote %s (%d reflections)", out_path, len(ds))

    return ds
