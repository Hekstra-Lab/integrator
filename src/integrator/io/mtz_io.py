import logging
from pathlib import Path

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch

from .metadata import load_metadata

logger = logging.getLogger(__name__)


def write_mtz_from_preds(
    pred_data: dict,
    metadata_path: Path,
    data_dir: Path,
    out_path: Path,
    expt_file: Path | None = None,
    drop_unmapped_batch: bool = True,
):
    """Build and write an MTZ file from integrator predictions.

    Args:
        pred_data: dict with keys refl_ids, qi_mean, qi_var, qbg_mean.
        metadata_path: path to metadata.npy (has H,K,L, wavelength, xyzcal, etc.)
        data_dir: dataset directory holding dataset.yaml (cell + spacegroup).
        out_path: where to write the .mtz file.
        expt_file: the DIALS `.expt` handed to `laue.integrate`. Supply it to get
            the laue-dials `BATCH` convention (dense experiment index + 1) rather
            than our `image_num + 1`; see `expt_batch` for why they differ and
            what breaks downstream when they do.
        drop_unmapped_batch: with `expt_file`, drop reflections whose image is
            absent from the experiment list. Those images were dropped upstream,
            so laue-dials never assigned them a BATCH and the reference pipeline
            has no counterpart for them.
    """
    from .dataset import read_dataset_spec

    meta = load_metadata(metadata_path)
    spec = read_dataset_spec(data_dir)
    if spec is None or "crystal" not in spec:
        raise FileNotFoundError(
            f"dataset.yaml with a crystal block not found in {data_dir}"
        )
    crystal = spec["crystal"]

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

    keep = np.ones(len(refl_ids), dtype=bool)
    if "image_num" not in meta:
        batch = np.ones(len(refl_ids), dtype=np.int32)
        logger.warning("no image_num in metadata; BATCH set to 1 for all rows")
    elif expt_file is not None:
        from .expt_batch import batch_map_from_expt

        image_num = _meta("image_num").astype(np.int64)
        bmap = batch_map_from_expt(expt_file)
        lut_size = max(int(image_num.max()) + 1, max(bmap) + 1)
        lut = np.zeros(lut_size, dtype=np.int32)
        for num, b in bmap.items():
            lut[num] = b
        batch = lut[image_num]
        keep = batch > 0
        n_drop = int((~keep).sum())
        if n_drop:
            missing = sorted(set(image_num[~keep].tolist()))
            msg = (
                f"{n_drop} reflections are on images absent from "
                f"{Path(expt_file).name} (image_num {missing[:10]}"
                f"{'...' if len(missing) > 10 else ''})"
            )
            if drop_unmapped_batch:
                logger.info("%s; dropping them", msg)
            else:
                raise ValueError(msg)
    else:
        # image_num is the true image number; laue-dials BATCH is the dense
        # experiment index. They diverge as soon as anything was dropped upstream,
        # and BATCH drives both careless scaling and by-BATCH frame cuts.
        batch = _meta("image_num").astype(np.int32) + 1
        logger.warning(
            "BATCH written as image_num + 1; pass expt_file to match the "
            "laue-dials convention (BATCH = experiment index + 1)"
        )

    cell_params = crystal["cell"]
    cell = gemmi.UnitCell(*cell_params)

    sg_str = crystal.get(
        "space_group", crystal.get("space_group_symbol", "P1")
    )
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

    mask = (ds["SIGI"] > 0).to_numpy() & keep
    n_filtered = int((~mask).sum())
    if n_filtered > 0:
        logger.info(
            "filtered %d reflections (SIGI==0 or no BATCH in the experiment list)",
            n_filtered,
        )
        ds = ds[mask]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.write_mtz(str(out_path), skip_problem_mtztypes=True)
    logger.info("wrote %s (%d reflections)", out_path, len(ds))

    return ds
