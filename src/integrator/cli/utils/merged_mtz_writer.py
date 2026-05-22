"""Write merged MTZ from a scaling model checkpoint.

Extracts per-HKL Gamma parameters directly from the checkpoint's
HKL lookup table, computes mean intensity and sigma, and writes a
merged MTZ suitable for phenix.refine.
"""

import logging
from pathlib import Path

import gemmi
import numpy as np
import torch
import torch.nn.functional as F
import yaml

import reciprocalspaceship as rs

logger = logging.getLogger(__name__)


def _extract_table_params(state_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    raw_mu = state_dict["hkl_table.raw_mu.weight"].cpu().squeeze(-1)
    raw_fano = state_dict["hkl_table.raw_fano.weight"].cpu().squeeze(-1)

    eps = 1e-6
    k_min = 0.1

    mu = torch.exp(raw_mu)
    fano = F.softplus(raw_fano) + eps
    rate = 1.0 / fano
    k = mu * rate + k_min

    gamma_mean = (k / rate).numpy()
    gamma_var = (k / rate.pow(2)).numpy()
    gamma_std = np.sqrt(np.clip(gamma_var, 0, None))

    return gamma_mean, gamma_std


def _build_asu_hkl_table(
    metadata_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    meta = torch.load(metadata_path, weights_only=False, map_location="cpu")
    asu_id = meta["asu_id"].long().numpy()
    H = meta["H"].numpy()
    K = meta["K"].numpy()
    L = meta["L"].numpy()

    n_hkl = int(asu_id.max()) + 1
    canon_H = np.zeros(n_hkl, dtype=np.int32)
    canon_K = np.zeros(n_hkl, dtype=np.int32)
    canon_L = np.zeros(n_hkl, dtype=np.int32)
    seen = np.zeros(n_hkl, dtype=bool)

    for i in range(len(asu_id)):
        aid = asu_id[i]
        if not seen[aid]:
            canon_H[aid] = int(H[i])
            canon_K[aid] = int(K[i])
            canon_L[aid] = int(L[i])
            seen[aid] = True

    return canon_H, canon_K, canon_L, n_hkl


def write_merged_mtz_from_checkpoint(
    checkpoint_path: Path,
    metadata_path: Path,
    crystal_yaml_path: Path,
    out_path: Path,
) -> rs.DataSet:
    """Extract merged structure factors from a checkpoint and write MTZ."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state_dict = ckpt["state_dict"]

    I_mean, sig_I = _extract_table_params(state_dict)
    canon_H, canon_K, canon_L, n_hkl = _build_asu_hkl_table(metadata_path)

    if len(I_mean) != n_hkl:
        raise ValueError(
            f"Table has {len(I_mean)} entries but metadata has {n_hkl} "
            "unique asu_ids."
        )

    crystal = yaml.safe_load(crystal_yaml_path.read_text())
    cell = gemmi.UnitCell(*crystal["cell"])
    sg_str = crystal.get(
        "space_group", crystal.get("space_group_symbol", "P1")
    )
    sg_str = sg_str.split("(")[0].strip()
    spacegroup = gemmi.SpaceGroup(sg_str)

    ds = rs.DataSet(
        {
            "H": canon_H,
            "K": canon_K,
            "L": canon_L,
            "IMEAN": I_mean.astype(np.float64),
            "SIGIMEAN": sig_I.astype(np.float64),
        },
        cell=cell,
        spacegroup=spacegroup,
    ).infer_mtz_dtypes()

    mask = ds["SIGIMEAN"] > 0
    n_filtered = (~mask).sum()
    if n_filtered > 0:
        logger.info("Filtered %d reflections with SIGIMEAN==0", n_filtered)
        ds = ds[mask]

    # Convert anomalous Friedel-pair rows into I(+)/I(-) columns
    # for phenix.refine compatibility.
    ds.hkl_to_asu(anomalous=True)
    ds_anom = ds.unstack_anomalous()
    ds_anom.rename(
        columns={
            "IMEAN(+)": "I(+)",
            "SIGIMEAN(+)": "SIGI(+)",
            "IMEAN(-)": "I(-)",
            "SIGIMEAN(-)": "SIGI(-)",
        },
        inplace=True,
    )
    ds_anom.infer_mtz_dtypes(inplace=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_anom.write_mtz(str(out_path), skip_problem_mtztypes=True)
    logger.info(
        "Wrote %s (%d reflections, anomalous columns)", out_path, len(ds_anom)
    )

    return ds_anom
