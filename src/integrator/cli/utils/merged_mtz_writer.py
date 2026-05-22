"""Write merged MTZ from a scaling model checkpoint.

Extracts per-HKL Gamma parameters directly from the checkpoint's
HKL lookup table, computes mean intensity and sigma, and writes a
merged MTZ with anomalous columns I(+)/SIGI(+)/I(-)/SIGI(-) suitable
for phenix.refine.
"""

import logging
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

import reciprocalspaceship as rs
from reciprocalspaceship.utils import hkl_to_asu

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


def _to_anomalous_columns(
    H: np.ndarray,
    K: np.ndarray,
    L: np.ndarray,
    I_mean: np.ndarray,
    sig_I: np.ndarray,
    spacegroup: gemmi.SpaceGroup,
    cell: gemmi.UnitCell,
) -> rs.DataSet:
    """Convert Friedel-pair rows into I(+)/I(-) anomalous columns."""
    hkl = np.column_stack([H, K, L])
    hkl_asu, isym = hkl_to_asu(hkl, spacegroup)
    is_plus = (isym % 2 == 0)

    df = pd.DataFrame({
        "H": hkl_asu[:, 0],
        "K": hkl_asu[:, 1],
        "L": hkl_asu[:, 2],
        "I": I_mean,
        "SIGI": sig_I,
        "is_plus": is_plus,
    })

    plus = df[df["is_plus"]].set_index(["H", "K", "L"])[["I", "SIGI"]]
    minus = df[~df["is_plus"]].set_index(["H", "K", "L"])[["I", "SIGI"]]

    merged = plus.join(minus, lsuffix="(+)", rsuffix="(-)", how="outer")
    merged.columns = ["I(+)", "SIGI(+)", "I(-)", "SIGI(-)"]

    ds = rs.DataSet(
        merged.reset_index(), cell=cell, spacegroup=spacegroup
    ).infer_mtz_dtypes()

    return ds


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

    mask = sig_I > 0
    n_filtered = (~mask).sum()
    if n_filtered > 0:
        logger.info("Filtered %d reflections with SIGI==0", n_filtered)
        canon_H = canon_H[mask]
        canon_K = canon_K[mask]
        canon_L = canon_L[mask]
        I_mean = I_mean[mask]
        sig_I = sig_I[mask]

    ds = _to_anomalous_columns(
        canon_H, canon_K, canon_L,
        I_mean.astype(np.float64),
        sig_I.astype(np.float64),
        spacegroup, cell,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.write_mtz(str(out_path), skip_problem_mtztypes=True)
    logger.info(
        "Wrote %s (%d reflections, anomalous columns)", out_path, len(ds)
    )

    return ds
