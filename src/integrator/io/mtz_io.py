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
):
    """Build and write an MTZ file from integrator predictions.

    Args:
        pred_data: dict with keys refl_ids, qi_mean, qi_var, qbg_mean.
        metadata_path: path to metadata.npy (has H,K,L, wavelength, xyzcal, etc.)
        data_dir: dataset directory holding dataset.yaml (cell + spacegroup).
        out_path: where to write the .mtz file.
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

    if "image_num" in meta:
        batch = _meta("image_num").astype(np.int32) + 1
    else:
        batch = np.ones(len(refl_ids), dtype=np.int32)
        logger.warning("no image_num in metadata; BATCH set to 1 for all rows")

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

    mask = ds["SIGI"] > 0
    n_filtered = (~mask).sum()
    if n_filtered > 0:
        logger.info("filtered %d reflections with SIGI==0", n_filtered)
        ds = ds[mask]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.write_mtz(str(out_path), skip_problem_mtztypes=True)
    logger.info("wrote %s (%d reflections)", out_path, len(ds))

    return ds


def extract_merged_posterior(
    integrator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (alpha, beta, seen) for the per-HKL Gamma posterior on I_h."""
    name = type(integrator).__name__
    if not hasattr(integrator, "get_merged_qi"):
        raise NotImplementedError(
            f"{name} has no get_merged_qi(); diagnostic supports "
            "ConjugateMergingIntegrator "
        )

    with torch.no_grad():
        q = integrator.get_merged_qi()
    alpha = q.concentration.detach().cpu().numpy().astype(np.float64)
    beta = q.rate.detach().cpu().numpy().astype(np.float64)

    seen = None
    for _buf in ("buffer_seen", "feat_seen"):
        if hasattr(integrator, _buf):
            seen = (
                getattr(integrator, _buf).detach().cpu().numpy().astype(bool)
            )
            break
    if seen is None:
        seen = np.ones(len(alpha), dtype=bool)

    logger.info(
        "Extracted q(I_h): %d HKLs total, %d seen, alpha mean %.3g, beta mean %.3g",
        len(alpha),
        seen.sum(),
        alpha[seen].mean() if seen.any() else float("nan"),
        beta[seen].mean() if seen.any() else float("nan"),
    )
    return alpha, beta, seen


def load_crystal(data_dir: Path) -> tuple[gemmi.UnitCell, gemmi.SpaceGroup]:
    """Cell + space group from <data_dir>/dataset.yaml (the make_shoeboxes spec)."""
    from integrator.io import read_dataset_spec

    spec = read_dataset_spec(data_dir) or {}
    crystal = spec.get("crystal")
    if not crystal:
        raise FileNotFoundError(
            f"no `crystal` block in {data_dir}/dataset.yaml (re-run make_shoeboxes)"
        )
    cell = gemmi.UnitCell(*crystal["cell"])
    num = crystal.get("space_group_number")
    if num is not None:
        sg = gemmi.SpaceGroup(int(num))
    else:
        sg = gemmi.SpaceGroup(
            str(crystal["space_group"]).split("(")[0].strip()
        )
    return cell, sg


def load_hkl_table(
    data_dir: Path, cfg: dict, cell: gemmi.UnitCell, sg: gemmi.SpaceGroup
) -> np.ndarray:
    """Merge-id -> canonical-(h,k,l) table, derived from the metadata."""
    from integrator.io import load_metadata, miller_index_columns

    ref_name = cfg["data_loader"]["args"]["shoebox_file_names"]["reference"]
    meta = load_metadata(data_dir / ref_name)
    anomalous = bool(cfg["integrator"]["args"].get("anomalous", True))
    cellp = [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma]
    _, _, tables = miller_index_columns(
        meta["H"],
        meta["K"],
        meta["L"],
        space_group=sg,
        cell=cellp,
        anomalous=anomalous,
    )
    key = "miller_idx_unfriedelized" if anomalous else "miller_idx_friedelized"
    return tables[key].astype(np.int32)


def write_merged_mtz(
    alpha: np.ndarray,
    beta: np.ndarray,
    seen: np.ndarray,
    hkl: np.ndarray,
    cell: gemmi.UnitCell,
    sg: gemmi.SpaceGroup,
    out_path: Path,
) -> rs.DataSet:
    """Write anomalous merged MTZ with both intensity and amplitude columns.

    For each HKL h, q(I_h) = Gamma(alpha_h, beta_h):
        E[I_h]   = alpha / beta
        Var[I_h] = alpha / beta^2
        SIGI     = sqrt(Var[I_h])

    Amplitude propagation (delta method on F = sqrt(I)):
        E[F_h] ≈ sqrt(E[I_h])         (small-uncertainty approx)
        SIGF   ≈ SIGI / (2 * E[F_h])
    """
    if len(alpha) != len(hkl):
        raise ValueError(
            f"alpha has {len(alpha)} entries but asu_id_to_hkl has {len(hkl)}"
        )

    mask = seen & (beta > 0)
    if not mask.any():
        raise RuntimeError("no HKLs with observations in the buffer")

    I_mean = (alpha / beta).astype(np.float64)
    I_var = (alpha / beta.clip(min=1e-12) ** 2).astype(np.float64)
    sigI = np.sqrt(np.clip(I_var, 0.0, None))

    F_mean = np.sqrt(np.clip(I_mean, 0.0, None))
    sigF = sigI / (2.0 * np.clip(F_mean, 1e-12, None))

    H, K, L = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    ds = rs.DataSet(
        {
            "H": rs.DataSeries(H[mask], dtype="H"),
            "K": rs.DataSeries(K[mask], dtype="H"),
            "L": rs.DataSeries(L[mask], dtype="H"),
            "F": rs.DataSeries(F_mean[mask], dtype="F"),
            "SIGF": rs.DataSeries(sigF[mask], dtype="Q"),
            "I": rs.DataSeries(I_mean[mask], dtype="J"),
            "SIGI": rs.DataSeries(sigI[mask], dtype="Q"),
        },
        cell=cell,
        spacegroup=sg,
        merged=True,
    ).set_index(["H", "K", "L"])

    ds = ds[ds["SIGI"] > 0]
    ds = ds.unstack_anomalous()

    anom_order = [
        "F(+)",
        "SIGF(+)",
        "F(-)",
        "SIGF(-)",
        "I(+)",
        "SIGI(+)",
        "I(-)",
        "SIGI(-)",
    ]
    ordered = [c for c in anom_order if c in ds.columns] + [
        c for c in ds.columns if c not in anom_order
    ]
    ds = ds[ordered]

    # TODO: check why this happens
    # Drop rows where ALL of F(+), F(-), I(+), I(-) are NaN
    # gaps, beam stop, etc.).
    drop_check = [
        c for c in ("F(+)", "F(-)", "I(+)", "I(-)") if c in ds.columns
    ]
    if drop_check:
        before = len(ds)
        ds = ds.dropna(subset=drop_check, how="all")
        dropped = before - len(ds)
        if dropped:
            logger.info(
                "Dropped %d all-NaN rows from unstacked MTZ "
                "(systematic absences / unobserved HKLs)",
                dropped,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.write_mtz(str(out_path), skip_problem_mtztypes=True)
    logger.info(
        "Wrote %s: %d reflections (anomalous), cell=%s, sg=%s",
        out_path,
        len(ds),
        cell,
        sg.hm,
    )

    if {"F(+)", "F(-)"}.issubset(ds.columns):
        pair = ds[["F(+)", "F(-)"]].dropna()
        if len(pair) > 100:
            fp = pair["F(+)"].to_numpy()
            fm = pair["F(-)"].to_numpy()
            corr = float(np.corrcoef(fp, fm)[0, 1])
            mean_F = 0.5 * (fp + fm)
            rel_diff = np.abs(fp - fm) / np.clip(mean_F, 1e-12, None)
            logger.info("Friedel-pair sanity:")
            logger.info("  pearson(F(+), F(-)) = %.4f   (good: > 0.99)", corr)
            logger.info(
                "  median |ΔF|/F        = %.3f   (good: 0.01-0.03 for HEWL S-anomalous)",
                float(np.median(rel_diff)),
            )
            logger.info(
                "  mean   |ΔF|/F        = %.3f",
                float(np.mean(rel_diff)),
            )
            logger.info(
                "  N Friedel pairs      = %d",
                len(pair),
            )

    return ds
