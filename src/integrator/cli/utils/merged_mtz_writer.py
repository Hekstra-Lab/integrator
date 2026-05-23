"""Write merged MTZ from a scaling model checkpoint.

Extracts per-HKL Gamma parameters directly from the checkpoint's
HKL lookup table, computes mean intensity and sigma, and writes a
merged MTZ with anomalous columns I(+)/SIGI(+)/I(-)/SIGI(-) suitable
for phenix.refine.

Uses the canonical HKL from `asu_id_to_hkl.pt` (produced by
`prepare_asu_ids.py`) and `rs.DataSet.unstack_anomalous` to correctly
split Friedel pairs into anomalous columns -- the same approach used
by careless and abismal.
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


def _detect_table_type(state_dict: dict) -> str:
    """Detect whether the checkpoint uses Gamma or amplitude table."""
    if "hkl_table.raw_fano.weight" in state_dict:
        return "gamma"
    if "hkl_table.raw_sigma.weight" in state_dict:
        return "amplitude"
    raise KeyError("Cannot detect HKL table type from checkpoint keys.")


def _extract_gamma_params(
    state_dict: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract Gamma table params. Returns (I_mean, sig_I, k, rate)."""
    raw_mu = state_dict["hkl_table.raw_mu.weight"].cpu().squeeze(-1)
    raw_fano = state_dict["hkl_table.raw_fano.weight"].cpu().squeeze(-1)

    eps = 1e-6
    k_min = 0.1

    mu = torch.exp(raw_mu)
    fano = F.softplus(raw_fano) + eps
    rate = 1.0 / fano
    k = mu * rate + k_min

    gamma_mean = (k / rate).numpy()
    gamma_std = np.sqrt((k / rate.pow(2)).numpy())

    return gamma_mean, gamma_std, k.numpy(), rate.numpy()


def _extract_amplitude_params(
    state_dict: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract amplitude table params. Returns (F_mean, sig_F, I_mean, sig_I).

    Uses exact FoldedNormal moments for F = |X|, X ~ N(mu, sigma).
    """
    from scipy.special import erfc
    from scipy.stats import norm as sp_norm

    raw_mu = state_dict["hkl_table.raw_mu.weight"].cpu().squeeze(-1).numpy()
    raw_sigma = state_dict["hkl_table.raw_sigma.weight"].cpu().squeeze(-1)
    sigma = (F.softplus(raw_sigma) + 1e-6).numpy()
    mu = raw_mu

    # FoldedNormal mean: E[|X|] = sigma*sqrt(2/pi)*exp(-mu^2/(2*sigma^2)) + mu*(1 - 2*Phi(-mu/sigma))
    t = mu / np.maximum(sigma, 1e-12)
    F_mean = (
        sigma * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * t ** 2)
        + mu * (1.0 - erfc(t / np.sqrt(2.0)))
    )
    F_mean = np.maximum(F_mean, 0.0)

    # I = E[F^2] = E[X^2] = mu^2 + sigma^2
    I_mean = mu ** 2 + sigma ** 2

    # Var[F] = E[F^2] - E[F]^2
    F_var = np.maximum(I_mean - F_mean ** 2, 0.0)
    sig_F = np.sqrt(F_var)

    # sig_I from delta method: Var[X^2] = E[X^4] - E[X^2]^2
    # For Normal: E[X^4] = mu^4 + 6*mu^2*sigma^2 + 3*sigma^4
    EX4 = mu ** 4 + 6.0 * mu ** 2 * sigma ** 2 + 3.0 * sigma ** 4
    sig_I = np.sqrt(np.maximum(EX4 - I_mean ** 2, 0.0))

    return F_mean.astype(np.float64), sig_F.astype(np.float64), I_mean.astype(np.float64), sig_I.astype(np.float64)


def _load_asu_hkl(
    asu_id_to_hkl_path: Path,
) -> np.ndarray:
    """Load the canonical (H, K, L) per asu_id from `asu_id_to_hkl.pt`.

    Returns an (n_hkl, 3) int32 array where row i is the canonical
    Miller index for asu_id=i.  These are the symmetry-canonical forms
    produced by `prepare_asu_ids.py` and preserve the Friedel distinction
    when `--anomalous` was used.
    """
    id_to_hkl = torch.load(
        asu_id_to_hkl_path, weights_only=False, map_location="cpu"
    )
    return id_to_hkl.numpy().astype(np.int32)


def _build_anomalous_dataset(
    hkl: np.ndarray,
    F_mean: np.ndarray,
    sig_F: np.ndarray,
    I_mean: np.ndarray,
    sig_I: np.ndarray,
    spacegroup: gemmi.SpaceGroup,
    cell: gemmi.UnitCell,
) -> rs.DataSet:
    """Build a merged anomalous DataSet using `unstack_anomalous`.

    This follows the same pattern used by careless and abismal:

    1. Create an `rs.DataSet` with the canonical HKL (which include
       both Friedel-plus and Friedel-minus forms for acentric
       reflections when anomalous asu_ids are used).
    2. Set the HKL as the index with `set_index(['H', 'K', 'L'])`.
    3. Call `unstack_anomalous()` which internally maps HKL to the
       non-anomalous ASU via `hkl_to_asu`, uses M/ISYM to identify
       Friedel pairs, and produces separate (+)/(-) columns.

    For centric reflections, `unstack_anomalous` correctly sets
    I(-) = I(+) since centrics have no Bijvoet difference.
    """
    ds = rs.DataSet(
        {
            "H": rs.DataSeries(hkl[:, 0], dtype="H"),
            "K": rs.DataSeries(hkl[:, 1], dtype="H"),
            "L": rs.DataSeries(hkl[:, 2], dtype="H"),
            "F": rs.DataSeries(F_mean, dtype="F"),
            "SIGF": rs.DataSeries(sig_F, dtype="Q"),
            "I": rs.DataSeries(I_mean, dtype="J"),
            "SIGI": rs.DataSeries(sig_I, dtype="Q"),
        },
        cell=cell,
        spacegroup=spacegroup,
        merged=True,
    )
    ds = ds.set_index(["H", "K", "L"])

    ds = ds.unstack_anomalous()

    # Reorder columns so phenix sees F(+)/SIGF(+)/F(-)/SIGF(-) first,
    # then I(+)/SIGI(+)/I(-)/SIGI(-) -- matching DIALS/careless convention.
    anom_keys = [
        "F(+)", "SIGF(+)", "F(-)", "SIGF(-)",
        "I(+)", "SIGI(+)", "I(-)", "SIGI(-)",
    ]
    reorder = [k for k in anom_keys if k in ds.columns]
    reorder += [k for k in ds.columns if k not in reorder]
    ds = ds[reorder]

    return ds


def write_merged_mtz_from_checkpoint(
    checkpoint_path: Path,
    metadata_path: Path,
    crystal_yaml_path: Path,
    out_path: Path,
    asu_id_to_hkl_path: Path | None = None,
) -> rs.DataSet:
    """Extract merged structure factors from a checkpoint and write MTZ.

    Args:
        checkpoint_path: path to the Lightning checkpoint.
        metadata_path: path to `metadata.pt` (used as fallback for HKL).
        crystal_yaml_path: path to `crystal.yaml` with cell/space_group.
        out_path: where to write the merged MTZ.
        asu_id_to_hkl_path: path to `asu_id_to_hkl.pt` produced by
            `prepare_asu_ids.py`.  Contains the canonical (H,K,L) per
            asu_id that preserves Friedel distinction.  If None, looks
            for it next to `metadata_path`.
    """
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state_dict = ckpt["state_dict"]

    table_type = _detect_table_type(state_dict)
    if table_type == "amplitude":
        F_mean, sig_F, I_mean, sig_I = _extract_amplitude_params(state_dict)
    else:
        from scipy.special import gammaln

        I_mean, sig_I, k, rate = _extract_gamma_params(state_dict)
        F_mean = np.exp(
            gammaln(k + 0.5) - gammaln(k)
        ) / np.sqrt(np.maximum(rate, 1e-12))
        F_var = np.maximum(I_mean - F_mean ** 2, 0.0)
        sig_F = np.sqrt(F_var)

    # Load canonical HKL from asu_id_to_hkl.pt (preferred) or fall back
    # to metadata.  The asu_id_to_hkl.pt file stores the true canonical
    # forms from prepare_asu_ids.py that preserve Friedel distinction.
    if asu_id_to_hkl_path is None:
        asu_id_to_hkl_path = metadata_path.parent / "asu_id_to_hkl.pt"

    if asu_id_to_hkl_path.exists():
        hkl = _load_asu_hkl(asu_id_to_hkl_path)
        n_hkl = len(hkl)
        logger.info(
            "Loaded %d canonical HKL from %s", n_hkl, asu_id_to_hkl_path
        )
    else:
        logger.warning(
            "%s not found; falling back to first-observed HKL from "
            "metadata (anomalous signal may be lost).",
            asu_id_to_hkl_path,
        )
        hkl, n_hkl = _build_asu_hkl_table_fallback(metadata_path)

    if len(I_mean) != n_hkl:
        raise ValueError(
            f"Table has {len(I_mean)} entries but HKL map has {n_hkl} "
            "unique asu_ids."
        )

    crystal = yaml.safe_load(crystal_yaml_path.read_text())
    cell = gemmi.UnitCell(*crystal["cell"])
    sg_str = crystal.get(
        "space_group", crystal.get("space_group_symbol", "P1")
    )
    sg_str = sg_str.split("(")[0].strip()
    spacegroup = gemmi.SpaceGroup(sg_str)

    mask = sig_F > 0
    n_filtered = (~mask).sum()
    if n_filtered > 0:
        logger.info("Filtered %d reflections with SIGF==0", n_filtered)
        hkl = hkl[mask]
        F_mean = F_mean[mask]
        sig_F = sig_F[mask]
        I_mean = I_mean[mask]
        sig_I = sig_I[mask]

    ds = _build_anomalous_dataset(
        hkl,
        F_mean.astype(np.float64),
        sig_F.astype(np.float64),
        I_mean.astype(np.float64),
        sig_I.astype(np.float64),
        spacegroup,
        cell,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.write_mtz(str(out_path), skip_problem_mtztypes=True)
    logger.info(
        "Wrote %s (%d reflections, anomalous columns)", out_path, len(ds)
    )

    return ds


def _build_asu_hkl_table_fallback(
    metadata_path: Path,
) -> tuple[np.ndarray, int]:
    """Fallback: get the first observed (H, K, L) per asu_id from metadata.

    This may not preserve the Friedel distinction if the first observed
    HKL for both members of a Friedel pair happen to be in the same
    symmetry-equivalent form.  Prefer using `asu_id_to_hkl.pt` instead.
    """
    meta = torch.load(metadata_path, weights_only=False, map_location="cpu")
    asu_id = meta["asu_id"].long().numpy()
    H = meta["H"].numpy()
    K = meta["K"].numpy()
    L = meta["L"].numpy()

    n_hkl = int(asu_id.max()) + 1
    canon = np.zeros((n_hkl, 3), dtype=np.int32)
    seen = np.zeros(n_hkl, dtype=bool)

    for i in range(len(asu_id)):
        aid = asu_id[i]
        if not seen[aid]:
            canon[aid] = [int(H[i]), int(K[i]), int(L[i])]
            seen[aid] = True

    return canon, n_hkl
