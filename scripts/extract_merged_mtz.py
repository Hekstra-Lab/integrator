"""Extract merged structure factors from a trained scaling model checkpoint.

Reads the HKL lookup table directly from the checkpoint, computes
Gamma mean/variance per unique ASU reflection, and writes a merged
MTZ suitable for phenix.refine.

Usage
-----
    uv run python scripts/extract_merged_mtz.py \\
        --config configs/scaling_hewl.yaml \\
        --checkpoint wandb/.../epoch=99-step=....ckpt \\
        --out merged.mtz

The output MTZ contains one row per unique ASU reflection with columns:
    H, K, L, IMEAN, SIGIMEAN
"""

import argparse
import logging
from pathlib import Path

import gemmi
import numpy as np
import torch
import torch.nn.functional as F
import yaml

import reciprocalspaceship as rs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_table_params(state_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compute Gamma mean and std from the HKL table's raw parameters."""
    raw_mu = state_dict["hkl_table.raw_mu.weight"].cpu().squeeze(-1)
    raw_fano = state_dict["hkl_table.raw_fano.weight"].cpu().squeeze(-1)

    eps = 1e-6
    k_min = 0.1

    mu = F.softplus(raw_mu) + eps
    fano = F.softplus(raw_fano) + eps
    rate = 1.0 / fano
    k = mu * rate + k_min

    gamma_mean = (k / rate).numpy()
    gamma_var = (k / rate.pow(2)).numpy()
    gamma_std = np.sqrt(np.clip(gamma_var, 0, None))

    return gamma_mean, gamma_std


def build_asu_hkl_table(
    metadata_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Get the canonical (H, K, L) for each asu_id from metadata.

    Returns the first observed (H, K, L) per asu_id, which is one of
    the symmetry-equivalent forms.
    """
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

    if not seen.all():
        n_missing = (~seen).sum()
        logger.warning("%d asu_ids have no observations", n_missing)

    return canon_H, canon_K, canon_L, n_hkl


def main():
    parser = argparse.ArgumentParser(
        description="Extract merged MTZ from a scaling model checkpoint."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("merged.mtz"))
    parser.add_argument(
        "--crystal-yaml",
        type=Path,
        default=None,
        help="Path to crystal.yaml with cell and space_group. "
        "If not given, looks in data_dir.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data_loader"]["args"]["data_dir"])
    ref_name = (
        config["data_loader"]["args"]
        .get("shoebox_file_names", {})
        .get("reference", "metadata.pt")
    )
    metadata_path = data_dir / ref_name

    crystal_yaml = args.crystal_yaml or (data_dir / "crystal.yaml")
    if not crystal_yaml.exists():
        raise FileNotFoundError(
            f"{crystal_yaml} not found. Create it with cell and space_group, "
            "or pass --crystal-yaml."
        )
    crystal = yaml.safe_load(crystal_yaml.read_text())

    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    state_dict = ckpt["state_dict"]

    logger.info("Extracting structure factors from HKL table")
    I_mean, sig_I = extract_table_params(state_dict)

    logger.info("Building (H,K,L) mapping from %s", metadata_path)
    canon_H, canon_K, canon_L, n_hkl = build_asu_hkl_table(metadata_path)

    if len(I_mean) != n_hkl:
        raise ValueError(
            f"Table has {len(I_mean)} entries but metadata has {n_hkl} "
            "unique asu_ids — checkpoint and data are mismatched."
        )

    cell_params = crystal["cell"]
    cell = gemmi.UnitCell(*cell_params)
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.write_mtz(str(args.out), skip_problem_mtztypes=True)
    logger.info(
        "Wrote %s: %d reflections, cell=%s, sg=%s",
        args.out,
        len(ds),
        cell,
        spacegroup.hm,
    )


if __name__ == "__main__":
    main()
