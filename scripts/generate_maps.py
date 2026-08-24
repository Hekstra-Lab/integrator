"""Generate electron density maps from a refinement model checkpoint.

Produces CCP4 map files using gemmi - no phenix dependency.

Maps generated:
  1. Model density (F_calc only) - the model's prediction of electron density
  2. 2Fo-Fc map - uses DIALS I_prf for Fo, model for Fc and phases
  3. Fo-Fc difference map - reveals model errors
  4. Anomalous difference map - F_calc(+) - F_calc(-)

Usage
-----
    uv run python scripts/generate_maps.py \
        --config configs/variational_refinement_hewl.yaml \
        --checkpoint path/to/epoch.ckpt \
        --out-dir maps/
"""

import argparse
import logging
import math
from pathlib import Path

import gemmi
import numpy as np
import torch
import torch.nn.functional as F
import yaml

if not hasattr(gemmi.UnitCell, "fractionalization_matrix"):
    gemmi.UnitCell.fractionalization_matrix = property(
        lambda self: self.frac.mat
    )
if not hasattr(gemmi.UnitCell, "orthogonalization_matrix"):
    gemmi.UnitCell.orthogonalization_matrix = property(
        lambda self: self.orth.mat
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EIGHT_PI_SQ = 8.0 * math.pi**2


def setup_sfcalculator(config, state_dict):
    from SFC_Torch import SFcalculator as SFC

    int_args = config["integrator"]["args"]
    data_dir = Path(config["data_loader"]["args"]["data_dir"])
    pdb_path = int_args["pdb_path"]
    if not Path(pdb_path).is_absolute():
        pdb_path = str(data_dir / pdb_path)

    sfc = SFC(
        pdbmodel=pdb_path,
        dmin=int_args["dmin"],
        anomalous=True,
        wavelength=int_args.get("wavelength", 1.0),
        device="cpu",
    )
    sfc.inspect_data()

    if "atom_pos_mu" in state_dict:
        sfc.atom_pos_orth = state_dict["atom_pos_mu"].cpu()
        raw_log_sigma = state_dict["atom_raw_log_sigma"].cpu()
        sigma = F.softplus(raw_log_sigma) + 1e-6
        sfc.atom_b_iso = EIGHT_PI_SQ * sigma.pow(2)
    else:
        sfc.atom_pos_orth = state_dict["atom_pos"].cpu()
        sfc.atom_b_iso = state_dict["atom_b_iso"].cpu()

    return sfc


def write_map(grid, path, normalize=True):
    if normalize:
        grid.normalize()
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(path))
    logger.info("Wrote %s", path)


def build_mtz(hkl, f_amp, f_phase_deg, cell, spacegroup, columns=None):
    """Build a gemmi.Mtz from arrays."""
    mtz = gemmi.Mtz()
    mtz.cell = cell
    mtz.spacegroup = spacegroup
    mtz.add_dataset("HKL_base")
    mtz.add_column("H", "H")
    mtz.add_column("K", "H")
    mtz.add_column("L", "H")

    col_names = columns or [("F", "F"), ("PHI", "P")]
    for name, mtz_type in col_names:
        mtz.add_column(name, mtz_type)

    n = len(hkl)
    data = np.zeros((n, 3 + len(col_names)), dtype=np.float32)
    data[:, 0] = hkl[:, 0]
    data[:, 1] = hkl[:, 1]
    data[:, 2] = hkl[:, 2]
    data[:, 3] = f_amp
    data[:, 4] = f_phase_deg

    mtz.set_data(data)
    return mtz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("maps"))
    parser.add_argument("--sample-rate", type=float, default=3.0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(config["data_loader"]["args"]["data_dir"])
    crystal = yaml.safe_load((data_dir / "crystal.yaml").read_text())
    cell = gemmi.UnitCell(*crystal["cell"])
    sg_str = crystal.get("space_group", "P1").split("(")[0].strip()
    spacegroup = gemmi.SpaceGroup(sg_str)

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    state_dict = ckpt["state_dict"]
    sfc = setup_sfcalculator(config, state_dict)

    # ---------------------------------------------------------------
    # 1. Model density map (F_calc)
    # ---------------------------------------------------------------
    logger.info("Computing F_calc...")
    Fc_plus = sfc.calc_fprotein(Return=True).detach()
    hkl = sfc.Hasu_array.astype(np.int32)

    Fc_amp = Fc_plus.abs().numpy().astype(np.float32)
    Fc_phase = np.degrees(torch.angle(Fc_plus).numpy()).astype(np.float32)

    mtz_calc = build_mtz(hkl, Fc_amp, Fc_phase, cell, spacegroup)
    grid = mtz_calc.transform_f_phi_to_map("F", "PHI", sample_rate=args.sample_rate)
    write_map(grid, args.out_dir / "model_density.ccp4")

    # ---------------------------------------------------------------
    # 2 & 3. 2Fo-Fc and Fo-Fc maps (using DIALS I_prf for Fo)
    # ---------------------------------------------------------------
    ref_name = (
        config["data_loader"]["args"]
        .get("shoebox_file_names", {})
        .get("reference", "metadata.pt")
    )
    meta = torch.load(data_dir / ref_name, weights_only=False, map_location="cpu")

    if "intensity.prf.value" in meta:
        logger.info("Computing Fo from DIALS profile-fitted intensities...")
        int_args = config["integrator"]["args"]

        # Build asu_id -> SFC index mapping
        asu_id_to_hkl_path = int_args["asu_id_to_hkl_path"]
        if not Path(asu_id_to_hkl_path).is_absolute():
            asu_id_to_hkl_path = str(data_dir / asu_id_to_hkl_path)
        id_to_hkl = torch.load(asu_id_to_hkl_path, weights_only=False, map_location="cpu")

        from integrator.model.scaling.refinement_integrator import _build_hasu_lookup
        hasu_lookup = _build_hasu_lookup(sfc.Hasu_array, spacegroup)

        # Merge DIALS I_prf per SFC HKL index (merging Friedel mates)
        n_hkl = len(hkl)
        I_sum_w = np.zeros(n_hkl, dtype=np.float64)
        w_sum = np.zeros(n_hkl, dtype=np.float64)

        asu_ids = meta["asu_id"].long().numpy()
        I_prf = meta["intensity.prf.value"].numpy()
        var_prf = meta["intensity.prf.variance"].numpy()

        for i in range(len(asu_ids)):
            aid = asu_ids[i]
            if aid >= len(id_to_hkl):
                continue
            h, k, l = int(id_to_hkl[aid, 0]), int(id_to_hkl[aid, 1]), int(id_to_hkl[aid, 2])
            key = (h, k, l)
            if key not in hasu_lookup:
                continue
            sfc_idx = hasu_lookup[key]
            v = var_prf[i]
            if v > 0:
                w = 1.0 / v
                I_sum_w[sfc_idx] += w * I_prf[i]
                w_sum[sfc_idx] += w

        I_merged = np.where(w_sum > 0, I_sum_w / w_sum, 0.0)
        Fo_amp = np.sqrt(np.clip(I_merged, 0, None)).astype(np.float32)

        # Scale Fo to Fc (linear scale factor in resolution shells)
        observed = (w_sum > 0) & (Fc_amp > 0) & (Fo_amp > 0)
        if observed.sum() > 100:
            k_scale = np.sum(Fo_amp[observed] * Fc_amp[observed]) / np.sum(Fc_amp[observed] ** 2)
            Fc_scaled = Fc_amp * k_scale
        else:
            Fc_scaled = Fc_amp

        # 2Fo-Fc coefficients
        twofofc_amp = np.where(observed, 2 * Fo_amp - Fc_scaled, 0.0).astype(np.float32)
        twofofc_amp = np.clip(twofofc_amp, 0, None)

        mtz_2fofc = build_mtz(hkl, twofofc_amp, Fc_phase, cell, spacegroup)
        grid = mtz_2fofc.transform_f_phi_to_map("F", "PHI", sample_rate=args.sample_rate)
        write_map(grid, args.out_dir / "2fofc.ccp4")

        # Fo-Fc coefficients
        fofc_amp = np.where(observed, Fo_amp - Fc_scaled, 0.0).astype(np.float32)

        mtz_fofc = build_mtz(hkl, np.abs(fofc_amp), Fc_phase, cell, spacegroup)
        # For negative Fo-Fc, flip the phase by 180 degrees
        fofc_phase = np.where(fofc_amp >= 0, Fc_phase, Fc_phase + 180.0).astype(np.float32)
        mtz_fofc = build_mtz(hkl, np.abs(fofc_amp), fofc_phase, cell, spacegroup)
        grid = mtz_fofc.transform_f_phi_to_map("F", "PHI", sample_rate=args.sample_rate)
        write_map(grid, args.out_dir / "fofc.ccp4")

        logger.info(
            "Fo stats: %d observed, k_scale=%.3f, mean|Fo|=%.1f, mean|Fc|=%.1f",
            observed.sum(), k_scale if observed.sum() > 100 else 0,
            Fo_amp[observed].mean(), Fc_amp[observed].mean(),
        )
    else:
        logger.warning("No intensity.prf.value in metadata - skipping Fo-Fc maps")

    # ---------------------------------------------------------------
    # 4. Anomalous difference map
    # ---------------------------------------------------------------
    logger.info("Computing anomalous map...")
    original_hasu = sfc.Hasu_array.copy()
    sfc.Hasu_array = -original_hasu
    Fc_minus = sfc.calc_fprotein(Return=True).detach()
    sfc.Hasu_array = original_hasu

    dano = (Fc_plus.abs() - Fc_minus.abs()).numpy().astype(np.float32)
    panom = (np.degrees(torch.angle(Fc_plus).numpy()) - 90.0).astype(np.float32)

    mtz_anom = build_mtz(hkl, np.abs(dano), panom, cell, spacegroup)
    # Flip phase for negative DANO
    anom_phase = np.where(dano >= 0, panom, panom + 180.0).astype(np.float32)
    mtz_anom = build_mtz(hkl, np.abs(dano), anom_phase, cell, spacegroup)
    grid = mtz_anom.transform_f_phi_to_map("F", "PHI", sample_rate=args.sample_rate)
    write_map(grid, args.out_dir / "anomalous.ccp4")

    logger.info("")
    logger.info("Open in Coot:")
    logger.info("  coot --pdb refined.pdb \\")
    logger.info("       --map %s \\", args.out_dir / "2fofc.ccp4")
    logger.info("       --map %s", args.out_dir / "fofc.ccp4")
    logger.info("")
    logger.info("Or PyMOL:")
    logger.info("  load refined.pdb")
    logger.info("  load %s, map_2fofc", args.out_dir / "2fofc.ccp4")
    logger.info("  load %s, map_fofc", args.out_dir / "fofc.ccp4")
    logger.info("  load %s, map_anom", args.out_dir / "anomalous.ccp4")
    logger.info("  isomesh mesh_2fo, map_2fofc, 1.0, refined, carve=2.0")
    logger.info("  isomesh mesh_pos, map_fofc, 3.0, refined, carve=2.0")
    logger.info("  isomesh mesh_neg, map_fofc, -3.0, refined, carve=2.0")
    logger.info("  isomesh mesh_anom, map_anom, 3.0, refined, carve=2.0")


if __name__ == "__main__":
    main()
