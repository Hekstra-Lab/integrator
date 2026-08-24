"""Diagnose a POLYCHROMATIC (Laue) merging checkpoint: MTZ + refine + anom peaks.

The poly analogue of scripts/diagnose_merging.py. Loads the last checkpoint of a
poly amortized-merging run (integrator `amortized_merging` with `scale_laue_mlp`,
data loader `polychromatic_data`), repopulates the per-HKL Gamma posterior
`q(I_h)` via finalize_merge over the grouped predict loader, writes an anomalous
MTZ (I + F columns), then runs a TWO-STAGE phenix.refine on the merged MTZ followed
by anomalous-difference map peak heights at the known scatterer (S/I) sites -- the
metric that matters for these data (the analysis half of refltorch/scripts/
laue_output/{submit_refinement,submit_analysis,anomalous_peak_heights}.py).

Phenix 2.0 (not 1.x): the effs are the phenix-2.0 `data_manager {}` form (the
reference_data/custom_refinement_param_{1,2}_poly.eff templates, poly copies of the
mono reference_data/phenix.eff), with $MTZFILE/$MODELFILE/$RFREE substituted at run
time. Both stages read the SAME merged data + a fixed R-free set; only the MODEL is
handed forward (stage 1 rigid-body -> stage 2 sites+ADP), so there's no MTZ /
column-name handoff. The model's intensities I(+),SIGI(+),... are refined with
French-Wilson exactly as the mono pipeline, so the anomalous map (ANOM/PANOM) and
the peak heights are directly comparable to the mono diagnostic.

Differences from the monochromatic diagnostic (everything else is identical, the
merged posterior is model-agnostic):
  - sigma inflation uses `intensity.sum.*` (Laue stills have no `intensity.prf.*`),
    with auto-fallback to prf if present.
  - the SCALE-VS-WAVELENGTH report: the LaueMLPScale's learned effective spectrum
    G(lambda). This is the one poly-specific failure mode -- if the scale did not
    learn the incident spectrum, observations at different wavelengths cannot merge
    and the MTZ is meaningless. Look at its dynamic range first.
  - two-stage refine (mono is single-stage). Defaults (model, eff templates, R-free,
    env) point at the cluster reference_data paths; override with
    --pdb/--ref-pdb/--eff1/--eff2/--rfree/--phenix-env/--elements.

Usage:
    python scripts/diagnose_poly_merging.py RUN_DIR [--skip-phenix]

Outputs land in RUN_DIR/diagnostics/:
    merged.mtz
    scale_spectrum.csv
    refine1/{refine.eff, phenix.log, ...}, refine2/{...}     (if phenix runs)
    anom_peaks.csv          (S/I anomalous map heights, ranked)
    summary.txt
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import subprocess
import time
from pathlib import Path

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch
import yaml

# gemmi 0.7.x compat (some rs internals expect these)
if not hasattr(gemmi.UnitCell, "fractionalization_matrix"):
    gemmi.UnitCell.fractionalization_matrix = property(
        lambda self: self.frac.mat
    )
if not hasattr(gemmi.UnitCell, "orthogonalization_matrix"):
    gemmi.UnitCell.orthogonalization_matrix = property(
        lambda self: self.orth.mat
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ====================================================================
# Rundir / checkpoint discovery
# ====================================================================


def load_run_metadata(run_dir: Path) -> tuple[dict, dict]:
    """Read run_paths.yaml and the run's config_log.yaml."""
    meta_path = run_dir / "run_paths.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_paths.yaml not found in {run_dir}")
    meta = yaml.safe_load(meta_path.read_text())
    cfg_path = Path(meta["config"])
    if not cfg_path.exists():
        raise FileNotFoundError(f"config_log.yaml not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    return cfg, meta


def find_last_checkpoint(meta: dict) -> Path:
    """Locate last.ckpt (or the latest epoch=*.ckpt) for this run."""
    log_dir = Path(meta["wandb"]["log_dir"])
    ckpt_dir = log_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoints dir not found: {ckpt_dir}")
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last.resolve()
    epoch_ckpts = sorted(ckpt_dir.glob("epoch=*.ckpt")) or sorted(
        ckpt_dir.glob("*.ckpt")
    )
    if not epoch_ckpts:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir}")
    return epoch_ckpts[-1].resolve()


# ====================================================================
# Model loading + posterior extraction
# ====================================================================


def load_integrator(cfg: dict, checkpoint_path: Path):
    from integrator.utils.factory_utils import construct_integrator

    # skip_warmstart: the checkpoint supplies the trained profile basis.
    model = construct_integrator(cfg, skip_warmstart=True)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state = ckpt["state_dict"]
    model_state = model.state_dict()
    compat = {
        k: v
        for k, v in state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    missing = [
        k
        for k in model_state
        if k not in compat
        and k not in ("alpha_buffer", "beta_buffer", "buffer_seen")
    ]
    if missing:
        logger.warning(
            "load_state_dict: %d keys missing from checkpoint, kept model init",
            len(missing),
        )
    model.load_state_dict(compat, strict=False)
    model.eval()
    return model


def extract_merged_posterior(
    integrator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (alpha, beta, seen) for the per-HKL Gamma posterior on I_h."""
    name = type(integrator).__name__
    if not hasattr(integrator, "get_merged_qi"):
        raise NotImplementedError(f"{name} has no get_merged_qi().")
    with torch.no_grad():
        q = integrator.get_merged_qi()
    alpha = q.concentration.detach().cpu().numpy().astype(np.float64)
    beta = q.rate.detach().cpu().numpy().astype(np.float64)

    seen = None
    for _buf in ("buffer_seen", "feat_seen"):
        if hasattr(integrator, _buf):
            seen = getattr(integrator, _buf).detach().cpu().numpy().astype(bool)
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


# ====================================================================
# Poly-specific: learned scale vs wavelength (effective spectrum G(lambda))
# ====================================================================


def report_scale_spectrum(
    integrator, data_dir: Path, cfg: dict, out_csv: Path, n_grid: int = 60
) -> Path | None:
    """Report the LaueMLPScale's learned effective spectrum G(lambda).

    Evaluates the scale MLP over a wavelength grid at the dataset's MEDIAN geometry
    (x, y, d), with the per-image log-scale term EXCLUDED (it is additive in
    log-space and wavelength-independent). The result is the learned wavelength
    dependence of the scale up to a constant -- the incident spectrum the merge
    divides out. A flat curve (dynamic range ~1x) means the scale never learned the
    spectrum and the MTZ is meaningless.
    """
    from integrator.model.scaling.chebyshev_scale import LaueMLPScale

    sf = integrator.scale_fn
    if not isinstance(sf, LaueMLPScale):
        logger.info(
            "Scale is %s, not LaueMLPScale -- skipping spectrum diagnostic.",
            type(sf).__name__,
        )
        return None

    ref_name = cfg["data_loader"]["args"]["shoebox_file_names"]["reference"]
    md = torch.load(data_dir / ref_name, weights_only=False, map_location="cpu")
    for k in ("wavelength", "xyzcal.px.0", "xyzcal.px.1", "d"):
        if k not in md:
            logger.warning("metadata missing %r -- skipping spectrum.", k)
            return None

    x = md["xyzcal.px.0"].float().median()
    y = md["xyzcal.px.1"].float().median()
    d = md["d"].float().median()
    lam = md["wavelength"].float()
    lam_min, lam_max = float(lam.min()), float(lam.max())

    device = next(integrator.parameters()).device
    grid = torch.linspace(lam_min, lam_max, n_grid)
    with torch.no_grad():
        feats = torch.stack(
            [
                (grid - sf.lam_mid.cpu()) / sf.lam_half.cpu(),
                ((x - sf.beam_cx.cpu()) / sf.r_max.cpu()).expand(n_grid),
                ((y - sf.beam_cy.cpu()) / sf.r_max.cpu()).expand(n_grid),
                ((d - sf.d_min.cpu()) / (sf.d_max.cpu() - sf.d_min.cpu())).expand(
                    n_grid
                ),
            ],
            dim=-1,
        ).to(device)
        log_g = sf.net(feats).squeeze(-1).clamp(-15.0, 15.0).cpu()
    g = torch.exp(log_g)
    g_rel = (g / g[0].clamp(min=1e-9)).numpy()
    grid_np = grid.numpy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wavelength", "G_rel"])
        for wl, gv in zip(grid_np, g_rel, strict=True):
            w.writerow([float(wl), float(gv)])

    dyn = float(g_rel.max() / max(g_rel.min(), 1e-9))
    logger.info("Learned scale vs wavelength (median geometry, no per-image term):")
    logger.info("  lambda range  = %.3f - %.3f A", lam_min, lam_max)
    logger.info(
        "  G(lambda) dynamic range = %.2fx (max/min) -- ~1x means the scale "
        "did NOT learn the spectrum",
        dyn,
    )
    logger.info("  wrote %s", out_csv)
    return out_csv


# ====================================================================
# Sigma inflation (poly: intensity.sum.* with auto-fallback to prf)
# ====================================================================


def compute_sigma_inflation_per_hkl(
    metadata_path: Path,
    alpha: np.ndarray,
    beta: np.ndarray,
    seen: np.ndarray,
    n_hkl: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-HKL inflation = observed per-obs CV / model CV (1/sqrt(alpha)).

    Uses the summed integrated intensity for Laue (`intensity.sum.value`), falling
    back to the profile-fit value if present.
    """
    logger.info("Loading metadata for sigma inflation: %s", metadata_path)
    metadata = torch.load(metadata_path, weights_only=False, map_location="cpu")

    val_key = (
        "intensity.prf.value"
        if "intensity.prf.value" in metadata
        else "intensity.sum.value"
    )
    var_key = (
        "intensity.prf.variance"
        if "intensity.prf.variance" in metadata
        else "intensity.sum.variance"
    )
    for key in ("asu_id", val_key):
        if key not in metadata:
            raise KeyError(
                f"metadata.pt missing '{key}' -- cannot compute inflation."
            )

    asu_ids = metadata["asu_id"].long().numpy()
    I_obs = metadata[val_key].float().numpy()
    var_obs = (
        metadata[var_key].float().numpy()
        if var_key in metadata
        else np.ones_like(I_obs)
    )
    logger.info("  using observed intensity key: %s", val_key)

    good = (
        (var_obs > 0) & np.isfinite(I_obs) & np.isfinite(var_obs) & (I_obs > 0)
    )
    asu_ids = asu_ids[good]
    I_obs = I_obs[good]

    observed_cv = np.full(n_hkl, np.nan, dtype=np.float64)
    n_obs = np.zeros(n_hkl, dtype=np.int64)

    sort_idx = np.argsort(asu_ids, kind="stable")
    sorted_ids = asu_ids[sort_idx]
    sorted_I = I_obs[sort_idx]
    change = np.concatenate([[True], sorted_ids[1:] != sorted_ids[:-1]])
    change_idx = np.flatnonzero(change)
    ends = np.concatenate([change_idx[1:], [len(sorted_ids)]])
    for start, end in zip(change_idx, ends, strict=False):
        n = end - start
        h = int(sorted_ids[start])
        n_obs[h] = n
        if n < 2:
            continue
        I_group = sorted_I[start:end]
        mean_I = I_group.mean()
        if mean_I > 0:
            observed_cv[h] = I_group.std(ddof=1) / mean_I

    model_cv = np.where(
        seen & (alpha > 0),
        1.0 / np.sqrt(np.clip(alpha, 1e-12, None)),
        np.nan,
    )
    ratio = observed_cv / np.clip(model_cv, 1e-12, None)
    inflation = np.where(np.isfinite(ratio) & (ratio > 1.0), ratio, 1.0)

    finite = np.isfinite(observed_cv) & np.isfinite(model_cv)
    if finite.any():
        logger.info(
            "Inflation over %d HKLs with >=2 obs: median observed CV=%.3f, "
            "model CV=%.4f, median inflation=%.2f, p90=%.2f",
            int(finite.sum()),
            float(np.median(observed_cv[finite])),
            float(np.median(model_cv[finite])),
            float(np.median(inflation[finite])),
            float(np.percentile(inflation[finite], 90)),
        )
    return inflation, n_obs


def apply_chi2_inflation(
    ds: rs.DataSet, chi_sq: np.ndarray, hkl_table: np.ndarray
) -> rs.DataSet:
    """Multiply (SIGF, SIGI) per Friedel mate by sqrt(max(1, chi^2)) per asu_id."""
    canon_to_asu = {
        tuple(hkl_table[i].tolist()): i for i in range(len(hkl_table))
    }
    infl_plus = np.ones(len(ds), dtype=np.float64)
    infl_minus = np.ones(len(ds), dtype=np.float64)
    for row_idx, (h, k, l) in enumerate(ds.index):
        a_plus = canon_to_asu.get((int(h), int(k), int(l)))
        a_minus = canon_to_asu.get((int(-h), int(-k), int(-l)))
        if a_plus is not None and np.isfinite(chi_sq[a_plus]):
            infl_plus[row_idx] = np.sqrt(max(1.0, chi_sq[a_plus]))
        if a_minus is not None and np.isfinite(chi_sq[a_minus]):
            infl_minus[row_idx] = np.sqrt(max(1.0, chi_sq[a_minus]))
    for col, infl in (
        ("SIGI(+)", infl_plus),
        ("SIGF(+)", infl_plus),
        ("SIGI(-)", infl_minus),
        ("SIGF(-)", infl_minus),
    ):
        if col in ds.columns:
            ds[col] = ds[col] * infl
    return ds


# ====================================================================
# MTZ assembly
# ====================================================================


def load_crystal(data_dir: Path) -> tuple[gemmi.UnitCell, gemmi.SpaceGroup]:
    crystal_yaml = data_dir / "crystal.yaml"
    if not crystal_yaml.exists():
        raise FileNotFoundError(f"crystal.yaml not found: {crystal_yaml}")
    crystal = yaml.safe_load(crystal_yaml.read_text())
    cell = gemmi.UnitCell(*crystal["cell"])
    sg = gemmi.SpaceGroup(
        crystal.get("space_group", crystal.get("space_group_symbol", "P1"))
        .split("(")[0]
        .strip()
    )
    return cell, sg


def load_hkl_table(data_dir: Path) -> np.ndarray:
    p = data_dir / "asu_id_to_hkl.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run prepare_asu_ids.py for this dataset."
        )
    return (
        torch.load(p, weights_only=False, map_location="cpu")
        .numpy()
        .astype(np.int32)
    )


def write_merged_mtz(
    alpha: np.ndarray,
    beta: np.ndarray,
    seen: np.ndarray,
    hkl: np.ndarray,
    cell: gemmi.UnitCell,
    sg: gemmi.SpaceGroup,
    out_path: Path,
) -> rs.DataSet:
    """Write an anomalous merged MTZ with intensity and amplitude columns.

    q(I_h) = Gamma(alpha, beta): E[I]=alpha/beta, Var[I]=alpha/beta^2.
    Amplitude via the delta method: E[F]~sqrt(E[I]), SIGF~SIGI/(2 E[F]).
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
        "F(+)", "SIGF(+)", "F(-)", "SIGF(-)",
        "I(+)", "SIGI(+)", "I(-)", "SIGI(-)",
    ]
    ordered = [c for c in anom_order if c in ds.columns] + [
        c for c in ds.columns if c not in anom_order
    ]
    ds = ds[ordered]

    drop_check = [
        c for c in ("F(+)", "F(-)", "I(+)", "I(-)") if c in ds.columns
    ]
    if drop_check:
        before = len(ds)
        ds = ds.dropna(subset=drop_check, how="all")
        if before - len(ds):
            logger.info(
                "Dropped %d all-NaN rows (absences / unobserved HKLs)",
                before - len(ds),
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.write_mtz(str(out_path), skip_problem_mtztypes=True)
    logger.info(
        "Wrote %s: %d reflections (anomalous), cell=%s, sg=%s",
        out_path, len(ds), cell, sg.hm,
    )

    if {"F(+)", "F(-)"}.issubset(ds.columns):
        pair = ds[["F(+)", "F(-)"]].dropna()
        if len(pair) > 100:
            fp = pair["F(+)"].to_numpy()
            fm = pair["F(-)"].to_numpy()
            corr = float(np.corrcoef(fp, fm)[0, 1])
            rel_diff = np.abs(fp - fm) / np.clip(0.5 * (fp + fm), 1e-12, None)
            logger.info("Friedel-pair sanity:")
            logger.info("  pearson(F(+), F(-)) = %.4f   (good: > 0.99)", corr)
            logger.info(
                "  median |dF|/F        = %.3f   (good: 0.01-0.03 for S-anom)",
                float(np.median(rel_diff)),
            )
            logger.info("  N Friedel pairs      = %d", len(pair))
    return ds


# ====================================================================
# Phenix two-stage refinement + anomalous peaks (refltorch Laue recipe,
# phenix 2.0 command syntax)
# ====================================================================


# Defaults. The two-stage effs are phenix 2.0 (data_manager) templates with
# $MTZFILE/$MODELFILE/$RFREE placeholders -- the poly copies of the mono eff.
PHENIX_ENV = "/n/lab_storage/hekstra_lab/garden_2/phenix-2.0-5729/phenix_env.sh"
_REF = "/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/reference_data"
REFINE_PDB = f"{_REF}/shaken_9b7c.pdb"          # starting model (stage 1 input)
SITES_PDB = f"{_REF}/9b7c.pdb"                   # true model: anomalous map sites
RFREE_MTZ = f"{_REF}/rfree.mtz"                  # fixed R-free set (as the mono eff)
EFF1 = f"{_REF}/custom_refinement_param_1_poly.eff"
EFF2 = f"{_REF}/custom_refinement_param_2_poly.eff"
ANOM_ELEMENTS = "[S,I]"


def _render_eff(template: Path, out: Path, mtz: Path, model: Path, rfree: Path):
    """Substitute $MTZFILE/$MODELFILE/$RFREE in a data_manager eff template."""
    text = (
        template.read_text()
        .replace("$MTZFILE", str(mtz))
        .replace("$MODELFILE", str(model))
        .replace("$RFREE", str(rfree))
    )
    out.write_text(text)


def _run_phenix(eff: Path, work_dir: Path, phenix_env: Path) -> bool:
    """Run phenix.refine on a rendered eff inside work_dir. Logs to phenix.log."""
    cmd = (
        f'source "{phenix_env}" && cd "{work_dir}" && '
        f'phenix.refine "{eff}" overwrite=true'
    )
    proc = subprocess.run(
        cmd, shell=True, executable="/bin/bash",
        capture_output=True, text=True, timeout=7200,
    )
    (work_dir / "phenix.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        logger.error(
            "phenix.refine failed in %s (exit %d) -- last 1500 chars:\n%s",
            work_dir.name, proc.returncode,
            (proc.stdout + proc.stderr)[-1500:],
        )
    return proc.returncode == 0


def run_two_stage_refine(
    mtz: Path,
    model: Path,
    eff1: Path,
    eff2: Path,
    rfree: Path,
    phenix_env: Path,
    refine_dir: Path,
) -> Path | None:
    """Two-stage phenix.refine on the merged MTZ (phenix 2.0 data_manager effs).

    Both stages read the SAME data + R-free; only the MODEL is handed forward
    (stage 1's refined PDB -> stage 2's input), so there's no MTZ/column-name
    handoff. Each stage's eff is rendered from its template by substituting
    $MTZFILE/$MODELFILE/$RFREE (all absolutized). Stage 1 is rigid body; stage 2
    is the full sites+ADP refinement that emits the anomalous map. Returns the
    refine2 dir on success, else None.
    """
    refine1 = refine_dir / "refine1"
    refine2 = refine_dir / "refine2"
    refine1.mkdir(parents=True, exist_ok=True)
    refine2.mkdir(parents=True, exist_ok=True)
    mtz, model, rfree = mtz.resolve(), model.resolve(), rfree.resolve()
    eff1, eff2, phenix_env = eff1.resolve(), eff2.resolve(), phenix_env.resolve()

    eff1_r = refine1 / "refine.eff"
    _render_eff(eff1, eff1_r, mtz, model, rfree)
    logger.info("Stage 1 (rigid body) ...")
    if not _run_phenix(eff1_r, refine1, phenix_env):
        return None
    pdbs = sorted(refine1.glob("*.pdb"), key=lambda p: p.stat().st_mtime)
    if not pdbs:
        logger.error("stage 1 produced no .pdb in %s", refine1)
        return None
    stage1_model = pdbs[-1]
    logger.info("stage 1 -> %s", stage1_model.name)

    eff2_r = refine2 / "refine.eff"
    _render_eff(eff2, eff2_r, mtz, stage1_model, rfree)
    logger.info("Stage 2 (sites + ADP) ...")
    if not _run_phenix(eff2_r, refine2, phenix_env):
        return None
    return refine2


def find_anom_map_mtz(refine2_dir: Path) -> tuple[Path, str] | None:
    """The refine2 MTZ with the anomalous-difference map, plus its phase label.

    Phenix 2.0 emits ANOM + PANOM by default; an explicit eff may use PHANOM.
    Scans by column label (filename serial varies). Returns (mtz, phase_label).
    """
    for m in sorted(refine2_dir.glob("*.mtz")):
        try:
            labels = gemmi.read_mtz_file(str(m)).column_labels()
        except Exception:
            continue
        if "ANOM" in labels:
            for ph in ("PANOM", "PHANOM"):
                if ph in labels:
                    return m, ph
    return None


def get_anom_peak_heights(
    mtz_filename: str,
    pdb_filename: str,
    atom_list: str,
    phase_label: str = "PANOM",
) -> tuple[list[str], list[float]]:
    """Anomalous-difference map height (sigma) at each scatterer site.

    Ported from refltorch/scripts/laue_output/anomalous_peak_heights.py: builds the
    ANOM/<phase> map, normalizes, and reads the value at each selected atom
    (averaged over symmetry equivalents).
    """
    mtz_file = gemmi.read_mtz_file(mtz_filename)
    st = gemmi.read_pdb(pdb_filename)
    real_grid = mtz_file.transform_f_phi_to_map(
        "ANOM", phase_label, sample_rate=3.0
    )
    real_grid.normalize()
    sel = gemmi.Selection(f"{atom_list}")
    anom_atoms = list(sel.copy_model_selection(st[0]).all())

    anom_res, anom_peaks = [], []
    for cra in anom_atoms:
        ops = real_grid.spacegroup.operations()
        atom = cra.atom
        eq_points = []
        for op in ops:
            sg_mapped = op.apply_to_xyz(st.cell.fractionalize(atom.pos).tolist())
            tmp = sg_mapped - np.floor(np.array(sg_mapped))
            eq_points.append(gemmi.Fractional(*tmp))
        peak_value = [
            real_grid.get_value(
                round(p.x * real_grid.nu),
                round(p.y * real_grid.nv),
                round(p.z * real_grid.nw),
            )
            for p in eq_points
        ]
        anom_res.append(f"{cra.residue.name} {cra.residue.seqid.num}")
        anom_peaks.append(round(float(np.average(peak_value)), 3))
    return anom_res, anom_peaks


def parse_final_r(refine_dir: Path) -> dict[str, float]:
    """Final R-work/R-free from a stage's phenix.log (last 'Final R-work')."""
    log = refine_dir / "phenix.log"
    if not log.exists():
        return {}
    for line in reversed(log.read_text().splitlines()):
        if "Final R-work" in line:
            nums = re.findall(r"\d+\.\d+", line)
            if len(nums) >= 2:
                return {
                    "r_work_final": float(nums[0]),
                    "r_free_final": float(nums[1]),
                }
    return {}


# ====================================================================
# Main
# ====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose a polychromatic (Laue) merging checkpoint."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--skip-phenix", action="store_true",
        help="Write MTZ + diagnostics but don't run phenix.refine.",
    )
    parser.add_argument(
        "--chi2-inflation", action="store_true",
        help="Inflate SIGI/SIGF by per-HKL observed vs model CV (off by default).",
    )
    parser.add_argument(
        "--no-finalize-merge", action="store_true",
        help="Skip the converged-encoder merge pass. The amortized model's "
        "alpha/beta buffers are non-persistent, so skipping yields the Wilson "
        "prior (garbage) -- only for checkpoints with persisted buffers.",
    )
    parser.add_argument(
        "--cell", type=str, default=None,
        help="override the merged-MTZ unit cell, 'a,b,c,al,be,ga' (default: "
        "from crystal.yaml). Use when crystal.yaml's cell disagrees with the "
        "refinement model/eff -- the data, model, and eff must share a cell or "
        "the anomalous map won't align with the atom sites.",
    )
    # Phenix recipe (refltorch Laue defaults; phenix 2.0). Override per machine.
    parser.add_argument(
        "--pdb", type=str, default=REFINE_PDB,
        help="model to refine (the shaken reference).",
    )
    parser.add_argument(
        "--ref-pdb", type=str, default=SITES_PDB,
        help="model whose S/I sites the anomalous map is read at (true model).",
    )
    parser.add_argument("--eff1", type=str, default=EFF1)
    parser.add_argument("--eff2", type=str, default=EFF2)
    parser.add_argument(
        "--rfree", type=str, default=RFREE_MTZ,
        help="fixed R-free flags MTZ (same set as the mono eff).",
    )
    parser.add_argument("--phenix-env", type=str, default=PHENIX_ENV)
    parser.add_argument(
        "--elements", type=str, default=ANOM_ELEMENTS,
        help="gemmi element selection for the anomalous sites (default [S,I]).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run dir: %s", run_dir)
    cfg, meta = load_run_metadata(run_dir)
    ckpt = args.checkpoint or find_last_checkpoint(meta)
    logger.info("Checkpoint: %s", ckpt)

    integrator = load_integrator(cfg, ckpt)
    logger.info("Loaded %s", type(integrator).__name__)

    # Repopulate the per-HKL posterior over the full dataset with the converged
    # encoder. The amortized model's merge buffers are NON-persistent, so this is
    # REQUIRED; it needs complete HKL groups per batch, hence the grouped poly
    # predict loader (group_by_asu_id must be set in the config).
    if hasattr(integrator, "finalize_merge") and not args.no_finalize_merge:
        from integrator.utils import construct_data_loader

        if torch.cuda.is_available():
            integrator.to(torch.device("cuda"))
        logger.info("Finalizing merge over the dataset (converged encoder)")
        data_loader = construct_data_loader(cfg)
        data_loader.setup()
        try:
            finalize_loader = data_loader.predict_dataloader(grouped=True)
        except TypeError:
            finalize_loader = data_loader.predict_dataloader()
        t0 = time.time()
        integrator.finalize_merge(finalize_loader)
        logger.info("finalize_merge done in %.1f s", time.time() - t0)
    else:
        logger.warning(
            "finalize_merge NOT run -- merge buffers are non-persistent, so the "
            "MTZ may reflect the Wilson prior."
        )

    alpha, beta, seen = extract_merged_posterior(integrator)

    # Guard: if q(I_h) is still the Wilson prior everywhere seen, the merge
    # didn't populate (wrong loader / stale code) -> the MTZ is invalid.
    if seen.any():
        a_W = float(getattr(integrator, "alpha_W", 1.0))
        at_prior = (
            np.isclose(alpha[seen], a_W, atol=1e-4)
            & np.isclose(beta[seen], 1.0, atol=1e-4)
        ).mean()
        if at_prior > 0.95:
            logger.error(
                "q(I_h) is the Wilson prior for %.0f%% of seen HKLs -- "
                "finalize_merge did NOT populate the buffers; the MTZ is INVALID.",
                100 * at_prior,
            )

    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])
    cell, sg = load_crystal(data_dir)
    if args.cell:
        vals = [float(x) for x in args.cell.replace(" ", "").split(",")]
        cell = gemmi.UnitCell(*vals)
        logger.info("Overriding merged-MTZ cell -> %s", cell)
    hkl = load_hkl_table(data_dir)

    # Poly-specific: did the scale learn the incident spectrum G(lambda)?
    spectrum_csv = report_scale_spectrum(
        integrator, data_dir, cfg, diag_dir / "scale_spectrum.csv"
    )

    mtz_path = diag_dir / "merged.mtz"
    ds_raw = write_merged_mtz(alpha, beta, seen, hkl, cell, sg, mtz_path)

    if args.chi2_inflation:
        try:
            metadata_path = (
                data_dir
                / cfg["data_loader"]["args"]["shoebox_file_names"]["reference"]
            )
            inflation, _ = compute_sigma_inflation_per_hkl(
                metadata_path, alpha, beta, seen, len(hkl)
            )
            ds_inflated = apply_chi2_inflation(ds_raw.copy(), inflation**2, hkl)
            mtz_path_inflated = diag_dir / "merged_inflated.mtz"
            ds_inflated.write_mtz(
                str(mtz_path_inflated), skip_problem_mtztypes=True
            )
            logger.info("Wrote %s (inflated sigmas)", mtz_path_inflated)
            mtz_path = mtz_path_inflated
        except (KeyError, FileNotFoundError) as e:
            logger.warning("Skipping chi2 inflation: %s", e)

    summary_lines = [
        f"Poly merging diagnostic for {run_dir.name}",
        f"Checkpoint: {ckpt.name}",
        f"Merged MTZ: {mtz_path}",
        f"HKLs in MTZ: {seen.sum()} / {len(alpha)} ({100 * seen.mean():.1f}%)",
        f"Scale spectrum: {spectrum_csv}" if spectrum_csv else "Scale spectrum: (skipped)",
        "",
    ]

    def _finish():
        summary = "\n".join(summary_lines)
        (diag_dir / "summary.txt").write_text(summary)
        print("\n" + "=" * 60 + "\n" + summary + "\n" + "=" * 60)

    # Phenix: two-stage refine on the merged MTZ (phenix 2.0 data_manager effs),
    # then anomalous peak heights at the known S/I sites. Skipped (MTZ +
    # diagnostics still written) on --skip-phenix or any missing input path.
    needed = {
        "refine model": args.pdb,
        "sites PDB": args.ref_pdb,
        "eff1": args.eff1,
        "eff2": args.eff2,
        "rfree MTZ": args.rfree,
        "phenix env": args.phenix_env,
    }
    missing = {k: v for k, v in needed.items() if not Path(v).exists()}
    if args.skip_phenix or missing:
        reason = (
            "--skip-phenix"
            if args.skip_phenix
            else "missing " + ", ".join(f"{k} ({v})" for k, v in missing.items())
        )
        logger.warning("Phenix skipped (%s). MTZ + diagnostics written.", reason)
        summary_lines.append(f"[phenix skipped: {reason}]")
        _finish()
        return

    refine2 = run_two_stage_refine(
        mtz_path, Path(args.pdb), Path(args.eff1), Path(args.eff2),
        Path(args.rfree), Path(args.phenix_env), diag_dir,
    )
    if refine2 is None:
        summary_lines.append("[phenix] two-stage refine FAILED (see refineN/phenix.log)")
        _finish()
        return

    r = parse_final_r(refine2)
    summary_lines.append(
        f"[refine] Rwork={r.get('r_work_final', '?')} "
        f"Rfree={r.get('r_free_final', '?')}"
    )

    found = find_anom_map_mtz(refine2)
    if found is None:
        logger.error(
            "No ANOM/<phase> map MTZ in %s -- check `ls %s/*.mtz` and its columns.",
            refine2, refine2,
        )
        summary_lines.append("[peaks] no ANOM map MTZ found in refine2/")
        _finish()
        return

    map_mtz, phase_label = found
    res, heights = get_anom_peak_heights(
        str(map_mtz), str(args.ref_pdb), args.elements, phase_label
    )
    ranked = sorted(zip(res, heights, strict=True), key=lambda t: -t[1])
    peaks_csv = diag_dir / "anom_peaks.csv"
    with open(peaks_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["residue", "peak_sigma"])
        w.writerows(ranked)
    logger.info(
        "Anomalous peak heights (ANOM/%s in %s, sites=%s) -> %s",
        phase_label, map_mtz.name, args.elements, peaks_csv,
    )
    for rr, hh in ranked[:12]:
        logger.info("  %-14s %6.2f sigma", rr, hh)
    summary_lines.append(
        f"[peaks] {args.elements} from {map_mtz.name} (wrote {peaks_csv.name}):"
    )
    summary_lines += [f"    {rr:<14} {hh:6.2f} sigma" for rr, hh in ranked[:12]]
    _finish()


if __name__ == "__main__":
    main()
