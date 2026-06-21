"""Diagnose a merging checkpoint: extract MTZ + run phenix.refine variants.

Loads the last checkpoint from a training run, reads the per-HKL Gamma
posterior `q(I_h)` from the model's buffer (`get_merged_qi()` after a
`finalize_merge` pass), writes an anomalous MTZ with intensity + amplitude
columns, and runs phenix.refine in two configurations:

    F_noFW : F(+),SIGF(+),... as amplitudes      (phenix FW off)
    I_FW   : I(+),SIGI(+),... as intensities     (french_wilson_scale=True)

i.e. the model's amplitudes directly vs the model's intensities with phenix's
French-Wilson (I -> F). For each variant, runs `rs.find_peaks` for anomalous
peaks. For the `AmortizedMergingIntegrator`. The crystal and the merge-id ->
canonical-HKL table are read from <data_dir>/dataset.yaml + the metadata
(make_shoeboxes output); no crystal.yaml / asu_id_to_hkl.pt needed.

Usage:
    uv run python scripts/diagnose_merging.py RUN_DIR \
        --eff-template /path/to/phenix.eff
    # or --skip-phenix to write only merged.mtz

Outputs land in RUN_DIR/diagnostics/:
    merged.mtz
    F_noFW/{phenix.eff, refine_*, peaks.csv}
    I_FW/{phenix.eff,   refine_*, peaks.csv}
    summary.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch
import yaml

# gemmi 0.7.x compat (SFcalculator and some rs internals expect these)
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


def checkpoint_dir(meta: dict) -> Path:
    """Resolve the run's checkpoint directory from run_paths.yaml.

    Checkpoints live under `<log_dir>/checkpoints`, where `log_dir` is the W&B
    files dir (`wandb.log_dir`) for W&B runs, else the top-level `log_dir`.
    """
    log_dir = (meta.get("wandb") or {}).get("log_dir") or meta.get("log_dir")
    if log_dir is None:
        raise KeyError("run_paths.yaml has neither wandb.log_dir nor log_dir")
    ckpt_dir = Path(log_dir) / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoints dir not found: {ckpt_dir}")
    return ckpt_dir


def _epoch_of(ckpt: Path) -> int:
    """Epoch number parsed from a Lightning `epoch=NNNN.ckpt` name (-1 if none)."""
    m = re.search(r"epoch=(\d+)", ckpt.name)
    return int(m.group(1)) if m else -1


def discover_checkpoints(meta: dict) -> list[Path]:
    """All per-epoch checkpoints for a run, sorted by epoch (ascending).

    Excludes the `last.ckpt` symlink (it duplicates the newest epoch ckpt).
    """
    ckpt_dir = checkpoint_dir(meta)
    ckpts = [
        p for p in ckpt_dir.glob("epoch=*.ckpt") if not p.is_symlink()
    ]
    if not ckpts:
        raise FileNotFoundError(f"no epoch=*.ckpt in {ckpt_dir}")
    return sorted(ckpts, key=_epoch_of)


def find_last_checkpoint(meta: dict) -> Path:
    """Locate the last.ckpt for this run.

    Lightning saves it in `<log_dir>/checkpoints/last.ckpt` (or as a symlink).
    Fallback: the most recent `epoch=*.ckpt` in that dir.
    """
    ckpt_dir = checkpoint_dir(meta)
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last.resolve()
    epoch_ckpts = sorted(ckpt_dir.glob("epoch=*.ckpt"), key=_epoch_of)
    if not epoch_ckpts:
        epoch_ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not epoch_ckpts:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir}")
    return epoch_ckpts[-1].resolve()


def finalize_merge_over_dataset(integrator, cfg: dict) -> None:
    """Recompute the per-HKL merged posterior over the full dataset.

    The merge buffers are non-persistent (absent from the checkpoint), so this
    pass is REQUIRED before `get_merged_qi()` returns anything but the Wilson
    prior. Needs complete HKL groups per batch, so it uses the grouped predict
    loader (`group_by_asu_id`). No-op if the integrator has no `finalize_merge`.
    """
    if not hasattr(integrator, "finalize_merge"):
        logger.warning(
            "%s has no finalize_merge; merge buffers may be the Wilson prior.",
            type(integrator).__name__,
        )
        return
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
    logger.info("finalize_merge completed in %.1f s", time.time() - t0)


# ====================================================================
# Model loading + posterior extraction
# ====================================================================


def load_integrator(cfg: dict, checkpoint_path: Path):
    from integrator.utils.factory_utils import construct_integrator

    model = construct_integrator(cfg)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state = ckpt["state_dict"]
    model_state = model.state_dict()
    # Tolerate shape mismatches (e.g. profile basis warm-start sizes).
    compat = {
        k: v
        for k, v in state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    # The merge alpha/beta buffers are non-persistent (not in the checkpoint) and
    # are repopulated by finalize_merge below, so they're expected to be missing.
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
    """Return (alpha, beta, seen) for the per-HKL Gamma posterior on I_h.

    Works for any integrator exposing `get_merged_qi()` returning a Gamma.
    `seen` is a bool mask of HKLs that received at least one observation,
    read from `buffer_seen` (ConjugateMerging / Amortized - populated by
    finalize_merge) or `feat_seen` (DeepSets EMA features); True everywhere
    if neither buffer exists.
    """
    name = type(integrator).__name__
    if not hasattr(integrator, "get_merged_qi"):
        raise NotImplementedError(
            f"{name} has no get_merged_qi(); diagnostic supports "
            "ConjugateMergingIntegrator and DeepSetsMergingIntegrator"
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


# ====================================================================
# MTZ assembly
# ====================================================================


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
        sg = gemmi.SpaceGroup(str(crystal["space_group"]).split("(")[0].strip())
    return cell, sg


def load_hkl_table(
    data_dir: Path, cfg: dict, cell: gemmi.UnitCell, sg: gemmi.SpaceGroup
) -> np.ndarray:
    """Merge-id -> canonical-(h,k,l) table, derived from the metadata.

    Reuses `io.miller_index_columns` (the exact function make_shoeboxes used to
    assign the ids), so `table[i]` is the canonical HKL of merge id `i` in the
    model's buffer. `anomalous` (from the config) selects the unfriedelized table.
    """
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


def compute_sigma_inflation_per_hkl(
    metadata_path: Path,
    alpha: np.ndarray,
    beta: np.ndarray,
    seen: np.ndarray,
    n_hkl: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-HKL inflation factor: observed scatter / model-predicted scatter.

    Aimless-style χ² against DIALS variances is the *wrong* reference for our
    model - DIALS sigmas often have their own calibration and may be loose
    relative to the per-obs scatter, but our conjugate model's per-HKL CV
    `1/√α_h` is set by Poisson alone and is consistently too tight. What we
    want is to inflate the model's SIGI so its CV matches the actually
    observed per-obs spread.

    For each HKL h with n_h >= 2 DIALS observations:
        observed_CV_h = std(I_obs_i) / mean(I_obs_i)        (dimensionless)
        model_CV_h    = 1 / √α_h                            (model's CV)
        inflation_h   = max(1, observed_CV_h / model_CV_h)
                      = max(1, observed_CV_h · √α_h)

    Multiplying SIGI by `inflation_h` rescales the model's per-HKL posterior
    CV to match the actually observed per-obs spread. Doesn't change the
    posterior mean.

    Returns:
        inflation: (n_hkl,) inflation factor per asu_id (1.0 for HKLs without
                   enough observations or where model_CV already exceeds
                   observed_CV).
        n_obs:     (n_hkl,) number of observations per asu_id.
    """
    logger.info(
        "Loading metadata for sigma-inflation computation: %s", metadata_path
    )
    from integrator.io import load_metadata

    metadata = load_metadata(metadata_path)

    for key in ("miller_idx_unfriedelized", "intensity.prf.value"):
        if key not in metadata:
            raise KeyError(
                f"metadata missing '{key}' - cannot compute inflation. "
                "Skip without --chi2-inflation."
            )

    asu_ids = metadata["miller_idx_unfriedelized"].long().numpy()
    I_obs = metadata["intensity.prf.value"].float().numpy()
    var_obs = (
        metadata["intensity.prf.variance"].float().numpy()
        if "intensity.prf.variance" in metadata
        else np.ones_like(I_obs)
    )

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
    inflation = np.where(
        np.isfinite(ratio) & (ratio > 1.0),
        ratio,
        1.0,
    )

    finite = np.isfinite(observed_cv) & np.isfinite(model_cv)
    if finite.any():
        obs_cv_med = float(np.median(observed_cv[finite]))
        mod_cv_med = float(np.median(model_cv[finite]))
        logger.info(
            "Inflation stats over %d HKLs with ≥2 obs:", int(finite.sum())
        )
        logger.info(
            "  median observed CV     = %.3f (DIALS per-obs std/mean)",
            obs_cv_med,
        )
        logger.info("  median model CV        = %.4f (1/√α_h)", mod_cv_med)
        logger.info(
            "  median inflation       = %.2f  (obs CV / model CV)",
            float(np.median(inflation[finite])),
        )
        logger.info(
            "  p90    inflation       = %.2f",
            float(np.percentile(inflation[finite], 90)),
        )

    return inflation, n_obs


def apply_chi2_inflation(
    ds: rs.DataSet, chi_sq: np.ndarray, hkl_table: np.ndarray
) -> rs.DataSet:
    """Multiply (SIGF, SIGI) per Friedel mate by √max(1, χ²) per asu_id.

    Looks up the (+) and (-) `asu_id` for each MTZ row from the canonical
    HKL table built by prepare_asu_ids.py:
        asu_id_+ has canonical_hkl == (h, k, l)
        asu_id_- has canonical_hkl == (-h, -k, -l)
    """
    canon_to_asu = {
        tuple(hkl_table[i].tolist()): i for i in range(len(hkl_table))
    }

    inflation_plus = np.ones(len(ds), dtype=np.float64)
    inflation_minus = np.ones(len(ds), dtype=np.float64)
    missing_plus = 0
    missing_minus = 0

    for row_idx, (h, k, l) in enumerate(ds.index):
        a_plus = canon_to_asu.get((int(h), int(k), int(l)))
        a_minus = canon_to_asu.get((int(-h), int(-k), int(-l)))
        if a_plus is not None and np.isfinite(chi_sq[a_plus]):
            inflation_plus[row_idx] = np.sqrt(max(1.0, chi_sq[a_plus]))
        else:
            missing_plus += 1
        if a_minus is not None and np.isfinite(chi_sq[a_minus]):
            inflation_minus[row_idx] = np.sqrt(max(1.0, chi_sq[a_minus]))
        else:
            missing_minus += 1

    if missing_plus or missing_minus:
        logger.info(
            "χ² inflation: %d (+) rows and %d (-) rows had no χ² "
            "(unobserved or <2 obs) - left unchanged",
            missing_plus,
            missing_minus,
        )

    if "SIGI(+)" in ds.columns:
        ds["SIGI(+)"] = ds["SIGI(+)"] * inflation_plus
    if "SIGF(+)" in ds.columns:
        ds["SIGF(+)"] = ds["SIGF(+)"] * inflation_plus
    if "SIGI(-)" in ds.columns:
        ds["SIGI(-)"] = ds["SIGI(-)"] * inflation_minus
    if "SIGF(-)" in ds.columns:
        ds["SIGF(-)"] = ds["SIGF(-)"] * inflation_minus

    return ds


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

    # Drop rows where ALL of F(+), F(-), I(+), I(-) are NaN.
    # These appear when unstack_anomalous creates index slots for HKL pairs
    # whose neither Friedel mate was observed (systematic absences, low-res
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

    # Friedel-pair sanity stats. Real anomalous signal should give:
    #   pearson(F(+), F(-)) > 0.99
    #   median |F(+) - F(-)| / mean(F(+), F(-)) ≈ 1-3% for sulfur K-edge
    # Much larger values mean (+)/(-) estimates are dominated by noise -
    # find_peaks will not resolve sulfur sites.
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


# ====================================================================
# Phenix eff generation
# ====================================================================


# Variants. Each tuple: (variant name, labels, array_type, french_wilson flag).
#   F_noFW: the model's amplitudes (F = sqrt(E[I])), phenix FW off.
#   I_FW:   the model's intensities, phenix does the French-Wilson (I -> F).
VARIANTS = [
    ("F_noFW", "F(+),SIGF(+),F(-),SIGF(-)", "amplitude", False),
    ("I_FW", "I(+),SIGI(+),I(-),SIGI(-)", "intensity", True),
]


def _set_array_type_star(line: str, star_token: str) -> str:
    """Move the `*` in an array_type listing to the requested token."""
    tokens = re.findall(r"\S+", line)
    out_tokens = []
    for t in tokens:
        bare = t.lstrip("*")
        out_tokens.append(f"*{bare}" if bare == star_token else bare)
    leading = line[: len(line) - len(line.lstrip())]
    return leading + " ".join(out_tokens) + "\n"


def render_eff(
    template: str,
    mtz_path: Path,
    labels: str,
    star_token: str,
    fw_scale: bool,
) -> str:
    """Render a phenix.eff from the template with the variant's substitutions.

    Modifies only the *first* miller_array block (the data file). The Rfree
    block is left intact. Also flips `french_wilson_scale` everywhere it
    appears.
    """
    text = template.replace("$MTZFILE", str(mtz_path.resolve()))

    miller_array_count = 0  # increments on each `miller_array {` we enter
    in_data_block = False  # True while inside the first miller_array block
    array_type_pending = False
    out_lines = []

    for line in text.splitlines(keepends=True):
        stripped = line.strip()

        if stripped.startswith("miller_array"):
            miller_array_count += 1
            in_data_block = miller_array_count == 1
            out_lines.append(line)
            continue

        if in_data_block and stripped.startswith("name ="):
            out_lines.append(
                re.sub(r'name = "[^"]*"', f'name = "{labels}"', line)
            )
            array_type_pending = True
            continue

        if in_data_block and array_type_pending:
            # array_type listing may span multiple lines via `\` continuation
            out_lines.append(_set_array_type_star(line, star_token))
            if not line.rstrip().endswith("\\"):
                array_type_pending = False
            continue

        if in_data_block and stripped.startswith("user_selected_labels"):
            out_lines.append(
                re.sub(
                    r'user_selected_labels = "[^"]*"',
                    f'user_selected_labels = "{labels}"',
                    line,
                )
            )
            in_data_block = False
            continue

        if "french_wilson_scale" in line:
            out_lines.append(
                re.sub(
                    r"french_wilson_scale\s*=\s*\w+",
                    f"french_wilson_scale = {fw_scale}",
                    line,
                )
            )
            continue

        out_lines.append(line)

    return "".join(out_lines)


# ====================================================================
# Phenix execution
# ====================================================================


def _phenix_env() -> str | None:
    return os.environ.get("PHENIX_ENV")


def run_phenix_refine(eff_path: Path, work_dir: Path, mtz_path: Path) -> bool:
    env = _phenix_env()
    if env is None:
        logger.warning(
            "PHENIX_ENV not set; skipping phenix.refine for %s", work_dir.name
        )
        return False
    cmd = (
        f"source {env} && cd {work_dir} && "
        f"phenix.refine {eff_path.resolve()} {mtz_path.resolve()} overwrite=true"
    )
    logger.info("Running phenix.refine for %s", work_dir.name)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            check=True,
            timeout=3600,
        )
        (work_dir / "phenix.stdout.log").write_text(proc.stdout)
        (work_dir / "phenix.stderr.log").write_text(proc.stderr)
        return True
    except subprocess.CalledProcessError as e:
        (work_dir / "phenix.stdout.log").write_text(e.stdout or "")
        (work_dir / "phenix.stderr.log").write_text(e.stderr or "")
        logger.error(
            "phenix.refine failed for %s (exit %d) - last 1000 chars stderr:\n%s",
            work_dir.name,
            e.returncode,
            (e.stderr or "")[-1000:],
        )
        return False


def run_find_peaks(work_dir: Path) -> Path | None:
    env = _phenix_env()
    if env is None:
        return None
    cmd = (
        f"source {env} && cd {work_dir} && "
        "rs.find_peaks *[0-9].mtz *[0-9].pdb "
        "-f ANOM -p PANOM -z 5.0 -o peaks.csv"
    )
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        (work_dir / "find_peaks.log").write_text(proc.stdout + proc.stderr)
        return work_dir / "peaks.csv"
    except subprocess.CalledProcessError as e:
        (work_dir / "find_peaks.log").write_text(
            (e.stdout or "") + (e.stderr or "")
        )
        logger.error("rs.find_peaks failed for %s", work_dir.name)
        return None


def parse_phenix_r_factors(work_dir: Path) -> dict[str, float]:
    """Read R-work/R-free from phenix.refine log."""
    log = work_dir / "phenix.stdout.log"
    if not log.exists():
        return {}
    text = log.read_text()
    r = {}
    for key, pat in [
        ("r_work_final", r"Final R-work\s*=\s*([\d.]+)"),
        ("r_free_final", r"Final R-free\s*=\s*([\d.]+)"),
        ("r_work_start", r"Start R-work\s*=\s*([\d.]+)"),
        ("r_free_start", r"Start R-free\s*=\s*([\d.]+)"),
    ]:
        m = re.search(pat, text)
        if m:
            r[key] = float(m.group(1))
    return r


def summarize_peaks(peaks_csv: Path | None, top_n: int = 10) -> str:
    if peaks_csv is None or not peaks_csv.exists():
        return "  (no peaks.csv)"
    try:
        import pandas as pd

        df = pd.read_csv(peaks_csv)
        if df.empty:
            return "  (no peaks)"
        zcol = "peak" if "peak" in df.columns else df.columns[-1]
        top = df.sort_values(zcol, ascending=False).head(top_n)
        lines = [f"  Top {len(top)} peaks (by {zcol}):"]
        for _, row in top.iterrows():
            lines.append(f"    {row.to_dict()}")
        return "\n".join(lines)
    except Exception as e:
        return f"  (parse failed: {e})"


# ====================================================================
# Main
# ====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose a merging checkpoint: extract MTZ + run phenix.refine variants"
    )
    parser.add_argument(
        "run_dir", type=Path, help="Path to training run directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint path (default: last.ckpt in run_dir)",
    )
    parser.add_argument(
        "--skip-phenix",
        action="store_true",
        help="Generate MTZ + eff files but don't run phenix.refine",
    )
    parser.add_argument(
        "--eff-template",
        type=Path,
        default=None,
        help="phenix.eff template path (else output.phenix_eff from the config)",
    )
    parser.add_argument(
        "--chi2-inflation",
        action="store_true",
        help="Inflate SIGI/SIGF using per-HKL observed vs model CV. "
        "Off by default - empirical metric still being calibrated.",
    )
    parser.add_argument(
        "--no-finalize-merge",
        action="store_true",
        help="Skip the converged-encoder merge pass. NOTE: the conjugate and "
        "amortized models keep their alpha/beta buffers non-persistent, so "
        "skipping yields the Wilson prior (garbage) - only meaningful for "
        "legacy checkpoints whose merge buffers were persisted.",
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

    # Recompute the per-HKL merged posterior over the full dataset with the
    # converged encoder. The conjugate and amortized merging models keep their
    # alpha/beta buffers NON-persistent (absent from the checkpoint), so this
    # pass is REQUIRED, not optional - without it get_merged_qi() returns the
    # Wilson prior. For the conjugate model finalize_merge does the two-pass
    # clean merge (per-HKL mean-field EM + calibrated quadrature). It needs
    # complete HKL groups per batch (group_by_asu_id), so it uses the grouped
    # predict loader. DeepSets (persistent feat-EMA buffers, no finalize_merge)
    # skips this block and uses the checkpoint buffers directly.
    if hasattr(integrator, "finalize_merge") and not args.no_finalize_merge:
        from integrator.utils import construct_data_loader

        if torch.cuda.is_available():
            integrator.to(torch.device("cuda"))
        logger.info("Finalizing merge over the dataset (converged encoder)")
        data_loader = construct_data_loader(cfg)
        data_loader.setup()  # builds full_dataset; predict_dataloader needs it
        # Merging models need complete HKL groups per batch; use the grouped
        # predict loader (fall back to the default for older data modules).
        try:
            finalize_loader = data_loader.predict_dataloader(grouped=True)
        except TypeError:
            finalize_loader = data_loader.predict_dataloader()
        t0 = time.time()
        integrator.finalize_merge(finalize_loader)
        logger.info("finalize_merge completed in %.1f s", time.time() - t0)
    else:
        logger.warning(
            "finalize_merge NOT run (no finalize_merge method or "
            "--no-finalize-merge). For conjugate/amortized models the merge "
            "buffers are non-persistent, so the MTZ will reflect the Wilson "
            "prior unless the checkpoint carries persistent buffers."
        )

    alpha, beta, seen = extract_merged_posterior(integrator)

    # Guard against a silent no-op: if the per-HKL posterior is still the Wilson
    # prior (alpha == alpha_W, beta == 1 everywhere it ran), finalize_merge did
    # not actually populate the buffers (stale code / wrong loader / skipped) and
    # the MTZ is meaningless. This is the "blazingly fast = old behavior" tell.
    if seen.any():
        a_W = float(getattr(integrator, "alpha_W", 1.0))
        at_prior = (
            np.isclose(alpha[seen], a_W, atol=1e-4)
            & np.isclose(beta[seen], 1.0, atol=1e-4)
        ).mean()
        if at_prior > 0.95:
            logger.error(
                "q(I_h) is the Wilson prior for %.0f%% of seen HKLs - "
                "finalize_merge did NOT populate the buffers. The deployed code "
                "is likely stale (pre-EMA-removal) or the loader was wrong; "
                "the MTZ is INVALID.",
                100 * at_prior,
            )

    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])
    cell, sg = load_crystal(data_dir)
    hkl = load_hkl_table(data_dir, cfg, cell, sg)

    mtz_path = diag_dir / "merged.mtz"
    ds_raw = write_merged_mtz(alpha, beta, seen, hkl, cell, sg, mtz_path)

    # Optional SIGI/SIGF inflation. Off by default - the metric uses
    # observed CV vs model CV per HKL, but isn't yet calibrated to give
    # find_peaks the right z-scale. Use --chi2-inflation to opt in.
    if args.chi2_inflation:
        try:
            metadata_path = (
                data_dir
                / cfg["data_loader"]["args"]["shoebox_file_names"]["reference"]
            )
            inflation, _ = compute_sigma_inflation_per_hkl(
                metadata_path, alpha, beta, seen, len(hkl)
            )
            ds_inflated = apply_chi2_inflation(
                ds_raw.copy(), inflation**2, hkl
            )
            mtz_path_inflated = diag_dir / "merged_inflated.mtz"
            ds_inflated.write_mtz(
                str(mtz_path_inflated), skip_problem_mtztypes=True
            )
            logger.info(
                "Wrote %s (χ²-inflated sigmas - used for phenix runs)",
                mtz_path_inflated,
            )
            # Phenix runs use the inflated MTZ
            mtz_path = mtz_path_inflated
        except (KeyError, FileNotFoundError) as e:
            logger.warning(
                "Skipping χ² inflation: %s. Phenix runs will use raw MTZ.",
                e,
            )

    # phenix.eff template: --eff-template, else output.phenix_eff from the cfg.
    eff_template_path = args.eff_template or cfg.get("output", {}).get(
        "phenix_eff"
    )
    template = ""
    if eff_template_path:
        eff_template_path = Path(eff_template_path)
        if not eff_template_path.exists():
            logger.error(
                "phenix.eff template not found at %s", eff_template_path
            )
            sys.exit(1)
        template = eff_template_path.read_text()
    elif not args.skip_phenix:
        logger.error(
            "no phenix.eff template: pass --eff-template PATH (or add "
            "output.phenix_eff to the config), or use --skip-phenix to write "
            "only the MTZ."
        )
        sys.exit(1)

    summary_lines = [
        f"Diagnostic summary for {run_dir.name}",
        f"Checkpoint: {ckpt.name}",
        f"Merged MTZ: {mtz_path}",
        f"HKLs in MTZ: {seen.sum()} / {len(alpha)} ({100 * seen.mean():.1f}%)",
        "",
    ]

    for variant, labels, star_token, fw in VARIANTS:
        work_dir = diag_dir / variant
        work_dir.mkdir(exist_ok=True)
        eff_path = work_dir / "phenix.eff"
        if template:
            eff_path.write_text(
                render_eff(template, mtz_path, labels, star_token, fw)
            )
            logger.info("Wrote %s", eff_path)

        if args.skip_phenix or not template:
            summary_lines.append(f"[{variant}] MTZ written (phenix skipped)")
            continue

        ok = run_phenix_refine(eff_path, work_dir, mtz_path)
        if not ok:
            summary_lines.append(f"[{variant}] phenix.refine FAILED")
            continue

        r = parse_phenix_r_factors(work_dir)
        peaks_csv = run_find_peaks(work_dir)
        peaks_text = summarize_peaks(peaks_csv)

        summary_lines.append(
            f"[{variant}] Rwork={r.get('r_work_final', '?')} "
            f"Rfree={r.get('r_free_final', '?')} "
            f"(start Rwork={r.get('r_work_start', '?')})"
        )
        summary_lines.append(peaks_text)
        summary_lines.append("")

    summary = "\n".join(summary_lines)
    (diag_dir / "summary.txt").write_text(summary)
    print("\n" + "=" * 60)
    print(summary)
    print("=" * 60)


if __name__ == "__main__":
    main()
