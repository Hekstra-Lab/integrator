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
import torch
import yaml

from integrator.io import (  # merged-MTZ helpers now live in the package
    extract_merged_posterior,
    load_crystal,
    load_hkl_table,
    write_merged_mtz,
)

# gemmi 0.7.x compat
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


# Rundir / checkpoint discovery


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
    ckpts = [p for p in ckpt_dir.glob("epoch=*.ckpt") if not p.is_symlink()]
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
    """Recompute the per-HKL merged posterior over the full dataset."""
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


# Model loading + posterior extraction
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


# Make phenix.eff files
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


# Run Phenix
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
    # Get refinement values
    for key, pat in [
        ("r_work_final", r"Final R-work\s*=\s*([\d.]+)"),
        ("r_free_final", r"Final R-work\s*=\s*[\d.]+\D+R-free\s*=\s*([\d.]+)"),
        ("r_work_start", r"Start R-work\s*=\s*([\d.]+)"),
        ("r_free_start", r"Start R-work\s*=\s*[\d.]+\D+R-free\s*=\s*([\d.]+)"),
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
        "--no-finalize-merge",
        action="store_true",
        help="Skip the converged-encoder merge pass.",
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
        logger.info("finalize_merge completed in %.1f s", time.time() - t0)
    else:
        logger.warning(
            "finalize_merge NOT run (no finalize_merge method or "
            "--no-finalize-merge). For conjugate/amortized models the merge "
            "buffers are non-persistent, so the MTZ will reflect the Wilson "
            "prior unless the checkpoint carries persistent buffers."
        )

    alpha, beta, seen = extract_merged_posterior(integrator)

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

    mtz_path = diag_dir / "merged.mtz"

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
