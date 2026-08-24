"""Diagnose a HierarchicalSVAEIntegrator run: model-as-integrator vs model-does-all.

The hierarchical SVAE does integration, scaling, and merging in one model, so it
emits BOTH a per-observation integrated `.refl` AND a scaled+merged `.mtz`. This
script runs the two downstream paths to completion and compares them head to
head, so you can see whether the model's *learned* scaling+merging matches or
beats letting DIALS do that job on the model's integrated intensities:

  Path A (integrate, then DIALS scale+merge):
        model -> integrated.refl
              -> dials.scale -> dials.merge -> phenix.refine -> rs.find_peaks
  Path B (model does everything):
        model -> merged.mtz
              -> phenix.refine -> rs.find_peaks

Both end in phenix.refine + anomalous peak finding, so R-work/R-free and the
sulfur anomalous peak heights are directly comparable. A third reference point
is your existing pure-DIALS run (DIALS integrate + scale + merge), if you have it.

The integrated `.refl` and merged `.mtz` are produced by `integrator.pred`
(which now writes both for this model); this script only orchestrates the DIALS
and Phenix steps and parses the results.

Usage
-----
    DIALS_ENV=/path/to/dials/setup.sh PHENIX_ENV=/path/to/phenix/setup.sh \
    uv run python scripts/diagnose_hierarchical_svae.py RUN_DIR [--last | --epoch N]

`RUN_DIR` must contain `run_paths.yaml` (written by `integrator.train`).
Outputs land in `RUN_DIR/diagnostics_hier/`:
    path_A_dials/   dials_scaled.{refl,expt}, merged_dials.mtz, phenix.*, peaks.csv
    path_B_model/   merged.mtz (copy), phenix.*, peaks.csv
    summary.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diagnose_hier")


# --------------------------------------------------------------------------- #
# Shell / environment helpers
# --------------------------------------------------------------------------- #
def _env(name: str, override: str | None) -> str | None:
    return override or os.environ.get(name)


def run_shell(
    cmd: str,
    work_dir: Path,
    log_name: str,
    source: str | None = None,
    timeout: int = 3600,
) -> bool:
    """Run a shell command (optionally sourcing an env activation script).

    stdout+stderr are captured to `work_dir/log_name`. Returns True on success.
    """
    full = f"cd {work_dir} && {cmd}"
    if source:
        full = f"source {source} && {full}"
    logger.info("[%s] %s", work_dir.name, cmd)
    try:
        proc = subprocess.run(
            full,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        (work_dir / log_name).write_text((proc.stdout or "") + (proc.stderr or ""))
        return True
    except subprocess.CalledProcessError as e:
        (work_dir / log_name).write_text((e.stdout or "") + (e.stderr or ""))
        logger.error(
            "FAILED (exit %d): %s\n  last stderr:\n%s",
            e.returncode,
            cmd.split()[0],
            (e.stderr or "")[-1200:],
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("TIMEOUT after %ds: %s", timeout, cmd.split()[0])
        return False


# --------------------------------------------------------------------------- #
# Run metadata / file location
# --------------------------------------------------------------------------- #
def load_run(run_dir: Path) -> tuple[dict, dict]:
    meta_path = run_dir / "run_paths.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"no run_paths.yaml in {run_dir}")
    meta = yaml.safe_load(meta_path.read_text())
    from integrator.utils import load_config

    cfg = load_config(meta["config"])
    return cfg, meta


def predictions_dir(meta: dict) -> Path:
    return Path(meta["wandb"]["log_dir"]).parent / "predictions"


def generate_refl_and_mtz(
    run_dir: Path, epoch: int | None, last: bool
) -> tuple[Path | None, Path | None]:
    """Run `integrator.pred` to write the integrated .refl and merged .mtz.

    Returns (refl_path, mtz_path) for the selected checkpoint's epoch dir.
    """
    sel = ["--last"] if last else ["--epoch", str(epoch)]
    cmd = [
        sys.executable, "-m", "integrator.cli.pred",
        "--run-dir", str(run_dir),
        "--write-refl",
        "--write-merged-mtz", "merged.mtz",
        *sel,
    ]
    logger.info("Generating .refl + .mtz: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (run_dir / "diagnostics_hier").mkdir(parents=True, exist_ok=True)
    (run_dir / "diagnostics_hier" / "pred.log").write_text(
        (proc.stdout or "") + (proc.stderr or "")
    )
    logger.info(
        "pred finished (exit %d) in %.1f s", proc.returncode, time.time() - t0
    )
    refl, mtz = locate_outputs(run_dir, epoch, last)
    # The .refl is written before the merged-MTZ step, so pred can exit non-zero
    # (e.g. n_hkl not set -> finalize_merge raises) yet still leave a usable .refl.
    if proc.returncode != 0:
        if refl is None:
            raise RuntimeError(
                "integrator.pred failed and produced no .refl; see "
                "diagnostics_hier/pred.log:\n" + (proc.stderr or "")[-2000:]
            )
        logger.warning(
            "integrator.pred exited %d but the .refl exists; the merged.mtz is "
            "likely missing (set n_hkl in the run config to enable Path B). "
            "See diagnostics_hier/pred.log.",
            proc.returncode,
        )
    return refl, mtz


def locate_outputs(
    run_dir: Path, epoch: int | None, last: bool
) -> tuple[Path | None, Path | None]:
    """Return (refl, mtz) for the selected epoch dir; either may be None."""
    _, meta = load_run(run_dir)
    pred_dir = predictions_dir(meta)
    epoch_dirs = sorted(pred_dir.glob("epoch_*"))
    if not epoch_dirs:
        return None, None
    if epoch is not None and not last:
        epoch_dir = pred_dir / f"epoch_{epoch:04d}"
    else:  # last
        epoch_dir = max(
            epoch_dirs, key=lambda p: int(re.search(r"(\d+)", p.name).group(1))
        )
    refls = list(epoch_dir.glob("preds_epoch_*.refl"))
    mtz = epoch_dir / "merged.mtz"
    refl = refls[0] if refls else None
    mtz = mtz if mtz.exists() else None
    logger.info("integrated.refl: %s", refl)
    logger.info("merged.mtz:      %s", mtz)
    return refl, mtz


# --------------------------------------------------------------------------- #
# Result parsing
# --------------------------------------------------------------------------- #
def parse_r_factors(work_dir: Path, log_name: str = "phenix.log") -> dict:
    log = work_dir / log_name
    if not log.exists():
        return {}
    text = log.read_text()
    out = {}
    for key, pat in [
        ("r_work", r"Final R-work\s*=\s*([\d.]+)"),
        ("r_free", r"Final R-free\s*=\s*([\d.]+)"),
    ]:
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def top_peaks(work_dir: Path, top_n: int = 8) -> str:
    csv = work_dir / "peaks.csv"
    if not csv.exists():
        return "    (no peaks.csv)"
    try:
        import pandas as pd

        df = pd.read_csv(csv)
        if df.empty:
            return "    (no peaks found)"
        zcol = "peakz" if "peakz" in df.columns else (
            "peak" if "peak" in df.columns else df.columns[-1]
        )
        top = df.sort_values(zcol, ascending=False).head(top_n)
        lines = [f"    top {len(top)} by {zcol}: "
                 + ", ".join(f"{v:.2f}" for v in top[zcol].tolist())]
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"    (peaks parse failed: {e})"


# --------------------------------------------------------------------------- #
# The two paths
# --------------------------------------------------------------------------- #
def path_a_dials(
    refl: Path,
    expt: Path,
    out: Path,
    dials_env: str | None,
    phenix_env: str | None,
    eff: Path,
) -> dict:
    """integrated.refl -> dials.scale -> dials.merge -> phenix.refine -> peaks."""
    out.mkdir(parents=True, exist_ok=True)
    result = {"path": "A: model integrate -> DIALS scale+merge -> phenix"}
    if dials_env is None:
        logger.warning("DIALS_ENV not set; skipping Path A.")
        result["status"] = "skipped (no DIALS_ENV)"
        return result

    scaled_refl = out / "dials_scaled.refl"
    scaled_expt = out / "dials_scaled.expt"
    merged_mtz = out / "merged_dials.mtz"

    ok = run_shell(
        f"dials.scale '{refl.resolve()}' '{expt.resolve()}' "
        f"output.reflections='{scaled_refl}' output.experiments='{scaled_expt}' "
        f"output.html='{out}/scaling.html' output.log='{out}/scaling.log'",
        out, "dials_scale.log", source=dials_env,
    )
    if not ok:
        result["status"] = "dials.scale failed"
        return result
    ok = run_shell(
        f"dials.merge '{scaled_refl}' '{scaled_expt}' "
        f"output.mtz='{merged_mtz}' output.log='{out}/merge.log' "
        f"output.html='{out}/merge.html'",
        out, "dials_merge.log", source=dials_env,
    )
    if not ok:
        result["status"] = "dials.merge failed"
        return result
    # DIALS merged MTZ carries intensities; run the I variants.
    result.update(
        _phenix_variants(merged_mtz, out, phenix_env, eff, ["I_FW", "I_noFW"])
    )
    result.setdefault("status", "ok")
    return result


def path_b_model(
    mtz: Path,
    out: Path,
    phenix_env: str | None,
    eff: Path,
) -> dict:
    """merged.mtz (the model's own scale+merge) -> phenix.refine -> peaks."""
    out.mkdir(parents=True, exist_ok=True)
    local_mtz = out / "merged.mtz"
    shutil.copy(mtz, local_mtz)
    result = {"path": "B: model integrate+scale+merge -> phenix"}
    # The model MTZ has both F and I; run all three so it is directly comparable
    # to how the amortized model was scored (F_noFW gave its best numbers).
    result.update(_phenix_variants(
        local_mtz, out, phenix_env, eff, ["F_noFW", "I_FW", "I_noFW"]
    ))
    result.setdefault("status", "ok")
    return result


def _set_array_type_star(line: str, star_token: str) -> str:
    """Move the `*` in an array_type listing to the requested token."""
    tokens = re.findall(r"\S+", line)
    out_tokens = [
        f"*{t.lstrip('*')}" if t.lstrip("*") == star_token else t.lstrip("*")
        for t in tokens
    ]
    leading = line[: len(line) - len(line.lstrip())]
    return leading + " ".join(out_tokens) + "\n"


def render_eff(template: str, mtz: Path, labels: str, star: str, fw: bool) -> str:
    """Render a phenix.eff: point the first miller_array at `mtz`, force the data
    labels + array_type, and set french_wilson_scale. Mirrors diagnose_merging so
    BOTH paths read INTENSITIES (I(+)/SIGI(+)/...) with French-Wilson -- the model
    MTZ also carries F columns with near-zero SIGF (a peaked posterior), which
    phenix would otherwise auto-select and badly over-weight.
    """
    text = template.replace("$MTZFILE", str(mtz.resolve()))
    count = 0
    in_data = False
    pending = False
    out_lines = []
    for line in text.splitlines(keepends=True):
        s = line.strip()
        if s.startswith("miller_array"):
            count += 1
            in_data = count == 1
            out_lines.append(line)
            continue
        if in_data and s.startswith("name ="):
            out_lines.append(re.sub(r'name = "[^"]*"', f'name = "{labels}"', line))
            pending = True
            continue
        if in_data and pending:
            out_lines.append(_set_array_type_star(line, star))
            if not line.rstrip().endswith("\\"):
                pending = False
            continue
        if in_data and s.startswith("user_selected_labels"):
            out_lines.append(re.sub(
                r'user_selected_labels = "[^"]*"',
                f'user_selected_labels = "{labels}"', line))
            in_data = False
            continue
        if "french_wilson_scale" in line:
            out_lines.append(re.sub(
                r"french_wilson_scale\s*=\s*\w+",
                f"french_wilson_scale = {fw}", line))
            continue
        out_lines.append(line)
    return "".join(out_lines)


def compare_merged_to_dials(model_mtz: Path, dials_mtz: Path) -> str:
    """CC of the model's merged I vs DIALS' merged I, joined on Miller index."""
    try:
        import numpy as np
        import reciprocalspaceship as rs

        def imean(ds):
            if "IMEAN" in ds.columns:
                return ds["IMEAN"].to_numpy(float)
            if {"I(+)", "I(-)"} <= set(ds.columns):
                return np.nanmean(
                    np.stack([ds["I(+)"].to_numpy(float),
                              ds["I(-)"].to_numpy(float)]), axis=0)
            if "I" in ds.columns:
                return ds["I"].to_numpy(float)
            return None

        m = rs.read_mtz(str(model_mtz))
        d = rs.read_mtz(str(dials_mtz))
        mi, di = m.copy(), d.copy()
        mi["_Im"] = imean(m)
        di["_Id"] = imean(d)
        if mi["_Im"] is None or di["_Id"] is None:
            return "  (no comparable I column in one of the MTZs)"
        j = mi[["_Im"]].join(di[["_Id"]], how="inner")
        x = j["_Im"].to_numpy(float)
        y = j["_Id"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if ok.sum() < 50:
            return f"  (only {int(ok.sum())} shared HKLs -- comparison unreliable)"
        cc = float(np.corrcoef(x[ok], y[ok])[0, 1])
        logcc = float(np.corrcoef(np.log(x[ok]), np.log(y[ok]))[0, 1])
        verdict = ("VALUES SOUND -> any bad R is structural (eff/columns/sigmas)"
                   if logcc > 0.9 else
                   "MERGE IS THE PROBLEM -> revisit nu / scale / retrain (ELBO)")
        return (f"  model-vs-DIALS merged I: n={int(ok.sum())} CC={cc:.3f} "
                f"logCC={logcc:.3f}  [{verdict}]")
    except Exception as e:  # noqa: BLE001
        return f"  (model-vs-DIALS CC failed: {e})"


# phenix.refine variants (matches diagnose_merging so results are comparable to
# how the amortized model was evaluated). The model's merged MTZ carries both F
# and I columns; F_noFW is the variant that gave the amortized model its best
# numbers, so it is the apples-to-apples baseline. (labels, array_type, FW)
_VARIANTS = {
    "F_noFW": ("F(+),SIGF(+),F(-),SIGF(-)", "amplitude", False),
    "I_FW": ("I(+),SIGI(+),I(-),SIGI(-)", "intensity", True),
    "I_noFW": ("I(+),SIGI(+),I(-),SIGI(-)", "intensity", False),
}


def _run_variant(mtz, out, phenix_env, template, name) -> dict:
    labels, star, fw = _VARIANTS[name]
    wd = out / name
    wd.mkdir(parents=True, exist_ok=True)
    rendered = wd / "refine.eff"
    rendered.write_text(render_eff(template, mtz, labels, star, fw))
    ok = run_shell(
        f"phenix.refine '{rendered.resolve()}' '{mtz.resolve()}' overwrite=true",
        wd, "phenix.log", source=phenix_env, timeout=5400,
    )
    if not ok:
        return {"status": "phenix.refine failed"}
    run_shell(
        "rs.find_peaks *[0-9].mtz *[0-9].pdb -f ANOM -p PANOM -z 5.0 -o peaks.csv",
        wd, "find_peaks.log", source=phenix_env, timeout=900,
    )
    r = parse_r_factors(wd)
    r["status"] = "ok"
    r["peaks"] = top_peaks(wd)
    return r


def _phenix_variants(
    mtz: Path, out: Path, phenix_env: str | None, eff: Path, names: list[str]
) -> dict:
    """Run phenix.refine in each named column/FW variant; return {name: result}."""
    if phenix_env is None:
        logger.warning("PHENIX_ENV not set; skipping phenix for %s", out.name)
        return {"status": "skipped (no PHENIX_ENV)"}
    template = eff.read_text()
    return {"variants": {
        n: _run_variant(mtz, out, phenix_env, template, n) for n in names
    }}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="training run dir (run_paths.yaml)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--last", action="store_true", help="use the last checkpoint (default)")
    g.add_argument("--epoch", type=int, default=None, help="use this epoch's checkpoint")
    p.add_argument("--dials-env", default=None, help="DIALS activation script (or $DIALS_ENV)")
    p.add_argument("--phenix-env", default=None, help="Phenix env script (or $PHENIX_ENV)")
    p.add_argument("--skip-generate", action="store_true",
                   help="reuse existing .refl/.mtz (skip integrator.pred)")
    p.add_argument("--skip-a", action="store_true", help="skip Path A (DIALS)")
    p.add_argument("--skip-b", action="store_true", help="skip Path B (model MTZ)")
    args = p.parse_args()

    last = args.last or args.epoch is None
    run_dir = args.run_dir.resolve()
    cfg, _ = load_run(run_dir)
    if cfg["integrator"]["name"] != "hierarchical_svae":
        logger.warning(
            "integrator is '%s', not 'hierarchical_svae'; the merged-MTZ path "
            "may not be available.", cfg["integrator"]["name"]
        )

    out_root = run_dir / "diagnostics_hier"
    out_root.mkdir(parents=True, exist_ok=True)
    dials_env = _env("DIALS_ENV", args.dials_env)
    phenix_env = _env("PHENIX_ENV", args.phenix_env)
    eff = Path(cfg["output"]["phenix_eff"])
    expt = Path(cfg["output"]["expt_file"])

    if args.skip_generate:
        refl, mtz = locate_outputs(run_dir, args.epoch, last)
    else:
        refl, mtz = generate_refl_and_mtz(run_dir, args.epoch, last)

    results = []
    if not args.skip_a:
        if refl is None:
            results.append({"path": "A: model integrate -> DIALS scale+merge -> "
                            "phenix", "status": "skipped (no .refl produced)"})
        else:
            results.append(path_a_dials(
                refl, expt, out_root / "path_A_dials", dials_env, phenix_env, eff
            ))
    if not args.skip_b:
        if mtz is None:
            results.append({"path": "B: model integrate+scale+merge -> phenix",
                            "status": "skipped (no merged.mtz -- set n_hkl in the "
                            "run config to enable it)"})
        else:
            results.append(path_b_model(
                mtz, out_root / "path_B_model", phenix_env, eff
            ))

    # Decisive check: do the model's merged intensities agree with DIALS'?
    # High CC -> the merge VALUES are sound (any bad R is structural/eff); low CC
    # -> the merge itself is the problem (training: nu, scale, ELBO maturity).
    cc_line = ""
    dials_mtz = out_root / "path_A_dials" / "merged_dials.mtz"
    if mtz is not None and dials_mtz.exists():
        cc_line = compare_merged_to_dials(mtz, dials_mtz)

    # Summary
    lines = ["=" * 72,
             "HierarchicalSVAE diagnostic: model-as-integrator vs model-does-all",
             "=" * 72,
             f"run_dir: {run_dir}",
             f"integrated.refl: {refl}",
             f"merged.mtz:      {mtz}",
             ""]
    if cc_line:
        lines += [cc_line, ""]
    for r in results:
        lines.append(r["path"])
        lines.append(f"    status : {r.get('status', '?')}")
        for name, v in r.get("variants", {}).items():
            if "r_work" in v:
                lines.append(
                    f"    [{name:7s}] R-work/R-free = "
                    f"{v.get('r_work', float('nan')):.4f} / "
                    f"{v.get('r_free', float('nan')):.4f}"
                )
                lines.append("    " + v.get("peaks", "").strip())
            else:
                lines.append(f"    [{name:7s}] {v.get('status', '?')}")
        lines.append("")
    lines.append(
        "Read: Path B at/above Path A on R-free and anomalous peaks means the "
        "model's learned scaling+merging matches/beats DIALS on the model's own "
        "integration -> the joint model earns its keep. Path A >> Path B means "
        "the merge is the weak link (revisit scale/nu/merge), not the integration."
    )
    summary = "\n".join(lines)
    (out_root / "summary.txt").write_text(summary)
    print("\n" + summary)
    logger.info("Wrote %s", out_root / "summary.txt")


if __name__ == "__main__":
    main()
