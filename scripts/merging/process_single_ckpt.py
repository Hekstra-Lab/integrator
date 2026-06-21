"""Evaluate ONE checkpoint of a merging run (one SLURM array task).

Picks `checkpoints[--index]` from `merging_eval_cfg.yaml`, finalizes the merge
over the dataset, writes a merged MTZ, and for each column variant runs
phenix.refine + rs.find_peaks (NO DIALS). Writes a per-checkpoint `result.json`
that compare_checkpoints.py aggregates across epochs.

All the heavy lifting (load, finalize, MTZ, eff render, phenix, peaks, R-factor
parse) is reused from the sibling `scripts/diagnose_merging.py`.

Usage (normally invoked by the SLURM array from submit_jobs.py):
    python process_single_ckpt.py --config /path/merging_eval_cfg.yaml --index 3
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import diagnose_merging as dm  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger(__name__)

_VARIANT_BY_NAME = {v[0]: v for v in dm.VARIANTS}


def _top_anomalous_peak(peaks_csv: Path | None) -> tuple[float, int]:
    """Return (top peak height, n peaks) from an rs.find_peaks CSV.

    The peak column is `peak` if present, else the last column. Empty/missing
    CSV -> (nan, 0).
    """
    if peaks_csv is None or not peaks_csv.exists():
        return float("nan"), 0
    try:
        import pandas as pd

        df = pd.read_csv(peaks_csv)
        if df.empty:
            return float("nan"), 0
        zcol = "peak" if "peak" in df.columns else df.columns[-1]
        return float(df[zcol].max()), int(len(df))
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not parse %s: %s", peaks_csv, exc)
        return float("nan"), 0


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate one merging checkpoint with phenix + find_peaks."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--index",
        type=int,
        required=True,
        help="Index into the config's `checkpoints` list (SLURM array task id).",
    )
    p.add_argument(
        "--stage",
        choices=["all", "finalize", "phenix"],
        default="all",
        help="all = finalize + phenix in one task (GPU). finalize = GPU MTZ "
        "only. phenix = CPU refine on an existing MTZ. Split frees the GPU "
        "before refinement.",
    )
    return p.parse_args()


def _select_checkpoint(eval_cfg: dict, index: int) -> tuple[Path, int, Path]:
    """Return (checkpoint, epoch, work_root) for this array index."""
    checkpoints = eval_cfg["checkpoints"]
    if not 0 <= index < len(checkpoints):
        raise IndexError(
            f"--index {index} out of range (0..{len(checkpoints) - 1})"
        )
    ckpt = Path(checkpoints[index])
    epoch = dm._epoch_of(ckpt)
    work_root = Path(eval_cfg["out_root"]) / f"epoch{epoch:04d}"
    work_root.mkdir(parents=True, exist_ok=True)
    return ckpt, epoch, work_root


def run_finalize(eval_cfg: dict, index: int) -> dict:
    """GPU stage: finalize the merge for this checkpoint and write the MTZ."""
    ckpt, epoch, work_root = _select_checkpoint(eval_cfg, index)
    logger.info("Finalize checkpoint %d epoch %d: %s", index, epoch, ckpt)

    cfg, _ = dm.load_run_metadata(Path(eval_cfg["run_dir"]))
    integrator = dm.load_integrator(cfg, ckpt)
    dm.finalize_merge_over_dataset(integrator, cfg)
    alpha, beta, seen = dm.extract_merged_posterior(integrator)

    data_dir = Path(eval_cfg["data_dir"])
    cell, sg = dm.load_crystal(data_dir)
    hkl = dm.load_hkl_table(data_dir, cfg, cell, sg)
    mtz_path = work_root / "merged.mtz"
    dm.write_merged_mtz(alpha, beta, seen, hkl, cell, sg, mtz_path)

    result = {
        "epoch": epoch,
        "checkpoint": str(ckpt),
        "index": index,
        "n_hkl_seen": int(seen.sum()),
        "n_hkl_total": int(len(alpha)),
        "mtz": str(mtz_path),
        "variants": {},
    }
    (work_root / "result.json").write_text(json.dumps(result, indent=2))
    logger.info("Wrote MTZ + partial result: %s", work_root)
    return result


def run_phenix(eval_cfg: dict, index: int) -> dict:
    """CPU stage: refine + find peaks on an already-written MTZ."""
    _, epoch, work_root = _select_checkpoint(eval_cfg, index)
    os.environ["PHENIX_ENV"] = eval_cfg["phenix_env"]

    result_path = work_root / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = {"epoch": epoch, "index": index, "variants": {}}
    mtz_path = work_root / "merged.mtz"
    if not mtz_path.exists():
        raise FileNotFoundError(
            f"{mtz_path} missing; run the finalize stage first."
        )

    template = Path(eval_cfg["eff_template"]).read_text()
    for vname in eval_cfg["variants"]:
        _, labels, star_token, fw = _VARIANT_BY_NAME[vname]
        work_dir = work_root / vname
        work_dir.mkdir(exist_ok=True)
        eff_path = work_dir / "phenix.eff"
        eff_path.write_text(
            dm.render_eff(template, mtz_path, labels, star_token, fw)
        )

        t0 = time.time()
        ok = dm.run_phenix_refine(eff_path, work_dir, mtz_path)
        r = dm.parse_phenix_r_factors(work_dir) if ok else {}
        peaks_csv = dm.run_find_peaks(work_dir) if ok else None
        top_peak, n_peaks = _top_anomalous_peak(peaks_csv)

        result["variants"][vname] = {
            "phenix_ok": ok,
            "r_work": r.get("r_work_final"),
            "r_free": r.get("r_free_final"),
            "r_work_start": r.get("r_work_start"),
            "top_anom_peak": top_peak,
            "n_anom_peaks": n_peaks,
            "seconds": round(time.time() - t0, 1),
        }
        logger.info(
            "[%s] phenix_ok=%s Rwork=%s Rfree=%s top_peak=%.2f n_peaks=%d",
            vname, ok, r.get("r_work_final"), r.get("r_free_final"),
            top_peak, n_peaks,
        )

    result_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s", result_path)
    return result


def main():
    args = parse_args()
    eval_cfg = yaml.safe_load(args.config.read_text())
    if args.stage in ("all", "finalize"):
        run_finalize(eval_cfg, args.index)
    if args.stage in ("all", "phenix"):
        run_phenix(eval_cfg, args.index)


if __name__ == "__main__":
    main()
