"""Evaluate ONE checkpoint of a merging run (one SLURM array task).

Picks `checkpoints[--index]` from `merging_eval_cfg.yaml`, finalizes the merge
over the dataset, writes a merged MTZ, and for each column variant runs
phenix.refine + rs.find_peaks (NO DIALS). Writes a per-checkpoint `result.json`
that compare_checkpoints.py aggregates across epochs.

All the heavy lifting (load, finalize, MTZ, eff render, phenix, peaks, R-factor
parse) is reused from the sibling `scripts/merge_eval.py`.

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
import merge_eval as dm  # noqa: E402

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
    p.add_argument(
        "--force",
        action="store_true",
        help="Redo work even if outputs already exist (default: skip done).",
    )
    return p.parse_args()


def _read_result(work_root: Path) -> dict | None:
    """Existing result.json for this checkpoint, or None if absent/unreadable."""
    p = work_root / "result.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _select_checkpoint(eval_cfg: dict, index: int) -> tuple[Path, int, Path]:
    """Return (checkpoint, epoch, work_root) for this array index."""
    checkpoints = eval_cfg["checkpoints"]
    if not 0 <= index < len(checkpoints):
        raise IndexError(
            f"--index {index} out of range (0..{len(checkpoints) - 1})"
        )
    ckpt = Path(checkpoints[index])
    epoch = dm._epoch_of(ckpt)
    # Integrator prediction layout: <output_root>/predictions/epoch_<NNNN>/.
    work_root = Path(eval_cfg["out_root"]) / f"epoch_{epoch:04d}"
    work_root.mkdir(parents=True, exist_ok=True)
    return ckpt, epoch, work_root


def write_obs_pred(integrator, cfg: dict, epoch: int, work_root: Path) -> None:
    """Write the per-obs `pred.parquet` into work_root (BatchPredWriter, one file).

    The merged qi is only correct on the grouped predict loader (each HKL
    complete per batch), so predict over `predict_dataloader(grouped=True)`.
    """
    from integrator.callbacks import BatchPredWriter
    from integrator.utils import construct_data_loader, construct_trainer

    data_loader = construct_data_loader(cfg)
    data_loader.setup()
    try:
        loader = data_loader.predict_dataloader(
            grouped=True, lightning_safe=True
        )
    except TypeError:
        loader = data_loader.predict_dataloader()

    pred_writer = BatchPredWriter(
        output_dir=work_root,
        write_interval="batch",
        epoch=epoch,
        partition=False,
    )
    # Match the predict CLI's _run_merging_predict: keep our grouped batch
    # sampler intact, else Lightning re-wraps it with a SequentialSampler ->
    # "'SequentialSampler' object is not subscriptable".
    trainer = construct_trainer(
        cfg, callbacks=[pred_writer], logger=False, use_distributed_sampler=False
    )
    trainer.predict(integrator, return_predictions=False, dataloaders=loader)


def run_finalize(eval_cfg: dict, index: int, force: bool = False) -> dict:
    """GPU stage: finalize the merge, write the MTZ + per-obs pred.parquet."""
    ckpt, epoch, work_root = _select_checkpoint(eval_cfg, index)

    # Skip the expensive merge pass if a valid MTZ + result already exist.
    existing = _read_result(work_root)
    if (
        not force
        and (work_root / "merged.mtz").exists()
        and (work_root / "pred.parquet").exists()
        and existing is not None
        and existing.get("n_hkl_seen")
    ):
        logger.info("Skip finalize epoch %d: MTZ + preds already present", epoch)
        return existing

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

    # Per-obs predictions (scaled_intensity + DIALS fingerprint) beside the MTZ.
    write_obs_pred(integrator, cfg, epoch, work_root)

    result = {
        "epoch": epoch,
        "checkpoint": str(ckpt),
        "index": index,
        "n_hkl_seen": int(seen.sum()),
        "n_hkl_total": int(len(alpha)),
        "mtz": str(mtz_path),
        "pred": str(work_root / "pred.parquet"),
        "variants": {},
    }
    (work_root / "result.json").write_text(json.dumps(result, indent=2))
    logger.info("Wrote MTZ + preds + partial result: %s", work_root)
    return result


def run_phenix(eval_cfg: dict, index: int, force: bool = False) -> dict:
    """CPU stage: refine + find peaks on an already-written MTZ."""
    _, epoch, work_root = _select_checkpoint(eval_cfg, index)
    os.environ["PHENIX_ENV"] = eval_cfg["phenix_env"]

    result_path = work_root / "result.json"
    result = _read_result(work_root) or {
        "epoch": epoch,
        "index": index,
        "variants": {},
    }
    result.setdefault("variants", {})
    mtz_path = work_root / "merged.mtz"
    if not mtz_path.exists():
        raise FileNotFoundError(
            f"{mtz_path} missing; run the finalize stage first."
        )

    template = Path(eval_cfg["eff_template"]).read_text()
    for vname in eval_cfg["variants"]:
        # Skip variants that already refined successfully.
        done = result["variants"].get(vname)
        if not force and done and done.get("phenix_ok"):
            logger.info("Skip %s epoch %d: already refined", vname, epoch)
            continue

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
        # Write after each variant so a crash keeps finished variants.
        result_path.write_text(json.dumps(result, indent=2))

    result_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s", result_path)
    return result


def main():
    args = parse_args()
    eval_cfg = yaml.safe_load(args.config.read_text())
    if args.stage in ("all", "finalize"):
        run_finalize(eval_cfg, args.index, args.force)
    if args.stage in ("all", "phenix"):
        run_phenix(eval_cfg, args.index, args.force)


if __name__ == "__main__":
    main()
