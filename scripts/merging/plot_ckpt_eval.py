"""Make anomalous / refinement / scatter plots from a finished ckpt_eval dir.

Standalone post-processor: it acts on the final `ckpt_eval/` directory (produced
by submit_jobs.py -> process_single_ckpt.py), pulling from each `epoch*/`
subdir only the files each plot needs:

    result.json          -> R-work/R-free + top anomalous peak vs epoch
    <variant>/*[0-9].mtz  -> per-site anomalous peak heights vs epoch (needs a
                             PDB + an anomalous-atom selection)
    merged.mtz           -> F(+) vs F(-) Friedel scatter

Decoupled from the submission / worker scripts so plots can be (re)made or
tweaked without rerunning phenix. Usage:

    python plot_ckpt_eval.py --run-dir RUN_DIR [--pdb ref.pdb --anom-atom-sel '[S]']
    python plot_ckpt_eval.py --ckpt-eval-dir /path/ckpt_eval --anom-atom-sel '[S]'
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

import merging_plots as mp

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger(__name__)


def resolve_ckpt_eval(run_dir: Path, override: Path | None) -> Path:
    """Where the per-checkpoint workers wrote (netscratch for a W&B run).

    override > eval config out_root > run_paths output_root/ckpt_eval > run_dir.
    """
    if override:
        return Path(override)
    cfg_file = run_dir / "merging_eval_cfg.yaml"
    if cfg_file.exists():
        c = yaml.safe_load(cfg_file.read_text()) or {}
        if c.get("out_root"):
            return Path(c["out_root"])
    meta_path = run_dir / "run_paths.yaml"
    if meta_path.exists():
        m = yaml.safe_load(meta_path.read_text()) or {}
        if m.get("output_root"):
            return Path(m["output_root"]) / "ckpt_eval"
    return run_dir / "ckpt_eval"


def load_results(ckpt_eval: Path) -> tuple[list[dict], list[dict]]:
    """Load every epoch*/result.json -> (rows per (epoch, variant), raw dicts)."""
    rows, results = [], []
    for rj in sorted(ckpt_eval.glob("epoch*/result.json")):
        try:
            res = json.loads(rj.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip %s: %s", rj, exc)
            continue
        res["_dir"] = str(rj.parent)
        results.append(res)
        for vname, v in (res.get("variants") or {}).items():
            rows.append(
                {
                    "epoch": res.get("epoch"),
                    "variant": vname,
                    "r_work": v.get("r_work"),
                    "r_free": v.get("r_free"),
                    "top_anom_peak": v.get("top_anom_peak"),
                    "n_anom_peaks": v.get("n_anom_peaks"),
                }
            )
    rows.sort(key=lambda r: (r["variant"] or "", r["epoch"] or -1))
    return rows, results


def _site_rows(results, pdb, anom_sel):
    """Per-site anomalous peak heights vs epoch, computed from the refined MTZs.

    For each checkpoint and variant, finds the phenix-refined map MTZ
    (`<variant>/*[0-9].mtz`) and samples the ANOM/PHANOM map at the selected
    atom sites. Done here (post-hoc), not in the worker.
    """
    rows = []
    for res in results:
        ep = res.get("epoch")
        work_root = Path(res["_dir"])
        for vname in (res.get("variants") or {}):
            mtzs = sorted((work_root / vname).glob("*[0-9].mtz"))
            if not mtzs:
                continue
            try:
                labels, heights = mp.get_anom_peak_heights(
                    mtzs[-1], pdb, anom_sel
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("site peaks skipped for %s/%s: %s", ep, vname, exc)
                continue
            for site, h in zip(labels, heights):
                rows.append(
                    {"epoch": ep, "site": f"{vname}:{site}", "height": h}
                )
    return rows


def make_all_plots(
    ckpt_eval: Path, *, pdb: str | None = None, anom_sel: str = ""
) -> list[Path]:
    """Write all eval plots into `ckpt_eval`; returns the saved paths."""
    rows, results = load_results(ckpt_eval)
    if not rows:
        raise FileNotFoundError(f"no result.json under {ckpt_eval}")
    logger.info("Loaded %d (epoch, variant) results", len(rows))

    saved = []
    fig, _ = mp.plot_r_values_vs_epoch(
        rows, title="Refinement R-factors vs checkpoint"
    )
    saved.append(mp.save_figure(fig, ckpt_eval / "r_factors_vs_epoch.png"))

    fig, _ = mp.plot_metric_vs_epoch(
        rows, "top_anom_peak", ylabel="top anomalous peak (sigma)",
        title="Top anomalous peak vs checkpoint",
    )
    saved.append(mp.save_figure(fig, ckpt_eval / "anom_peak_vs_epoch.png"))

    if pdb and anom_sel:
        site_rows = _site_rows(results, pdb, anom_sel)
        if site_rows:
            fig, _ = mp.plot_anom_sites_vs_epoch(
                site_rows, title="Anomalous peak height at sites vs checkpoint"
            )
            saved.append(
                mp.save_figure(fig, ckpt_eval / "anom_sites_vs_epoch.png")
            )

    # Friedel scatter from the latest checkpoint's merged MTZ.
    mtzs = sorted(
        (r.get("mtz") for r in results if r.get("mtz")), key=lambda p: p or ""
    )
    if mtzs and Path(mtzs[-1]).exists():
        try:
            fig, _ = mp.friedel_scatter_from_mtz(
                mtzs[-1], title="Friedel pairs (latest checkpoint)"
            )
            if fig is not None:
                saved.append(
                    mp.save_figure(fig, ckpt_eval / "friedel_scatter.png")
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Friedel scatter skipped: %s", exc)
    return saved


def _resolve_pdb(run_dir: Path, cli_pdb: str | None) -> str | None:
    """PDB from --pdb, else the eval config's pdb."""
    if cli_pdb:
        return cli_pdb
    cfg = run_dir / "merging_eval_cfg.yaml"
    if cfg.exists():
        return (yaml.safe_load(cfg.read_text()) or {}).get("pdb") or None
    return None


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot a finished ckpt_eval directory (anomalous / "
        "refinement / scatter)."
    )
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument(
        "--ckpt-eval-dir",
        type=Path,
        default=None,
        help="The ckpt_eval directory directly (else resolved from --run-dir).",
    )
    p.add_argument(
        "--pdb",
        type=str,
        default=None,
        help="Reference PDB for per-site anomalous peaks (else eval config pdb).",
    )
    p.add_argument(
        "--anom-atom-sel",
        type=str,
        default="",
        help="gemmi selection of anomalous scatterers, e.g. '[S]'. Needed for "
        "the per-site anomalous-peak-height plot.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.run_dir and not args.ckpt_eval_dir:
        raise SystemExit("pass --run-dir or --ckpt-eval-dir")
    run_dir = args.run_dir.resolve() if args.run_dir else None
    ckpt_eval = (
        Path(args.ckpt_eval_dir).resolve()
        if args.ckpt_eval_dir
        else resolve_ckpt_eval(run_dir, None)
    )
    if not ckpt_eval.exists():
        raise FileNotFoundError(f"no ckpt_eval at {ckpt_eval}")
    pdb = args.pdb or (_resolve_pdb(run_dir, None) if run_dir else None)

    saved = make_all_plots(ckpt_eval, pdb=pdb, anom_sel=args.anom_atom_sel)
    print("\n".join(["Wrote:"] + [f"  {p}" for p in saved]))


if __name__ == "__main__":
    main()
