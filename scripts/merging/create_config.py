import argparse
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

# Reuse the run/checkpoint discovery from the diagnostic script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import merge_eval as dm  # noqa: E402

from integrator.utils import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger(__name__)

# columns -> the merge_eval VARIANT names to run per checkpoint.
_COLUMN_VARIANTS = {
    "amplitude": ["F_noFW"],
    "intensity": ["I_FW"],
    "both": ["F_noFW", "I_FW"],
}


@dataclass
class MergingEvalConfig:
    """Everything the per-checkpoint worker needs, written to YAML."""

    run_dir: str
    data_dir: str
    checkpoints: list[str]
    variants: list[str]
    phenix_env: str
    eff_template: str
    pdb: str = ""
    chi2_inflation: bool = False
    out_root: str = ""  # per-ckpt outputs
    python_env: str = ""  # micromamba env activated by the SLURM worker
    mamba_setup: str = (
        "/n/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh"
    )
    extra: dict = field(default_factory=dict)


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare merging_eval_cfg.yaml: evaluate each checkpoint "
        "of a merging run with phenix.refine + rs.find_peaks (no DIALS)."
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--columns",
        choices=["amplitude", "intensity", "both"],
        default="both",
        help="Which merged columns phenix reads (default: both).",
    )
    p.add_argument(
        "--eff-template",
        type=str,
        default=None,
        help="phenix.eff template (else training output.phenix_eff, else "
        "$PHENIX_EFF).",
    )
    p.add_argument(
        "--pdb",
        type=str,
        default=None,
        help="Reference PDB for refinement (else output.pdb, else $PDB).",
    )
    p.add_argument(
        "--process-cfg",
        type=str,
        default=None,
        help="Post-processing YAML supplying phenix_eff / pdb / phenix_env "
        "(like the mono pipeline). Overrides the training config's output "
        "section; overridden by the explicit --eff-template/--pdb/--phenix-env.",
    )
    p.add_argument(
        "--phenix-env",
        type=str,
        default=None,
        help="Phenix env activation script (else $PHENIX_ENV).",
    )
    p.add_argument(
        "--python-env",
        type=str,
        default=os.environ.get("INTEGRATOR_ENV", "integrator-dev"),
        help="micromamba env the SLURM worker activates (default: "
        "$INTEGRATOR_ENV or integrator-dev).",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Where per-checkpoint outputs land (default: <output_root>/predictions).",
    )
    p.add_argument(
        "--chi2-inflation",
        action="store_true",
        help="Pass --chi2-inflation through to the worker.",
    )
    p.add_argument(
        "--last-only",
        action="store_true",
        help="Only the final checkpoint (quick smoke of the workflow).",
    )
    return p.parse_args()


def _resolve(cli, cfg_val, env_var, label, required=True):
    """First non-empty of CLI > training-config value > env var."""
    if cli:
        return cli
    if cfg_val:
        return str(cfg_val)
    env = os.environ.get(env_var) if env_var else None
    if env:
        return env
    if required:
        raise ValueError(
            f"Missing '{label}': pass --{label.replace('_', '-')}, set it in "
            f"the training config output section, or set ${env_var}."
        )
    return ""


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    logger.info("Run dir: %s", run_dir)

    cfg, meta = dm.load_run_metadata(run_dir)
    output_cfg = cfg.get("output", {}) or {}
    data_dir = cfg["data_loader"]["args"]["data_dir"]

    # Post-processing config (phenix_eff / pdb / phenix_env), like the mono
    # pipeline. Priority: CLI flag > --process-cfg > training output > env.
    proc_cfg = load_config(args.process_cfg) if args.process_cfg else {}

    def _cfg_val(key):
        return proc_cfg.get(key) or output_cfg.get(key)

    if args.last_only:
        checkpoints = [dm.find_last_checkpoint(meta)]
    else:
        checkpoints = dm.discover_checkpoints(meta)
    logger.info("Found %d checkpoint(s)", len(checkpoints))

    phenix_env = _resolve(
        args.phenix_env, _cfg_val("phenix_env"), "PHENIX_ENV", "phenix_env"
    )
    eff_template = _resolve(
        args.eff_template, _cfg_val("phenix_eff"), "PHENIX_EFF", "eff_template"
    )
    pdb = _resolve(args.pdb, _cfg_val("pdb"), "PDB", "pdb")

    # Eval lands in the integrator prediction layout (<output_root>/predictions/
    # epoch_<NNNN>/) so per-obs preds + merge + phenix sit beside the run's heavy
    # outputs. output_root = run_paths.yaml output_root (netscratch W&B dir for a
    # W&B run, else run_dir).
    output_root = meta.get("output_root") or str(run_dir)
    out_root = args.out_root or os.path.join(output_root, "predictions")

    config = MergingEvalConfig(
        run_dir=str(run_dir),
        data_dir=str(data_dir),
        checkpoints=[str(c) for c in checkpoints],
        variants=_COLUMN_VARIANTS[args.columns],
        phenix_env=phenix_env,
        eff_template=eff_template,
        pdb=pdb,
        chi2_inflation=args.chi2_inflation,
        out_root=out_root,
        python_env=args.python_env,
    )

    out = run_dir / "merging_eval_cfg.yaml"
    out.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    logger.info(
        "Wrote %s: %d checkpoints x %d variant(s) %s",
        out,
        len(checkpoints),
        len(config.variants),
        config.variants,
    )
    logger.info("Outputs will land under: %s", out_root)
    logger.info(
        "Next: python submit_jobs.py --run-dir %s --script-dir %s",
        run_dir,
        Path(__file__).resolve().parent,
    )


if __name__ == "__main__":
    main()
