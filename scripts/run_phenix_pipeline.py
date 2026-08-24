"""Run phenix.refine and anomalous peak finding on merged MTZ files.

Two modes:
  1. Submit mode (default): submit a SLURM array job, one task per epoch
  2. Worker mode (--index N): process a single epoch (called by the array job)

Usage
-----
    # Submit all epochs as parallel SLURM jobs:
    python scripts/run_phenix_pipeline.py --run-dir <run_dir>

    # Process a single epoch (called by SLURM):
    python scripts/run_phenix_pipeline.py --run-dir <run_dir> --index 3
"""

import argparse
import logging
import os
import re
import subprocess
import textwrap
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_phenix_env() -> str:
    env = os.environ.get("PHENIX_ENV")
    if env is None:
        raise RuntimeError(
            "PHENIX_ENV environment variable not set. "
            "Export it before running this script."
        )
    return env


# (columns -> (miller labels, array_type "*" token)). The model's merged MTZ
# carries posterior amplitudes F(+)/F(-) (E[sqrt(I)] under the Gamma) AND
# intensities I(+)/I(-); "amplitude" reads F directly with French-Wilson OFF (no
# second I->F), "intensity" reads I with French-Wilson ON (phenix's standard).
_COLUMN_SPECS = {
    "intensity": ("I(+),SIGI(+),I(-),SIGI(-)", "intensity"),
    "amplitude": ("F(+),SIGF(+),F(-),SIGF(-)", "amplitude"),
}


def _set_array_type_star(line: str, star_token: str) -> str:
    """Move the `*` in an array_type listing to the requested token."""
    tokens = re.findall(r"\S+", line)
    out_tokens = [
        f"*{t.lstrip('*')}" if t.lstrip("*") == star_token else t.lstrip("*")
        for t in tokens
    ]
    leading = line[: len(line) - len(line.lstrip())]
    return leading + " ".join(out_tokens) + "\n"


def _write_phenix_eff(
    template: Path,
    out_path: Path,
    mtz_path: Path,
    columns: str = "amplitude",
    fw_scale: bool = False,
) -> None:
    """Render the phenix.eff: substitute $MTZFILE and set the *first* (data)
    miller_array block to the requested columns + french_wilson_scale.

    Only the data block is touched (name / array_type `*` / user_selected_labels);
    the Rfree block is left intact. The MTZ is passed to phenix on the command
    line, so $MTZFILE substitution is for templates that also embed it.
    """
    labels, star_token = _COLUMN_SPECS[columns]
    with open(template) as f:
        text = f.read().replace("$MTZFILE", str(mtz_path.resolve()))

    miller_array_count = 0
    in_data_block = False
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

    with open(out_path, "w") as f:
        f.write("".join(out_lines))


def _get_config(run_dir: Path) -> tuple[dict, dict]:
    meta = yaml.safe_load((run_dir / "run_paths.yaml").read_text())
    config_path = meta["config"]
    with open(config_path) as f:
        return yaml.safe_load(f), meta


def _get_epoch_dirs(
    pred_dir: Path, epochs: str | None, mtz_name: str
) -> list[Path]:
    if epochs:
        dirs = [pred_dir / f"epoch_{int(e):04d}" for e in epochs.split(",")]
    else:
        dirs = sorted(pred_dir.glob("epoch_*"))
    return [d for d in dirs if (d / mtz_name).exists()]


def process_single_epoch(
    epoch_dir: Path,
    phenix_env: str,
    phenix_eff: Path,
    columns: str = "amplitude",
    fw_scale: bool = False,
) -> None:
    mtz_path = epoch_dir / "merged.mtz"
    phenix_out = epoch_dir / "phenix_out"
    phenix_out.mkdir(parents=True, exist_ok=True)

    updated_eff = phenix_out / "phenix.eff"
    _write_phenix_eff(
        phenix_eff, updated_eff, mtz_path, columns=columns, fw_scale=fw_scale
    )

    refine_cmd = (
        f"phenix.refine {updated_eff.resolve()} "
        f"{mtz_path.resolve()} overwrite=true"
    )
    peaks_cmd = (
        "rs.find_peaks *[0-9].mtz *[0-9].pdb "
        "-f ANOM -p PANOM -z 5.0 -o peaks.csv"
    )
    full_cmd = (
        f"source {phenix_env} && cd {phenix_out} && "
        f"{refine_cmd} && {peaks_cmd}"
    )

    logger.info("Processing %s", epoch_dir.name)
    try:
        subprocess.run(
            full_cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("Completed %s", epoch_dir.name)
    except subprocess.CalledProcessError as e:
        logger.error(
            "%s failed (exit %d):\n%s",
            epoch_dir.name,
            e.returncode,
            e.stderr[-2000:],
        )


def submit_array_job(
    run_dir: Path,
    epoch_dirs: list[Path],
    script_path: Path,
    columns: str = "amplitude",
    french_wilson: str = "auto",
) -> None:
    log_dir = run_dir / "phenix_logs"
    log_dir.mkdir(exist_ok=True)

    n_tasks = len(epoch_dirs) - 1

    job_script = textwrap.dedent(f"""\
        #!/bin/bash
        echo "Job ID: $SLURM_JOB_ID"
        echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
        echo "Started at: $(date)"

        source /n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
        micromamba activate refltorch

        python {script_path.resolve()} \\
            --run-dir "{run_dir.resolve()}" \\
            --columns {columns} \\
            --french-wilson {french_wilson} \\
            --index $SLURM_ARRAY_TASK_ID

        echo "Finished at: $(date)"
    """)

    job_path = run_dir / "phenix_array_job.sh"
    job_path.write_text(job_script)
    job_path.chmod(0o755)

    sbatch_cmd = [
        "sbatch",
        "--parsable",
        "--job-name=phenix_scaling",
        f"--output={log_dir}/phenix_%A_%a.out",
        f"--error={log_dir}/phenix_%A_%a.err",
        "--time=01:00:00",
        "--mem=8G",
        "--partition=shared",
        "--cpus-per-task=1",
        f"--array=0-{n_tasks}",
        str(job_path),
    ]

    job_id = subprocess.check_output(sbatch_cmd, text=True).strip()
    logger.info(
        "Submitted SLURM array job %s with %d tasks", job_id, n_tasks + 1
    )
    logger.info("Check status: squeue -u $USER")


def main():
    parser = argparse.ArgumentParser(
        description="Run phenix.refine + peak finding on merged MTZ files."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--mtz-name",
        default="merged.mtz",
        help="Name of merged MTZ in each epoch dir (default: merged.mtz)",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default=None,
        help="Comma-separated epoch numbers to process (default: all)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Process a single epoch by index (used by SLURM array tasks)",
    )
    parser.add_argument(
        "--columns",
        choices=["intensity", "amplitude"],
        default="amplitude",
        help="Merged columns phenix reads: 'amplitude' = F(+)/F(-) (the "
        "model's posterior amplitudes) with French-Wilson off; 'intensity' = "
        "I(+)/I(-) with French-Wilson on.",
    )
    parser.add_argument(
        "--french-wilson",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override french_wilson_scale ('auto' = off for amplitude, on "
        "for intensity).",
    )
    args = parser.parse_args()

    if args.french_wilson == "auto":
        fw_scale = args.columns == "intensity"
    else:
        fw_scale = args.french_wilson == "true"

    config, meta = _get_config(args.run_dir)
    phenix_eff = Path(config["output"]["phenix_eff"])
    phenix_env = _get_phenix_env()

    wandb_info = meta["wandb"]
    log_dir = Path(wandb_info["log_dir"])
    pred_dir = log_dir.parent / "predictions"

    epoch_dirs = _get_epoch_dirs(pred_dir, args.epochs, args.mtz_name)

    if not epoch_dirs:
        logger.error("No epoch directories with %s found", args.mtz_name)
        return

    if args.index is not None:
        process_single_epoch(
            epoch_dirs[args.index],
            phenix_env,
            phenix_eff,
            columns=args.columns,
            fw_scale=fw_scale,
        )
    else:
        submit_array_job(
            args.run_dir,
            epoch_dirs,
            Path(__file__),
            columns=args.columns,
            french_wilson=args.french_wilson,
        )


if __name__ == "__main__":
    main()
