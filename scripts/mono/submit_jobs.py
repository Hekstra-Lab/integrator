import argparse
import subprocess
import textwrap
from pathlib import Path

from integrator.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit DIALS+Phenix SLURM array job"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory containing parallel_config.json",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="dials_phenix_logs",
        help="Path to logging directory",
    )
    parser.add_argument(
        "--script-dir",
        type=str,
        help="Path to directory containing .py files; e.g. process_single_refl.py",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()

    # Config file containing env paths
    cfg_file = run_dir / "dials_phenix_cfg.yaml"

    if not cfg_file.exists():
        raise FileNotFoundError(cfg_file)

    cfg = load_config(cfg_file)

    num_files = len(cfg["refl_files"]) - 1
    if num_files < 0:
        raise ValueError("No .refl files found in config")

    logs_dir = Path(args.log_dir)
    logs_dir.mkdir(exist_ok=True)

    # Script directory
    script_dir = Path(args.script_dir)
    proc_py = (script_dir / "process_single.py").as_posix()

    # dials_phenix_job.sh
    dials_script = textwrap.dedent(
        f"""\
        #!/bin/bash
        mkdir -p logs

        echo "Job ID: $SLURM_JOB_ID"
        echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
        echo "Running on node: $HOSTNAME"
        echo "Started at: $(date)"
        echo "Working dir: {run_dir}"

        source /n/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
        micromamba activate refltorch

        python {proc_py} --config "{cfg_file}" --index $SLURM_ARRAY_TASK_ID

        echo "Finished at: $(date)"
        """
    )

    dials_script_path = Path("dials_phenix_job.sh")
    dials_script_path.write_text(dials_script)
    dials_script_path.chmod(0o755)

    # Plot script
    # plot_script = (script_dir / "compare_models.py").as_posix()

    # Script to generate figures
    # Depends on DIALS/PHENIX processing
    # upload_script = textwrap.dedent(
    #     f"""\
    #     #!/bin/bash
    #     echo "All array jobs completed. Generating plots"
    #     echo "Started at: $(date)"
    #
    #     source /n/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
    #     micromamba activate refltorch
    #
    #     python {plot_script} --run-dirs "{run_dir}"
    #
    #     echo "Finished at: $(date)"
    #     """
    # )

    # Handle paths
    # upload_script_path = Path("upload_wandb.sh")
    # upload_script_path.write_text(upload_script)
    # upload_script_path.chmod(0o755)

    # Submit array job
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        "--job-name=dials_phenix_parallel",
        f"--output={logs_dir}/dials_phenix_%A_%a.out",
        f"--error={logs_dir}/dials_phenix_%A_%a.err",
        "--time=10:00:00",
        "--mem=8G",
        "--partition=shared",
        "--cpus-per-task=1",
        f"--array=0-{num_files}",
        str(dials_script_path),
    ]

    job_id = subprocess.check_output(sbatch_cmd, text=True).strip()
    print(f"Submitted job array with tasks 0-{num_files}, Job ID: {job_id}")

    # Submit dependent job
    # upload_cmd = [
    #     "sbatch",
    #     "--parsable",
    #     f"--dependency=afterany:{job_id}",
    #     "--job-name=wandb_upload",
    #     f"--output={logs_dir}/plots_%j.out",
    #     f"--error={logs_dir}/plots_upload_%j.err",
    #     "--time=01:00:00",
    #     "--mem=80G",
    #     "--partition=seas_compute",
    #     "--cpus-per-task=1",
    #     str(upload_script_path),
    # ]

    # upload_job_id = subprocess.check_output(upload_cmd, text=True).strip()
    # print(
    #     f"Submitted WandB upload job with ID: {upload_job_id} (depends on {job_id})"
    # )
    print("Check status with: squeue -u $USER")


if __name__ == "__main__":
    main()
