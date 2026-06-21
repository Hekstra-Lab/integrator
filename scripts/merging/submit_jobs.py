"""Submit a SLURM array that evaluates every checkpoint of a merging run.

Analogue of refltorch's dials_output/submit_jobs.py: reads
`<run_dir>/merging_eval_cfg.yaml` (from create_config.py), launches one array
task per checkpoint running process_single_ckpt.py (finalize merge -> MTZ ->
phenix.refine -> rs.find_peaks), then a dependent aggregation job running
compare_checkpoints.py. No DIALS.

Usage:
    python create_config.py --run-dir RUN --eff-template EFF --pdb PDB
    python submit_jobs.py   --run-dir RUN --script-dir $(pwd)
"""

import argparse
import subprocess
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(
        description="Submit per-checkpoint phenix evaluation SLURM array."
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--script-dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Dir holding process_single_ckpt.py + compare_checkpoints.py "
        "(default: this script's dir).",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="SLURM log dir (default: <run_dir>/ckpt_eval_logs).",
    )
    p.add_argument(
        "--split",
        action="store_true",
        help="Two phases: a GPU array writes MTZs, then a dependent CPU array "
        "runs phenix (frees the GPU before refinement). Default: one array.",
    )
    # Finalize/coupled array runs the model -> default to a GPU.
    p.add_argument(
        "--partition",
        type=str,
        default="gpu",
    )
    p.add_argument(
        "--gpus", type=str, default="1", help="--gres=gpu:N (0=cpu)"
    )
    p.add_argument(
        "--time",
        type=str,
        default="0:15:00",
    )
    p.add_argument(
        "--mem",
        type=str,
        default="100G",
    )
    p.add_argument(
        "--cpus",
        type=str,
        default="16",
    )
    # Phenix array (CPU) knobs, used only with --split.
    p.add_argument(
        "--phenix-partition",
        type=str,
        default="shared",
    )
    p.add_argument(
        "--phenix-time",
        type=str,
        default="02:00:00",
    )
    p.add_argument(
        "--phenix-mem",
        type=str,
        default="16G",
    )
    p.add_argument(
        "--phenix-cpus",
        type=str,
        default="4",
    )
    p.add_argument(
        "--agg-partition",
        type=str,
        default="shared",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=0,
        help="Throttle the array to N concurrent tasks (0 = no limit). Useful "
        "to avoid hogging GPUs.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the job scripts but don't sbatch.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cfg_file = run_dir / "merging_eval_cfg.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"{cfg_file} (run create_config.py first)",
        )

    cfg = yaml.safe_load(cfg_file.read_text())
    n = len(cfg["checkpoints"])
    if n == 0:
        raise ValueError(
            "no checkpoints in config",
        )

    script_dir = Path(args.script_dir)
    worker = (script_dir / "process_single_ckpt.py").as_posix()
    aggregator = (script_dir / "compare_checkpoints.py").as_posix()
    mamba_setup = cfg.get(
        "mamba_setup",
        "",
    )
    python_env = cfg.get(
        "python_env",
        "integrator-dev",
    )

    # Absolute, so sbatch --output works regardless of the submitting cwd.
    logs_dir = (
        Path(args.log_dir).resolve()
        if args.log_dir
        else run_dir / "ckpt_eval_logs"
    )
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build line by line (no dedent): the multi-line activate block would
    # otherwise defeat dedent and leave #! indented, which sbatch rejects.
    activate = (
        ["source " + mamba_setup, "micromamba activate " + python_env]
        if mamba_setup
        else []
    )

    def _script(*body: str) -> str:
        return "\n".join(["#!/bin/bash", *activate, *body, ""])

    # One worker script; the stage (all|finalize|phenix) is positional arg $1.
    worker_path = run_dir / "merging_eval_job.sh"
    worker_path.write_text(
        _script(
            'echo "stage $1 task $SLURM_ARRAY_TASK_ID on $HOSTNAME ($(date))"',
            f'python {worker} --config "{cfg_file}" '
            "--index $SLURM_ARRAY_TASK_ID --stage $1",
        )
    )
    worker_path.chmod(0o755)

    agg_path = run_dir / "merging_eval_aggregate.sh"
    agg_path.write_text(
        _script(
            'echo "aggregating ($(date))"',
            f'python {aggregator} --run-dir "{run_dir}"',
        )
    )
    agg_path.chmod(0o755)

    array = f"0-{n - 1}"
    if args.max_concurrent > 0:
        array += f"%{args.max_concurrent}"

    def _array_cmd(name, partition, time, mem, cpus, gpus, stage, dependency):
        cmd = [
            "sbatch",
            "--parsable",
            f"--job-name={name}",
            f"--output={logs_dir}/{name}_%A_%a.out",
            f"--error={logs_dir}/{name}_%A_%a.err",
            f"--time={time}",
            f"--mem={mem}",
            f"--partition={partition}",
            f"--cpus-per-task={cpus}",
            f"--array={array}",
        ]
        if gpus and gpus != "0":
            cmd.append(
                f"--gres=gpu:{gpus}",
            )
        if dependency:
            cmd.append(
                f"--dependency={dependency}",
            )
        return cmd + [str(worker_path), stage]

    def _submit(cmd, label):
        print(f"{label}:", " ".join(cmd))
        if args.dry_run:
            return "DRYRUN"
        jid = subprocess.check_output(cmd, text=True).strip()
        print(
            f"  -> job {jid}",
        )
        return jid

    print(
        f"Array: {array}  ({n} checkpoints x {len(cfg['variants'])} variants)"
        f"  {'split (GPU finalize -> CPU phenix)' if args.split else 'coupled'}"
    )

    if args.split:
        # Phase 1 (GPU): write MTZs. Phase 2 (CPU): phenix, element-wise after 1.
        j1 = _submit(
            _array_cmd(
                "merge_finalize",
                args.partition,
                args.time,
                args.mem,
                args.cpus,
                args.gpus,
                "finalize",
                None,
            ),
            "finalize array",
        )
        j2 = _submit(
            _array_cmd(
                "merge_phenix",
                args.phenix_partition,
                args.phenix_time,
                args.phenix_mem,
                args.phenix_cpus,
                "0",
                "phenix",
                f"aftercorr:{j1}",
            ),
            "phenix array",
        )
        last = j2
    else:
        last = _submit(
            _array_cmd(
                "merging_eval",
                args.partition,
                args.time,
                args.mem,
                args.cpus,
                args.gpus,
                "all",
                None,
            ),
            "eval array",
        )

    agg_cmd = [
        "sbatch",
        "--parsable",
        "--job-name=merging_eval_agg",
        f"--output={logs_dir}/aggregate_%j.out",
        f"--error={logs_dir}/aggregate_%j.err",
        "--time=01:00:00",
        "--mem=16G",
        f"--partition={args.agg_partition}",
        "--cpus-per-task=1",
        f"--dependency=afterany:{last}",
        str(agg_path),
    ]
    _submit(
        agg_cmd,
        "aggregation",
    )
    if args.dry_run:
        print("[dry-run] not submitting. Scripts written to", run_dir)
    else:
        print(
            "Status: squeue -u $USER",
        )


if __name__ == "__main__":
    main()
