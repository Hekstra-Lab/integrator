import argparse
import subprocess
from pathlib import Path

from out_utils import DialsPhenixConfig as RunPaths
from refltorch.io import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phenix on a scaled.refl")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to dials_phenix_cfg.yaml",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="For multi file processing; denotes the ith .refl file to process",
    )

    return parser.parse_args()


def run_dials(
    dials_env: str,
    command: str,
):
    command = f"source {dials_env} && {command}"
    try:
        result = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            check=True,
        )
        return result

    except subprocess.CalledProcessError as e:
        print(f"Command failed with error code: {e.returncode}")
        print("Standard Output (stdout):")
        print(e.stdout if e.stdout else "No stdout output")
        print("Standard Error (stderr):")
        print(e.stderr if e.stderr else "No stderr output")
        raise


def run_phenix(
    phenix_env,
    mtz_file,
    phenix_eff,
):
    # Create the phenix directory first
    paren = Path(mtz_file).parent
    phenix_dir = paren / "phenix_out"
    phenix_dir.mkdir(parents=True, exist_ok=True)

    # Construct the phenix.refine command with proper escaping
    refine_command = f"phenix.refine {Path(phenix_eff).resolve()} {Path(mtz_file).resolve()} overwrite=true"

    # refined_mtz_out = phenix_dir + "/refine_001.mtz"
    refined_mtz_out = mtz_file

    # Construct the find_peaks command
    peaks_command = "rs.find_peaks *[0-9].mtz *[0-9].pdb -f ANOM -p PANOM -z 5.0 -o peaks.csv"

    # full_command = f"source {phenix_env} && cd {phenix_dir} && {refine_command} && {peaks_command} && cd {paired_ref_dir} && {paired_ref_command} && cd {paired_model_dir} && {paired_model_command} "
    full_command = f"source {phenix_env} && cd {phenix_dir} && {refine_command} && {peaks_command}"

    try:
        # Use subprocess.run instead of Popen for better error handling
        result = subprocess.run(
            full_command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,  # Capture both stdout and stderr
            text=True,  # Convert output to string
            check=True,  # Raise CalledProcessError on non-zero exit
        )
        print("Phenix command completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error code: {e.returncode}")
        print(
            "Command that failed:", full_command
        )  # Print the actual command for debugging
        print("Working directory:", phenix_dir)
        print("Standard Output:")
        print(e.stdout)
        print("Error Output:")
        print(e.stderr)
        raise


# Define the update_phenix_eff function to update Phenix configuration
def update_phenix_eff(
    phenix_eff_template,
    updated_phenix_eff,
    mtz_file,
):
    with open(phenix_eff_template) as file:
        lines = file.readlines()

    with open(updated_phenix_eff, "w") as file:
        for line in lines:
            if 'file = "$MTZFILE"' in line:
                line = f'      file = "{mtz_file}"\n'
            file.write(line)


def run_shell(cmd: str, *, cwd: Path | None = None):
    r = subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print("FAILED CMD:", cmd)
        print("CWD:", cwd)
        print("STDOUT:\n", r.stdout)
        print("STDERR:\n", r.stderr)
        raise RuntimeError(f"Command failed (exit {r.returncode})")
    return r


# Function to process a single reflection file
def process_single_refl(
    refl_file,
    expt_file,
    dials_env,
    phenix_env,
    phenix_eff,
    paired_ref_eff: str | None = None,
    paired_model_eff: str | None = None,
):
    print(f"Processing reflection file: {refl_file}")

    # dirs and filenames
    parent_dir = Path(refl_file).parent
    output_dir = parent_dir / "dials"

    print("output directory:", output_dir)

    scaled_refl_out = output_dir / "dials_scaled.refl"
    scaled_expt_out = output_dir / "dials_scaled.expt"

    # Make output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run dials.scale
    scale_command = (
        f"dials.scale '{refl_file}' '{expt_file}' "
        f"output.reflections='{scaled_refl_out}' "
        f"output.experiments='{scaled_expt_out}' "
        f"output.html='{output_dir}/scaling.html' "
        f"output.log='{output_dir}/scaling.log' "
    )
    print("Executing scale command:", scale_command)
    run_dials(dials_env, scale_command)

    # Extract refl_ids flagged as outlier_in_scaling -> scaling_outliers.parquet
    outliers_out = output_dir / "scaling_outliers.parquet"
    extract_script = (
        Path(__file__).resolve().parent / "extract_scaling_outliers.py"
    )
    outliers_command = (
        f"dials.python '{extract_script}' "
        f"--refl '{scaled_refl_out}' "
        f"--output '{outliers_out}'"
    )
    print("Executing outlier extraction:", outliers_command)
    run_dials(dials_env, outliers_command)

    # Run dials.merge
    merge_mtz_out = output_dir / "merged.mtz"
    merge_command = (
        f"dials.merge '{scaled_refl_out}' '{scaled_expt_out}' "
        f"output.log='{output_dir}/merged.log' "
        f"output.html='{output_dir}/merged.html' "
        f"output.mtz='{merge_mtz_out}'"
    )
    print("Executing merge command:", merge_command)
    run_dials(dials_env, merge_command)

    # Update phenix.eff and run phenix
    updated_phenix_eff = output_dir / "phenix_updated.eff"
    print("updated phenix.eff: ", updated_phenix_eff)
    update_phenix_eff(phenix_eff, updated_phenix_eff, merge_mtz_out)

    run_phenix(
        phenix_env,
        merge_mtz_out,
        updated_phenix_eff,
    )

    print(f"Completed processing for {refl_file}")


def main():
    args = parse_args()

    cfg_path = args.config
    file_index = args.index

    config = load_config(cfg_path)
    paths = RunPaths(**config)

    if file_index >= len(paths.refl_files):
        raise ValueError(f"file_index out of range: file_index={file_index}")

    refl_file = paths.refl_files[file_index]
    process_single_refl(
        refl_file=refl_file,
        expt_file=paths.expt_file,
        dials_env=paths.dials_env,
        phenix_env=paths.phenix_env,
        phenix_eff=paths.phenix_eff,
    )


# Main execution for single file processing
if __name__ == "__main__":
    main()
