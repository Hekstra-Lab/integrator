import json
import subprocess
import sys
from pathlib import Path


def run_dials(dials_env, command):
    full_command = f"source {dials_env} && {command}"

    try:
        result = subprocess.run(
            full_command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            check=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        # Print more detailed error messages
        print(f"Command failed with error code: {e.returncode}")
        print("Standard Output (stdout):")
        print(e.stdout if e.stdout else "No stdout output")
        print("Standard Error (stderr):")
        print(e.stderr if e.stderr else "No stderr output")
        raise


def run_phenix(phenix_env, mtz_file, phenix_eff, paired_ref_eff, paired_model_eff):
    # Create the phenix directory first
    phenix_dir = str(Path(mtz_file).parent) + "/phenix_out"
    Path(phenix_dir).mkdir(parents=True, exist_ok=True)

    # Create the paired refinement directory
    paired_ref_dir = phenix_dir + "/paired_ref"
    Path(paired_ref_dir).mkdir(parents=True, exist_ok=True)

    paired_model_dir = phenix_dir + "/paired_model"
    Path(paired_model_dir).mkdir(parents=True, exist_ok=True)

    # Construct the phenix.refine command with proper escaping
    refine_command = f"phenix.refine {Path(phenix_eff).resolve()} {Path(mtz_file).resolve()} overwrite=true"

    refined_mtz_out = phenix_dir + "/refine_001.mtz"
    updated_paired_model_eff = paired_model_dir + "/updated_paired_model.eff"

    update_phenix_eff(paired_model_eff, updated_paired_model_eff, refined_mtz_out)

    # Paired refinement commands
    # the reference always uses the same mtz, but needs new pdb
    paired_ref_command = (
        f"phenix.refine {Path(paired_ref_eff).resolve()} ../*[0-9].pdb overwrite=true"
    )

    # the model always uses the same pdb but needs new mtz
    paired_model_command = (
        f"phenix.refine {Path(updated_paired_model_eff).resolve()}  overwrite=true"
    )

    # Construct the find_peaks command
    peaks_command = (
        "rs.find_peaks *[0-9].mtz *[0-9].pdb -f ANOM -p PANOM -z 5.0 -o peaks.csv"
    )

    full_command = f"source {phenix_env} && cd {phenix_dir} && {refine_command} && {peaks_command} && cd {paired_ref_dir} && {paired_ref_command} && cd {paired_model_dir} && {paired_model_command} "

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
def update_phenix_eff(phenix_eff_template, updated_phenix_eff, mtz_file):
    with open(phenix_eff_template) as file:
        lines = file.readlines()

    with open(updated_phenix_eff, "w") as file:
        for line in lines:
            if 'file = "$MTZFILE"' in line:
                line = f'      file = "{mtz_file}"\n'
            file.write(line)


# Function to process a single reflection file
def process_single_refl(
    refl_file,
    expt_file,
    dials_env,
    phenix_env,
    phenix_eff,
    paired_ref_eff,
    paired_model_eff,
):
    print(f"Processing reflection file: {refl_file}")

    parent_dir = Path(refl_file).parent
    integration_type = Path(refl_file).name.replace("_.refl", "")
    output_dir = parent_dir / f"dials_out_{integration_type}"
    print("output directory:", output_dir)

    scaled_refl_out = output_dir / f"dials_out_{integration_type}_scaled.refl"
    scaled_expt_out = output_dir / f"dials_out_{integration_type}_scaled.expt"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run dials.scale
    scale_command = (
        f"dials.scale '{refl_file}' '{expt_file}' "
        f"output.reflections='{scaled_refl_out}' "
        f"output.experiments='{scaled_expt_out}' "
        f"output.html='{parent_dir}/dials_out/scaling.html' "
        f"output.html='{output_dir}/scaling.html' "
        f"output.log='{output_dir}/scaling.log' "
    )
    print("Executing scale command:", scale_command)
    run_dials(dials_env, scale_command)

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
        paired_ref_eff=paired_ref_eff,
        paired_model_eff=paired_model_eff,
    )

    print(f"Completed processing for {refl_file}")


# Main execution for single file processing
if __name__ == "__main__":
    # Get the configuration file path from the first argument
    if len(sys.argv) < 2:
        print("Usage: python process_single.py <config_file> <file_index>")
        sys.exit(1)

    config_file = sys.argv[1]
    file_index = int(sys.argv[2])

    # Load the configuration
    with open(config_file) as f:
        config = json.load(f)

    # Extract parameters from config
    refl_file_paths = config["refl_files"]
    phenix_eff = config["phenix_eff"]
    dials_env = config["dials_env"]
    phenix_env = config["phenix_env"]
    expt_file = config["expt_file"]
    paired_ref_eff = config["paired_ref_eff"]
    paired_model_eff = config["paired_model_eff"]

    # Get the specific reflection file to process
    if file_index >= len(refl_file_paths):
        print(
            f"Error: file_index {file_index} out of range. Only {len(refl_file_paths)} files available."
        )
        sys.exit(1)

    refl_file = refl_file_paths[file_index]

    # Process the single reflection file
    process_single_refl(
        refl_file,
        expt_file,
        dials_env,
        phenix_env,
        phenix_eff,
        paired_ref_eff,
        paired_model_eff,
    )
