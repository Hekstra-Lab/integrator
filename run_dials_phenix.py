import subprocess
from pathlib import Path

glob.glob(prediction_path + "epoch*/reflections/*.refl")

# TODO: these should be passed as an argument
dials_environment = "/Applications/dials-v3-16-1/dials_env.sh"
expt_file = "/Users/luis/dials_out/816_sbgrid_HEWL/pass1/integrated.expt"


def run_dials(dials_environment, command):
    full_command = f"source {dials_environment} && {command}"

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

        # Optionally re-raise the exception if you want it to propagate
        raise


def run_phenix(phenix_env, mtz_file, pdb_file):
    command = f"source {phenix_env} && phenix.refine {pdb_file} {mtz_file} miller_array.labels.name=F\(+\),F\(-\) overwrite=true "
    phenix_dir = str(Path(mtz_file).parent) + "/phenix_out"
    Path(phenix_dir).mkdir(parents=True, exist_ok=True)

    # command = ' '.join(command)
    command += (
        f";rs.find_peaks *[0-9].mtz *[0-9].pdb -f ANOM -p PANOM -z 5.0 -o peaks.csv"
    )

    try:
        result = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            cwd=phenix_dir,
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


# refl_files = glob.glob(prediction_path + "epoch*/reflections/*.refl")
refl_files = glob.glob(
    "/Users/luis/integratorv3/integrator/lightning_logs/version_68/predictions/epoch*/reflections/*.refl"
)
refl_file = refl_files[0]


for refl_file in refl_files:
    parent_dir = Path(refl_file).parent.parent.__str__()
    scaled_refl_out = parent_dir + "/dials_out/scaled.refl"
    scaled_expt_out = parent_dir + "/dials_out/scaled.expt"
    # make output dir
    Path(parent_dir + "/dials_out").mkdir(parents=True, exist_ok=True)

    scale_command = (
        f"dials.scale {refl_file} {expt_file} "
        f"output.reflections='{scaled_refl_out}' "
        f"output.experiments='{scaled_expt_out}' "
        f"output.html='{parent_dir}/dials_out/scaling.html' "
        f"output.log='{parent_dir}/dials_out/scaling.log'"
    )

    run_dials(dials_environment, scale_command)

    merge_command = (
        f"dials.merge {scaled_refl_out} {scaled_expt_out} "
        f"output.log='{parent_dir}/dials_out/merged.log' "
        f"output.html='{parent_dir}/dials_out/merged.html' "
        f"output.mtz='{parent_dir}/dials_out/merged.mtz'"
    )

    run_dials(dials_environment, merge_command)

    mtz_file = parent_dir + "/dials_out/merged.mtz"
    pdb = "/Users/luis/Downloads/7l84.pdb"

    phenix_env = "/Applications/phenix-1.21.1-5286/phenix_env.sh"
    run_phenix(phenix_env, mtz_file, pdb)


subprocess.run(
    "rs.find_peaks *[0-9].mtz *[0-9].pdb -f ANOM -p PANOM -z 5.0 -o peaks.csv",
    shell=True,
    cwd="./lightning_logs/version_68/predictions/epoch_1/dials_out/phenix_out/",
)
