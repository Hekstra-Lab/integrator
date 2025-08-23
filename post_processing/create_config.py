import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="find reflection files")
    parser.add_argument("--path", type = str)
    args = parser.parse_args()

    path = Path(args.path)
    refl_files = {"refl_files": [x.as_posix() for x in list(path.glob("**/*_.refl"))]}

    # Define your parameters here
    config = {
        "refl_files": refl_files["refl_files"],
        "phenix_eff": "/n/hekstra_lab/people/aldama/pass1/phenix.eff",
        "dials_env": "/n/hekstra_lab/people/aldama/software/dials-v3-16-1/dials_env.sh",
        "phenix_env": "/n/hekstra_lab/garden_backup/phenix-1.21/phenix-1.21.1-5286/phenix_env.sh",
        "expt_file": "/n/holylabs/LABS/hekstra_lab/Users/laldama/integratorv2/integrator/logs/DIALS_/CNNResNetSoftmax_08_045/integrated.expt",
        "pdb": "/n/hekstra_lab/people/aldama/pass1/9b7c.pdb"
    }


    # Ensure the paths are absolute
    for key in ["phenix_eff", "dials_env", "phenix_env", "expt_file", "pdb"]:
        config[key] = str(Path(config[key]).resolve())

    # Get the number of reflection files
    num_files = len(config["refl_files"])
    print(f"Found {num_files} reflection files to process")

    # Save the configuration to a JSON file
    config_file = "parallel_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to {config_file}")
    print(f"Use this file with the Slurm submission script to process {num_files} files in parallel")
