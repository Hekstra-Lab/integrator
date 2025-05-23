from integrator.callbacks import PredWriter
import yaml
import json
from pytorch_lightning.callbacks import Callback
import os
import glob
from integrator.utils import (
    load_config,
    create_integrator,
    create_integrator_from_checkpoint,
    create_data_loader,
    create_trainer,
    parse_args,
    override_config,
    clean_from_memory,
    predict_from_checkpoints,
    reflection_file_writer,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import torch
from integrator.model.encoders import ShoeboxEncoder
import subprocess

# from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from integrator.callbacks import (
    IntensityPlotter,
    MVNPlotter,
    UNetPlotter,
    Plotter,
    Plotter2,
    IntegratedPlotter,
)
from integrator.model import *
import lightning_fabric.utilities.cloud_io as cloud_io

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    dials_env = "/n/hekstra_lab/people/aldama/software/dials-v3-16-1/dials_env.sh "
    phenix_env = (
        "/n/hekstra_lab/garden_backup/phenix-1.21/phenix-1.21.1-5286/phenix_env.sh"
    )
    expt_file = "/n/holylabs/LABS/hekstra_lab/Users/laldama/integratorv2/integrator/logs/DIALS_/CNNResNetSoftmax_08_045/integrated.expt"
    pdb = (
        "/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/1dpx.pdb"
    )

    def flatten_config(config, parent_key="", sep="."):
        """Flatten nested config dict for cleaner W&B logging"""
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def get_git_info():
        try:
            commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode("ascii")
                .strip()
            )
            branch = (
                subprocess.check_output(
                    ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL
                )
                .decode("ascii")
                .strip()
            )
            dirty = (
                subprocess.check_output(
                    ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
                )
                .decode("ascii")
                .strip()
                != ""
            )
            return {"commit_hash": commit_hash, "branch": branch, "dirty": dirty}
        except Exception:
            return {"commit_hash": "unknown", "branch": "unknown", "dirty": False}

    class GitInfoCallback(Callback):
        def on_train_start(self, trainer, pl_module):
            log_dir = trainer.logger.log_dir
            git_info = get_git_info()
            with open(os.path.join(log_dir, "git_info.txt"), "w") as f:
                f.write(f"Commit hash: {git_info['commit_hash']}\n")
                f.write(f"Branch: {git_info['branch']}\n")
                f.write(f"Dirty: {git_info['dirty']}\n")

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

            # Optionally re-raise the exception if you want it to propagate
            raise

    def run_phenix(phenix_env, mtz_file, pdb_file):
        # Create the phenix directory first
        phenix_dir = str(Path(mtz_file).parent) + "/phenix_out"
        Path(phenix_dir).mkdir(parents=True, exist_ok=True)

        # Construct the phenix.refine command with proper escaping
        refine_command = (
            f"phenix.refine {pdb_file} {mtz_file} "
            f"'miller_array.labels.name=F(+),F(-)' "
            f"overwrite=true"
        )

        # Construct the find_peaks command
        peaks_command = (
            f"rs.find_peaks *[0-9].mtz *[0-9].pdb "
            f"-f ANOM -p PANOM -z 5.0 -o peaks.csv"
        )

        # Combine commands with proper sourcing
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

    def analysis(prediction_path, dials_env, phenix_env, pdb, expt_file):
        # refl_files = glob.glob(prediction_path + "epoch*/reflections/*.refl")

        p = Path(prediction_path).glob("epoch*/reflections/*.refl")

        for refl_file in p:
            # Convert paths to absolute paths to avoid any directory navigation issues
            parent_dir = Path(refl_file).parent.parent.absolute().__str__()
            integration_type = (refl_file.name).replace("_.refl", "")
            scaled_refl_out = parent_dir + f"/dials_out_{integration_type}/scaled.refl"
            scaled_expt_out = parent_dir + f"/dials_out_{integration_type}/scaled.expt"

            # Ensure output directory exists
            Path(parent_dir + f"/dials_out_{integration_type}").mkdir(
                parents=True, exist_ok=True
            )

            # Construct commands with proper quoting
            scale_command = (
                f"dials.scale '{refl_file}' '{expt_file}' "
                f"output.reflections='{scaled_refl_out}' "
                f"output.experiments='{scaled_expt_out}' "
                f"output.html='{parent_dir}/dials_out_{integration_type}/scaling.html' "
                f"output.log='{parent_dir}/dials_out_{integration_type}/scaling.log'"
            )

            print("Executing scale command:", scale_command)  # Debug print
            run_dials(dials_env, scale_command)

            merge_command = (
                f"dials.merge '{scaled_refl_out}' '{scaled_expt_out}' "
                f"output.log='{parent_dir}/dials_out_{integration_type}/merged.log' "
                f"output.html='{parent_dir}/dials_out_{integration_type}/merged.html' "
                f"output.mtz='{parent_dir}/dials_out_{integration_type}/merged.mtz'"
            )

            print("Executing merge command:", merge_command)  # Debug print
            run_dials(dials_env, merge_command)

            mtz_file = parent_dir + f"/dials_out_{integration_type}/merged.mtz"
            run_phenix(phenix_env, mtz_file, pdb)

    args = parse_args()

    # Load configuration file
    config = load_config(args.config)

    # override config options from command line
    override_config(args, config)

    # Create data loader
    data = create_data_loader(config)

    # Get gitinfo

    # Create callbacks
    pred_writer = PredWriter(
        output_dir=None,
        write_interval=config["trainer"]["params"]["callbacks"]["pred_writer"][
            "write_interval"
        ],
    )

    integrator_name = config["integrator"]["name"]
    if "image_encoder" in config["components"]:
        encoder_name = config["components"]["image_encoder"]["name"]
    else:
        encoder_name = config["components"]["encoder"]["name"]

    qbg_name = config["components"]["q_bg"]["name"]
    profile_name = config["components"]["profile"]["name"]
    pbg_name = config["components"]["loss"]["params"]["p_bg_name"]

    if "q_I" in config["components"]:
        qI_name = config["components"]["q_I"]["name"]
        pI_name = config["components"]["loss"]["params"]["p_I_name"]
    else:
        qI_name = ""
        pI_name = ""


    if "p_p" in config["components"]["loss"]["params"]:
        pp_name = config["components"]["loss"]["params"]["p_p"]["name"]
    else:
        pp_name = ""

    logger = WandbLogger(
        project="integrator_",
        name="Integrator_"
        + integrator_name
        + "_Encoder_"
        + encoder_name
        + "_I_"
        + qI_name
        + "_"
        + pI_name
        + "_Bg_"
        + qbg_name
        + "_"
        + pbg_name
        + "_P_"
        + profile_name
        + "_"
        + pp_name,
        save_dir="/n/hekstra_lab/people/aldama/lightning_logs"
    )

    if config["integrator"]["name"] == "mlp_integrator":
        plotter = UNetPlotter(
            num_profiles=10,
            plot_every_n_epochs=1,
        )

    elif config["integrator"]["name"] == "integrator6":
        plotter = Plotter2(
            num_profiles=10,
            plot_every_n_epochs=1,
        )


    elif config["integrator"]["name"] == "mvn_integrator":
        plotter = MVNPlotter(
            num_profiles=10,
            plot_every_n_epochs=1,
        )

    elif config['integrator']['name'] in {"integrator", "integrator2","integrator3","integrator4","integrator5"}:
        plotter = Plotter(
            num_profiles=10,
            plot_every_n_epochs=1,
        )

    elif config["integrator"]["name"] == "lrmvn_integrator":
        plotter = UNetPlotter(
                num_profiles=10,
                plot_every_n_epochs=1,
                )
    else:
        # plotter = IntensityPlotter(num_profiles=10)
        plotter = IntegratedPlotter(
            num_profiles=10,
            plot_every_n_epochs=1,
        )

    ## create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.experiment.dir + "/checkpoints",  # when using wandb logger
        filename="{epoch}-{val_loss:.2f}",
#        every_n_epochs=config["trainer"]["params"]["check_val_every_n_epoch"],
        save_top_k=20,
        monitor='val_loss',
        mode='min',
        save_last="link",
    )

    # Create a logger
    #    logger = TensorBoardLogger(save_dir='lightning_logs',name='integrator')

    # Create trainer
    trainer = create_trainer(
        config,
        data,
        callbacks=[
            pred_writer,
            checkpoint_callback,
            plotter,
        ],
        logger=logger,
    )

    # os.makedirs(trainer.logger.log_dir,exist_ok=True)
    # log_dirr = trainer.logger.log_dir
    os.makedirs(trainer.logger.experiment.dir, exist_ok=True)

    log_dirr = trainer.logger.experiment.dir

    git_info = get_git_info()

    save_git_info = os.path.join(log_dirr, "git_info.txt")
    logger.log_hyperparams(git_info)

    logger.log_hyperparams(config)

    with open(save_git_info, "w") as file:
        json.dump(git_info, file)

    if git_info["dirty"]:
        diff = subprocess.check_output(["git", "diff"]).decode("utf-8")
        with open(os.path.join(log_dirr, "uncommitted.diff"), "w") as f:
            f.write(diff)

    # Create integrator model
    integrator = create_integrator(config)

    #checkpoint_path =  '/n/hekstra_lab/people/aldama/lightning_logs/wandb/run-20250504_125302-7tkfopqd/files/checkpoints/epoch=95-val_loss=0.00.ckpt'

    #checkpoint = torch.load(
    #        checkpoint_path,
    #        weights_only=False
    #        )
#
#    integrator = create_integrator(config)
#    integrator.load_state_dict(checkpoint['state_dict'])

    # Fit the model
    trainer.fit(
        integrator,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )


   # pred_checkpoint = torch.load(
   #         log_dirr + "/checkpoints/last.ckpt",
   #         weights_only=False
   #         )

   # integrator.load_state_dict(pred_checkpoint['state_dict'])

    # Predict
#    trainer.predict(
#        pred_integrator,
#        return_predictions=False,
#        dataloaders=data.predict_dataloader(),
#    )

    version_dir = log_dirr
    path = os.path.join(version_dir, "checkpoints", "epoch*.ckpt")

    # override to stop new version dirs from being created
    config["trainer"]["params"]["logger"] = False

    # clean from memory
    clean_from_memory(pred_writer, pred_writer, pred_writer, checkpoint_callback)

    save_config = os.path.join(log_dirr, "config_copy.yaml")
    config['data_loader']['params']['batch_size'] = 1000
    config['data_loader']['params']['subset_size'] = None

    with open(save_config, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    config = load_config(save_config)

    pred_integrator = create_integrator(config)

    # predict from checkpoints
    predict_from_checkpoints(config, trainer, pred_integrator, data, version_dir, path)

    # write refl files
    prediction_path = version_dir + "/predictions/"
    prediction_directories = glob.glob(prediction_path + "epoch*")
    prediction_files = glob.glob(prediction_path + "epoch*/*.pt")

    reflection_file_writer(
        prediction_directories,
        prediction_files,
        config["output"]["refl_file"],
    )

    #analysis(prediction_path, dials_env, phenix_env, pdb, expt_file)

    # reflection_file_writer(prediction_directories, prediction_files)
    # for ckpt in glob.glob(path):
    # epoch = re.search(r"epoch=(\d+)", ckpt).group(0)
    # epoch = epoch.replace("=", "_")
    # ckpt_dir = version_dir + "/predictions/" + epoch
    # Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # # prediction writer for current checkpoint
    # pred_writer = PredWriter(
    # output_dir=ckpt_dir,
    # write_interval=config["trainer"]["params"]["callbacks"]["pred_writer"][
    # "write_interval"
    # ],
    # )
    # print('after predwriter')
    # log_memory()
    # trainer.callbacks = [pred_writer]

# #        trainer = create_trainer(
# #            config,
# #            data,
# #            callbacks=[
# #                pred_writer,
# #            ],
# #        )
# #        print('created_new_trainer')
# #        print(f'checkpoint:{ckpt}')
# #        log_memory()


# checkpoint = torch.load(ckpt,map_location='cpu')
# pred_integrator.load_state_dict(checkpoint['state_dict'])
# pred_integrator.to(torch.device('cuda'))
# pred_integrator.eval()

# print('created integrator from checkpoint')
# log_memory()

# print('running trainer.predict')
# trainer.predict(
# pred_integrator,
# return_predictions=False,
# dataloaders=data.predict_dataloader(),
# )

# clean_from_memory(pred_writer, pred_writer, pred_writer)


# predict from checkpoints

# predict_from_checkpoints(config, data, version_dir, path)

# prediction_path = version_dir + "/predictions/"
# prediction_directories = glob.glob(prediction_path + "epoch*")
# prediction_files = glob.glob(prediction_path + "epoch*/*.pt")

# reflection_file_writer(prediction_directories, prediction_files)
