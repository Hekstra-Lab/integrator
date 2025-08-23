import glob
import json
import os
import subprocess

import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint

# from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from integrator.callbacks import (
    Plotter,
    PredWriter,
    assign_labels,
)
from integrator.utils import (
    clean_from_memory,
    create_data_loader,
    create_integrator,
    create_trainer,
    load_config,
    override_config,
    parse_args,
    predict_from_checkpoints,
    reflection_file_writer,
)

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
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
        write_interval=config["trainer"]["params"]["callbacks"]["pred_writer"]["write_interval"],
    )

    integrator_name = config["integrator"]["name"]

    logger = WandbLogger(
        project="integrator_2025-07",
        name="Integrator_" + integrator_name,
        save_dir="/n/netscratch/hekstra_lab/Lab/laldama/lightning_logs",
    )

    # assign labels to samples
    assign_labels(dataset=data, save_dir=logger.experiment.dir)

    plotter = Plotter(
        n_profiles=10,
        plot_every_n_epochs=1,
        d=config["logger"]["d"],
        h=config["logger"]["h"],
        w=config["logger"]["w"],
    )

    ## create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.experiment.dir + "/checkpoints",  # when using wandb logger
        filename="{epoch}-{val_loss:.2f}",
        every_n_epochs=config["trainer"]["params"]["check_val_every_n_epoch"],
        save_top_k=-1,
        save_last="link",
    )

    # Create a logger
    #    logger = TensorBoardLogger(save_dir='lightning_logs',name='integrator')

    # Create trainer
    trainer = create_trainer(
        config,
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

    # checkpoint_path =  '/n/hekstra_lab/people/aldama/lightning_logs/wandb/run-20250504_125302-7tkfopqd/files/checkpoints/epoch=95-val_loss=0.00.ckpt'

    # checkpoint = torch.load(
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
    integrator.train_df.write_csv(version_dir + "/avg_train_metrics.csv")
    integrator.val_df.write_csv(version_dir + "/avg_val_metrics.csv")
    path = os.path.join(version_dir, "checkpoints", "epoch*.ckpt")

    # override to stop new version dirs from being created
    config["trainer"]["params"]["logger"] = False

    # clean from memory
    clean_from_memory(pred_writer, pred_writer, pred_writer, checkpoint_callback)

    save_config = os.path.join(log_dirr, "config_copy.yaml")
    config["data_loader"]["params"]["batch_size"] = 1000
    config["data_loader"]["params"]["subset_size"] = None

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

    # Submit the script using sbatch
    try:
        result = subprocess.run(
            ["sbatch", "run_parallel.sh"], check=True, capture_output=True, text=True
        )
        print("Submission successful!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Submission failed.")
        print("Return code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
