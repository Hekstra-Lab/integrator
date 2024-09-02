import os
import json
import datetime
import yaml
import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from integrator import ShoeboxDataModule
from integrator.layers import Standardize
from integrator.model import (
    Integrator,
    CNNResNet,
    FcResNet,
    MVNProfile,
    SoftmaxProfile,
    BackgroundDistribution,
    IntensityDistribution,
    Loss,
    Decoder,
)
from integrator.utils import OutWriter
from integrator.utils import Plotter
from integrator.utils.logger import init_tensorboard_logger

import os

from pytorch_lightning.loggers import TensorBoardLogger


def init_tensorboard_logger(save_dir="logs", name="tensorboard_logs"):
    """
    Initialize a TensorBoard logger.

    Args:
        save_dir (str): Directory where the logs will be saved.
        name (str): Name of the logger.

    Returns:
        TensorBoardLogger: Initialized logger.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the TensorBoard logger
    logger = TensorBoardLogger(save_dir=save_dir, name=name)
    return logger


def get_experiment_counter(model_type, profile_type, base_dir="logs/outputs"):
    """Get the next experiment number for the given date, model type, and profile type."""
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    counter_file = os.path.join(base_dir, "experiment_counters.json")

    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)

    if os.path.exists(counter_file):
        with open(counter_file, "r") as f:
            counters = json.load(f)
    else:
        counters = {}

    key = f"{date_str}_{model_type}_{profile_type}"

    if key in counters:
        counters[key] += 1
    else:
        counters[key] = 1

    # Save the updated counters back to the file
    with open(counter_file, "w") as f:
        json.dump(counters, f)

    return counters[key]


def get_encoder(
    config,
):
    encoder_type = config.get("encoder_type")

    if encoder_type == "CNNResNet":
        return CNNResNet(
            depth=config.get("depth", 10),
            dmodel=config.get("dmodel", 64),
            feature_dim=config.get("feature_dim", 7),
            Z=config.get("Z", 3),
            H=config.get("H", 21),
            W=config.get("W", 21),
            dropout=config.get("dropout", None),
        )
    elif encoder_type == "FcResNet":
        return FcResNet(
            depth=config.get("depth", 10),
            dmodel=config.get("dmodel", 64),
            feature_dim=config.get("feature_dim", 7),
            dropout=config.get("dropout", None),
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def get_profile(config):
    profile_type = config.get("profile_type")

    if profile_type == "MVNProfile":
        return MVNProfile(dmodel=config.get("dmodel", 64), rank=config.get("rank", 3))
    elif profile_type == "SoftmaxProfile":
        return SoftmaxProfile(
            input_dim=config.get("input_dim", 64),
            rank=config.get("rank", 3),
            channels=config.get("channels", 3),
            height=config.get("height", 21),
            width=config.get("width", 21),
        )
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Resolve paths
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    config["shoebox_data"] = os.path.join(base_path, config["shoebox_data"])
    config["metadata"] = os.path.join(base_path, config["metadata"])
    config["dead_pixel_mask"] = os.path.join(base_path, config["dead_pixel_mask"])

    return config


def resolve_path(path):
    """Resolve environment variables in paths."""
    return os.path.expandvars(path)


def generate_experiment_dir(config, base_dir="logs/outputs"):
    """Generate a unique directory for each experiment with a running counter."""
    model_type = config.get("encoder_type", "UnknownModel")
    profile_type = config.get("profile_type", "UnknownProfile")
    date_str = datetime.datetime.now().strftime("%Y%m%d")

    # Get the experiment number
    experiment_number = get_experiment_counter(model_type, profile_type, base_dir)

    # Format the experiment directory name
    experiment_name = f"{model_type}_{profile_type}_{date_str}_{experiment_number:03d}"
    experiment_dir = os.path.join(base_dir, experiment_name)

    # Ensure that all necessary directories exist
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "out"), exist_ok=True)
    os.makedirs(os.path.join("logs", "tensorboard_logs"), exist_ok=True)

    return experiment_dir


def train(config, resume_from_checkpoint=None, log_dir="logs/outputs"):
    # Generate a unique directory for this experiment
    experiment_dir = generate_experiment_dir(config, log_dir)

    # TensorBoard logger
    logger = TensorBoardLogger(save_dir="logs", name="tensorboard_logs")

    # Setup data module
    data_module = ShoeboxDataModule(
        shoebox_data=config["shoebox_data"],
        metadata=config["metadata"],
        dead_pixel_mask=config["dead_pixel_mask"],
        batch_size=config["batch_size"],
        val_split=config["val_split"],
        test_split=config["test_split"],
        num_workers=config["num_workers"],
        include_test=config["include_test"],
        subset_size=config["subset_size"],
        single_sample_index=config["single_sample_index"],
    )

    data_module.setup()

    # Initialize model components
    encoder = get_encoder(config)
    profile = get_profile(config)
    standardize = Standardize()
    decoder = Decoder()

    loss = Loss(
        p_I_scale=config["p_I_scale"],
        p_bg_scale=config["p_bg_scale"],
        p_I=torch.distributions.exponential.Exponential(0.1),
        p_bg=torch.distributions.normal.Normal(0.0, 0.3),
    )

    q_bg = BackgroundDistribution(
        config["dmodel"], q_bg=torch.distributions.gamma.Gamma
    )
    q_I = IntensityDistribution(config["dmodel"], q_I=torch.distributions.gamma.Gamma)

    # Define the directory to save images
    images_dir = os.path.join(experiment_dir, "out", "images")
    os.makedirs(images_dir, exist_ok=True)

    integrator_model = Integrator(
        encoder,
        profile,
        q_bg,
        q_I,
        decoder,
        loss,
        standardize,
        encoder_type=config.get("encoder_type", "UnknownEncoder"),
        profile_type=config.get("profile_type", "UnknownProfile"),
        total_steps=None,
        max_epochs=config["epochs"],
        dmodel=config["dmodel"],
        batch_size=config["batch_size"],
        rank=config["rank"],
        C=config["C"],
        Z=config.get("Z", 3),
        H=config.get("H", 21),
        W=config.get("W", 21),
        lr=0.001,
        images_dir=images_dir,  # Pass the images_dir to the Integrator
    )

    # Setup logging and checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=os.path.join(experiment_dir, "checkpoints"),
        filename="integrator-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    progress_bar = TQDMProgressBar(refresh_rate=1)

    # Setup trainer
    trainer = Trainer(
        max_epochs=config["epochs"],
        accelerator="cpu",
        devices=1,
        num_nodes=1,
        precision=32,
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, progress_bar],
        logger=logger,
        log_every_n_steps=10,
    )

    # Start training
    trainer.fit(integrator_model, data_module)

    return integrator_model, experiment_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script for integrator model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs to use for training (0 for CPU).",
    )
    parser.add_argument(
        "--resume", type=str, help="Path to checkpoint to resume training."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/outputs",
        help="Directory where TensorBoard logs will be saved.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Ensure the log_dir exists before writing to it
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    model, experiment_dir = train(
        config, resume_from_checkpoint=args.resume, log_dir=args.log_dir
    )

    # Ensure the output directory exists
    output_refl_dir = os.path.join(experiment_dir, "out")

    # Output file paths
    output_refl_file = os.path.join(output_refl_dir, "out.refl")

    outwriter = OutWriter(
        model,
        "/Users/luis/integratorv2/integrator/data/simulated_hewl_816/reflections_.refl",
        output_refl_file,
    )

    outwriter.write_output()

    # Extract encoder_type, profile_type, and batch_size from config
    encoder_type = config.get("encoder_type", "UnknownEncoder")
    profile_type = config.get("profile_type", "UnknownProfile")
    batch_size = config.get("batch_size", "UnknownBatchSize")

    # Plotting
    plotter = Plotter(
        output_refl_file, output_refl_dir, encoder_type, profile_type, batch_size
    )
    plotter.plot_uncertainty(save=True, display=False)
    plotter.plot_intensities(save=True, display=False)

    config_copy_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f)
    model_summary = str(model)

    summary_path = os.path.join(experiment_dir, "model_summary.txt")
    with open(summary_path, "w") as f:
        f.write(model_summary)
