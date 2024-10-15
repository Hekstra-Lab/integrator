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
    DirichletProfile,
    Loss,
    Decoder,
)
from integrator.utils import OutWriter
import integrator.utils as utils
from integrator.utils.logger import init_tensorboard_logger

# torch.set_float32_matmul_precision('high')


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


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Resolve paths
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    config["data_dir"] = os.path.join(
        base_path, os.path.dirname(config["shoebox_data"])
    )

    return config


def get_encoder(config):
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
    elif profile_type == "DirichletProfile":
        return DirichletProfile(
            dmodel=config.get("dmodel", 64),
            num_components=config.get("num_components", 3 * 21 * 21),
        )
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


def get_prior_distribution(config):
    """Create a torch distribution for priors based on the config."""
    dist_name = config["distribution"]
    params = {k: v for k, v in config.items() if k != "distribution"}

    return getattr(torch.distributions, dist_name)(**params)


def train(config, resume_from_checkpoint=None, log_dir="logs/outputs"):
    # Generate a unique directory for this experiment
    experiment_dir = generate_experiment_dir(config, log_dir)

    # TensorBoard logger
    logger = TensorBoardLogger(save_dir="logs", name="tensorboard_logs")

    # Setup data module
    data_module = ShoeboxDataModule(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        val_split=config["val_split"],
        test_split=config["test_split"],
        num_workers=config["num_workers"],
        include_test=config["include_test"],
        subset_size=config["subset_size"],
        single_sample_index=config["single_sample_index"],
        cutoff=config["cutoff"],
    )

    data_module.setup()

    train_size = len(data_module.train_dataloader())
    val_size = len(data_module.val_dataloader())
    gradient_steps = config["gradient_steps"]
    batch_size = config["batch_size"]
    epochs = (gradient_steps * batch_size) // train_size

    # Initialize model components
    encoder = get_encoder(config)
    profile = get_profile(config)
    standardize = Standardize()
    decoder = Decoder()

    loss = Loss(
        p_I_scale=config["p_I_scale"],
        p_bg_scale=config["p_bg_scale"],
        p_p_scale=config["p_p_scale"],
        p_I=get_prior_distribution(config["p_I"]),
        p_bg=get_prior_distribution(config["p_bg"]),
        p_p=get_prior_distribution(config["p_p"]),
    )

    q_bg = BackgroundDistribution(
        config["dmodel"],
        q_bg=getattr(torch.distributions, config["q_bg"]["distribution"]),
    )
    q_I = IntensityDistribution(
        config["dmodel"],
        q_I=getattr(torch.distributions, config["q_I"]["distribution"]),
    )

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
        encoder_type=config["encoder_type"],
        profile_type=config["profile_type"],
        total_steps=config["total_steps"],
        # max_epochs=config["epochs"],
        max_epochs=epochs,
        dmodel=config["dmodel"],
        batch_size=config["batch_size"],
        rank=config["rank"],
        C=config["C"],
        Z=config["Z"],
        H=config["H"],
        W=config["W"],
        lr=config["learning_rate"],
        images_dir=images_dir,
    )

    # Load weights from checkpoint if provided
    if resume_from_checkpoint:
        print(f"Loading weights from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint)
        integrator_model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Weights loaded successfully")

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
        max_epochs=epochs,
        accelerator=config["accelerator"],
        devices=1,
        num_nodes=1,
        precision=config["precision"],
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, progress_bar],
        logger=logger,
        log_every_n_steps=10,
    )

    # Start training
    trainer.fit(integrator_model, data_module)

    val_predictions = integrator_model.evaluate_full_dataset(
        data_module.val_dataloader()
    )

    train_predictions = integrator_model.evaluate_full_dataset(
        data_module.train_dataloader()
    )
    return train_predictions, val_predictions, experiment_dir, integrator_model

    # return integrator_model, experiment_dir


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
        "--resume", type=str, help="Path to checkpoint to resume training."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/outputs",
        help="Directory where logs will be saved.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Ensure the log_dir exists before writing to it
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    train_preds, val_preds, experiment_dir, model = train(
        config, resume_from_checkpoint=args.resume, log_dir=args.log_dir
    )

    # Ensure the output directory exists
    output_refl_dir = os.path.join(experiment_dir, "out")

    # Output file paths
    output_refl_file = os.path.join(output_refl_dir, "NN.refl")
    output_refl_file2 = os.path.join(output_refl_dir, "DIALS_sum_NN.refl")
    output_refl_file3 = os.path.join(output_refl_dir, "DIALS_sum_NN_subset.refl")

    outwriter = OutWriter(
        train_preds,
        val_preds,
        os.path.join(config["data_path"], config["dataset_path"]),
        output_refl_file,
        out_file_name2=output_refl_file2,
        out_file_name3=output_refl_file3,
    )

    sel = outwriter.write_output()

    # Extract encoder_type, profile_type, and batch_size from config
    encoder_type = config.get("encoder_type", "UnknownEncoder")
    profile_type = config.get("profile_type", "UnknownProfile")
    batch_size = config.get("batch_size", "UnknownBatchSize")

    # Plotting
    utils.plot_intensities(
        nn_refl=output_refl_file,
        title="NNProfileIntensity vs. DIALSIntensity",
        dials_refl=os.path.join(config["data_path"], config["dataset_path"]),
        sel=sel,
        output_dir=output_refl_dir,
        encoder_type=encoder_type,
        profile_type=profile_type,
        batch_size=batch_size,
        out_png_filename="I_weighted_sum.png",
        intensity_column="I_weighted_sum",
        save=True,
        display=False,
    )

    utils.plot_intensities(
        nn_refl=output_refl_file2,
        title="NNProfileIntensity vs. DIALSIntensity",
        dials_refl=os.path.join(config["data_path"], config["dataset_path"]),
        sel=sel,
        output_dir=output_refl_dir,
        encoder_type=encoder_type,
        profile_type=profile_type,
        batch_size=batch_size,
        out_png_filename="intensity_comparison.png",
        intensity_column="I_weighted_sum",
        save=True,
        display=False,
    )

    config_copy_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f)
    model_summary = str(model)

    summary_path = os.path.join(experiment_dir, "model_summary.txt")
    with open(summary_path, "w") as f:
        f.write(model_summary)
