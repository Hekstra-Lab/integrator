import torch
import glob
import json
import datetime
from integrator.model import (
    MVNProfile,
    DirichletProfile,
    SoftmaxProfile,
    CNNResNet,
    MVNProfile,
    SoftmaxProfile,
    DirichletProfile,
)
import os


def get_profile(config):
    """
    Function to get the correct profile based on the config.
    """
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

    if dist_name == "Dirichlet":
        concentration_shape = config["concentration_shape"]
        concentration = torch.ones(*concentration_shape)
        return torch.distributions.Dirichlet(concentration)

    params = {k: v for k, v in config.items() if k != "distribution"}

    return getattr(torch.distributions, dist_name)(**params)


def get_experiment_counter(model_type, profile_type, base_dir="logs/outputs"):
    """Get the next experiment number for the given date, model type, and profile type."""
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    counter_file = os.path.join(base_dir, "experiment_counters.json")

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

    with open(counter_file, "w") as f:
        json.dump(counters, f)

    return counters[key]


def generate_experiment_dir(config, base_dir="logs/outputs"):
    """Generate a unique directory for each experiment with a running counter."""
    model_type = config.get("encoder_type", "UnknownModel")
    profile_type = config.get("profile_type", "UnknownProfile")
    date_str = datetime.datetime.now().strftime("%Y%m%d")

    experiment_number = get_experiment_counter(model_type, profile_type, base_dir)

    experiment_name = f"{model_type}_{profile_type}_{date_str}_{experiment_number:03d}"
    experiment_dir = os.path.join(base_dir, experiment_name)

    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "out"), exist_ok=True)
    os.makedirs(os.path.join("logs", "tensorboard_logs"), exist_ok=True)

    return experiment_dir


def get_most_recent_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoint_files:
        return None
    return max(checkpoint_files, key=os.path.getmtime)


def get_encoder(config):
    """

    Args:
        config ():

    Raises:
        ValueError:

    Returns:

    """
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
