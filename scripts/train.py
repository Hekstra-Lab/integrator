import os
import glob
import pytorch_lightning
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
    CNNResNet, FcResNet, MVNProfile,Integrator, SoftmaxProfile, BackgroundDistribution,
    IntensityDistribution, DirichletProfile, Loss, Decoder
)
from integrator.utils import OutWriter
import integrator.utils as utils

torch.set_float32_matmul_precision('high')

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

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

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

    if dist_name == "Dirichlet":
        concentration_shape = config["concentration_shape"]
        concentration = torch.ones(*concentration_shape)
        return torch.distributions.Dirichlet(concentration)

    params = {k: v for k, v in config.items() if k != "distribution"}

    return getattr(torch.distributions, dist_name)(**params)

def train(config, resume_from_checkpoint=None, log_dir="logs/outputs"):
    experiment_dir = generate_experiment_dir(config, log_dir)

    logger = TensorBoardLogger(save_dir="logs", name="tensorboard_logs")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = get_encoder(config)
    profile = get_profile(config)
    standardize = Standardize()
    decoder = Decoder(dirichlet=config.get("dirichlet", False))

    alpha = torch.ones(3 * 21 * 21) * 0.5
    alpha = alpha.to(device)

    loss = Loss(
        p_I_scale=config["p_I_scale"],
        p_bg_scale=config["p_bg_scale"],
        p_p_scale=config["p_p_scale"],
        p_I=get_prior_distribution(config["p_I"]),
        p_bg=get_prior_distribution(config["p_bg"]),
        p_p=torch.distributions.dirichlet.Dirichlet(alpha),
        device=device,
    )

    q_bg = BackgroundDistribution(
        config["dmodel"],
        q_bg=getattr(torch.distributions, config["q_bg"]["distribution"]),
    )
    q_I = IntensityDistribution(
        config["dmodel"],
        q_I=getattr(torch.distributions, config["q_I"]["distribution"]),
    )

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
        max_epochs=config["epochs"],
        dmodel=config["dmodel"],
        batch_size=config["batch_size"],
        rank=config["rank"],
        C=config["C"],
        Z=config["Z"],
        H=config["H"],
        W=config["W"],
        lr=config["learning_rate"],
        images_dir=images_dir,
        dirichlet=config.get("dirichlet", False),
    )

    if resume_from_checkpoint:
        print(f"Loading weights from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint)
        integrator_model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Weights loaded successfully")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(experiment_dir, "checkpoints"),
        filename="integrator-{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=1,
    )

    progress_bar = TQDMProgressBar(refresh_rate=1)

    trainer = Trainer(
        max_epochs=config["epochs"],
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

    trainer.fit(integrator_model, data_module)

    return integrator_model, experiment_dir

def evaluate(model, data_module, experiment_dir, config):
    trainer = Trainer(
        accelerator=config["accelerator"],
        devices=1,
        num_nodes=1,
        precision=config["precision"],
    )

    print("Running evaluation on the full dataset...")
    predictions = trainer.predict(model, dataloaders=data_module.predict_dataloader())

    def process_predictions(predictions):
        processed = {k: [] for k in predictions[0].keys()}
        for batch in predictions:
            for k, v in batch.items():
                processed[k].append(v)
        return {k: torch.cat(v) for k, v in processed.items()}

    all_preds = process_predictions(predictions)

    output_refl_dir = os.path.join(experiment_dir, "out")
    os.makedirs(output_refl_dir, exist_ok=True)
    output_refl_file = os.path.join(output_refl_dir, "NN.refl")
    output_refl_file2 = os.path.join(output_refl_dir, "DIALS_sum_NN.refl")
    output_refl_file3 = os.path.join(output_refl_dir, "DIALS_sum_NN_subset.refl")

    input_refl_file = os.path.join(config["data_dir"], config["refl_file"])

    outwriter = OutWriter(
        all_preds,
        input_refl_file,
        output_refl_file,
        out_file_name2=output_refl_file2,
        out_file_name3=output_refl_file3,
    )

    sel = outwriter.write_output()

    utils.plot_intensities(
        nn_refl=output_refl_file,
        title="NNWeightedSum: vs. DIALSIntensity",
        dials_refl=input_refl_file,
        sel=sel,
        output_dir=output_refl_dir,
        encoder_type=config.get("encoder_type", "UnknownEncoder"),
        profile_type=config.get("profile_type", "UnknownProfile"),
        batch_size=config.get("batch_size", "UnknownBatchSize"),
        out_png_filename="I_weighted_sum.png",
        intensity_column="I_weighted_sum",
        save=True,
        display=False,
    )

    utils.plot_intensities(
        nn_refl=output_refl_file2,
        title="NNMaskedSum vs. DIALSIntensity",
        dials_refl=input_refl_file,
        sel=sel,
        output_dir=output_refl_dir,
        encoder_type=config.get("encoder_type", "UnknownEncoder"),
        profile_type=config.get("profile_type", "UnknownProfile"),
        batch_size=config.get("batch_size", "UnknownBatchSize"),
        out_png_filename="I_masked_sum.png",
        intensity_column="I_masked_sum",
        save=True,
        display=False,
    )

    utils.plot_intensities(
        nn_refl=output_refl_file2,
        title="NNProfileIntensity vs. DIALSIntensity",
        dials_refl=input_refl_file,
        sel=sel,
        output_dir=output_refl_dir,
        encoder_type=config.get("encoder_type", "UnknownEncoder"),
        profile_type=config.get("profile_type", "UnknownProfile"),
        batch_size=config.get("batch_size", "UnknownBatchSize"),
        out_png_filename="intensity_comparison.png",
        intensity_column="I_weighted_sum",
        save=True,
        display=False,
    )

    return sel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training and evaluation script for integrator model.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the config file.")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training.")
    parser.add_argument("--log_dir", type=str, default="logs/outputs", help="Directory where logs will be saved.")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on trained model")
    args = parser.parse_args()

    config = load_config(args.config)

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    if not args.evaluate:
        model, experiment_dir = train(config, resume_from_checkpoint=args.resume, log_dir=args.log_dir)
    else:
        if args.resume:
            weights_path = args.resume
            experiment_dir = os.path.dirname(os.path.dirname(weights_path))
        else:
            raise ValueError("Please provide a checkpoint file using --resume when using --evaluate")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Checkpoint file not found: {weights_path}")

        model = Integrator.load_from_checkpoint(weights_path)

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

    sel = evaluate(model, data_module, experiment_dir,config)

    config_copy_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f)

    if not args.evaluate:
        model_summary = str(model)
        summary_path = os.path.join(experiment_dir, "model_summary.txt")
        with open(summary_path, "w") as f:
            f.write(model_summary)

    print("Evaluation and output generation complete.")

