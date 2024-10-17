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
    CNNResNet,
    FcResNet,
    MVNProfile,
    Integrator,
    SoftmaxProfile,
    BackgroundDistribution,
    IntensityDistribution,
    DirichletProfile,
    Loss,
    Decoder,
)
from integrator.utils import (
    OutWriter,
    get_profile,
    get_prior_distribution,
    get_experiment_counter,
    generate_experiment_dir,
    get_encoder,
    get_most_recent_checkpoint,
)

import integrator.utils as utils


torch.set_float32_matmul_precision("high")


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def train(config, resume_from_checkpoint=None, log_dir="logs/outputs"):
    """

    Args:
        config ():
        resume_from_checkpoint ():
        log_dir ():

    Returns:

    """
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
    """

    Args:
        model ():
        data_module ():
        experiment_dir ():
        config ():

    Returns:

    """
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
    parser = argparse.ArgumentParser(
        description="Training and evaluation script for integrator model."
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
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # Always run training (or resume from checkpoint)
    model, experiment_dir = train(
        config, resume_from_checkpoint=args.resume, log_dir=args.log_dir
    )

    # Find the most recent checkpoint file
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    final_weights_path = get_most_recent_checkpoint(checkpoint_dir)

    if final_weights_path is None:
        raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")

    print(f"Using checkpoint: {final_weights_path}")

    # Recreate the model components
    encoder = get_encoder(config)
    profile = get_profile(config)
    q_bg = BackgroundDistribution(
        config["dmodel"],
        q_bg=getattr(torch.distributions, config["q_bg"]["distribution"]),
    )
    q_I = IntensityDistribution(
        config["dmodel"],
        q_I=getattr(torch.distributions, config["q_I"]["distribution"]),
    )
    decoder = Decoder(dirichlet=config.get("dirichlet", False))
    loss = Loss(
        p_I_scale=config["p_I_scale"],
        p_bg_scale=config["p_bg_scale"],
        p_p_scale=config["p_p_scale"],
        p_I=get_prior_distribution(config["p_I"]),
        p_bg=get_prior_distribution(config["p_bg"]),
        p_p=torch.distributions.dirichlet.Dirichlet(torch.ones(3 * 21 * 21) * 0.5),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    standardize = Standardize()

    # Load the model from checkpoint
    model = Integrator.load_from_checkpoint(
        final_weights_path,
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
        images_dir=os.path.join(experiment_dir, "out", "images"),
        dirichlet=config.get("dirichlet", False),
    )
    model.eval()

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

    # Run evaluation
    sel = evaluate(model, data_module, experiment_dir, config)

    # Save config and model summary
    config_copy_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f)

    model_summary = str(model)
    summary_path = os.path.join(experiment_dir, "model_summary.txt")
    with open(summary_path, "w") as f:
        f.write(model_summary)

    print("Training, evaluation, and output generation complete.")
