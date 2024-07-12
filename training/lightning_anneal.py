import torch
import random
import argparse
import pickle
import os
import polars as plrs
import numpy as np
from integrator.io import SimulatedData, SimulatedDataModule
from integrator.models import (
    Encoder,
    PoissonLikelihoodV2,
    DistributionBuilder,
    IntegratorModelSim,
)
from torch.utils.data import DataLoader
from rs_distributions import distributions as rsd
from integrator.layers import Standardize
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("high")


def main(args):
    print(f"anneal: {args.anneal}")
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)

    # Hyperparameters
    depth = args.depth
    dmodel = args.dmodel
    feature_dim = args.feature_dim
    beta = args.beta
    mc_samples = args.mc_samples
    max_size = args.max_size
    eps = args.eps
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs

    # Load training data
    weak_data = torch.load(args.weak_data_path)
    strong_data = torch.load(args.strong_data_path)

    merged_data = {}

    for key in weak_data:
        if key in strong_data:
            merged_data[key] = weak_data[key] + strong_data[key]
        else:
            merged_data[key] = weak_data[key]

    loaded_data_ = plrs.DataFrame(merged_data).sample(fraction=1, shuffle=True)

    num_voxels = [x.size(0) for x in loaded_data_["coords"]]
    max_voxel = np.max(num_voxels)
    min_voxel = np.min(num_voxels)

    # %%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Variational distributions
    intensity_dist = torch.distributions.gamma.Gamma
    background_dist = torch.distributions.gamma.Gamma
    prior_I = torch.distributions.exponential.Exponential(rate=torch.tensor(0.5))
    concentration = torch.tensor([1.0], device=device)
    rate = torch.tensor([1.0], device=device)
    prior_bg = torch.distributions.gamma.Gamma(concentration, rate)

    prior_I = torch.distributions.exponential.Exponential(rate=torch.tensor(0.01))
    p_I_scale = args.p_I_scale
    p_bg_scale = args.p_bg_scale

    concentration = torch.tensor([1.0])
    rate = torch.tensor([1])
    prior_bg = torch.distributions.gamma.Gamma(
        torch.tensor([1.0], device=device), torch.tensor([1], device=device)
    )
    p_bg_scale = 1

    data_module = SimulatedDataModule(loaded_data_, max_voxel, batch_size=batch_size)
    data_module.setup()

    train_loader_len = len(data_module.train_dataloader())

    standardization = Standardize(max_counts=train_loader_len)
    encoder = Encoder(depth, dmodel, feature_dim, dropout=None)
    distribution_builder = DistributionBuilder(
        dmodel, intensity_dist, background_dist, eps, beta
    )
    poisson_loss = PoissonLikelihoodV2(
        beta=beta,
        eps=eps,
        prior_I=prior_I,
        prior_bg=prior_bg,
        concentration=concentration,
        rate=rate,
        p_I_scale=p_I_scale,
        p_bg_scale=p_bg_scale,
    )

    total_steps = 1000 * train_loader_len
    model = IntegratorModelSim(
        encoder,
        distribution_builder,
        poisson_loss,
        standardization,
        min_voxel,
        total_steps=total_steps,
        n_cycle=args.n_cycle,
        lr=learning_rate,
        anneal=args.anneal,
    )

    logger = TensorBoardLogger(args.output_dir, name="integrator_model")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(args.output_dir, "checkpoints/"),
        filename="integrator-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    progress_bar = TQDMProgressBar(refresh_rate=20)

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices="auto",
        num_nodes=1,
        strategy="ddp",
        precision="16-mixed",
        accumulate_grad_batches=4,
        callbacks=[checkpoint_callback, progress_bar],
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)

    # Save weights
    torch.save(
        model.state_dict(), os.path.join(args.output_dir, "integrator_weights.pth")
    )

    results = {
        "train_preds": model.training_preds,
        "test_preds": model.validation_preds,
        "train_avg_loss": model.train_avg_loss,
        "test_avg_loss": model.validation_avg_loss,
    }

    with open(os.path.join(args.output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for IntegratorModel")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--n_cycle", type=int, default=4, help="Number of cycles")
    parser.add_argument("--depth", type=int, default=10, help="Depth of the encoder")
    parser.add_argument("--dmodel", type=int, default=64, help="Model dimension")
    parser.add_argument("--feature_dim", type=int, default=7, help="Feature dimension")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--anneal", action="store_true")
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Beta parameter for the Poisson likelihood",
    )
    parser.add_argument(
        "--mc_samples", type=int, default=100, help="Number of Monte Carlo samples"
    )
    parser.add_argument(
        "--max_size", type=int, default=1024, help="Maximum size for padding"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-5, help="Epsilon value for numerical stability"
    )
    parser.add_argument(
        "--weak_data_path", type=str, required=True, help="Path to the weak data file"
    )
    parser.add_argument(
        "--strong_data_path",
        type=str,
        required=True,
        help="Path to the strong data file",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to store the outputs"
    )
    parser.add_argument(
        "--p_I_scale",
        type=float,
        default=0.001,
        help="Intensity prior distribution weight",
    )
    parser.add_argument(
        "--p_bg_scale",
        type=float,
        default=0.001,
        help="Background prior distribution weight",
    )

    args = parser.parse_args()
    main(args)
