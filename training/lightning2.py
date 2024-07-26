# Run with:
# python lightning2.py --epochs 10  --dmodel 32 --batch_size 100 --output_dir ./ --learning_rate .001 --p_I_scale .0001 --p_bg_scale .0001 --subset_ratio 1

import torch
from dials.array_family import flex
import argparse
import pickle
import os
import numpy as np
from integrator.io import RotationDataModule
from integrator.models import (
    Encoder,
    PoissonLikelihoodV2,
    DistributionBuilder,
)
from rs_distributions import distributions as rsd
from integrator.layers import Standardize
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import polars as pl
from integrator.layers import Standardize
from integrator.models import IntegratorModel

torch.set_float32_matmul_precision("high")


def main(args):
    # Hyperparameters
    depth = args.depth
    dmodel = args.dmodel
    feature_dim = args.feature_dim
    dropout = args.dropout
    beta = 1.0
    mc_samples = args.mc_samples
    max_size = args.max_size
    eps = args.eps
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    subset_ratio = args.subset_ratio

    # Directory with .refl files
    shoebox_dir = "/n/holylabs/LABS/hekstra_lab/Users/laldama/integrator_/rotation_data_examples/data_temp/temp"
    # Set device to cpu or gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_module = RotationDataModule(
        shoebox_dir=shoebox_dir,
        batch_size=batch_size,
        subset_ratio=subset_ratio,
        num_workers=16,
    )
    data_module.setup()

    train_loader_len = len(data_module.train_dataloader())

    # Variational distributions

    intensity_dist = torch.distributions.gamma.Gamma

    background_dist = torch.distributions.gamma.Gamma

    prior_I = torch.distributions.exponential.Exponential(rate=torch.tensor(1.0))

    concentration = torch.tensor([1.0], device=device)

    rate = torch.tensor([1.0], device=device)

    prior_bg = torch.distributions.gamma.Gamma(concentration, rate)

    # Instantiate standardization, encoder, distribution builder, and likelihood
    standardization = Standardize(max_counts=train_loader_len)
    encoder = Encoder(depth, dmodel, feature_dim, dropout=dropout)
    distribution_builder = DistributionBuilder(
        dmodel, intensity_dist, background_dist, eps, beta
    )
    poisson_loss = PoissonLikelihoodV2(
        beta=beta,
        eps=eps,
        prior_I=prior_I,
        prior_bg=prior_bg,
        p_I_scale=args.p_I_scale,
        p_bg_scale=args.p_bg_scale,
    )

    total_steps = 1000 * train_loader_len
    print(train_loader_len)
    print(total_steps)

    model = IntegratorModel(
        encoder,
        distribution_builder,
        poisson_loss,
        standardization,
        total_steps=total_steps,
        n_cycle=args.n_cycle,
        lr=learning_rate,
        anneal=args.anneal,
    )

    logger = TensorBoardLogger(
        os.path.join(args.output_dir, "tb_logs"), name="integrator_model"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=os.path.join(args.output_dir, "checkpoints/"),
        filename="integrator-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    progress_bar = TQDMProgressBar(refresh_rate=1)

    trainer = Trainer(
        max_epochs=epochs,
        devices=2,
        num_nodes=1,
        accelerator="ddp",
        precision="16-mixed",
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        #strategy = DDPStrategy(find_unused_parameters=False)
        callbacks=[checkpoint_callback, progress_bar],
        logger=logger,
        log_every_n_steps=1,
    )

    trainer.fit(model, data_module)

    # %%
    # Code to store outputs

    # intensity prediction array
    intensity_preds = np.array(model.training_preds["q_I_mean"])

    # Reflection id array
    # refl_ids = np.array(model.training_preds["refl_id"])

    # Table ids
    tbl_ids = np.unique(np.array(model.training_preds["tbl_id"]))

    # Training predictions
    train_res_df = pl.DataFrame(
        {
            "tbl_id": model.training_preds["tbl_id"],
            "refl_id": model.training_preds["refl_id"],
            "q_I_mean": model.training_preds["q_I_mean"],
            "q_I_stddev": model.training_preds["q_I_stddev"],
        }
    )

    # Validation predictions
    val_res_df = pl.DataFrame(
        {
            "tbl_id": model.validation_preds["tbl_id"],
            "refl_id": model.validation_preds["refl_id"],
            "q_I_mean": model.validation_preds["q_I_mean"],
            "q_I_stddev": model.validation_preds["q_I_stddev"],
        }
    )

    # Concatenate train_res_df and val_res_df
    res_df = pl.concat([train_res_df, val_res_df])

    # Iterate over reflection id
    for tbl_id in tbl_ids:
        sel = np.asarray([False] * len(data_module.full_dataset.refl_tables[tbl_id]))

        filtered_df = res_df.filter(res_df["tbl_id"] == tbl_id)

        # Reflection ids
        reflection_ids = filtered_df["refl_id"].to_list()

        # Intensity predictions
        intensity_preds = filtered_df["q_I_mean"].to_list()
        intensity_stddev = filtered_df["q_I_stddev"].to_list()

        for id in reflection_ids:
            sel[id] = True

        refl_temp_tbl = data_module.full_dataset.refl_tables[tbl_id].select(
            flex.bool(sel)
        )

        refl_temp_tbl["intensity.sum.value"] = flex.double(intensity_preds)

        refl_temp_tbl["intensity.sum.variance"] = flex.double(intensity_stddev)

        # save the updated reflection table
        refl_temp_tbl.as_file(f"integrator_preds_{tbl_id}.refl")

    # Save weights
    torch.save(
        model.state_dict(), os.path.join(args.output_dir, "integrator_weights.pth")
    )

    # Function to recursively move tensors to CPU
    def move_to_cpu(data):
        if isinstance(data, dict):
            return {key: move_to_cpu(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [move_to_cpu(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(move_to_cpu(item) for item in data)
        elif torch.is_tensor(data):
            return data.cpu()
        else:
            return data

    results = {
        "train_preds": model.training_preds,
        "test_preds": model.validation_preds,
        "train_avg_loss": model.train_avg_loss,
        "test_avg_loss": model.validation_avg_loss,
    }
    results_cpu = move_to_cpu(results)

    with open(os.path.join(args.output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results_cpu, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for IntegratorModel")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--n_cycle",
        type=int,
        default=4,
        help="Number of cycles",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="Depth of the encoder",
    )
    parser.add_argument(
        "--dmodel",
        type=int,
        default=64,
        help="Model dimension",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=7,
        help="Feature dimension",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate",
    )
    parser.add_argument(
        "--anneal",
        action="store_true",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Beta parameter for the Poisson likelihood",
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=100,
        help="Number of Monte Carlo samples",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=1024,
        help="Maximum size for padding",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="Epsilon value for numerical stability",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store the outputs",
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
    parser.add_argument("--subset_ratio", type=float, default=0.1, help="Subset ratio")

    args = parser.parse_args()
    main(args)
