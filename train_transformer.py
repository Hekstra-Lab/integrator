import torch
import rs_distributions.distributions as rsd

from integrator.models.integrator_mvn_3d_cnn import BackgroundIndicator
from integrator.models.mvn_3d_transformer import IntegratorTransformer
from integrator.utils import OutWriter

import argparse
import os
import pickle
import json

torch.set_float32_matmul_precision("medium")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "hyperparameters.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    model = IntegratorTransformer(
        depth=args.depth,
        dmodel=args.dmodel,
        feature_dim=args.feature_dim,
        dropout=args.dropout,
        beta=args.beta,
        mc_samples=args.mc_samples,
        max_size=args.max_size,
        eps=args.eps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        intensity_dist=torch.distributions.gamma.Gamma,
        background_dist=torch.distributions.gamma.Gamma,
        prior_I=torch.distributions.exponential.Exponential(rate=torch.tensor(1.0)),
        prior_bg=torch.distributions.exponential.Exponential(rate=torch.tensor(1.0)),
        device=args.device,
        shoebox_file=args.shoebox_file,
        metadata_file=args.metadata_file,
        dead_pixel_mask_file=args.dead_pixel_mask_file,
        subset_size=args.subset_size,
        p_I_scale=args.p_I_scale,
        p_bg_scale=args.p_bg_scale,
        num_components=args.num_components,
        bg_indicator=None,
        patch_size=7,
        img_size=21,
        num_hiddens=24,
        mlp_num_hiddens=48,
        num_heads=2,
        num_blks=2,
        emb_dropout=0.1,
        blk_dropout=0.1,
        lr=0.1,
    )

    data_module = model.LoadData(
        val_split=args.val_split,
        test_split=args.test_split,
    )
    trainer, integrator_model = model.BuildModel(
            img_size=args.img_size,
            patch_size=args.patch_size,
            num_hiddens=args.num_hiddens,
            mlp_num_hiddens=args.mlp_num_hiddens,
            num_heads=args.num_heads,
            num_blks=args.num_blks,
            emb_dropout=args.emb_dropout,
            blk_dropout=args.blk_dropout,
            lr=args.lr,
            )
    trainer.fit(integrator_model, data_module)

    outwriter = OutWriter(
        integrator_model,
        args.refl_file_name,
        os.path.join(args.out_dir, args.out_filename),
    )
    outwriter.write_output()

    torch.save(
        integrator_model.state_dict(),
        os.path.join(args.out_dir, "integrator_weights.pth"),
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
        "train_avg_loss": integrator_model.train_avg_loss,
        "test_avg_loss": integrator_model.validation_avg_loss,
    }

    results_cpu = move_to_cpu(results)

    with open(os.path.join(args.out_dir, "results.pkl"), "wb") as f:
        pickle.dump(results_cpu, f)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--depth",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--dmodel",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--intensity_dist",
        type=torch.distributions.gamma.Gamma,
    )
    parser.add_argument(
        "--background_dist",
        type=torch.distributions.gamma.Gamma,
    )
    parser.add_argument(
        "--prior_I",
        type=torch.distributions.exponential.Exponential,
    )
    parser.add_argument(
        "--prior_bg",
        type=torch.distributions.exponential.Exponential,
    )
    parser.add_argument(
        "--device",
        type=torch.device,
    )
    parser.add_argument(
        "--shoebox_file",
        type=str,
        default="./data/hewl_816/samples.pt",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="./data/hewl_816/metadata.pt",
    )
    parser.add_argument(
        "--dead_pixel_mask_file",
        type=str,
        default="./data/hewl_816/masks.pt",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=410000,
    )
    parser.add_argument(
        "--p_I_scale",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--p_bg_scale",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--num_components",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--refl_file_name",
        default="./data/hewl_816/reflections_.refl",
        type=str,
    )
    parser.add_argument(
        "--out_filename",
        default="out.refl",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./out/out_transformer/",
    )
    parser.add_argument(
        "--bg_indicator",
        type=BackgroundIndicator(dmodel=64),
        help="Background indicator",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.3,
        help="Validation split",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Test split",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=21,
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--num_hiddens",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--mlp_num_hiddens",
        type=int,
        default=48,
    )
    parser.add_argument(
        "--num_blks",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--emb_dropout",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--blk_dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
    )



    args = parser.parse_args()

    main(args)
