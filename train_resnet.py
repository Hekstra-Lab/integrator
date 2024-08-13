import torch
import rs_distributions.distributions as rsd

from integrator.models.integrator_mvn_3d_cnn import BackgroundIndicator
from integrator.models.mvn_3d_cnn import IntegratorCNN
from integrator.utils import OutWriter

import argparse
import pickle
import os
import json

torch.set_float32_matmul_precision("medium")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "hyperparameters.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    model = IntegratorCNN(
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
        bg_indicator=args.bg_indicator,
        #        num_workers = 4
    )

    data_module = model.LoadData()

    trainer, integrator_model = model.BuildModel(
        use_bn=args.use_bn,
        conv1_in_channel=args.conv1_in_channel,
        conv1_out_channel=args.conv1_out_channel,
        conv1_kernel_size=args.conv1_kernel_size,
        conv1_stride=args.conv1_stride,
        conv1_padding=args.conv1_padding,
        layer1_num_blocks=args.layer1_num_blocks,
        conv2_out_channel=args.conv2_out_channel,
        layer2_stride=args.layer2_stride,
        layer2_num_blocks=args.layer2_num_blocks,
        maxpool_in_channel=args.maxpool_in_channel,
        maxpool_out_channel=args.maxpool_out_channel,
        maxpool_kernel_size=args.maxpool_kernel_size,
        maxpool_stride=args.maxpool_stride,
        maxpool_padding=args.maxpool_padding,
        precision=args.precision,
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
        default=None,
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
        default=3,
    )

    parser.add_argument(
        "--refl_file_name",
        default="./data/hewl_816/reflections_.refl",
        type=str,
    )

    parser.add_argument(
        "--out_filename",
        type=str,
        default="out.refl",
    )

    parser.add_argument("--out_dir", type=str, default="./out/out_resnet/")

    parser.add_argument(
        "--bg_indicator",
        type=bool,
        default=False,
        help="Background indicator",
    )
    parser.add_argument(
        "--use_bn",
        type=bool,
        default=False,
        help="Use batch normalization",
    )
    parser.add_argument(
        "--conv1_in_channel",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--conv1_out_channel",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--conv1_kernel_size",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--conv1_stride",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--conv1_padding",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--layer1_num_blocks",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--conv2_out_channel",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--layer2_stride",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--layer2_num_blocks",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--maxpool_in_channel",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--maxpool_out_channel",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--maxpool_kernel_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--maxpool_stride",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--maxpool_padding",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
    )

    args = parser.parse_args()

    main(args)
