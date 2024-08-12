import torch
from rs_distributions import distributions as rsd
from integrator.models.mixture_model_3d_mvn import MixtureModel3DMVN
from integrator.models.mixture_model_3d_mvn import BackgroundIndicator
import argparse
import os
import pickle
from integrator.utils import OutWriter
import json

def main(args):

    os.makedirs(args.out_dir,exist_ok=True)

    with open(os.path.join(args.out_dir,'hyperparameters.json'),'w') as f:
        json.dump(vars(args),f,indent=4)

    model = MixtureModel3DMVN(
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
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        shoebox_file = args.shoebox_file,
        metadata_file = args.metadata_file,
        dead_pixel_mask = args.dead_pixel_mask,
        subset_size=args.subset_size,
        p_I_scale=args.p_I_scale,
        p_bg_scale=args.p_bg_scale,
        num_components=args.num_components,
        bg_indicator=None,
    )

    data_module = model.LoadData()
    trainer, integrator_model = model.BuildModel()
    trainer.fit(integrator_model, data_module)

    # Write outputs to reflection table
    outwriter = OutWriter(
        integrator_model,
        args.refl_file_name,
        os.path.join(args.out_dir, args.out_filename),
    )
    outwriter.write_output()

    # Save weights
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
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
        default=8,
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
        "--out_dir",
        type=str,
        required=True,
        default='./out/out_mlp/',
        help="Directory to store the outputs",
    )

    parser.add_argument(
        "--p_I_scale",
        type=float,
        default=0.0001,
        help="Intensity prior distribution weight",
    )
    
    parser.add_argument(
        "--p_bg_scale",
        type=float,
        default=0.0001,
        help="Background prior distribution weight",
    )

    parser.add_argument(
            "--subset_ratio", 
        type=float, 
        default=1, 
        help="Subset ratio"
            )

    parser.add_argument(
        "--num_components",
        type=int,
        default=1,
        help="Number of mixture components",
    )

    parser.add_argument(
        "--refl_file_name",
        type=str,
        default='./data/hewl_816/reflections_.refl',
        help="Reflection file name",
    )

    parser.add_argument(
        "--out_filename",
        type=str,
        default='out.refl',
        help="Output filename",
    )

    parser.add_argument(
        "--subset_size",
        type=int,
        default=410000,
        help="Cardinatlity of data subset",
        )

    parser.add_argument(
        "--bg_indicator",
        type=BackgroundIndicator(),
        default = None,
        help="Background indicator",
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
        "--dead_pixel_mask", 
        type=str, 
        default="./data/hewl_816/masks.pt",
        )

    args = parser.parse_args()
    main(args)

# %%

# arg_defaults = {
    # "learning_rate": 0.001,
    # "batch_size": 10,
    # "epochs": 1000,
    # "n_cycle": 4,
    # "depth": 10,
    # "dmodel": 64,
    # "feature_dim": 7,
    # "dropout": None,
    # "anneal": False,
    # "beta": 1.0,
    # "mc_samples": 100,
    # "max_size": 1024,
    # "eps": 1e-5,
    # "out_dir": "./",
    # "p_I_scale": 0.0001,
    # "p_bg_scale": 0.0001,
    # "subset_ratio": 0.1,
    # "num_components": 3,
    # "subset_size": 2,
    # "bg_indicator": BackgroundIndicator(),
# }

# args = argparse.Namespace()

# for key, value in arg_defaults.items():
    # if not hasattr(args, key):
        # setattr(args, key, value)

# main(args)


# %%

# # Model Specifications
# model = MixtureModel3DMVN(
    # depth=10,
    # dmodel=32,
    # feature_dim=7,
    # dropout=0.5,
    # beta=1.0,
    # mc_samples=100,
    # max_size=1024,
    # eps=1e-5,
    # batch_size=10,
    # learning_rate=0.001,
    # epochs=10,
    # intensity_dist=torch.distributions.gamma.Gamma,
    # background_dist=torch.distributions.gamma.Gamma,
    # prior_I=torch.distributions.exponential.Exponential(rate=torch.tensor(0.05)),
    # prior_bg=rsd.FoldedNormal(0, 0.1),
    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # shoebox_file="./samples.pt",
    # metadata_file="./metadata.pt",
    # dead_pixel_mask_file="./masks.pt",
    # subset_size=100,
    # p_I_scale=0.0001,
    # p_bg_scale=0.0001,
    # num_components=3,
    # bg_indicator=BackgroundIndicator(),
# )

# # Load Data
# data_module = model.LoadData()

# # Build the model and trainer
# trainer, integrator_model = model.BuildModel()

# # Train the model
# trainer.fit(integrator_model, data_module)

# # Write outputs
# outwriter = OutWriter(
    # integrator_model, "reflections_.refl", "integrator_preds_test_2024-08-08.refl"
# )
# outwriter.write_output()
