import torch
import rs_distributions.distributions as rsd

from integrator.models.integrator_mvn_3d_cnn import BackgroundIndicator
from integrator.models.mvn_3d_transformer import IntegratorTransformer
from integrator.utils import OutWriter

import argparse
import pickle
import os


def main(args):
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
        prior_I=torch.distributions.exponential.Exponential(rate=torch.tensor(0.05)),
        prior_bg=rsd.FoldedNormal(0, 0.1),
        device=args.device,
        shoebox_file=args.shoebox_file,
        metadata_file=args.metadata_file,
        dead_pixel_mask_file=args.dead_pixel_mask_file,
        subset_size=args.subset_size,
        p_I_scale=args.p_I_scale,
        p_bg_scale=args.p_bg_scale,
        num_components=args.num_components,
        bg_indicator=BackgroundIndicator(dmodel=args.dmodel),
    )

    data_module = model.LoadData()
    trainer, integrator_model = model.BuildModel()
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
        "train_preds": integrator_model.training_preds,
        "test_preds": integrator_model.validation_preds,
        "train_avg_loss": integrator_model.train_avg_loss,
        "test_avg_loss": integrator_model.validation_avg_loss,
    }

    results_cpu = move_to_cpu(results)

    with open(os.path.join(args.out_dir, "results.pkl"), "wb") as f:
        pickle.dump(results_cpu, f)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--dmodel", type=int, default=64)
    parser.add_argument("--feature_dim", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--mc_samples", type=int, default=100)
    parser.add_argument("--max_size", type=int, default=1024)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--intensity_dist", type=torch.distributions.gamma.Gamma)
    parser.add_argument("--background_dist", type=torch.distributions.gamma.Gamma)
    parser.add_argument("--prior_I", type=torch.distributions.exponential.Exponential)
    parser.add_argument("--prior_bg", type=rsd.FoldedNormal)
    parser.add_argument("--device", type=torch.device)
    parser.add_argument("--shoebox_file", type=str, default="./samples.pt")
    parser.add_argument("--metadata_file", type=str, default="./metadata.pt")
    parser.add_argument("--dead_pixel_mask_file", type=str, default="./masks.pt")
    parser.add_argument("--subset_size", type=int, default=10)
    parser.add_argument("--p_I_scale", type=float, default=0.0001)
    parser.add_argument("--p_bg_scale", type=float, default=0.0001)
    parser.add_argument("--num_components", type=int, default=5)
    parser.add_argument("--refl_file_name", type=str)
    parser.add_argument("--out_filename", type=str)
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument(
        "--bg_indicator",
        type=BackgroundIndicator(dmodel=128),
        help="Background indicator",
    )
    args = parser.parse_args()

    main(args)


# %%
# # Model Specifications
model = IntegratorTransformer(
    depth=10,
    dmodel=64,
    feature_dim=7,
    dropout=0.5,
    beta=1.0,
    mc_samples=100,
    max_size=1024,
    eps=1e-5,
    batch_size=10,
    learning_rate=0.001,
    epochs=10,
    intensity_dist=torch.distributions.gamma.Gamma,
    background_dist=torch.distributions.gamma.Gamma,
    prior_I=torch.distributions.exponential.Exponential(rate=torch.tensor(0.05)),
    prior_bg=rsd.FoldedNormal(0, 0.1),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    shoebox_file="./samples.pt",
    metadata_file="./metadata.pt",
    dead_pixel_mask_file="./masks.pt",
    subset_size=100,
    p_I_scale=0.0001,
    p_bg_scale=0.0001,
    num_components=3,
    bg_indicator=None,
)

# Load Data
data_module = model.LoadData()

# Build the model and trainer
trainer, integrator_model = model.BuildModel()

# Train the model
trainer.fit(integrator_model, data_module)


# Write outputs
outwriter = OutWriter(
    integrator_model, "reflections_.refl", "integrator_preds_test_2024-08-08.refl"
)

outwriter.write_output()
