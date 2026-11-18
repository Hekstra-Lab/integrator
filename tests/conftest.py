import pytest
import torch

SIMULATED_COLS = [
    "shoebox_median",
    "shoebox_var",
    "shoebox_mean",
    "shoebox_min",
    "shoebox_max",
    "intensity",
    "background",
    "refl_ids",
    "is_test",
    "group_label",
    "profile_group_label",
    "d",
]

N_SAMPLES = 500
D, H, W = 3, 21, 21
N_PIXELS = D * H * W  # 1323


@pytest.fixture()
def sim_reference():
    """Dict with all SIMULATED_COLS keys as tensors (N_SAMPLES,)."""
    torch.manual_seed(0)
    ref = {}
    for col in SIMULATED_COLS:
        if col == "is_test":
            ref[col] = torch.rand(N_SAMPLES) < 0.1
        elif col in ("refl_id", "refl_ids"):
            ref[col] = torch.arange(N_SAMPLES, dtype=torch.float32)
        elif col == "group_label":
            ref[col] = torch.randint(0, 5, (N_SAMPLES,))
        else:
            ref[col] = torch.rand(N_SAMPLES)
    ref["d"] = torch.rand(N_SAMPLES) * 5.0 + 1.0
    return ref


@pytest.fixture()
def sim_data_dir(tmp_path, sim_reference):
    """Write counts.pt, masks.pt, stats_anscombe.pt, reference.pt to tmp dir."""
    torch.manual_seed(0)

    counts = torch.poisson(torch.full((N_SAMPLES, N_PIXELS), 5.0))
    masks = torch.ones(N_SAMPLES, N_PIXELS)
    stats = torch.tensor([counts.mean().item(), counts.var().item()])
    reference = sim_reference

    torch.save(counts, tmp_path / "counts.pt")
    torch.save(masks, tmp_path / "masks.pt")
    torch.save(stats, tmp_path / "stats_anscombe.pt")
    torch.save(reference, tmp_path / "reference.pt")

    return tmp_path


@pytest.fixture()
def sim_config(sim_data_dir):
    """Config dict matching the current schema, CPU-only, minimal epochs."""
    encoder_out = 64

    return {
        "integrator": {
            "name": "hierarchical",
            "args": {
                "data_dim": "3d",
                "d": D,
                "h": H,
                "w": W,
                "encoder_out": encoder_out,
                "mc_samples": 4,
                "predict_keys": "default",
            },
        },
        "optimizer": {"lr": 0.001, "weight_decay": 0.0},
        "encoders": [
            {
                "name": "profile_encoder",
                "args": {
                    "data_dim": "3d",
                    "in_channels": 1,
                    "input_shape": [D, H, W],
                    "encoder_out": encoder_out,
                    "conv1_out_channels": 16,
                    "conv1_kernel_size": [1, 3, 3],
                    "conv1_padding": [0, 1, 1],
                    "norm1_num_groups": 4,
                    "pool_kernel_size": [1, 2, 2],
                    "pool_stride": [1, 2, 2],
                    "conv2_out_channels": 32,
                    "conv2_kernel_size": [3, 3, 3],
                    "conv2_padding": [0, 0, 0],
                    "norm2_num_groups": 4,
                },
            },
            # modelc needs 4 intensity encoders: k_i, r_i, k_bg, r_bg
            *[
                {
                    "name": "intensity_encoder",
                    "args": {
                        "data_dim": "3d",
                        "in_channels": 1,
                        "encoder_out": encoder_out,
                        "conv1_out_channels": 16,
                        "conv1_kernel_size": [3, 3, 3],
                        "conv1_padding": [1, 1, 1],
                        "norm1_num_groups": 4,
                        "pool_kernel_size": [1, 2, 2],
                        "pool_stride": [1, 2, 2],
                        "conv2_out_channels": 32,
                        "conv2_kernel_size": [3, 3, 3],
                        "conv2_padding": [0, 0, 0],
                        "norm2_num_groups": 4,
                        "conv3_out_channels": 64,
                        "conv3_kernel_size": [3, 3, 3],
                        "conv3_padding": [1, 1, 1],
                        "norm3_num_groups": 8,
                    },
                }
                for _ in range(4)
            ],
        ],
        "surrogates": {
            "qp": {
                "name": "learned_basis_profile",
                "args": {
                    "input_dim": encoder_out,
                    "latent_dim": 8,
                    "output_dim": N_PIXELS,
                    "init_std": 0.5,
                },
            },
            "qbg": {
                "name": "gamma",
                "args": {
                    "reparameterization": "shape_rate",
                    "in_features": encoder_out,
                    "eps": 0.1,
                },
            },
            "qi": {
                "name": "gamma",
                "args": {
                    "reparameterization": "shape_rate",
                    "in_features": encoder_out,
                    "eps": 0.1,
                },
            },
        },
        "loss": {
            "name": "monochromatic_wilson",
            "args": {
                "bg_rate": 1.0,
                "bg_concentration": 1.0,
                "n_bins": 5,
            },
        },
        "data_loader": {
            "name": "rotation_data",
            "args": {
                "data_dir": str(sim_data_dir),
                "batch_size": 10,
                "validation_split": 0.2,
                "num_workers": 0,
                "include_test": False,
                "subset_size": 100,
                "resolution_cutoff": None,
                "shoebox_file_names": {
                    "counts": "counts.pt",
                    "masks": "masks.pt",
                    "stats": "stats_anscombe.pt",
                    "reference": "reference.pt",
                },
                "transform": "standardization",
            },
        },
        "trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
            "precision": "32",
            "check_val_every_n_epoch": 1,
            "log_every_n_steps": 1,
            "deterministic": False,
            "enable_checkpointing": False,
        },
    }


@pytest.fixture()
def sim_data_dir_anscombe(tmp_path, sim_reference):
    """Like sim_data_dir but saves Anscombe-transformed stats in stats_anscombe.pt."""
    torch.manual_seed(0)

    counts = torch.poisson(torch.full((N_SAMPLES, N_PIXELS), 5.0))
    masks = torch.ones(N_SAMPLES, N_PIXELS)
    # Mask out first pixel for all samples to test mask behaviour
    masks[:, 0] = 0.0

    anscombe_transformed = 2 * (counts + 0.375).sqrt()
    stats = torch.tensor(
        [anscombe_transformed.mean().item(), anscombe_transformed.var().item()]
    )

    torch.save(counts, tmp_path / "counts.pt")
    torch.save(masks, tmp_path / "masks.pt")
    torch.save(stats, tmp_path / "stats_anscombe.pt")
    torch.save(sim_reference, tmp_path / "reference.pt")

    return tmp_path
