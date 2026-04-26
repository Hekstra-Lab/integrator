import pytest
import torch

from integrator.data_loaders.data_module import SIMULATED_COLS

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
            # ~10 % flagged for test
            ref[col] = torch.rand(N_SAMPLES) < 0.1
        elif col in ("refl_id", "refl_ids"):
            ref[col] = torch.arange(N_SAMPLES, dtype=torch.float32)
        else:
            ref[col] = torch.rand(N_SAMPLES)
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
            "name": "modela",
            "args": {
                "data_dim": "3d",
                "d": D,
                "h": H,
                "w": W,
                "lr": 0.001,
                "encoder_out": encoder_out,
                "mc_samples": 4,
                "weight_decay": 0.0,
                "renyi_scale": 0.0,
                "predict_keys": "default",
            },
        },
        "encoders": [
            {
                "name": "shoebox_encoder",
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
            },
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
                "name": "folded_normal",
                "args": {"in_features": encoder_out, "eps": 0.1},
            },
            "qi": {
                "name": "folded_normal",
                "args": {"in_features": encoder_out, "eps": 0.1},
            },
        },
        "loss": {
            "name": "default",
            "args": {
                "mc_samples": 4,
                "eps": 1e-6,
                "pi_cfg": {
                    "name": "gamma",
                    "params": {"concentration": 1.0, "rate": 0.5},
                    "weight": 0.01,
                },
                "pbg_cfg": {
                    "name": "gamma",
                    "params": {"concentration": 1.0, "rate": 0.5},
                    "weight": 0.01,
                },
                "pprf_cfg": {
                    "name": "dirichlet",
                    "params": {"concentration": 1.0, "shape": [D, H, W]},
                    "weight": 0.01,
                },
            },
        },
        "data_loader": {
            "name": "simulated_data",
            "args": {
                "data_dir": str(sim_data_dir),
                "batch_size": 10,
                "val_split": 0.2,
                "test_split": 0.1,
                "num_workers": 0,
                "include_test": False,
                "subset_size": 100,
                "cutoff": None,
                "shoebox_file_names": {
                    "counts": "counts.pt",
                    "masks": "masks.pt",
                    "stats": "stats_anscombe.pt",
                    "reference": "reference.pt",
                    "standardized_counts": None,
                },
                "D": D,
                "H": H,
                "W": W,
                "anscombe": False,
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


@pytest.fixture()
def per_bin_data_dir(tmp_path, sim_reference):
    """Write simulated data + per-bin prior buffers for PerBinLoss tests."""
    torch.manual_seed(0)
    n_groups = 5

    counts = torch.poisson(torch.full((N_SAMPLES, N_PIXELS), 5.0))
    masks = torch.ones(N_SAMPLES, N_PIXELS)
    stats = torch.tensor([counts.mean().item(), counts.var().item()])

    # assign group labels
    sim_reference["group_label"] = torch.randint(
        0, n_groups, (N_SAMPLES,)
    ).float()

    torch.save(counts, tmp_path / "counts.pt")
    torch.save(masks, tmp_path / "masks.pt")
    torch.save(stats, tmp_path / "stats_anscombe.pt")
    torch.save(sim_reference, tmp_path / "reference.pt")

    # per-bin prior buffers
    torch.save(torch.rand(n_groups) + 0.1, tmp_path / "tau_per_group.pt")
    torch.save(torch.rand(n_groups) + 0.1, tmp_path / "bg_rate_per_group.pt")

    return tmp_path


@pytest.fixture()
def per_bin_sim_config(per_bin_data_dir):
    """Config for HierarchicalIntegratorB + PerBinLoss."""
    import copy

    encoder_out = 64

    intensity_enc = {
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

    return {
        "integrator": {
            "name": "hierarchicalB",
            "args": {
                "data_dim": "3d",
                "d": D,
                "h": H,
                "w": W,
                "lr": 0.001,
                "encoder_out": encoder_out,
                "mc_samples": 4,
                "weight_decay": 0.0,
                "renyi_scale": 0.0,
                "predict_keys": [
                    "refl_ids",
                    "is_test",
                    "qi_mean",
                    "qi_var",
                    "qbg_mean",
                    "qbg_var",
                    "group_label",
                    "tau_per_refl",
                    "intensity",
                    "background",
                ],
            },
        },
        "encoders": [
            {
                "name": "shoebox_encoder",
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
            copy.deepcopy(intensity_enc),  # k
            copy.deepcopy(intensity_enc),  # r
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
                "name": "gammaC",
                "args": {"in_features": encoder_out, "eps": 1e-6},
            },
            "qi": {
                "name": "gammaC",
                "args": {"in_features": encoder_out, "eps": 1e-6},
            },
        },
        "loss": {
            "name": "per_bin",
            "args": {
                "mc_samples": 4,
                "eps": 1e-6,
                "tau_per_group": str(per_bin_data_dir / "tau_per_group.pt"),
                "bg_rate_per_group": str(
                    per_bin_data_dir / "bg_rate_per_group.pt"
                ),
                "pprf_weight": 0.005,
                "pbg_weight": 0.5,
                "pi_weight": 1.0,
            },
        },
        "data_loader": {
            "name": "simulated_data",
            "args": {
                "data_dir": str(per_bin_data_dir),
                "batch_size": 10,
                "val_split": 0.2,
                "test_split": 0.1,
                "num_workers": 0,
                "include_test": False,
                "subset_size": 100,
                "cutoff": None,
                "shoebox_file_names": {
                    "counts": "counts.pt",
                    "masks": "masks.pt",
                    "stats": "stats_anscombe.pt",
                    "reference": "reference.pt",
                    "standardized_counts": None,
                },
                "D": D,
                "H": H,
                "W": W,
                "anscombe": False,
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
