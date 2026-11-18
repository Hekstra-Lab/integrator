import atexit
import shutil
import tempfile

import torch
import yaml

from integrator.data_loaders.data_module import DEFAULT_DS_COLS


def _make_temp_data_dir():
    path = tempfile.mkdtemp()
    atexit.register(lambda: shutil.rmtree(path, ignore_errors=True))
    return path


def _generate_data_files(
    data_dir: str,
    depth: int = 3,
    height: int = 21,
    width: int = 21,
    dataset_size: int = 1000,
) -> dict[str, str]:
    n_pixels = depth * height * width

    counts = torch.randint(
        0, 10, (dataset_size, n_pixels), dtype=torch.float32
    )
    masks = torch.randint(0, 2, (dataset_size, n_pixels))
    stats = torch.tensor([0.0, 1.0])
    concentration = counts.mean(0)

    metadata = {}
    for c in DEFAULT_DS_COLS:
        metadata[c] = torch.randn(dataset_size)
    metadata["refl_ids"] = torch.randint(
        low=0, high=10000, size=(dataset_size,)
    )

    fnames = {
        "data_dir": data_dir,
        "masks": data_dir + "/masks.pt",
        "counts": data_dir + "/counts.pt",
        "reference": data_dir + "/reference.pt",
        "stats": data_dir + "/stats.pt",
        "concentration": data_dir + "/concentration.pt",
    }

    torch.save(counts, fnames["counts"])
    torch.save(masks, fnames["masks"])
    torch.save(metadata, fnames["reference"])
    torch.save(stats, fnames["stats"])
    torch.save(concentration, fnames["concentration"])

    return fnames


def _build_test_config(data_dir: str, batch_size: int = 100) -> dict:
    return {
        "integrator": {
            "name": "integrator",
            "args": {
                "data_dim": "3d",
                "d": 3,
                "h": 21,
                "w": 21,
            },
        },
        "encoders": [
            {
                "name": "profile_encoder",
                "args": {
                    "data_dim": "3d",
                    "in_channels": 1,
                    "input_shape": [3, 21, 21],
                    "encoder_out": 64,
                    "conv1_out_channels": 64,
                    "conv1_kernel_size": [1, 3, 3],
                    "conv1_padding": [0, 1, 1],
                    "norm1_num_groups": 4,
                    "pool_kernel_size": [1, 2, 2],
                    "pool_stride": [1, 2, 2],
                    "conv2_out_channels": 128,
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
                    "encoder_out": 64,
                    "conv1_out_channels": 64,
                    "conv1_kernel_size": [1, 3, 3],
                    "conv1_padding": [0, 1, 1],
                    "norm1_num_groups": 4,
                    "pool_kernel_size": [1, 2, 2],
                    "pool_stride": [1, 2, 2],
                    "conv2_out_channels": 128,
                    "conv2_kernel_size": [3, 3, 3],
                    "conv2_padding": [0, 0, 0],
                    "norm2_num_groups": 4,
                    "conv3_out_channels": 256,
                    "conv3_kernel_size": [3, 3, 3],
                    "conv3_padding": [1, 1, 1],
                    "norm3_num_groups": 8,
                },
            },
            {
                "name": "intensity_encoder",
                "args": {
                    "data_dim": "3d",
                    "in_channels": 1,
                    "encoder_out": 64,
                    "conv1_out_channels": 64,
                    "conv1_kernel_size": [1, 3, 3],
                    "conv1_padding": [0, 1, 1],
                    "norm1_num_groups": 4,
                    "pool_kernel_size": [1, 2, 2],
                    "pool_stride": [1, 2, 2],
                    "conv2_out_channels": 128,
                    "conv2_kernel_size": [3, 3, 3],
                    "conv2_padding": [0, 0, 0],
                    "norm2_num_groups": 4,
                    "conv3_out_channels": 256,
                    "conv3_kernel_size": [3, 3, 3],
                    "conv3_padding": [1, 1, 1],
                    "norm3_num_groups": 8,
                },
            },
        ],
        "surrogates": {
            "qp": {
                "name": "dirichlet",
                "args": {"eps": 1e-6, "in_features": 64, "sbox_shape": [3, 21, 21]},
            },
            "qbg": {"name": "gammaA", "args": {"eps": 1e-6, "in_features": 64}},
            "qi": {"name": "gammaA", "args": {"eps": 1e-6, "in_features": 64}},
        },
        "loss": {
            "name": "default",
            "args": {
                "mc_samples": 100,
                "eps": 1e-6,
                "pprf_cfg": {
                    "name": "dirichlet",
                    "params": {"concentration": 0.01, "shape": [3, 21, 21]},
                    "weight": 0.001,
                },
                "pbg_cfg": {
                    "name": "gamma",
                    "params": {"concentration": 0.01, "rate": 0.001},
                    "weight": 0.5,
                },
                "pi_cfg": {
                    "name": "gamma",
                    "params": {"concentration": 0.01, "rate": 0.001},
                    "weight": 0.5,
                },
            },
        },
        "data_loader": {
            "name": "default",
            "args": {
                "data_dir": data_dir,
                "batch_size": batch_size,
                "validation_split": 0.3,
                "test_split": 0.0,
                "num_workers": 0,
                "include_test": False,
                "subset_size": 100,
                "resolution_cutoff": None,
                "shoebox_file_names": {
                    "data_dir": data_dir,
                    "counts": "counts.pt",
                    "masks": "masks.pt",
                    "stats": "stats.pt",
                    "reference": "reference.pt",
                    "standardized_counts": None,
                },
                "D": 3,
                "H": 21,
                "W": 21,
                "anscombe": True,
            },
        },
        "trainer": {
            "max_epochs": 10,
            "accelerator": "cpu",
            "devices": 1,
            "logger": True,
            "precision": "32",
            "check_val_every_n_epoch": 2,
            "log_every_n_steps": 1,
            "deterministic": False,
            "enable_checkpointing": True,
        },
        "logger": {"d": 3, "h": 21, "w": 21},
        "output": {"refl_file": "reflections_.refl"},
    }


data_dir = _make_temp_data_dir()
data_files = _generate_data_files(data_dir=data_dir)
cfg_dict = _build_test_config(data_dir=data_dir)
cfg_path = data_dir + "/test_yaml.yaml"

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg_dict, f, sort_keys=False)
