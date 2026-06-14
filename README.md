# Integrator
An amortized variational inference model to integrate diffraction data.

## Prerequisites

The installation requires a python environment manager. 
I use [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), and the following instructions assume it. 
Another popular manager is [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), but I do not use it nor have experience with it. 

## Installation

The project depends on DIALS and laue-DIALS alongside PyTorch.
DIALS (and dxtbx/cctbx) and PyTorch must be installed from conda-forge.
Pick the environment file that matches your machine:

| File                   | Env               | For                                    |
| ---                    | ---               | ---                                    |
| `environment.yml`      | `integrator`      | CPU runtime (local, non-CUDA)          |
| `environment-cuda.yml` | `integrator-cuda` | CUDA / GPU runtime (cluster, linux-64) |
| `environment-dev.yml`  | `integrator-dev`  | CPU + test/lint/logging tooling (development)  |
| `environment-cuda-dev.yml` | `integrator-cuda-dev` | CUDA / GPU + dev tooling (development on GPU) |

```bash
# CPU runtime
micromamba env create -f environment.yml
micromamba activate integrator

# CUDA runtime (GPU machine)
micromamba env create -f environment-cuda.yml
micromamba activate integrator-cuda

# Development (tests, ruff, mypy)
micromamba env create -f environment-dev.yml
micromamba activate integrator-dev

# Development on a GPU machine
micromamba env create -f environment-cuda-dev.yml
micromamba activate integrator-cuda-dev
```

Each installs DIALS, laue-DIALS, PyTorch, the integrator package (editable), and the upstream reciprocalspaceship build required to read DIALS `.refl` files. 
The four files share the same dependency set and differ only in the PyTorch build and the dev tooling, so keep the shared lines in sync when editing one.

### Without micromamba

If DIALS and laue-DIALS are already provided by your environment, install just the Python package and its remaining dependencies with pip:

```bash
pip install -e ".[dev]" \
  "reciprocalspaceship @ git+https://github.com/rs-station/reciprocalspaceship"
```

The reciprocalspaceship git URL is required because pip would otherwise install the PyPI release, which lacks the DIALS `.refl` fix.
uv is optional and is not required by this project; if you use it, `uv pip install` works the same way.
