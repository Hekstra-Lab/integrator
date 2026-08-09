# Integrator
An amortized variational inference model to integrate diffraction data.

## Prerequisites

The installation requires a python environment manager. 
I use [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), and the following instructions assume it. 
Another popular manager is [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), but I do not use it nor have experience with it. 

## Installation

The project depends on DIALS and laue-DIALS alongside PyTorch.
DIALS (and dxtbx/cctbx) and PyTorch are installed from conda-forge.
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
## MFX Workflow

This repository includes an end-to-end workflow for applying the variational-inference Integrator to experimental MFX serial femtosecond crystallography data from LCLS.

The MFX workflow extends the base Integrator framework with:

- scalable Jungfrau shoebox extraction from cctbx.xfel `.refl/.expt` files
- chunked preprocessing and GPU training for datasets containing more than 100 million reflections
- alternative intensity posteriors including FoldedNormal and LogNormal
- Normal and Student-t observation likelihoods for floating-point and negative-valued detector data
- Monte Carlo KL fallback for posterior/prior combinations without a built-in analytic KL divergence
- MFX prediction write-back from Integrator intensities and uncertainties to the original reflection tables
- downstream cctbx.xfel scaling/merging and PHENIX refinement for direct crystallographic comparison
- diagnostic tools for shoeboxes, learned spot profiles, training behavior, and resolution-dependent merging statistics

### Current MFX study

The current production study uses experiment `mfx101555026`, covering runs 269–289 excluding 275:

- **124,975** indexed MFX `.refl/.expt` pairs
- **~115 million** extracted 25 × 25 Jungfrau shoeboxes
- two production Integrator models using **ASINH** and **sqrt-squareplus** preprocessing with a **FoldedNormal intensity posterior**
- matched cctbx.xfel baselines for apples-to-apples comparison
- evaluation through **CC½, Rint, Rsplit, R-work, and R-free**

The current results establish an end-to-end proof of concept: Integrator can scale to a full experimental MFX dataset, produce probabilistic intensity estimates, write them back into the crystallographic workflow, and generate refinable structures. The cctbx.xfel baseline currently performs better overall, while the elevated high-resolution Integrator CC½ behavior remains an active validation question.

See the full workflow, commands, configs, Slurm jobs, PHIL files, diagnostics, and results here:

- [`workflows/mfx101555026`](workflows/mfx101555026/README.md) — complete MFX cctbx.xfel → Integrator → scaling/merging → PHENIX workflow