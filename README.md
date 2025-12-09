# Integrator

An amortized variational inference model to integrate diffraction data.

## Installation 

```bash
# Create a new environment using conda or micromamba
micromamba create -n integrator python=3.12 \

# Install DIALS and PyTorch using micromamba
micromamba install -c conda-forge dials pytorch

# Install uv
pip install uv

# install integrator
uv pip install -e ".[dev]"
```
