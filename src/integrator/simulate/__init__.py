"""Simulation utilities for generating synthetic shoebox datasets."""

from .generate import save_dataset, simulate
from .profiles import h_to_physical_params, h_to_profile, sample_profiles

__all__ = [
    "h_to_physical_params",
    "h_to_profile",
    "sample_profiles",
    "simulate",
    "save_dataset",
]
