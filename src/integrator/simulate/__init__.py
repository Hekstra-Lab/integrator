"""Simulation utilities for generating synthetic shoebox datasets."""

from .generate import save_dataset, simulate
from .profiles import sample_profiles

__all__ = ["sample_profiles", "simulate", "save_dataset"]
