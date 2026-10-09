"""A torch reimplementation of the DIALS scaling and merging algorithms.

Built so the scale target, error model and merge can be modified and
compared numerically without a cctbx runtime. Symmetry comes from
`gemmi`/`reciprocalspaceship` rather than `cctbx.miller`; gradients come
from autograd rather than the hand-assembled sparse Jacobians DIALS
carries.

Fidelity target is numerical agreement with DIALS to a stated tolerance,
not bit-for-bit reproduction. Known deliberate divergences are recorded
in `NOTES.md`.
"""

from .french_wilson import french_wilson
from .io import Experiment, ScalingModelRef, load_experiment, load_reflections
from .merge import merge, r_factors, separate_bijvoet
from .scale_model import PhysicalScalingModel, ScaleFeatures, build_features
from .smoother import GaussianSmoother1D
from .stats import (
    MergingStats,
    ShellStats,
    cc_anom,
    cc_half_random,
    cc_half_sigma_tau,
    cc_star,
    completeness,
    merging_statistics,
    r_split,
    resolution_bins,
    weighted_cc_half,
)
from .symmetry import build_asu_map, complete_set
from .types import AsuMap, MergedData, Reflections

__all__ = [
    "AsuMap",
    "Experiment",
    "GaussianSmoother1D",
    "MergedData",
    "MergingStats",
    "PhysicalScalingModel",
    "Reflections",
    "ScaleFeatures",
    "ScalingModelRef",
    "ShellStats",
    "build_asu_map",
    "build_features",
    "cc_anom",
    "cc_half_random",
    "cc_half_sigma_tau",
    "cc_star",
    "complete_set",
    "completeness",
    "french_wilson",
    "load_experiment",
    "load_reflections",
    "merge",
    "merging_statistics",
    "r_factors",
    "r_split",
    "resolution_bins",
    "separate_bijvoet",
    "weighted_cc_half",
]
