"""Extension subpackage: unsupervised scale + merge methods.

Standalone, additive modules that operate on a trained integrator's
per-observation intensities. Nothing here modifies the integrator, the loss,
the factory, or the registry -- import explicitly when you want it.
"""

from integrator.merge_ext.negative_control_merge import (
    build_merge_groups,
    merge_anomalous,
    scale_and_merge,
    tukey_biweight,
)

__all__ = [
    "build_merge_groups",
    "merge_anomalous",
    "scale_and_merge",
    "tukey_biweight",
]
