"""Reflection-table / metadata / MTZ serialization for the integrator pipeline.

Single home for converting between DIALS `.refl` files, per-reflection metadata
(`.pt`/`.npy`), model predictions, and MTZ output. DIALS itself is imported
lazily inside the `.refl` writer, so importing this package does not require a
DIALS install.
"""

from .dtypes import (
    DEFAULT_EXCLUDED_COLS,
    DEFAULT_REFL_COLS,
    SCALAR_DTYPES,
    VECTOR_COLUMNS,
)
from .metadata import load_metadata, refl_as_pt
from .mtz_io import write_mtz_from_preds
from .pred_io import get_pred_files, write_refl_from_preds
from .refl_io import unstack_preds, write_refl_with_predictions

__all__ = [
    "DEFAULT_REFL_COLS",
    "DEFAULT_EXCLUDED_COLS",
    "SCALAR_DTYPES",
    "VECTOR_COLUMNS",
    "load_metadata",
    "refl_as_pt",
    "write_mtz_from_preds",
    "get_pred_files",
    "write_refl_from_preds",
    "unstack_preds",
    "write_refl_with_predictions",
]
