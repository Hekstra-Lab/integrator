"""Figure and report generation for training runs."""

from .figure_data import (
    BasisRecorder,
    LatentRecorder,
    TrackedRecorder,
    load_basis,
    load_latents,
    load_tracked,
    select_tracked,
)
from .figure_style import paper_style, save_animation, save_figure

__all__ = [
    "BasisRecorder",
    "LatentRecorder",
    "TrackedRecorder",
    "build_report",
    "load_basis",
    "load_latents",
    "load_tracked",
    "paper_style",
    "save_animation",
    "save_figure",
    "select_tracked",
]


def __getattr__(name):
    # `build_report` pulls in plotly, a dev-only dependency; keep it out of
    # the import path for the matplotlib figures.
    if name == "build_report":
        from .report import build_report

        return build_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
