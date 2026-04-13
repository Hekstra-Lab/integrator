from dataclasses import dataclass


@dataclass
class OutputConfig:
    refl_file: str
    dials_merge_html: str | None = None
    phenix_refine_log: str | None = None
    anomalous_peaks: str | None = None

    def __post_init__(self):
        if not self.refl_file:
            raise ValueError("refl_file must be a non-empty path")
