from dataclasses import dataclass


@dataclass
class OutputConfig:
    """Output artifact paths produced by a training or scaling run.

    Attributes:
        refl_file: Path to the written reflection table; must be non-empty.
        dials_merge_html: Optional path to the DIALS merge report.
        phenix_refine_log: Optional path to the `phenix.refine` log.
        anomalous_peaks: Optional path to the anomalous-peak summary.
    """

    refl_file: str
    dials_merge_html: str | None = None
    phenix_refine_log: str | None = None
    anomalous_peaks: str | None = None

    def __post_init__(self):
        if not self.refl_file:
            raise ValueError("refl_file must be a non-empty path")
