from dataclasses import dataclass


@dataclass
class OutputConfig:
    """Output artifact paths produced by a training or scaling run.

    Attributes:
        refl_file: Path to the written reflection table; must be non-empty.
    """

    refl_file: str

    def __post_init__(self):
        if not self.refl_file:
            raise ValueError("refl_file must be a non-empty path")
