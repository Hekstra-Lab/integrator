from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataFileNames:
    data_dir: str
    counts: str
    masks: str
    stats: str
    reference: str
    standardized_counts: str | None = None

    def __post_init__(self):
        for name in (
            "counts",
            "masks",
            "stats",
            "reference",
        ):
            p = self._resolve(getattr(self, name))
            if not p.is_file():
                raise FileNotFoundError(f"{name} file not found: {p}")

        if self.standardized_counts is not None:
            p = self._resolve(self.standardized_counts)
            if not p.is_file():
                raise FileNotFoundError(f"standardized_counts file not found: {p}")

    def _resolve(self, fname: str) -> Path:
        return Path(self.data_dir) / fname


@dataclass
class DataLoaderArgs:
    data_dir: str
    batch_size: int
    val_split: float
    test_split: float
    num_workers: int
    include_test: bool
    subset_size: int
    cutoff: int | None
    use_metadata: bool
    shoebox_file_names: DataFileNames
    D: int
    H: int
    W: int
    anscombe: bool

    def __post_init__(self):
        if self.batch_size < 0:
            raise ValueError(
                f"""
                    Batch size must be an integer greater than 0, but
                    batch_size={self.batch_size} was passed
                """
            )


@dataclass
class DataLoaderConfig:
    name: str
    args: DataLoaderArgs
