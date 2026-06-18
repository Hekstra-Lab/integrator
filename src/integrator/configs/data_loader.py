from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataFileNames:
    """Filenames

    Attributes:
        data_dir: Directory the other filenames are resolved against.
        counts: Filename of the raw photon-count tensor.
        masks: Filename of the per-pixel validity mask tensor.
        reference: Filename of the reference reflection table.
        standardized_counts: Optional filename of pre-standardized counts; `None` to standardize on the fly.
    """

    data_dir: str
    counts: str
    masks: str
    reference: str
    standardized_counts: str | None = None

    def __post_init__(self):
        for name in (
            "counts",
            "masks",
            "reference",
        ):
            p = self._resolve(getattr(self, name))
            if not p.is_file():
                raise FileNotFoundError(f"{name} file not found: {p}")

        if self.standardized_counts is not None:
            p = self._resolve(self.standardized_counts)
            if not p.is_file():
                raise FileNotFoundError(
                    f"standardized_counts file not found: {p}"
                )

    def _resolve(self, fname: str) -> Path:
        return Path(self.data_dir) / fname


@dataclass
class DataLoaderArgs:
    """Constructor arguments for the shoebox data module.

    Attributes:
        data_dir: Root directory holding the shoebox tensors; must exist.
        batch_size: Number of reflections per batch; must be non-negative.
        val_split: Fraction of data held out for validation.
        test_split: Fraction of data held out for testing.
        num_workers: Number of `DataLoader` worker processes.
        include_test: Whether to materialize the test split.
        subset_size: Cap on the number of reflections loaded, or `None`.
        cutoff: Optional resolution cutoff, or `None` for no cutoff.
        shoebox_file_names: Mapping locating the tensors on disk (see `DataFileNames`).
        D: Shoebox depth in pixels.
        H: Shoebox height in pixels.
        W: Shoebox width in pixels.
        transform: Count transform `anscombe`, `log1p`, or `standardization`. `log1p` feeds raw `log1p`
            counts to the encoder (the scvi-tools recipe for skewed-count VAEs); `standardization`
            mean-subtracts and std-divides the raw counts.
        min_valid_pixels: Drop shoeboxes with fewer than this many valid pixels.
        get_dxyz: Also load per-pixel dxyz offsets (rotation data only).
        single_sample_index: Restrict the dataset to a single reflection index (debugging; rotation only).
    """

    data_dir: str
    batch_size: int
    val_split: float
    test_split: float
    num_workers: int
    include_test: bool
    subset_size: int | None
    cutoff: int | None
    shoebox_file_names: DataFileNames
    D: int
    H: int
    W: int
    transform: str | None = None
    min_valid_pixels: int = 10
    get_dxyz: bool = False
    single_sample_index: int | None = None

    def __post_init__(self):
        if self.batch_size < 0:
            raise ValueError(
                f"""
                    Batch size must be an integer greater than 0, but
                    batch_size={self.batch_size} was passed
                """
            )
        if not Path(self.data_dir).exists():
            raise ValueError(
                f"The data directory does not exist: data_dir={self.data_dir}"
            )


@dataclass
class DataLoaderConfig:
    """Registry selection for the data module: a `name` plus its typed `args`.

    Attributes:
        name: Registry key naming the data-module class to construct.
        args: Data-module constructor arguments.
    """

    name: str
    args: DataLoaderArgs
