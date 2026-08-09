"""
ChunkedDataModule — loads one pre-processed chunk at a time.

Chunk layout (produced by integrator.preprocess)
─────────────────────────────────────────────────
chunk_NNNNN/
  counts.npy          float32  (N, pixels)   raw pixel counts
  shoebox.npy         float32  (N, pixels)   transform-applied (model input)
  mask.npy            bool     (N, pixels)   valid-pixel mask
  metadata.npz        all numeric + bool columns:
                        row_id, is_train, is_val, image_id, d, lp,
                        group_label, profile_group_label, refl_ids, ...
  strings.parquet     all string columns (source_refl, source_expt, ...);
                      omitted entirely when no string columns exist

manifest.yaml (authoritative schema, version 2)
  version: 2
  numeric_columns: [image_id, d, group_label, ...]   # list of column names
  string_columns:  [source_refl, source_expt]         # list of column names
  chunks:
    - dir: chunk_00000
      n_rows: 50000
      n_train: ...
      n_val: ...
      n_unique_images: 42      # unique image_id values in this chunk
      min_image_id: 0          # inclusive  (null when image_id absent)
      max_image_id: 199        # inclusive  (null when image_id absent)

Batch design
────────────
ChunkedDataset yields complete mini-batches (not individual reflections).
The DataLoader is created with batch_size=None so items from __iter__ are
returned as-is without additional collation overhead.  This eliminates the
per-reflection Python loop that would otherwise dominate CPU time.

Each mini-batch contains rows from a single chunk only.  Shuffling is applied
at two levels during training:
  1. chunk order is randomised each epoch (shuffle_chunks=True)
  2. rows within each chunk are shuffled before batching (shuffle_within=True)
Cross-chunk batch mixing (requiring a multi-chunk buffer) is deferred.

Worker distribution
───────────────────
With num_workers > 0 each worker receives a disjoint subset of chunk
directories (round-robin by index) so no chunk is processed more than once
per epoch.  persistent_workers=False is required: with IterableDataset,
worker iterators are not reliably reset between epochs when persistent.

Prediction ordering
───────────────────
predict_dataloader() forces num_workers=0.  With IterableDataset and
num_workers > 0 the DataLoader interleaves batches from workers in an
undefined order, breaking ID-preserving write-back.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import polars as pl
from pytorch_lightning import LightningDataModule
import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


class ChunkedDataset(IterableDataset):
    """Yields pre-formed mini-batches from chunk directories.

    Each call to __iter__ loads one chunk directory at a time into RAM,
    applies optional row-filtering and shuffling, then slices into mini-batches
    using numpy — one torch conversion per batch, not per reflection.

    Each mini-batch contains rows from a single chunk.  Training shuffles
    chunk order each epoch and rows within each chunk before batching.
    Validation and prediction do not shuffle.
    """

    def __init__(
        self,
        chunk_dirs: list[Path],
        numeric_columns: list[
            str
        ],  # ordered list of numeric/bool column names
        string_columns: list[str],  # ordered list of string column names
        filter_key: str | None,  # "is_train" | "is_val" | None (all rows)
        shuffle_chunks: bool = False,
        shuffle_within: bool = False,
        batch_size: int = 32,
    ):
        self.chunk_dirs = list(chunk_dirs)
        self.numeric_columns = list(numeric_columns)
        self.string_columns = list(string_columns)
        self.filter_key = filter_key
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within = shuffle_within
        self.batch_size = batch_size

    @staticmethod
    def _load_chunk(
        chunk_dir: Path,
        numeric_columns: list[str],
        string_columns: list[str],
    ) -> dict:
        """Load all arrays for one chunk directory into RAM.

        Reads:
          counts.npy      — float32 (N, pixels)
          shoebox.npy     — float32 (N, pixels)
          mask.npy        — bool    (N, pixels)
          metadata.npz    — all numeric + bool columns (row_id, is_train, is_val,
                            and every name in numeric_columns).  Loaded via a
                            context manager; arrays are .copy()-ed before the
                            file handle is released.
          strings.parquet — all string columns.  Raises FileNotFoundError if
                            string_columns is non-empty but the file is absent.

        Raises:
            FileNotFoundError: strings.parquet missing with non-empty string_columns.
            KeyError: metadata.npz is missing a required key (is_train, is_val,
                      row_id, or any column in numeric_columns).
            ValueError: shoebox.npy or mask.npy first-dimension does not match
                        counts.npy; any array length does not match counts.shape[0];
                        or strings.parquet is missing a column listed in
                        string_columns.
        """
        d = chunk_dir
        chunk: dict = {
            "counts": np.load(d / "counts.npy"),
            "shoebox": np.load(d / "shoebox.npy"),
            "mask": np.load(d / "mask.npy"),
        }

        # Validate that shoebox.npy and mask.npy have the same first-dimension
        # row count as counts.npy.  Catches truncated pixel-array writes.
        n_rows = chunk["counts"].shape[0]
        for key in ("shoebox", "mask"):
            if chunk[key].shape[0] != n_rows:
                raise ValueError(
                    f"Shape mismatch in {d}: "
                    f"counts has {n_rows} rows but '{key}.npy' has "
                    f"{chunk[key].shape[0]} rows."
                )

        # Load all numeric + bool metadata.
        # Context manager closes the NpzFile promptly; .copy() detaches each
        # array from the underlying mmap before the file handle is released.
        with np.load(d / "metadata.npz", allow_pickle=False) as meta:
            # Verify all required keys are present before accessing any of them.
            required_keys = {"is_train", "is_val", "row_id", *numeric_columns}
            missing_keys = required_keys - set(meta.files)
            if missing_keys:
                raise KeyError(
                    f"metadata.npz in {d} is missing key(s): "
                    f"{sorted(missing_keys)}.\n"
                    "Re-run  integrator.preprocess  to regenerate chunks."
                )
            chunk["is_train"] = meta["is_train"].copy()
            chunk["is_val"] = meta["is_val"].copy()
            chunk["row_id"] = meta["row_id"].copy()
            for col in numeric_columns:
                chunk[col] = meta[col].copy()

        # Load string columns.  Raise immediately if the file is declared in
        # the manifest but physically absent (e.g. truncated preprocess run).
        strings_path = d / "strings.parquet"
        if string_columns:
            if not strings_path.exists():
                raise FileNotFoundError(
                    f"strings.parquet not found in {d} but "
                    f"string_columns={string_columns!r} are declared in "
                    "manifest.yaml.\n"
                    "Re-run  integrator.preprocess  to regenerate chunks."
                )
            str_df = pl.read_parquet(strings_path)
            # Verify every declared string column exists in the file.
            missing_cols = [
                col for col in string_columns if col not in str_df.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"strings.parquet in {d} is missing column(s): "
                    f"{missing_cols}.\n"
                    "Re-run  integrator.preprocess  to regenerate chunks."
                )
            for col in string_columns:
                chunk[col] = str_df[col].to_list()

        # Validate all lengths against counts.shape[0].
        # Catches truncated writes (disk-full or interrupted preprocess run).
        for key in ("is_train", "is_val", "row_id", *numeric_columns):
            if len(chunk[key]) != n_rows:
                raise ValueError(
                    f"Length mismatch in {d}: "
                    f"counts has {n_rows} rows but '{key}' has "
                    f"{len(chunk[key])} rows."
                )
        for col in string_columns:
            if len(chunk[col]) != n_rows:
                raise ValueError(
                    f"Length mismatch in {d}: "
                    f"counts has {n_rows} rows but string column '{col}' has "
                    f"{len(chunk[col])} rows."
                )

        return chunk

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dirs = list(self.chunk_dirs)

        # Each chunk assigned to exactly one worker — no duplication.
        if worker_info is not None:
            dirs = [
                d
                for j, d in enumerate(dirs)
                if j % worker_info.num_workers == worker_info.id
            ]

        if self.shuffle_chunks:
            random.shuffle(dirs)

        nc = self.numeric_columns
        sc = self.string_columns
        bs = self.batch_size

        for chunk_dir in dirs:
            chunk = self._load_chunk(chunk_dir, nc, sc)
            n = len(chunk["counts"])

            if self.filter_key is not None:
                idx = np.flatnonzero(chunk[self.filter_key].astype(bool))
            else:
                idx = np.arange(n)

            if self.shuffle_within:
                idx = idx[np.random.permutation(len(idx))]

            # Yield mini-batches as numpy slices — one torch conversion per batch.
            for b_start in range(0, len(idx), bs):
                b_idx = idx[b_start : b_start + bs]

                counts_b = torch.from_numpy(chunk["counts"][b_idx]).float()
                shoebox_b = torch.from_numpy(chunk["shoebox"][b_idx]).float()
                mask_b = torch.from_numpy(chunk["mask"][b_idx]).bool()

                meta_b: dict = {}
                for col in nc:
                    # dtype preserved: int64->torch.int64, float32->torch.float32
                    meta_b[col] = torch.as_tensor(chunk[col][b_idx])
                for col in sc:
                    meta_b[col] = [chunk[col][i] for i in b_idx]  # list[str]

                # row_id always included for traceability
                meta_b["row_id"] = torch.as_tensor(chunk["row_id"][b_idx])

                yield counts_b, shoebox_b, mask_b, meta_b


class ChunkedDataModule(LightningDataModule):
    """LightningDataModule that loads pre-processed chunks one at a time.

    Requires a chunk directory produced by ``integrator.preprocess`` (v6 layout,
    manifest version 2).  The train/val split is baked into each chunk's
    metadata.npz arrays (is_train, is_val) and is never re-randomised at
    training time.

    YAML config example::

        data_loader:
          name: chunked_rotation_data
          args:
            chunk_dir: /data/mfx/chunks
            batch_size: 256
            num_workers: 4
    """

    # Explicit opt-in sentinel checked by assign_labels() in prediction_writer.py.
    # assign_labels() is always called with the DataModule directly (train.py:424).
    # ChunkedDataModule handles its own train/val split at preprocess time, so the
    # expensive refl_id scan in assign_labels() must be skipped entirely.
    _CHUNKED_DATA_MODULE: bool = True

    def __init__(
        self,
        chunk_dir: str | Path,
        batch_size: int = 10,
        num_workers: int = 0,
    ):
        super().__init__()
        self.chunk_dir = Path(chunk_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Set in setup(); numeric_columns / string_columns exposed so
        # inject_binning_labels() can verify group_label is present before
        # training starts.
        self.chunk_dirs: list[Path] = []
        self.numeric_columns: list[str] = []
        self.string_columns: list[str] = []
        self.n_images: int | None = None

    def setup(self, stage: str | None = None) -> None:
        manifest_path = self.chunk_dir / "manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.yaml found in {self.chunk_dir}.\n"
                "Run  integrator.preprocess  first."
            )
        m = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

        # Require manifest version 2 (v6 layout: metadata.npz + strings.parquet).
        # Version 1 uses per-column col_*.npy / str_*.txt files which are no
        # longer supported.  Fail fast rather than crashing mid-epoch with an
        # obscure FileNotFoundError.
        version = m.get("version", 1)
        if version != 2:
            raise ValueError(
                f"manifest.yaml declares version {version}; version 2 is "
                "required.\nRe-run  integrator.preprocess  to regenerate "
                "chunks with the v6 layout."
            )

        string_columns = list(m.get("string_columns", []))
        required_files = [
            "counts.npy",
            "shoebox.npy",
            "mask.npy",
            "metadata.npz",
        ]
        if string_columns:
            required_files.append("strings.parquet")

        logger.info(
            "ChunkedDataModule.setup: verifying %d chunk(s) in %s ...",
            len(m["chunks"]),
            self.chunk_dir,
        )
        for entry in m["chunks"]:
            cdir = self.chunk_dir / entry["dir"]
            for fname in required_files:
                if not (cdir / fname).exists():
                    raise FileNotFoundError(
                        f"Required file '{fname}' missing from {cdir}.\n"
                        "Re-run  integrator.preprocess  to regenerate chunks."
                    )

        self.chunk_dirs = [self.chunk_dir / e["dir"] for e in m["chunks"]]
        self.numeric_columns = list(m.get("numeric_columns", []))
        self.string_columns = string_columns
        self.n_images = m.get("n_images")

        logger.info(
            "ChunkedDataModule: %d chunk(s)  total=%s  train=%s  val=%s  n_images=%s",
            len(self.chunk_dirs),
            m.get("n_valid_rows", "?"),
            m.get("n_train", "?"),
            m.get("n_val", "?"),
            self.n_images,
        )

    def _make_loader(
        self,
        filter_key: str | None,
        shuffle_chunks: bool,
        shuffle_within: bool,
        num_workers: int,
        string_columns: list[str] | None = None,
    ) -> DataLoader:
        # Allow callers to override which string columns are loaded.
        # Train and val pass [] to skip strings.parquet entirely (those columns
        # are only needed during prediction for write-back).  Predict passes
        # self.string_columns to get source_refl / source_expt.
        sc = self.string_columns if string_columns is None else string_columns
        ds = ChunkedDataset(
            chunk_dirs=sorted(self.chunk_dirs),
            numeric_columns=self.numeric_columns,
            string_columns=sc,
            filter_key=filter_key,
            shuffle_chunks=shuffle_chunks,
            shuffle_within=shuffle_within,
            batch_size=self.batch_size,
        )
        return DataLoader(
            ds,
            batch_size=None,  # ChunkedDataset yields complete batches
            num_workers=num_workers,
            pin_memory=False,  # pin_memory=True causes _PinMemoryThread deadlock
            # at epoch transitions with IterableDataset;
            # host→GPU transfer time is negligible for this
            # workload (disk-I/O + GPU-compute bound).
            # persistent_workers=True for num_workers>0: workers live for the full
            # training run and reset their iterator at each epoch, avoiding the
            # fork-after-CUDA deadlock that occurs when workers are killed and
            # respawned at every epoch boundary with persistent_workers=False.
            # Must be False for num_workers=0 (no workers to persist).
            persistent_workers=num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(
            filter_key="is_train",
            shuffle_chunks=True,
            shuffle_within=True,
            num_workers=self.num_workers,
            string_columns=[],  # source_refl/source_expt unused during training
        )

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(
            filter_key="is_val",
            shuffle_chunks=False,
            shuffle_within=False,
            num_workers=self.num_workers,
            string_columns=[],  # source_refl/source_expt unused during validation
        )

    def predict_dataloader(self) -> DataLoader:
        """All valid rows in deterministic sorted-chunk order.

        num_workers is forced to 0: with IterableDataset and num_workers > 0
        the DataLoader interleaves batches from workers in an undefined order,
        breaking the ordering guarantee required for ID-preserving write-back.
        """
        return self._make_loader(
            filter_key=None,
            shuffle_chunks=False,
            shuffle_within=False,
            num_workers=0,
            string_columns=self.string_columns,  # needed for write-back
        )
