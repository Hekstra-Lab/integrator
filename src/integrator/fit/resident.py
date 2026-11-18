"""Shoeboxes held on the device, so a sweep pass costs no host transfer.

The block sweep touches every pixel once per pass, and 2,664,858 x 625 is
1.7e9 terms. At roughly ten operations each that is milliseconds of arithmetic
and tens of milliseconds of memory traffic, so the 61.7 s a streaming pass took
on hewl_1118 was the dataloader and the host-to-device copy, not the work.

Counts are `int32` and masks are `bool` on disk, which is 6.66 GB and 1.67 GB
for this dataset: 8.3 GB of a 20 GB slice, leaving room for the per-chunk
working buffers. Nothing is held in the standardized form the encoders read,
because the sweep never uses it -- that alone is another 6.66 GB of host memory
and the Anscombe pass that builds it.

File order is the state order here, so the sweep needs no separate indexing
pass to map reflection ids onto rows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

_logger = logging.getLogger(__name__)

__all__ = ["ResidentShoeboxes", "device_free_bytes"]

_METADATA_KEYS = (
    "d",
    "wavelength",
    "xyzcal.px.0",
    "xyzcal.px.1",
    "image_num",
    "refl_ids",
    "H",
    "K",
    "L",
)


def _load_reference(path) -> dict:
    """Metadata from a `.pt` or a `.npy` reference file, as a dict of tensors.

    hewl_1118 stores a `torch.save` dict; hewl_9b7c's scaling dataset stores a
    pickled 0-d object array holding the same mapping, which `np.load` returns
    and `torch.load` cannot read. Both appear as `shoebox_file_names.reference`
    in a config, so the loader dispatches on the suffix rather than requiring
    the caller to know which dataset it has.
    """
    path = Path(path)
    if path.suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
        raw = obj.item() if obj.shape == () else obj
        if not hasattr(raw, "items"):
            raise ValueError(
                f"{path} is a {type(raw).__name__}, not a mapping of columns"
            )
        out = {}
        for k, v in raw.items():
            if isinstance(v, torch.Tensor):
                out[k] = v
            elif isinstance(v, np.ndarray) and v.dtype != object:
                out[k] = torch.from_numpy(np.ascontiguousarray(v))
        return out
    return torch.load(path, weights_only=False, map_location="cpu")


def device_free_bytes(device) -> int:
    """Free device memory, or 0 when there is no device to ask."""
    if device is None or torch.device(device).type != "cuda":
        return 0
    free, _total = torch.cuda.mem_get_info(torch.device(device))
    return int(free)


class ResidentShoeboxes:
    """Counts, masks and metadata on the device, iterated in chunks.

    Calling the instance returns a fresh iterator of
    `(obs_index, counts, mask, metadata)` with every tensor already on the
    device, which is the interface `run_sweep` consumes.

    `metadata_keys` selects which columns of the reference file are held, for
    callers that need more than the sweep does; keys absent from the file are
    skipped. Left unset it is `_METADATA_KEYS`, so the sweep is unaffected.
    """

    def __init__(
        self,
        data_dir,
        device,
        chunk: int = 65536,
        min_valid_pixels: int = 0,
        counts_file: str = "counts.npy",
        masks_file: str = "masks.npy",
        reference_file: str = "metadata.pt",
        block: int = 262144,
        metadata_keys: tuple[str, ...] | None = None,
    ) -> None:
        data_dir = Path(data_dir)
        self.device = torch.device(device)
        self.chunk = int(chunk)
        keys = _METADATA_KEYS if metadata_keys is None else tuple(metadata_keys)

        meta = _load_reference(data_dir / reference_file)
        counts_path = data_dir / counts_file
        src = np.load(counts_path, mmap_mode="r")
        msk = np.load(data_dir / masks_file, mmap_mode="r")
        n, p = int(src.shape[0]), int(src.shape[1])

        keep = None
        if min_valid_pixels > 0:
            valid = np.zeros(n, dtype=np.int32)
            for s in range(0, n, block):
                valid[s : s + block] = np.asarray(msk[s : s + block]).sum(1)
            keep = valid >= int(min_valid_pixels)
            if bool(keep.all()):
                keep = None
            else:
                _logger.info(
                    "min_valid_pixels=%d drops %d of %d shoeboxes",
                    min_valid_pixels, int((~keep).sum()), n,
                )
        rows = np.nonzero(keep)[0] if keep is not None else None
        self.n = int(rows.size) if rows is not None else n

        need = self.n * p * 4 + self.n * p
        free = device_free_bytes(self.device)
        if free and need > 0.85 * free:
            raise RuntimeError(
                f"shoeboxes need {need / 1e9:.1f} GB on {self.device} but only "
                f"{free / 1e9:.1f} GB is free; run without --resident"
            )

        self.counts = torch.empty(
            (self.n, p), dtype=torch.int32, device=self.device
        )
        self.mask = torch.empty((self.n, p), dtype=torch.bool, device=self.device)
        out = 0
        for s in range(0, n, block):
            sl = slice(s, min(s + block, n))
            idx = None if rows is None else rows[(rows >= sl.start) & (rows < sl.stop)]
            c = np.asarray(src[sl])
            m = np.asarray(msk[sl])
            if idx is not None:
                c = c[idx - sl.start]
                m = m[idx - sl.start]
            k = c.shape[0]
            if k == 0:
                continue
            # a basic slice of a memmap is a read-only view; copy before
            # handing it to torch, which otherwise warns about writability
            self.counts[out : out + k] = torch.from_numpy(
                np.array(c, dtype=np.int32, copy=True)
            ).to(self.device, non_blocking=True)
            self.mask[out : out + k] = torch.from_numpy(
                np.array(m, dtype=bool, copy=True)
            ).to(self.device, non_blocking=True)
            out += k
        if out != self.n:
            raise RuntimeError(f"loaded {out} shoeboxes, expected {self.n}")

        self.metadata: dict[str, Tensor] = {}
        for key in keys:
            if key not in meta:
                continue
            v = meta[key].reshape(-1)
            if rows is not None:
                v = v[torch.from_numpy(rows)]
            self.metadata[key] = v.to(self.device).float()

        _logger.info(
            "resident: %d shoeboxes x %d pixels on %s (%.2f GB), %d metadata columns",
            self.n, p, self.device, need / 1e9, len(self.metadata),
        )

    def __len__(self) -> int:
        return self.n

    def __call__(self):
        for s in range(0, self.n, self.chunk):
            e = min(s + self.chunk, self.n)
            idx = torch.arange(s, e, device=self.device)
            md = {k: v[s:e] for k, v in self.metadata.items()}
            yield idx, self.counts[s:e], self.mask[s:e], md
