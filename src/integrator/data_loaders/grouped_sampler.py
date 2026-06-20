"""Grouped sampler for Deep Sets merging.

`GroupedAsuIdBatchSampler` yields batches by:
    1. Shuffling the set of unique HKLs.
    2. For each batch, picking HKLs in order until the cumulative number
       of observations reaches `batch_size`.
    3. Yielding all observation indices for those HKLs.

"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import BatchSampler, Sampler

logger = logging.getLogger(__name__)


class GroupedAsuIdBatchSampler(BatchSampler):
    """Yield batches containing all observations of selected HKLs.

    Args:
        miller_idx: int tensor of length N — HKL id for each observation in
            the dataset.
        indices: subset of dataset indices to draw from (e.g. train_idx).
            If None, uses all indices [0, N).
        batch_size: approximate target batch size in observations.
        shuffle: shuffle HKL order each epoch.
        drop_last: whether to drop the trailing partial batch.
        max_obs_per_hkl: cap the number of observations per HKL in a batch
            (None means use all). For very common HKLs, all-obs can blow
            up the batch.
        seed: RNG seed for shuffling. None → unseeded.
    """

    def __init__(
        self,
        miller_idx: torch.Tensor,
        indices: torch.Tensor | None = None,
        batch_size: int = 2048,
        shuffle: bool = True,
        drop_last: bool = False,
        max_obs_per_hkl: int | None = None,
        seed: int | None = None,
    ):
        if indices is None:
            indices = torch.arange(len(miller_idx), dtype=torch.long)
        else:
            indices = torch.as_tensor(indices, dtype=torch.long)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_obs_per_hkl = max_obs_per_hkl
        self.seed = seed
        self._epoch = 0

        # Restrict to provided indices and build per-HKL index lists
        sub_asu = miller_idx[indices].long().numpy()
        idx_np = indices.long().numpy()

        # Group indices by HKL
        order = np.argsort(sub_asu, kind="stable")
        sorted_asu = sub_asu[order]
        sorted_indices = idx_np[order]
        # Split points where asu_id changes
        change = np.concatenate([[True], sorted_asu[1:] != sorted_asu[:-1]])
        change_idx = np.flatnonzero(change)
        # End of each group
        ends = np.concatenate([change_idx[1:], [len(sorted_asu)]])

        self._hkl_groups: list[np.ndarray] = [
            sorted_indices[start:end]
            for start, end in zip(change_idx, ends, strict=True)
        ]
        self._unique_asu = sorted_asu[change_idx]
        self._n_hkls = len(self._hkl_groups)
        self._total_obs = len(sub_asu)

        logger.info(
            "GroupedAsuIdBatchSampler: %d obs across %d unique HKLs "
            "(mean obs/HKL = %.1f)",
            self._total_obs,
            self._n_hkls,
            self._total_obs / max(self._n_hkls, 1),
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            if self.seed is not None:
                g = torch.Generator()
                g.manual_seed(self.seed + self._epoch)
                perm = torch.randperm(self._n_hkls, generator=g).numpy()
            else:
                perm = np.random.permutation(self._n_hkls)
        else:
            perm = np.arange(self._n_hkls)

        for chunk in np.array_split(perm, len(self)):
            batch: list[int] = []
            for h in chunk:
                obs = self._hkl_groups[h]
                if (
                    self.max_obs_per_hkl is not None
                    and len(obs) > self.max_obs_per_hkl
                ):
                    # Random subset (without replacement) for very common HKLs
                    obs = np.random.choice(
                        obs, size=self.max_obs_per_hkl, replace=False
                    )
                batch.extend(int(x) for x in obs)
            yield batch

    def __len__(self) -> int:
        n = self._total_obs // self.batch_size
        if not self.drop_last and self._total_obs % self.batch_size:
            n += 1
        return max(1, min(n, self._n_hkls))


class GroupedAsuIdSampler(Sampler[int]):
    """Yields single indices in HKL-grouped order.

    Each epoch shuffles HKL order, then yields all observation indices
    of HKL_0, then all of HKL_1, etc. Used with `DataLoader(..., sampler=...,
    batch_size=B)` instead of `batch_sampler=`, which gives Lightning the
    standard `dataloader.sampler` + `dataloader.batch_size` surface area
    it expects when re-resolving val dataloaders mid-training.

    Consecutive batches of size B from this sampler each contain
    consecutive HKLs in the shuffle order — mostly complete groups, with
    at most one HKL split across each batch boundary. For HEWL val
    (~5 obs/HKL, batch_size=2048), that's ~436 complete HKLs per batch.

    Args:
        miller_idx: int tensor of length N — HKL id for each observation.
        indices: subset of dataset indices to draw from. If None, uses all.
        shuffle: shuffle HKL order each epoch.
        seed: RNG seed.
    """

    def __init__(
        self,
        miller_idx: torch.Tensor,
        indices: torch.Tensor | None = None,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        if indices is None:
            indices = torch.arange(len(miller_idx), dtype=torch.long)
        else:
            indices = torch.as_tensor(indices, dtype=torch.long)

        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        sub_asu = miller_idx[indices].long().numpy()
        idx_np = indices.long().numpy()

        order = np.argsort(sub_asu, kind="stable")
        sorted_asu = sub_asu[order]
        sorted_indices = idx_np[order]
        change = np.concatenate([[True], sorted_asu[1:] != sorted_asu[:-1]])
        change_idx = np.flatnonzero(change)
        ends = np.concatenate([change_idx[1:], [len(sorted_asu)]])
        self._hkl_groups: list[np.ndarray] = [
            sorted_indices[start:end]
            for start, end in zip(change_idx, ends, strict=True)
        ]
        self._n_hkls = len(self._hkl_groups)
        self._total_obs = len(sub_asu)

        logger.info(
            "GroupedAsuIdSampler: %d obs across %d unique HKLs "
            "(mean obs/HKL = %.1f)",
            self._total_obs,
            self._n_hkls,
            self._total_obs / max(self._n_hkls, 1),
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        if self.shuffle:
            if self.seed is not None:
                g = torch.Generator()
                g.manual_seed(self.seed + self._epoch)
                perm = torch.randperm(self._n_hkls, generator=g).numpy()
            else:
                perm = np.random.permutation(self._n_hkls)
        else:
            perm = np.arange(self._n_hkls)

        for h in perm:
            for i in self._hkl_groups[h]:
                yield int(i)

    def __len__(self) -> int:
        return self._total_obs
