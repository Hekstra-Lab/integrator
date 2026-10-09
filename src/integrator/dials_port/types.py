"""Core containers for the DIALS scale/merge port.

All tensors are `torch.float64` on CPU unless stated otherwise.
Integer columns (`hkl`, group ids, dataset ids) are `torch.int64`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

DTYPE = torch.float64
IDTYPE = torch.int64


@dataclass
class Reflections:
    """Unmerged reflection observations from one or more sweeps.

    Mirrors the subset of a DIALS reflection table that the scaling and
    merging algorithms actually consume.

    Attributes:
        hkl: `(N, 3)` Miller indices as recorded (not ASU-mapped).
        intensity: `(N,)` observed intensity.
        variance: `(N,)` observed variance.
        d: `(N,)` resolution in Angstrom.
        phi: `(N,)` rotation coordinate used by the smoothers. This is
            the `xyzobs.px.value` z-component (image number), matching
            `PhysicalScalingModel`.
        s1: `(N, 3)` diffracted beam vectors in the lab frame.
        partiality: `(N,)` partiality, or all-ones when absent.
        lp: `(N,)` Lorentz-polarization factor.
        qe: `(N,)` detector efficiency.
        dataset_id: `(N,)` sweep/experiment identifier.
        scale: `(N,)` inverse scale factor. Filled by the refiner; holds
            DIALS' `inverse_scale_factor` when loaded from `scaled.refl`.
    """

    hkl: torch.Tensor
    intensity: torch.Tensor
    variance: torch.Tensor
    d: torch.Tensor
    phi: torch.Tensor
    s1: torch.Tensor
    partiality: torch.Tensor
    lp: torch.Tensor
    qe: torch.Tensor
    dataset_id: torch.Tensor
    scale: torch.Tensor

    def __len__(self) -> int:
        return int(self.hkl.shape[0])

    @property
    def sigma(self) -> torch.Tensor:
        """Standard deviation implied by `variance`."""
        return torch.sqrt(self.variance)

    def select(self, mask: torch.Tensor) -> Reflections:
        """Return a new `Reflections` keeping rows where `mask` is true.

        Args:
            mask: boolean or integer index tensor.

        Returns:
            A new instance; the original is left unmodified.
        """
        return Reflections(
            **{
                name: getattr(self, name)[mask]
                for name in self.__dataclass_fields__
            }
        )

    def with_scale(self, scale: torch.Tensor) -> Reflections:
        """Return a copy carrying a new inverse scale factor."""
        return replace(self, scale=scale)


@dataclass
class AsuMap:
    """Symmetry reduction of a set of Miller indices.

    Two groupings are carried simultaneously, because DIALS runs the
    identical weighted-mean code over both: the anomalous grouping keeps
    Friedel mates apart, the Laue grouping pools them.

    Attributes:
        asu_hkl: `(N, 3)` ASU-mapped Miller indices.
        isym: `(N,)` symmetry operation index from `rs.utils.hkl_to_asu`.
        plus: `(N,)` bool, true for the I(+) hemisphere (`isym` odd).
        centric: `(N,)` bool.
        epsilon: `(N,)` multiplicity-of-reflection epsilon factor.
        sys_absent: `(N,)` bool, systematically absent.
        group_id: `(N,)` anomalous group index in `[0, n_groups)`.
        group_id_laue: `(N,)` Friedel-pooled group index.
        n_groups: number of distinct anomalous groups.
        n_groups_laue: number of distinct Laue groups.
    """

    asu_hkl: torch.Tensor
    isym: torch.Tensor
    plus: torch.Tensor
    centric: torch.Tensor
    epsilon: torch.Tensor
    sys_absent: torch.Tensor
    group_id: torch.Tensor
    group_id_laue: torch.Tensor
    n_groups: int
    n_groups_laue: int

    def groups(self, anomalous: bool) -> tuple[torch.Tensor, int]:
        """Return `(group_id, n_groups)` for the requested grouping."""
        if anomalous:
            return self.group_id, self.n_groups
        return self.group_id_laue, self.n_groups_laue


@dataclass
class MergedData:
    """Result of merging symmetry equivalents.

    Attributes:
        hkl: `(M, 3)` unique ASU Miller indices, one row per group.
        intensity: `(M,)` merged intensity.
        sigma: `(M,)` merged standard deviation.
        multiplicity: `(M,)` number of contributing observations.
        d: `(M,)` resolution of the merged reflection.
        centric: `(M,)` bool.
        plus: `(M,)` bool; meaningful only for an anomalous merge.
    """

    hkl: torch.Tensor
    intensity: torch.Tensor
    sigma: torch.Tensor
    multiplicity: torch.Tensor
    d: torch.Tensor
    centric: torch.Tensor
    plus: torch.Tensor

    def __len__(self) -> int:
        return int(self.hkl.shape[0])
