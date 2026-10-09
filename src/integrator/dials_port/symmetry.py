"""Symmetry reduction, the one irreducible cctbx dependency.

DIALS reaches `cctbx.miller` for ASU mapping, centric flags and complete
set enumeration. All three are available from `gemmi` and
`reciprocalspaceship`, which this package uses instead so that nothing
here needs a cctbx runtime.

The anomalous/Laue distinction is the whole of DIALS' anomalous
handling: `merge_equivalents` is run twice over two different grouping
keys, and nothing else differs. See `AsuMap` for both groupings.
"""

from __future__ import annotations

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch

from .types import DTYPE, IDTYPE, AsuMap


def as_spacegroup(spacegroup: str | int | gemmi.SpaceGroup) -> gemmi.SpaceGroup:
    """Coerce a space group specifier to a `gemmi.SpaceGroup`.

    Accepts Hermann-Mauguin symbols, ITC numbers, and the Hall symbols
    that DIALS writes into `.expt` files as `space_group_hall_symbol`
    (e.g. `"P 4w 2c"`, which is not a valid HM symbol). Also tolerates
    the `"P 43 21 2 (No. 96)"` form.

    Args:
        spacegroup: Hermann-Mauguin symbol, Hall symbol, ITC number, or
            an existing instance.

    Returns:
        The corresponding `gemmi.SpaceGroup`.

    Raises:
        ValueError: if the symbol resolves as neither HM nor Hall.
    """
    if isinstance(spacegroup, gemmi.SpaceGroup):
        return spacegroup
    if isinstance(spacegroup, int):
        return gemmi.SpaceGroup(spacegroup)

    name = str(spacegroup).split("(")[0].strip()
    try:
        return gemmi.SpaceGroup(name)
    except ValueError:
        pass

    found = gemmi.find_spacegroup_by_ops(gemmi.symops_from_hall(name))
    if found is None:
        raise ValueError(f"Unknown space group: {spacegroup!r}")
    return found


def _row_ids(rows: np.ndarray) -> tuple[np.ndarray, int]:
    """Assign a contiguous id to each distinct row of an integer array.

    Uses a structured-void view so the uniqueness test is a single
    `np.unique` over opaque bytes rather than a Python loop.

    Args:
        rows: `(N, K)` C-contiguous integer array.

    Returns:
        `(ids, n_unique)` with `ids` of shape `(N,)` in `[0, n_unique)`.
    """
    contig = np.ascontiguousarray(rows)
    void = contig.view([("", contig.dtype)] * contig.shape[1])
    _, inverse = np.unique(void.ravel(), return_inverse=True)
    inverse = inverse.astype(np.int64, copy=False)
    return inverse, int(inverse.max()) + 1 if inverse.size else 0


def build_asu_map(
    hkl: torch.Tensor,
    spacegroup: str | int | gemmi.SpaceGroup,
) -> AsuMap:
    """Map Miller indices to the ASU and build both grouping keys.

    Friedel parity comes from the `isym` returned by
    `rs.utils.hkl_to_asu`: odd values are the I(+) hemisphere. This is
    the convention already used elsewhere in this repository.

    Per-reflection symmetry predicates (`centric`, `epsilon`,
    `sys_absent`) are evaluated once per distinct Laue-group index and
    broadcast back, since the `gemmi` accessors are per-call Python.

    Args:
        hkl: `(N, 3)` integer Miller indices.
        spacegroup: space group of the crystal.

    Returns:
        An `AsuMap` carrying both the anomalous and Laue groupings.
    """
    sg = as_spacegroup(spacegroup)
    ops = sg.operations()

    h_np = hkl.detach().cpu().numpy().astype(np.int32, copy=False)
    asu_hkl, isym = rs.utils.hkl_to_asu(h_np, sg)
    asu_hkl = np.ascontiguousarray(asu_hkl.astype(np.int64))
    isym = np.asarray(isym, dtype=np.int64)
    plus = (isym % 2) == 1

    laue_ids, n_laue = _row_ids(asu_hkl)

    # Evaluate the symmetry predicates on representatives only.
    rep = np.zeros(n_laue, dtype=np.int64)
    rep[laue_ids] = np.arange(laue_ids.size, dtype=np.int64)
    rep_hkl = asu_hkl[rep]

    centric_u = np.empty(n_laue, dtype=bool)
    epsilon_u = np.empty(n_laue, dtype=np.float64)
    absent_u = np.empty(n_laue, dtype=bool)
    for i, (h, k, ll) in enumerate(rep_hkl):
        idx = (int(h), int(k), int(ll))
        centric_u[i] = ops.is_reflection_centric(idx)
        epsilon_u[i] = float(ops.epsilon_factor(idx))
        absent_u[i] = ops.is_systematically_absent(idx)

    centric = centric_u[laue_ids]

    # Anomalous groups split each Laue group by hemisphere, but ONLY for
    # acentrics. A centric reflection satisfies h ~ -h under the
    # rotation group alone, so cctbx's `map_to_asu` sends both Friedel
    # mates to a single index even with `anomalous_flag` set. Splitting
    # centrics by parity would invent spurious extra groups and inflate
    # the group count above cctbx's.
    hemisphere = np.where(centric, 0, plus.astype(np.int64))
    anom_key = laue_ids * 2 + hemisphere
    uniq_anom, anom_ids = np.unique(anom_key, return_inverse=True)
    anom_ids = anom_ids.astype(np.int64, copy=False)
    n_anom = int(uniq_anom.size)

    return AsuMap(
        asu_hkl=torch.from_numpy(asu_hkl).to(IDTYPE),
        isym=torch.from_numpy(isym).to(IDTYPE),
        plus=torch.from_numpy(plus),
        centric=torch.from_numpy(centric_u[laue_ids]),
        epsilon=torch.from_numpy(epsilon_u[laue_ids]).to(DTYPE),
        sys_absent=torch.from_numpy(absent_u[laue_ids]),
        group_id=torch.from_numpy(anom_ids).to(IDTYPE),
        group_id_laue=torch.from_numpy(laue_ids).to(IDTYPE),
        n_groups=n_anom,
        n_groups_laue=n_laue,
    )


def complete_set(
    cell: tuple[float, float, float, float, float, float],
    spacegroup: str | int | gemmi.SpaceGroup,
    d_min: float,
    d_max: float | None = None,
    anomalous: bool = False,
) -> torch.Tensor:
    """Enumerate all symmetry-unique reflections in a resolution range.

    Replaces `cctbx.miller.set.complete_set`, which is what DIALS uses
    as the completeness denominator.

    Args:
        cell: unit cell parameters `(a, b, c, alpha, beta, gamma)`.
        spacegroup: space group of the crystal.
        d_min: high resolution limit in Angstrom, inclusive.
        d_max: low resolution limit in Angstrom, inclusive. `None` means
            no low-resolution cut.
        anomalous: keep Friedel mates as distinct reflections.

    Returns:
        `(M, 3)` integer tensor of unique Miller indices, systematic
        absences removed.
    """
    sg = as_spacegroup(spacegroup)
    uc = gemmi.UnitCell(*cell)
    ops = sg.operations()

    # `gemmi.make_miller_array` rejects a `None` low-resolution limit, so
    # the documented "no cut" case has to be spelled as 0.0.
    hkl = gemmi.make_miller_array(
        uc, sg, d_min, 0.0 if d_max is None else d_max, unique=True
    )
    hkl = np.asarray(hkl, dtype=np.int64)

    keep = np.array(
        [
            not ops.is_systematically_absent((int(h), int(k), int(ll)))
            for h, k, ll in hkl
        ],
        dtype=bool,
    )
    hkl = hkl[keep]

    if anomalous:
        acentric = np.array(
            [
                not ops.is_reflection_centric((int(h), int(k), int(ll)))
                for h, k, ll in hkl
            ],
            dtype=bool,
        )
        hkl = np.concatenate([hkl, -hkl[acentric]], axis=0)

    return torch.from_numpy(np.ascontiguousarray(hkl)).to(IDTYPE)


def resolution(
    hkl: torch.Tensor,
    cell: tuple[float, float, float, float, float, float],
) -> torch.Tensor:
    """Compute d-spacing for Miller indices.

    Args:
        hkl: `(N, 3)` integer Miller indices.
        cell: unit cell parameters `(a, b, c, alpha, beta, gamma)`.

    Returns:
        `(N,)` resolution in Angstrom.
    """
    uc = gemmi.UnitCell(*cell)
    h_np = hkl.detach().cpu().numpy().astype(np.int64)
    d = np.array(
        [uc.calculate_d((int(h), int(k), int(ll))) for h, k, ll in h_np],
        dtype=np.float64,
    )
    return torch.from_numpy(d).to(DTYPE)
