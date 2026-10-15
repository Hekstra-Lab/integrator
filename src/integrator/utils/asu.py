"""Map Miller indices to asymmetric-unit identifiers.

Two groupings are in use.  The `anomalous` grouping keeps Friedel mates
apart, so an acentric reflection and its mate carry different ids; the
`pooled` grouping merges them.  A centric reflection is its own Friedel
mate, so it is never split -- negating it would turn one reflection into
two ids and invent an anomalous difference that cannot exist.
"""

import numpy as np
import reciprocalspaceship as rs


def canonical_hkl(hkl, spacegroup, anomalous: bool = True) -> np.ndarray:
    """Canonical ASU representative of each observed Miller index.

    Args:
        hkl: Observed Miller indices, shape (N, 3).
        spacegroup: `gemmi.SpaceGroup`.
        anomalous: Keep Friedel mates apart.

    Returns:
        (N, 3) int32 canonical indices.  With `anomalous`, acentric
        Friedel-minus observations (ISYM even) are negated; centrics stay
        in the plus form.
    """
    hkl = np.asarray(hkl, dtype=np.int32)
    asu_hkl, isym = rs.utils.hkl_to_asu(hkl, spacegroup)
    if not anomalous:
        return asu_hkl.astype(np.int32)

    centric = np.asarray(rs.utils.is_centric(asu_hkl, spacegroup), dtype=bool)
    is_minus = (isym % 2 == 0) & ~centric
    canon = asu_hkl.copy()
    canon[is_minus] = -canon[is_minus]
    return canon.astype(np.int32)


def _table_from_ids(
    asu_id: np.ndarray, canon: np.ndarray, n_hkl: int
) -> tuple[np.ndarray, int, int]:
    """First-seen index per id, plus how many observations disagree."""
    table = np.zeros((n_hkl, 3), dtype=np.int32)
    filled = np.zeros(n_hkl, dtype=bool)

    order = np.argsort(asu_id, kind="stable")
    first = order[np.unique(asu_id[order], return_index=True)[1]]
    table[asu_id[first]] = canon[first]
    filled[asu_id[first]] = True

    n_disagree = int((canon != table[asu_id]).any(axis=1).sum())
    return table, n_disagree, int((~filled).sum())


def resolve_asu_id_table(
    asu_id, hkl, spacegroup, n_hkl: int | None = None
) -> tuple[np.ndarray, dict]:
    """Miller index per existing `asu_id`, without reassigning any id.

    The grouping is read off the data rather than assumed: the anomalous
    canonicalization is kept when every observation of every id agrees on
    it, otherwise the pooled one.  Ids already in a trained model's
    `hkl_table` keep their meaning, which is what separates this from
    rebuilding the ids from scratch.

    Args:
        asu_id: Per-observation id, shape (N,).
        hkl: Per-observation Miller indices, shape (N, 3).
        spacegroup: `gemmi.SpaceGroup`.
        n_hkl: Table length.  Defaults to `max(asu_id) + 1`.

    Returns:
        `(table, info)`.  `table[i]` is the canonical index of `asu_id == i`;
        rows for ids with no observations are zero.  `info` carries the
        grouping chosen, the disagreement count under each grouping, and
        the number of unobserved ids.

    Raises:
        ValueError: if neither grouping is consistent, which means the ids
            do not come from a symmetry canonicalization of these indices.
    """
    asu_id = np.asarray(asu_id, dtype=np.int64)
    hkl = np.asarray(hkl, dtype=np.int32)
    if n_hkl is None:
        n_hkl = int(asu_id.max()) + 1

    anom = canonical_hkl(hkl, spacegroup, anomalous=True)
    pooled = canonical_hkl(hkl, spacegroup, anomalous=False)

    t_anom, d_anom, n_missing = _table_from_ids(asu_id, anom, n_hkl)
    t_pooled, d_pooled, _ = _table_from_ids(asu_id, pooled, n_hkl)

    info = {
        "n_obs": int(len(asu_id)),
        "n_hkl": n_hkl,
        "n_disagree_anomalous": d_anom,
        "n_disagree_pooled": d_pooled,
        "n_unobserved_ids": n_missing,
    }

    if d_anom == 0:
        info["grouping"] = "anomalous"
        return t_anom, info
    if d_pooled == 0:
        info["grouping"] = "pooled"
        return t_pooled, info

    raise ValueError(
        "asu_id is consistent with neither grouping: "
        f"{d_anom} of {len(asu_id)} observations disagree with the "
        f"anomalous canonicalization and {d_pooled} with the pooled one. "
        "The ids were probably built from a different space group or a "
        "different reflection table."
    )
