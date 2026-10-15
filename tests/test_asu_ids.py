"""The map from `asu_id` to a Miller index.

An `asu_id` is the index into a trained model's `hkl_table`, so the map has to
be resolved for the ids that exist rather than rebuilt with a fresh numbering.
The rule also has to leave centrics alone: a centric reflection is its own
Friedel mate, and negating it splits one reflection across two ids and invents
an anomalous difference that cannot exist.
"""

import gemmi
import numpy as np
import pytest
import reciprocalspaceship as rs

from integrator.utils.asu import canonical_hkl, resolve_asu_id_table

SG = "P 43 21 2"


def _observations(seed: int = 1, n: int = 800) -> np.ndarray:
    """A reflection table: every symmetry mate, and every Friedel mate."""
    sg = gemmi.SpaceGroup(SG)
    rng = np.random.default_rng(seed)
    base = rng.integers(-9, 10, size=(n, 3))
    base = base[~np.all(base == 0, axis=1)]
    base = base[~rs.utils.is_absent(base, sg)]
    out = []
    for h in base:
        for op in sg.operations().sym_ops:
            m = np.array(op.rot, dtype=np.float64) / op.DEN
            g = (h @ m).astype(np.int32)
            out.append(g)
            out.append(-g)
    return np.asarray(out, dtype=np.int32)


def test_centrics_are_never_split_by_the_anomalous_grouping():
    """A centric is its own Friedel mate, so it owns exactly one id."""
    sg = gemmi.SpaceGroup(SG)
    obs = _observations()
    asu_hkl, _ = rs.utils.hkl_to_asu(obs, sg)
    centric = np.asarray(rs.utils.is_centric(asu_hkl, sg), dtype=bool)
    assert centric.any()

    canon = canonical_hkl(obs, sg, anomalous=True)
    assert np.array_equal(canon[centric], asu_hkl[centric])

    rows = {tuple(r) for r in canon}
    assert not any(tuple(-r) in rows for r in canon[centric])


def test_acentrics_split_exactly_two_ways():
    """The anomalous grouping doubles the acentric count and nothing else."""
    sg = gemmi.SpaceGroup(SG)
    obs = _observations()
    anom = canonical_hkl(obs, sg, anomalous=True)
    pooled = canonical_hkl(obs, sg, anomalous=False)
    centric = np.asarray(rs.utils.is_centric(pooled, sg), dtype=bool)

    n_anom = len({tuple(r) for r in anom[~centric]})
    n_pooled = len({tuple(r) for r in pooled[~centric]})
    assert n_anom == 2 * n_pooled


@pytest.mark.parametrize("anomalous", [True, False])
def test_the_table_is_resolved_without_renumbering_the_ids(anomalous):
    """Existing ids keep their meaning; only their Miller index is recovered."""
    sg = gemmi.SpaceGroup(SG)
    obs = _observations()
    keys, ids = np.unique(
        canonical_hkl(obs, sg, anomalous=anomalous), axis=0, return_inverse=True
    )

    table, info = resolve_asu_id_table(ids, obs, sg)
    assert info["grouping"] == ("anomalous" if anomalous else "pooled")
    assert info["n_unobserved_ids"] == 0
    assert np.array_equal(table, keys.astype(np.int32))


def test_ids_from_another_table_are_refused_rather_than_guessed():
    """A wrong map would quietly scramble the MTZ, so it has to raise."""
    sg = gemmi.SpaceGroup(SG)
    obs = _observations()
    _, ids = np.unique(canonical_hkl(obs, sg), axis=0, return_inverse=True)
    rng = np.random.default_rng(0)
    scrambled = rng.permutation(ids)

    with pytest.raises(ValueError, match="consistent with neither grouping"):
        resolve_asu_id_table(scrambled, obs, sg)
