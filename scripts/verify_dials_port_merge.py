"""Check `dials_port.merge` and `.french_wilson` against cctbx/DIALS.

Merges a scaled HEWL dataset with both implementations and reports the
relative disagreement in merged intensity, merged sigma, the merging
R-factors and the French-Wilson output.

Both paths are deterministic, so the only expected disagreement is
floating-point summation order; anything above `~1e-12` relative points
at an algorithmic difference.

Run from the repository root with the DIALS-enabled interpreter::

    /Users/luis/micromamba/envs/integrator-dev/bin/python \\
        scripts/verify_dials_port_merge.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "src")

from integrator.dials_port import io, symmetry  # noqa: E402
from integrator.dials_port.french_wilson import (  # noqa: E402
    french_wilson,
)
from integrator.dials_port.merge import (  # noqa: E402
    merge,
    r_factors,
    scale_observations,
)

DEFAULT_NPZ = (
    "/private/tmp/claude-501/-Users-luis-integrator/"
    "a69c47c1-a7b1-409e-b127-f918bd14acbc/scratchpad/hewl_scaled.npz"
)
DEFAULT_EXPT = "/Users/luis/dials_out/816_sbgrid_HEWL/pass1/scaled.expt"


def rel_stats(mine: np.ndarray, theirs: np.ndarray) -> dict[str, float]:
    """Relative difference summary between two aligned arrays."""
    denom = np.maximum(np.abs(theirs), 1e-30)
    rel = np.abs(mine - theirs) / denom
    return {
        "max": float(rel.max()),
        "median": float(np.median(rel)),
        "n": int(rel.size),
        "max_abs": float(np.abs(mine - theirs).max()),
    }


def show(label: str, stats: dict[str, float]) -> None:
    print(
        f"  {label:<28s} n={stats['n']:>7d}  "
        f"max_rel={stats['max']:.3e}  med_rel={stats['median']:.3e}  "
        f"max_abs={stats['max_abs']:.3e}"
    )


def cctbx_merge(
    hkl: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
    cell: tuple[float, ...],
    space_group: str,
    anomalous: bool,
    use_internal_variance: bool,
):
    """Merge with real cctbx, returning the `merge_equivalents` object."""
    from cctbx import crystal, miller, sgtbx  # noqa: PLC0415
    from cctbx.array_family import flex  # noqa: PLC0415

    sym = crystal.symmetry(
        unit_cell=tuple(cell),
        space_group_info=sgtbx.space_group_info(
            f"Hall: {space_group.strip()}"
        ),
        assert_is_compatible_unit_cell=False,
    )
    indices = flex.miller_index(
        [(int(h), int(k), int(ll)) for h, k, ll in hkl]
    )
    ms = miller.set(
        crystal_symmetry=sym, indices=indices, anomalous_flag=anomalous
    )
    arr = miller.array(ms, data=flex.double(intensity))
    arr.set_observation_type_xray_intensity()
    arr.set_sigmas(flex.double(sigma))
    return arr.merge_equivalents(use_internal_variance=use_internal_variance)


def align(
    mine_hkl: np.ndarray,
    mine_plus: np.ndarray,
    mine_centric: np.ndarray,
    theirs_hkl: np.ndarray,
    space_group: str,
    anomalous: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Match rows of two merged sets by (ASU index, hemisphere).

    cctbx and `reciprocalspaceship` do not have to agree on which
    member of an orbit represents it, so both index sets are pushed
    through the same ASU map before being paired. For a Laue merge the
    hemisphere is not part of the key; for an anomalous merge it is,
    except on centrics, which occupy a single group.

    Returns:
        `(mine_order, theirs_order)`, index arrays selecting the shared
        rows of each set in a common order.
    """
    their_map = symmetry.build_asu_map(
        torch.from_numpy(theirs_hkl.astype(np.int64)), space_group
    )
    their_hkl = their_map.asu_hkl.numpy()
    if anomalous:
        their_side = np.where(
            their_map.centric.numpy(), 0, their_map.plus.numpy()
        ).astype(np.int64)
        my_side = np.where(mine_centric, 0, mine_plus).astype(np.int64)
    else:
        their_side = np.zeros(len(their_hkl), dtype=np.int64)
        my_side = np.zeros(len(mine_hkl), dtype=np.int64)

    their_key = np.concatenate([their_hkl, their_side[:, None]], axis=1)
    my_key = np.concatenate([mine_hkl, my_side[:, None]], axis=1)

    def pack(a: np.ndarray) -> np.ndarray:
        c = np.ascontiguousarray(a)
        return c.view([("", c.dtype)] * c.shape[1]).ravel()

    mk, tk = pack(my_key), pack(their_key)
    common = np.intersect1d(mk, tk)
    if common.size != mk.size or common.size != tk.size:
        print(
            f"  WARNING: row sets differ "
            f"(mine={mk.size}, theirs={tk.size}, common={common.size})"
        )
    m_sort = np.argsort(mk)
    t_sort = np.argsort(tk)
    return (
        m_sort[np.searchsorted(mk[m_sort], common)],
        t_sort[np.searchsorted(tk[t_sort], common)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", default=DEFAULT_NPZ)
    parser.add_argument("--expt", default=DEFAULT_EXPT)
    parser.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args()

    refl = io.load_npz(args.npz)
    expt = io.load_experiment(args.expt)
    sg = expt.space_group
    print(f"loaded {len(refl)} observations, space group {sg!r}")
    print(f"cell {tuple(round(c, 4) for c in expt.cell)}")

    scale = refl.scale.numpy()
    print(
        f"inverse scale factor range "
        f"[{scale.min():.4g}, {scale.max():.4g}], "
        f"n_nonpositive={int((scale <= 0).sum())}"
    )

    i_scaled, var_scaled = scale_observations(refl)
    sig_scaled = torch.sqrt(var_scaled)
    hkl = refl.hkl.numpy()

    worst = 0.0
    for anomalous in (False, True):
        amap = symmetry.build_asu_map(refl.hkl, sg)
        for uiv in (False, True):
            tag = f"anomalous={anomalous} internal_var={uiv}"
            print(f"\n[{tag}]")

            mine = merge(
                refl, amap, anomalous=anomalous, use_internal_variance=uiv
            )
            ref = cctbx_merge(
                hkl,
                i_scaled.numpy(),
                sig_scaled.numpy(),
                expt.cell,
                sg,
                anomalous,
                uiv,
            )
            ref_arr = ref.array()
            t_hkl = np.asarray(ref_arr.indices(), dtype=np.int64)
            t_i = np.asarray(ref_arr.data(), dtype=np.float64)
            t_s = np.asarray(ref_arr.sigmas(), dtype=np.float64)
            t_n = np.asarray(ref.redundancies().data(), dtype=np.int64)

            mo, to = align(
                mine.hkl.numpy(),
                mine.plus.numpy(),
                mine.centric.numpy(),
                t_hkl,
                sg,
                anomalous,
            )
            print(f"  merged reflections: mine={len(mine)} cctbx={len(t_i)}")

            s_i = rel_stats(mine.intensity.numpy()[mo], t_i[to])
            s_s = rel_stats(mine.sigma.numpy()[mo], t_s[to])
            show("intensity", s_i)
            show("sigma", s_s)
            worst = max(worst, s_i["max"], s_s["max"])

            n_bad = int((mine.multiplicity.numpy()[mo] != t_n[to]).sum())
            print(f"  multiplicity mismatches:     {n_bad}")
            worst = max(worst, float(n_bad))

            if uiv:
                continue

            rf = r_factors(refl, amap, anomalous=anomalous)
            cc = {
                "rmerge": ref.r_merge(),
                "rmeas": ref.r_meas(),
                "rpim": ref.r_pim(),
                "rint": ref.r_int(),
            }
            for key in ("rmerge", "rmeas", "rpim", "rint"):
                rel = abs(rf[key] - cc[key]) / max(abs(cc[key]), 1e-30)
                print(
                    f"  {key:<7s} mine={rf[key]:.10f} "
                    f"cctbx={cc[key]:.10f}  rel={rel:.3e}"
                )
                worst = max(worst, rel)

    print("\n[french_wilson, anomalous=False]")
    amap = symmetry.build_asu_map(refl.hkl, sg)
    mine = merge(refl, amap)
    # Take d from the merged index and cell, as cctbx does, so the
    # comparison isolates the French-Wilson algorithm from any
    # difference in the per-observation `d` column.
    mine.d = symmetry.resolution(mine.hkl, expt.cell)
    worst = max(worst, compare_french_wilson(mine, expt.cell, sg))

    print(f"\nworst relative disagreement overall: {worst:.3e}")
    ok = worst <= args.tol
    print("PASS" if ok else f"FAIL (tolerance {args.tol:.1e})")
    return 0 if ok else 1


def compare_french_wilson(
    merged, cell: tuple[float, ...], space_group: str
) -> float:
    """Compare the torch French-Wilson against the DIALS original."""
    from cctbx import crystal, miller, sgtbx  # noqa: PLC0415
    from cctbx.array_family import flex  # noqa: PLC0415
    from dials.algorithms.merging.french_wilson import (  # noqa: PLC0415
        french_wilson as dials_fw,
    )

    sym = crystal.symmetry(
        unit_cell=tuple(cell),
        space_group_info=sgtbx.space_group_info(
            f"Hall: {space_group.strip()}"
        ),
        assert_is_compatible_unit_cell=False,
    )
    hkl = merged.hkl.numpy()
    ms = miller.set(
        crystal_symmetry=sym,
        indices=flex.miller_index(
            [(int(h), int(k), int(ll)) for h, k, ll in hkl]
        ),
        anomalous_flag=False,
    )
    arr = miller.array(ms, data=flex.double(merged.intensity.numpy()))
    arr.set_observation_type_xray_intensity()
    arr.set_sigmas(flex.double(merged.sigma.numpy()))

    # cctbx's centric flags must agree, or the branches diverge trivially.
    their_centric = np.asarray(arr.centric_flags().data(), dtype=bool)
    n_centric_bad = int((their_centric != merged.centric.numpy()).sum())
    print(f"  centric-flag mismatches:     {n_centric_bad}")

    ref = dials_fw(arr)
    ref_hkl = np.asarray(ref.indices(), dtype=np.int64)
    ref_f = np.asarray(ref.data(), dtype=np.float64)
    ref_sf = np.asarray(ref.sigmas(), dtype=np.float64)

    def pack(a: np.ndarray) -> np.ndarray:
        c = np.ascontiguousarray(a.astype(np.int64))
        return c.view([("", c.dtype)] * c.shape[1]).ravel()

    def compare(got: dict, label: str) -> float:
        valid = got["valid"].numpy()
        mk, tk = pack(hkl[valid]), pack(ref_hkl)
        common = np.intersect1d(mk, tk)
        if common.size != mk.size or common.size != tk.size:
            print(
                f"  WARNING: FW row sets differ "
                f"(mine={mk.size}, dials={tk.size}, common={common.size})"
            )
        ms_, ts_ = np.argsort(mk), np.argsort(tk)
        mo = ms_[np.searchsorted(mk[ms_], common)]
        to = ts_[np.searchsorted(tk[ts_], common)]
        print(f"  -- {label}: kept mine={int(valid.sum())} dials={len(ref_f)}")
        s_f = rel_stats(got["F"].numpy()[valid][mo], ref_f[to])
        s_sf = rel_stats(got["sigF"].numpy()[valid][mo], ref_sf[to])
        show("F", s_f)
        show("sigF", s_sf)
        missing = float(
            abs(common.size - mk.size) + abs(common.size - tk.size)
        )
        return max(s_f["max"], s_sf["max"], missing)

    # The equal-count binning takes its edges from the data, so a large
    # share of reflections land exactly on an edge and a one-ulp change
    # in the coordinate moves them between shells. Feeding cctbx's own
    # d*^3 tests the algorithm; deriving it from `d` tests the whole
    # path and shows how much that sensitivity is worth in practice.
    shared = torch.from_numpy(
        np.asarray(arr.d_star_cubed().data(), dtype=np.float64)
    )
    algo = compare(
        french_wilson(merged, d_star_cubed=shared), "cctbx d*^3 (algorithm)"
    )
    end_to_end = compare(french_wilson(merged), "own d*^3 (end to end)")
    print(f"  binning sensitivity costs {end_to_end:.2e} relative")
    return algo


if __name__ == "__main__":
    raise SystemExit(main())
