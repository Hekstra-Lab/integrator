from pathlib import Path

import numpy as np
import reciprocalspaceship as rs
import reciprocalspaceship.io as rs_io
import torch

from .dtypes import DEFAULT_EXCLUDED_COLS, DEFAULT_REFL_COLS


def _to_numpy(v):
    return v.numpy() if torch.is_tensor(v) else np.asarray(v)


def save_data(obj, path) -> Path:
    """Save a tensor or dict-of-tensors.

    Writes `.npy` by default; writes `.pt` only when path ends
    in `.pt`. Returns the path actually written.
    """
    p = Path(path)
    if p.suffix == ".pt":
        torch.save(obj, p)
        return p
    p = p.with_suffix(".npy")
    if isinstance(obj, dict):
        np.save(
            p, {k: _to_numpy(v) for k, v in obj.items()}, allow_pickle=True
        )
    else:
        np.save(p, _to_numpy(obj))
    return p


def data_path(path) -> Path | None:
    p = Path(path)
    npy = p.with_suffix(".npy")
    if npy.exists():
        return npy
    pt = p.with_suffix(".pt")
    if pt.exists():
        return pt
    return p if p.exists() else None


def load_data(path, map_location="cpu"):
    """Load a tensor or dict-of-tensors"""
    target = data_path(path) or Path(path)
    if target.suffix == ".npy":
        arr = np.load(target, allow_pickle=True)
        if arr.dtype == object:
            obj = arr.item()
            if isinstance(obj, dict):
                return {k: torch.as_tensor(v) for k, v in obj.items()}
            return torch.as_tensor(obj)
        return torch.as_tensor(arr)
    try:
        return torch.load(target, weights_only=True, map_location=map_location)
    except Exception:
        return torch.load(
            target, weights_only=False, map_location=map_location
        )


def load_metadata(path, map_location="cpu") -> dict:
    """Load a per-reflection metadata dict."""
    return load_data(path, map_location=map_location)


def refl_as_pt(
    refl,
    column_names: list[str] = DEFAULT_REFL_COLS,
    excluded_columns: list[str] = DEFAULT_EXCLUDED_COLS,
    out_dir: Path | None = None,
    out_fname: str = "metadata.npy",
) -> dict:
    ds = rs_io.read_dials_stills(
        refl,
        extra_cols=column_names,
    )
    assert isinstance(ds, rs.DataSet)

    data = {}
    for k, v in ds.items():
        if k not in excluded_columns:
            data[k] = torch.tensor(v, dtype=torch.float32)

    if out_dir is not None:
        fname = Path(out_dir) / out_fname
    else:
        fname = Path(out_fname)
    save_data(data, fname)
    return data


def _contiguous_group_ids(
    hkl: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Contiguous integer id per unique (h, k, l) row.

    Returns `(inverse, n_unique, table)` where `inverse[i]` is the id of row i
    and `table[id]` is the (h, k, l) of that id (sorted-unique order).
    """
    uniq, inverse = np.unique(hkl, axis=0, return_inverse=True)
    inverse = inverse.astype(np.int64)
    n = int(inverse.max()) + 1 if inverse.size else 0
    return inverse, n, uniq.astype(np.int32)


def miller_index_columns(
    H,
    K,
    L,
    space_group,
    cell=None,
    anomalous: bool = False,
) -> tuple[dict, dict, dict]:
    """Friedel-pooled / -separate Miller-index group ids + Friedel flags.

    Maps each `(H, K, L)` to its asymmetric-unit representative
    (`rs.utils.hkl_to_asu`) and assigns contiguous integer ids the merging model
    batches and merges over:

        miller_idx_friedelized    - Friedel-pooled id: both mates (H and -H)
                                    share an id (Friedel's law applied).
        miller_idx_unfriedelized  - Friedel-separate id: I(+) and I(-) get
                                    distinct ids (only when `anomalous`).

    Also returns `friedel_plus` (the I(+) member, ISYM odd) and `centric`.

    Args:
        H, K, L: per-observation Miller indices (int-like array or tensor).
        space_group: `gemmi.SpaceGroup`, or an int number / Hermann-Mauguin str.
        cell: 6 unit-cell params (only used for centric labeling); a placeholder
            is used when None.
        anomalous: also emit `miller_idx_unfriedelized`.

    Returns:
        `(columns, counts, hkl_tables)`: `columns` maps each name to a torch
        tensor; `counts` has `n_friedelized` and (if `anomalous`)
        `n_unfriedelized`; `hkl_tables` maps each id column to an `(n_id, 3)`
        int array whose row `i` is the canonical `(h, k, l)` of id `i`.
    """
    import gemmi
    import reciprocalspaceship as rs

    if isinstance(space_group, gemmi.SpaceGroup):
        sg = space_group
    elif isinstance(space_group, int):
        sg = gemmi.SpaceGroup(space_group)
    else:
        sg = gemmi.SpaceGroup(str(space_group).split("(")[0].strip())

    H = _to_numpy(H).astype(np.int32).ravel()
    K = _to_numpy(K).astype(np.int32).ravel()
    L = _to_numpy(L).astype(np.int32).ravel()
    hkl = np.stack([H, K, L], axis=1)

    # ISYM odd = F(+)/hasu form, even = F(-). The pooled (friedelized) canonical
    # is the asu representative with no sign flip, so both mates share it.
    asu_hkl, isym = rs.utils.hkl_to_asu(hkl, sg)
    friedel_plus = isym % 2 == 1
    fried_ids, n_fried, fried_table = _contiguous_group_ids(asu_hkl)

    cellp = list(cell) if cell is not None else [1.0, 1.0, 1.0, 90.0, 90.0, 90.0]
    ds = rs.DataSet(
        {"H": H, "K": K, "L": L},
        cell=gemmi.UnitCell(*cellp),
        spacegroup=sg,
    ).set_index(["H", "K", "L"])
    centric = ds.label_centrics()["CENTRIC"].to_numpy().astype(bool)

    columns = {
        "miller_idx_friedelized": torch.from_numpy(fried_ids),
        "friedel_plus": torch.from_numpy(np.ascontiguousarray(friedel_plus)),
        "centric": torch.from_numpy(np.ascontiguousarray(centric)),
    }
    counts = {"n_friedelized": n_fried}
    hkl_tables = {"miller_idx_friedelized": fried_table}
    if anomalous:
        # Split I(+)/I(-) by flipping the sign of ACENTRIC F(-) observations.
        # Centrics have I(+) == I(-) by symmetry, so they must NOT be split --
        # they keep their pooled rep (one anomalous id). Without the `~centric`
        # guard, centrics whose F(-) form gets an even ISYM are over-split, which
        # inflates the anomalous reflection count by ~the centric count.
        canon = asu_hkl.copy()
        is_minus = (isym % 2 == 0) & (~centric)
        canon[is_minus] = -canon[is_minus]
        anom_ids, n_anom, anom_table = _contiguous_group_ids(canon)
        columns["miller_idx_unfriedelized"] = torch.from_numpy(anom_ids)
        counts["n_unfriedelized"] = n_anom
        hkl_tables["miller_idx_unfriedelized"] = anom_table
    return columns, counts, hkl_tables
