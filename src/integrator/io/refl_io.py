import numpy as np
# dials.array_family.flex is imported locally inside functions that need it.
# Do NOT import it at module level: integrator-cuda-dev does not have dials.


def unstack_preds(preds: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate per-batch prediction lists into single arrays."""
    return {k: np.concatenate(v) for k, v in preds.items()}


def write_refl_with_predictions(
    refl_file,
    out_file,
    refl_ids,
    i_value,
    i_variance,
    bg_mean,
):
    """Overwrite the intensity/background columns of a DIALS `.refl`.

    Args:
        refl_file: source `.refl` (must carry a `refl_ids` column).
        out_file: destination `.refl`.
        refl_ids: integer ids selecting/ordering the prediction rows.
        i_value: predicted intensity (written to intensity.prf/sum.value).
        i_variance: predicted variance (written to intensity.prf/sum.variance).
        bg_mean: predicted background (written to background.mean).
    """
    from dials.array_family import flex
    from dxtbx import flumpy

    rt = flex.reflection_table.from_file(str(refl_file))
    if "refl_ids" not in rt:
        raise KeyError(
            f"{refl_file} has no 'refl_ids' column; expected a .refl written "
            "by integrator.mksbox"
        )

    pred_ids = np.asarray(refl_ids, dtype=np.int64)
    order = np.argsort(pred_ids)
    pred_ids = pred_ids[order]
    i_value = np.asarray(i_value, dtype=np.float64)[order]
    i_variance = np.asarray(i_variance, dtype=np.float64)[order]
    bg_mean = np.asarray(bg_mean, dtype=np.float64)[order]

    table_ids = flumpy.to_numpy(rt["refl_ids"]).astype(np.int64)
    keep = np.isin(table_ids, pred_ids)
    rt = rt.select(flumpy.from_numpy(keep))
    rt = rt.select(flex.sort_permutation(rt["refl_ids"]))

    sel_ids = flumpy.to_numpy(rt["refl_ids"]).astype(np.int64)
    if sel_ids.shape[0] != pred_ids.shape[0] or not np.array_equal(
        sel_ids, pred_ids
    ):
        raise ValueError(
            "refl_ids in the .refl do not match the prediction refl_ids after "
            f"selection ({sel_ids.shape[0]} table rows vs "
            f"{pred_ids.shape[0]} predictions)"
        )

    rt["intensity.prf.value"] = flumpy.from_numpy(i_value)
    rt["intensity.prf.variance"] = flumpy.from_numpy(i_variance)
    rt["intensity.sum.value"] = flumpy.from_numpy(i_value)
    rt["intensity.sum.variance"] = flumpy.from_numpy(i_variance)
    rt["background.mean"] = flumpy.from_numpy(bg_mean)

    rt.as_file(str(out_file))
    return rt


def _as_numpy(v):
    """Return tensor/list/array metadata as a NumPy array."""
    try:
        import torch

        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(v)


def _as_text(v):
    """Normalize NumPy scalar/bytes/path-like values to a Python string."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, bytes):
        return v.decode()
    return str(v)


def _mfx_source_name(metadata, key, row, image_number, suffix):
    """Return source file name from metadata, with image-number fallback."""
    if key in metadata:
        name = _as_text(_as_numpy(metadata[key])[row])
        if name:
            return name
    return f"idx-data_{int(image_number):05d}_integrated{suffix}"


def _copy_or_link_expt(src, dst, copy_expt=False):
    """Copy or symlink a matching .expt file into the write-back output."""
    import os
    import shutil

    if dst.exists() or dst.is_symlink():
        return
    if copy_expt:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def write_mfx_refl_with_predictions(
    pred_data,
    metadata_path,
    original_refl_dir,
    out_dir,
    variance_floor=1.0e-6,
    copy_expt=False,
):
    """Write Integrator predictions into many original MFX .refl files.

    MFX write-back extension (Thao): Luis's original write_refl_with_predictions()
    assumes one source .refl with a refl_ids column. This function supports the
    MFX/cctbx case where one prediction table maps back to many files:

        prediction refl_ids -> metadata row -> source_refl/source_expt or
        image_num/image_index -> reflection_id row inside that source .refl

    Required prediction columns: refl_ids, qi_mean, qi_var, qbg_mean.
    Required metadata columns: refl_ids, reflection_id, plus either
    source_refl/source_expt or image_num/image_index.
    """
    from pathlib import Path

    from dials.array_family import flex
    from dxtbx import flumpy

    from .metadata import load_metadata

    metadata_path = Path(metadata_path)
    original_refl_dir = Path(original_refl_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path)

    pred_ids = np.asarray(pred_data["refl_ids"], dtype=np.int64)
    pred_order = np.argsort(pred_ids)
    pred_ids = pred_ids[pred_order]
    qi_mean = np.asarray(pred_data["qi_mean"], dtype=np.float64)[pred_order]
    qi_var = np.asarray(pred_data["qi_var"], dtype=np.float64)[pred_order]
    qbg_mean = np.asarray(pred_data["qbg_mean"], dtype=np.float64)[pred_order]
    qi_var = np.clip(qi_var, variance_floor, None)

    metadata_ids = _as_numpy(metadata["refl_ids"]).astype(np.int64)
    metadata_order = np.argsort(metadata_ids)
    metadata_ids_sorted = metadata_ids[metadata_order]

    # BatchPredWriter stores refl_ids as float32, which loses precision for
    # values > 2^23 (~8.4 M). Large IDs get rounded; at the boundary this
    # can push the maximum value one step beyond the metadata range.
    # Clamp before searchsorted so edge over-counts map to the nearest valid
    # metadata row instead of raising an out-of-bounds error.
    meta_lo = int(metadata_ids_sorted[0])
    meta_hi = int(metadata_ids_sorted[-1])
    n_oob = int(np.sum((pred_ids < meta_lo) | (pred_ids > meta_hi)))
    if n_oob > 0:
        import warnings

        warnings.warn(
            f"{n_oob} pred refl_ids are outside metadata range "
            f"[{meta_lo}, {meta_hi}] (float32 boundary rounding). "
            "Clamping to nearest valid ID.",
            UserWarning,
            stacklevel=2,
        )
        pred_ids = np.clip(pred_ids, meta_lo, meta_hi)

    found = np.searchsorted(metadata_ids_sorted, pred_ids)
    found = np.clip(found, 0, len(metadata_ids_sorted) - 1)
    if not np.array_equal(metadata_ids_sorted[found], pred_ids):
        n_bad = int((metadata_ids_sorted[found] != pred_ids).sum())
        raise ValueError(
            f"{n_bad:,} prediction refl_ids do not match any metadata refl_id "
            "after clamping. Check that the correct metadata.npy is being used."
        )
    metadata_rows = metadata_order[found]

    if "reflection_id" not in metadata:
        raise KeyError(
            "metadata.npy must contain reflection_id for MFX write-back"
        )
    reflection_id = _as_numpy(metadata["reflection_id"]).astype(np.int64)

    if "image_num" in metadata:
        image_numbers = _as_numpy(metadata["image_num"]).astype(np.int64)
    elif "image_index" in metadata:
        image_numbers = _as_numpy(metadata["image_index"]).astype(np.int64)
    elif "source_refl" in metadata:
        image_numbers = np.full(len(metadata_ids), -1, dtype=np.int64)
    else:
        raise KeyError(
            "metadata.npy must contain source_refl/source_expt or image_num/image_index"
        )

    row_in_source = reflection_id[metadata_rows]

    # Pull source_refl / source_expt via vectorized fancy-index — one C call,
    # no per-row Python loop. metadata["source_refl"] is the source of truth
    # for filenames; run-prefixed names (e.g. r0269_idx-data_01418_integrated.refl)
    # are NOT reconstructible from image_num alone.
    source_refl_all = _as_numpy(metadata["source_refl"])
    source_expt_all = _as_numpy(metadata["source_expt"])
    source_refl_subset = source_refl_all[metadata_rows]
    source_expt_subset = source_expt_all[metadata_rows]

    # Encode unique source_refl strings to sorted int IDs — one C pass.
    # Groupby via argsort on encoded IDs: O(N log N), no O(N x n_files) scan.
    unique_refl, encoded = np.unique(source_refl_subset, return_inverse=True)
    order = np.argsort(encoded, kind="stable")
    sorted_encoded = encoded[order]
    boundaries = np.flatnonzero(sorted_encoded[1:] != sorted_encoded[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))

    written = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        group = order[start:end]
        refl_name = _as_text(unique_refl[i])

        # Validate exactly one .expt per .refl group.
        expt_names = np.unique(source_expt_subset[group])
        if len(expt_names) != 1:
            raise ValueError(
                f"Expected one source_expt for {refl_name}, "
                f"got {expt_names.tolist()}"
            )
        expt_name = _as_text(expt_names[0])

        src_refl = original_refl_dir / refl_name
        if not src_refl.exists():
            raise FileNotFoundError(f"source .refl not found: {src_refl}")

        rt = flex.reflection_table.from_file(str(src_refl))
        n_rows = len(rt)
        rows = row_in_source[group].astype(np.int64)
        if np.any(rows < 0) or np.any(rows >= n_rows):
            raise IndexError(
                f"reflection_id out of range for {src_refl}: "
                f"table has {n_rows} rows"
            )

        def _set_column(name, values):
            if name in rt:
                arr = flumpy.to_numpy(rt[name]).astype(np.float64, copy=True)
            else:
                arr = np.zeros(n_rows, dtype=np.float64)
            arr[rows] = values
            rt[name] = flumpy.from_numpy(arr)

        _set_column("intensity.prf.value", qi_mean[group])
        _set_column("intensity.prf.variance", qi_var[group])
        _set_column("intensity.sum.value", qi_mean[group])
        _set_column("intensity.sum.variance", qi_var[group])
        _set_column("background.mean", qbg_mean[group])

        out_refl = out_dir / refl_name
        rt.as_file(str(out_refl))
        written.append(out_refl)

        src_expt = original_refl_dir / expt_name
        if not src_expt.exists():
            raise FileNotFoundError(f"source .expt not found: {src_expt}")
        _copy_or_link_expt(src_expt, out_dir / expt_name, copy_expt=copy_expt)

    return {
        "out_dir": out_dir,
        "n_prediction_rows": int(len(pred_ids)),
        "n_refl_files": int(len(written)),
        "written_refl_files": written,
    }
