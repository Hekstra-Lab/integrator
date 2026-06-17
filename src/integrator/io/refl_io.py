import numpy as np


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
