from pathlib import Path

import numpy as np
import reciprocalspaceship as rs
import reciprocalspaceship.io as io
import torch

SCALAR_DTYPES: dict = {
    "zeta": "double",
    "qe": "double",
    "profile.correlation": "double",
    "partiality": "double",
    "partial_id": "std::size_t",
    "panel": "std::size_t",
    "flags": "std::size_t",
    "num_pixels.valid": "int",
    "num_pixels.foreground": "int",
    "num_pixels.background_used": "int",
    "num_pixels.background": "int",
    "lp": "double",
    "intensity.prf.value": "double",
    "intensity.prf.variance": "double",
    "intensity.sum.value": "double",
    "intensity.sum.variance": "double",
    "imageset_id": "int",
    "entering": "bool",
    "d": "double",
    "background.mean": "double",
    "background.sum.value": "double",
    "background.sum.variance": "double",
    "refl_ids": "int",
}

VECTOR_COLUMNS = {
    "bbox": ("int6", 6),
    "s1": ("vec3<double>", 3),
    "xyzcal.mm": ("vec3<double>", 3),
    "xyzcal.px": ("vec3<double>", 3),
    "xyzobs.mm.value": ("vec3<double>", 3),
    "xyzobs.mm.variance": ("vec3<double>", 3),
    "xyzobs.px.value": ("vec3<double>", 3),
    "xyzobs.px.variance": ("vec3<double>", 3),
    "miller_index": ("cctbx::miller::index<>", 3),
}

# unique from the model
MODEL_OUT_KEYS = [
    "qbg_mean",
    "qbg_var",
    "qi_mean",
    "qi_var",
]

# Default columns from rs.io.read_dials_stills
# out will contain the following
FLOAT_COLS = [
    "zeta",
    "xyzobs.px.variance.0",
    "xyzobs.px.variance.1",
    "xyzobs.px.variance.2",
    "xyzobs.px.value.0",
    "xyzobs.px.value.1",
    "xyzobs.px.value.2",
    "xyzobs.mm.variance.0",
    "xyzobs.mm.variance.1",
    "xyzobs.mm.variance.2",
    "xyzobs.mm.value.0",
    "xyzobs.mm.value.1",
    "xyzobs.mm.value.2",
    "xyzcal.mm.0",
    "xyzcal.mm.1",
    "xyzcal.mm.2",
    "qe",
    "profile.correlation",
    "partiality",
    "lp",
    "intensity.prf.variance",
    "intensity.prf.value",
    "d",
    "background.sum.variance",
    "background.sum.value",
    "background.mean",
    "s1.0",
    "s1.1",
    "s1.2",
    "xyzcal.px.0",
    "xyzcal.px.1",
    "xyzcal.px.2",
    "intensity.sum.variance",
    "intensity.sum.value",
]

INT_COLS = [
    "refl_ids",
    "partial_id",
    "panel",
    "num_pixels.valid",
    "num_pixels.foreground",
    "num_pixels.background_used",
    "num_pixels.background",
    "imageset_id",
    "flags",
    "bbox.0",
    "bbox.1",
    "bbox.2",
    "bbox.3",
    "bbox.4",
    "bbox.5",
    "H",
    "K",
    "L",
]

BOOL_COLS = [
    "entering",
]

ALL_COLS = FLOAT_COLS + INT_COLS + BOOL_COLS

REFL_TAG: str = "dials::af::reflection_table"

REFL_VERSION: int = 1

DTYPE_TO_NUMPY = {
    "double": np.float64,
    "vec3<double>": np.float64,  # (N,3)
    "int": np.int32,
    "int6": np.int32,  # (N,6)
    "bool": np.uint8,  # 1 byte each
    "std::size_t": np.uint64,  # 8 bytes each
    "cctbx::miller::index<>": np.int32,  # (N,3) int32
}

# Mappings to invert DataFrame
SCALAR_DTYPES = {
    "zeta": "double",
    "qe": "double",
    "profile.correlation": "double",
    "partiality": "double",
    "partial_id": "std::size_t",
    "panel": "std::size_t",
    "flags": "std::size_t",
    "num_pixels.valid": "int",
    "num_pixels.foreground": "int",
    "num_pixels.background_used": "int",
    "num_pixels.background": "int",
    "lp": "double",
    "intensity.prf.value": "double",
    "intensity.prf.variance": "double",
    "intensity.sum.value": "double",
    "intensity.sum.variance": "double",
    "imageset_id": "int",
    "entering": "bool",
    "d": "double",
    "background.mean": "double",
    "background.sum.value": "double",
    "background.sum.variance": "double",
    "refl_ids": "int",
}

VECTOR_COLUMNS = {
    "bbox": ("int6", 6),
    "s1": ("vec3<double>", 3),
    "xyzcal.mm": ("vec3<double>", 3),
    "xyzcal.px": ("vec3<double>", 3),
    "xyzobs.mm.value": ("vec3<double>", 3),
    "xyzobs.mm.variance": ("vec3<double>", 3),
    "xyzobs.px.value": ("vec3<double>", 3),
    "xyzobs.px.variance": ("vec3<double>", 3),
    "miller_index": ("cctbx::miller::index<>", 3),
}


def _unstack_preds(
    preds: dict[str, list[np.ndarray]],
) -> dict:
    # combine epochs into a single array
    for k, v in preds.items():
        preds[k] = np.concatenate(v)
    return preds


def _extract_vector(preds, base, n):
    cols = [f"{base}.{i}" for i in range(n)]
    return np.stack([preds[c] for c in cols], axis=1)


def _extract_miller_index(preds):
    return np.stack(
        [preds["H"], preds["K"], preds["L"]],
        axis=1,
    )


def _cast_for_dials(
    arr: np.ndarray,
    dtype_name: str,
) -> np.ndarray:
    np_dtype = DTYPE_TO_NUMPY[dtype_name]
    return np.ascontiguousarray(arr.astype(np_dtype, copy=False))


def _dict_to_refl_columns(preds):
    columns = {}

    # vector columns
    for key, (dtype_name, n) in VECTOR_COLUMNS.items():
        if key == "miller_index":
            arr = _extract_miller_index(preds)
        else:
            arr = _extract_vector(preds, key, n)
        columns[key] = (_cast_for_dials(arr, dtype_name), dtype_name)

    # scalar columns
    for key, dtype_name in SCALAR_DTYPES.items():
        arr = np.asarray(preds[key])
        columns[key] = (_cast_for_dials(arr, dtype_name), dtype_name)

    return columns


DEFAULT_REFL_COLS = list(SCALAR_DTYPES.keys()) + list(VECTOR_COLUMNS.keys())
DEFAULT_EXCLUDED_COLS = ["BATCH", "PARTIAL"]


def refl_as_pt(
    refl,
    column_names: list[str] = DEFAULT_REFL_COLS,
    excluded_columns: list[str] = DEFAULT_EXCLUDED_COLS,
    out_dir: Path | None = None,
    out_fname: str = "metadata.pt",
) -> dict:
    ds = io.read_dials_stills(
        refl,
        extra_cols=column_names,
    )
    assert isinstance(ds, rs.DataSet)

    data = {}
    for k, v in ds.items():
        if k not in excluded_columns:
            data[k] = torch.tensor(v, dtype=torch.float32)

    # write to output directory if specified
    if out_dir is not None:
        fname = Path(out_dir) / out_fname
    else:
        fname = Path(out_fname)

    if fname.suffix == ".npy":
        # numpy-native: a single pickled dict of arrays (loadable without
        # torch). Mirrors the .npy counts/masks; read back via load_metadata.
        np.save(
            fname,
            {k: v.numpy() for k, v in data.items()},
            allow_pickle=True,
        )
    else:
        torch.save(data, fname)
    return data


def load_metadata(path, map_location="cpu") -> dict:
    """Load a per-reflection metadata dict from `.npy` or `.pt`.

    with open(filename, "wb") as f:
        f.write(msgpack.packb(outer, use_bin_type=True))


# %%
if __name__ == "__main__":
    import tempfile

    import msgpack
    import numpy as np
    import torch
    # generate an output tensor

    # outputs are done on a per
    out = torch.load(
        "/Users/luis/from_harvard_rc/run-20251217_003728-5mkojfi/predictions/epoch_1/preds.pt",
        weights_only=False,
    )

    # Generating an output pred.pt file
    n_epochs = 10  # number of prediction epochs
    batch_size = 100
    data = {}

    for c in MODEL_OUT_KEYS:
        data[c] = [np.random.random(batch_size) for x in range(n_epochs)]

    for c in ALL_COLS:
        if c in INT_COLS:
            data[c] = [
                np.random.randint(0, 100, batch_size).astype(np.float32)
                for x in range(n_epochs)
            ]
        elif c in BOOL_COLS:
            data[c] = [
                np.random.randint(0, 2, batch_size).astype(np.float32)
                for x in range(n_epochs)
            ]
        elif c in FLOAT_COLS:
            data[c] = [np.random.random(batch_size) for x in range(n_epochs)]

    with tempfile.TemporaryDirectory() as tdir:
        torch.save(data, "preds.pt")
