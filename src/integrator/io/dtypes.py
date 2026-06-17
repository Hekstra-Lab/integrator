"""DIALS reflection-table schema for the IO layer."""

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

DEFAULT_REFL_COLS = list(SCALAR_DTYPES.keys()) + list(VECTOR_COLUMNS.keys())
DEFAULT_EXCLUDED_COLS = ["BATCH", "PARTIAL"]
