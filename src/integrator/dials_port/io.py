"""Loading DIALS data without a cctbx runtime where possible.

A `.expt` file is plain JSON, so experiment geometry and the refined
scaling model are readable with the standard library alone. Reflection
tables are msgpack behind a DIALS-specific schema, so `load_reflections`
imports `dials.array_family` lazily and is the only entry point here
that needs the `integrator-dev` environment.

The intended workflow is to call `load_reflections` once, cache with
`dump_npz`, then work from `load_npz` in the plain project environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .types import DTYPE, IDTYPE, Reflections


@dataclass
class ScalingModelRef:
    """Refined DIALS scaling model, read back for verification.

    Attributes:
        model_id: `"physical"`, `"KB"` or `"array"`.
        components: parameter vector per component name.
        config: the `configuration_parameters` block verbatim.
    """

    model_id: str
    components: dict[str, np.ndarray]
    config: dict = field(default_factory=dict)

    @property
    def error_model(self) -> tuple[float, float]:
        """Refined `(a, b)` of the basic error model."""
        a, b = self.config["error_model_parameters"]
        return float(a), float(b)


@dataclass
class Experiment:
    """Geometry and crystal metadata from a `.expt` file.

    Attributes:
        cell: unit cell `(a, b, c, alpha, beta, gamma)`.
        space_group_hall: Hall symbol as written by DIALS.
        wavelength: beam wavelength in Angstrom.
        s0: `(3,)` incident beam vector, sample-to-source direction.
        rotation_axis: `(3,)` goniometer rotation axis.
        fixed_rotation: `(3, 3)` goniometer fixed rotation.
        setting_rotation: `(3, 3)` goniometer setting rotation.
        image_range: inclusive `(first, last)` image numbers.
        osc_range: `(start, end)` rotation in degrees.
        scaling_model: refined model, when the file has been scaled.
    """

    cell: tuple[float, float, float, float, float, float]
    space_group_hall: str
    wavelength: float
    s0: np.ndarray
    rotation_axis: np.ndarray
    fixed_rotation: np.ndarray
    setting_rotation: np.ndarray
    image_range: tuple[int, int]
    osc_range: tuple[float, float]
    scaling_model: ScalingModelRef | None = None

    @property
    def space_group(self) -> str:
        """Hall symbol, accepted directly by `gemmi.SpaceGroup`."""
        return self.space_group_hall


def _cell_from_real_space(
    a: list, b: list, c: list
) -> tuple[float, float, float, float, float, float]:
    """Derive cell parameters from real-space basis vectors."""
    va, vb, vc = (np.asarray(v, dtype=np.float64) for v in (a, b, c))
    la, lb, lc = (float(np.linalg.norm(v)) for v in (va, vb, vc))
    deg = 180.0 / np.pi

    def angle(u: np.ndarray, v: np.ndarray, nu: float, nv: float) -> float:
        return float(np.arccos(np.clip(u @ v / (nu * nv), -1.0, 1.0)) * deg)

    return (
        la,
        lb,
        lc,
        angle(vb, vc, lb, lc),
        angle(va, vc, la, lc),
        angle(va, vb, la, lb),
    )


def load_experiment(path: str | Path, index: int = 0) -> Experiment:
    """Read geometry and the refined scaling model from a `.expt`.

    Args:
        path: path to a DIALS `.expt` file.
        index: which experiment to read when the file holds several.

    Returns:
        The parsed `Experiment`.
    """
    doc = json.loads(Path(path).read_text())
    crystal = doc["crystal"][index]
    beam = doc["beam"][index]
    gonio = doc["goniometer"][index]
    scan = doc["scan"][index]

    # An unscaled `.expt` may carry an empty `scaling_model` list rather
    # than omitting the key.
    scaling_model = None
    if len(doc.get("scaling_model", [])) > index:
        raw = doc["scaling_model"][index]
        config = raw.get("configuration_parameters", {})
        components = {
            name: np.asarray(block["parameters"], dtype=np.float64)
            for name, block in raw.items()
            if isinstance(block, dict) and "parameters" in block
        }
        scaling_model = ScalingModelRef(
            model_id=raw.get("__id__", "unknown"),
            components=components,
            config=config,
        )

    # Newer DIALS writes a per-image oscillation array under
    # `scan.properties`; older files carry a two-element `oscillation`.
    # A scaled model, when present, records the range it was refined
    # over and is preferred so the smoother normalisation matches.
    def _osc_range() -> tuple[float, float]:
        if scaling_model is not None:
            recorded = scaling_model.config.get("valid_osc_range")
            if recorded:
                return float(recorded[0]), float(recorded[-1])
        per_image = scan.get("properties", {}).get("oscillation")
        if per_image:
            return float(per_image[0]), float(per_image[-1])
        legacy = scan.get("oscillation")
        if legacy:
            return float(legacy[0]), float(legacy[-1])
        return 0.0, 0.0

    return Experiment(
        cell=_cell_from_real_space(
            crystal["real_space_a"],
            crystal["real_space_b"],
            crystal["real_space_c"],
        ),
        space_group_hall=crystal["space_group_hall_symbol"],
        wavelength=float(beam["wavelength"]),
        s0=np.asarray(beam["direction"], dtype=np.float64),
        rotation_axis=np.asarray(gonio["rotation_axis"], dtype=np.float64),
        fixed_rotation=np.asarray(
            gonio["fixed_rotation"], dtype=np.float64
        ).reshape(3, 3),
        setting_rotation=np.asarray(
            gonio["setting_rotation"], dtype=np.float64
        ).reshape(3, 3),
        image_range=(
            int(scan["image_range"][0]),
            int(scan["image_range"][1]),
        ),
        osc_range=_osc_range(),
        scaling_model=scaling_model,
    )


# DIALS flag bits that exclude a reflection from scaling. Mirrors the
# composite `bad_for_scaling` flag used by `reduce_on_intensities`.
# Values verified against `flex.reflection_table.flags`.
_USER_EXCLUDED_IN_SCALING = 1 << 22
_OUTLIER_IN_SCALING = 1 << 23
_EXCLUDED_FOR_SCALING = 1 << 24
_BAD_FOR_SCALING = (
    _EXCLUDED_FOR_SCALING | _OUTLIER_IN_SCALING | _USER_EXCLUDED_IN_SCALING
)
assert _BAD_FOR_SCALING == 29360128


def load_reflections(
    path: str | Path,
    intensity: str = "scale",
    drop_bad_for_scaling: bool = True,
) -> Reflections:
    """Read a DIALS `.refl` into a `Reflections` container.

    Requires the DIALS environment; import is deferred so the rest of
    this package stays usable without it.

    Args:
        path: path to a `.refl` file.
        intensity: which intensity column to take, one of `"scale"`,
            `"sum"` or `"prf"`.
        drop_bad_for_scaling: remove reflections carrying any of the
            `bad_for_scaling` flag bits.

    Returns:
        The loaded observations. `scale` holds `inverse_scale_factor`
        when present, otherwise ones.

    Raises:
        KeyError: if the requested intensity column is absent.
    """
    from dials.array_family import flex  # noqa: PLC0415
    from dxtbx import flumpy  # noqa: PLC0415

    rt = flex.reflection_table.from_file(str(path))

    value_key = f"intensity.{intensity}.value"
    var_key = f"intensity.{intensity}.variance"
    if value_key not in rt:
        raise KeyError(f"{path} has no column {value_key!r}")

    if drop_bad_for_scaling:
        flags = flumpy.to_numpy(rt["flags"]).astype(np.uint64)
        keep = (flags & np.uint64(_BAD_FOR_SCALING)) == 0
        rt = rt.select(flumpy.from_numpy(keep))

    n = rt.size()

    def col(key: str, default: float | None = None) -> np.ndarray:
        if key in rt:
            return flumpy.to_numpy(rt[key]).astype(np.float64)
        if default is None:
            raise KeyError(f"{path} has no column {key!r}")
        return np.full(n, default, dtype=np.float64)

    hkl = np.asarray(rt["miller_index"], dtype=np.int64).reshape(n, 3)
    xyz = flumpy.to_numpy(rt["xyzobs.px.value"]).astype(np.float64)
    s1 = flumpy.to_numpy(rt["s1"]).astype(np.float64).reshape(n, 3)

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)).to(DTYPE)

    return Reflections(
        hkl=torch.from_numpy(np.ascontiguousarray(hkl)).to(IDTYPE),
        intensity=t(col(value_key)),
        variance=t(col(var_key)),
        d=t(col("d")),
        phi=t(xyz[:, 2]),
        s1=t(s1),
        partiality=t(col("partiality", 1.0)),
        lp=t(col("lp", 1.0)),
        qe=t(col("qe", 1.0)),
        dataset_id=torch.from_numpy(
            flumpy.to_numpy(rt["id"]).astype(np.int64)
        ).to(IDTYPE),
        scale=t(col("inverse_scale_factor", 1.0)),
    )


def dump_npz(refl: Reflections, path: str | Path) -> None:
    """Cache a `Reflections` to a `.npz`, for use without DIALS."""
    np.savez_compressed(
        str(path),
        **{
            name: getattr(refl, name).detach().cpu().numpy()
            for name in refl.__dataclass_fields__
        },
    )


def load_npz(path: str | Path) -> Reflections:
    """Load a `Reflections` cached by `dump_npz`."""
    with np.load(str(path)) as data:
        fields = {}
        for name in Reflections.__dataclass_fields__:
            arr = data[name]
            dtype = IDTYPE if arr.dtype.kind in "iu" else DTYPE
            fields[name] = torch.from_numpy(arr).to(dtype)
    return Reflections(**fields)
