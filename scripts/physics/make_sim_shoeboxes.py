"""Cut Laue shoeboxes from a nanoBragg simulation at the TRUE crystal orientation.

Derek Mendez's ecDHFR Laue simulation stores the orientation (`Umat`), reciprocal
cell (`Bmat`) and mosaic spread of every shot in an `.h5` beside each `.cbf`. Here
reflections are predicted from that truth (`integrator.physics.laue_predict`)
instead of from indexing, so a test of the physics prior has no indexing error in
it. Windows are placed exactly as `integrator.make_shoeboxes --laue` places them
(centred on `floor(xyzcal.px)`, slid inward at the panel edges), and the output is
the format `PolychromaticDataModule` reads.

By default the reflection set is predicted from that truth. `--from-refl` takes
it from a DIALS reflection table instead -- normally `laue.integrate`'s own
`integrated.refl` -- so that our arm integrates exactly the reflections
laue.integrate reported, at exactly its centroids and window placement. That is
what the per-reflection error comparison against the truth-mask sum needs: with
the footprints shared, the only thing left varying between the arms is the
estimator. The crystal metadata still
comes from the h5, because `physics.yaml` and `crystal_models.npz` describe the
simulation rather than the reflection set.

Two stages, because the environments that can read these CBFs and the one that
runs the integrator are different (see the dhfr_nanobragg_simulation notes):

    cut       ld env (dxtbx + h5py, numpy 1.x): counts.npy, masks.npy, metadata.npy,
              crystal_models.npz (per-image A, U, B, mosaic spread), physics.yaml
              (detector panel and beam, for the nanoBragg kernel)
    finalize  integrator env: neighbour-overlap masking, stats, concentration.npy,
              dataset.yaml (with a top-level `physics` block copied from physics.yaml)

Example:
    micromamba run -n ld python make_sim_shoeboxes.py cut --data-dir DATA --out OUT
    micromamba run -n integrator-cuda-dev python make_sim_shoeboxes.py finalize --out OUT
"""

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve()


def _load_predictor():
    """Import laue_predict by path, so `cut` runs without the integrator package."""
    candidates = [
        root / "src" / "integrator" / "physics" / "laue_predict.py"
        for root in (HERE.parents[1], HERE.parents[2])
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("laue_predict", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"laue_predict.py not found in {candidates}")


class _ReflSet:
    """The subset of a `predict_laue` result that `cut` consumes, from a refl table."""

    __slots__ = ("hkl", "xy", "d", "wavelength", "n_harmonics", "s1")

    def __init__(self, hkl, xy, d, wavelength, n_harmonics, s1):
        self.hkl = hkl
        self.xy = xy
        self.d = d
        self.wavelength = wavelength
        self.n_harmonics = n_harmonics
        self.s1 = s1


def _refl_sets_by_image(refl_path: str, cell: np.ndarray) -> dict:
    """Group a DIALS reflection table into one `_ReflSet` per image.

    `id` is the experiment index and the shots are named 1-based, so
    image_num = id + 1; `truth.expt` was built in sorted shot order and that
    mapping is checked there.

    The resolution is recomputed from the orthorhombic cell rather than read
    from the table, so it cannot disagree with the truth-mask binning.
    """
    from dials.array_family import flex

    refls = flex.reflection_table.from_file(refl_path)
    ids = refls["id"].as_numpy_array()
    hkl = np.asarray(refls["miller_index"], dtype=np.int64).reshape(-1, 3)
    xy = refls["xyzcal.px"].as_numpy_array()[:, :2]
    lam = (refls["wavelength"].as_numpy_array() if "wavelength" in refls
           else np.full(len(refls), np.nan))
    s1 = (refls["s1"].as_numpy_array() if "s1" in refls
          else np.zeros((len(refls), 3)))
    nh = (refls["harmonics"].as_numpy_array().astype(np.int64) + 1
          if "harmonics" in refls else np.ones(len(refls), dtype=np.int64))
    d = 1.0 / np.sqrt(
        (hkl[:, 0] / cell[0]) ** 2
        + (hkl[:, 1] / cell[1]) ** 2
        + (hkl[:, 2] / cell[2]) ** 2
    )
    out = {}
    for i in np.unique(ids):
        m = ids == i
        out[int(i) + 1] = _ReflSet(
            hkl[m], xy[m], d[m], lam[m], nh[m], s1[m]
        )
    print(f"{len(refls):,} reflections over {len(out)} images from {refl_path}")
    return out


def _window(x: int, y: int, n: int, n_fast: int, n_slow: int) -> tuple[int, int, int, int]:
    """Same placement as `make_shoeboxes._shift_bbox_xy` for a window that fits the panel."""
    half = n // 2
    x0 = min(max(x - half, 0), n_fast - n)
    y0 = min(max(y - half, 0), n_slow - n)
    return x0, x0 + n, y0, y0 + n


def cut(args):
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    import dxtbx
    import h5py
    from numpy.lib.format import open_memmap

    predict_laue = None if args.from_refl else _load_predictor().predict_laue
    data = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shots = sorted(data.glob(f"{args.prefix}*.cbf"))
    if args.max_images:
        shots = shots[: args.max_images]
    n = args.size

    refl_sets = None
    if args.from_refl:
        # the cell is only used to recompute d, and is orthorhombic here
        import h5py as _h5

        with _h5.File(shots[0].with_suffix(".h5"), "r") as f:
            B0 = f["Bmat"][()].reshape(3, 3)
        cell0 = np.array([1.0 / B0[0, 0], 1.0 / B0[1, 1], 1.0 / B0[2, 2]])
        refl_sets = _refl_sets_by_image(args.from_refl, cell0)

    per_image = []
    for cbf in shots:
        image_num = int(cbf.stem.split("_")[-1])
        with h5py.File(cbf.with_suffix(".h5"), "r") as f:
            U = f["Umat"][()].reshape(3, 3)
            B = f["Bmat"][()].reshape(3, 3)
            mos = float(f["mos_spread"][()])
        img = dxtbx.load(str(cbf))
        panel = img.get_detector()[0]
        beam = img.get_beam()
        s0_hat = -np.asarray(beam.get_sample_to_source_direction())
        pix0 = np.asarray(panel.get_origin())
        fdet, sdet = np.asarray(panel.get_fast_axis()), np.asarray(panel.get_slow_axis())
        n_fast, n_slow = panel.get_image_size()
        if refl_sets is not None:
            ref = refl_sets.get(image_num)
            if ref is None:
                print(f"{cbf.name}: no reflections in the table, skipped")
                continue
        else:
            ref = predict_laue(
                U @ B, s0_hat, args.lambda_min, args.lambda_max, args.dmin,
                pix0, fdet, sdet, panel.get_pixel_size()[0], n_fast, n_slow,
            )
        per_image.append(dict(cbf=cbf, image_num=image_num, U=U, B=B, mos=mos, ref=ref))
        print(f"{cbf.name}: {len(ref.hkl)} reflections")

    n_total = sum(len(p["ref"].hkl) for p in per_image)
    counts_mm = open_memmap(out / "counts.npy", mode="w+", dtype=np.int32, shape=(n_total, n * n))
    masks_mm = open_memmap(out / "masks.npy", mode="w+", dtype=np.bool_, shape=(n_total, n * n))
    cols: dict[str, list] = {k: [] for k in (
        "refl_ids", "image_num", "H", "K", "L", "d", "wavelength", "n_harmonics",
        "s1.0", "s1.1", "s1.2", "xyzcal.px.0", "xyzcal.px.1", "xyzcal.px.2",
        "bbox.0", "bbox.1", "bbox.2", "bbox.3", "bbox.4", "bbox.5",
        "intensity.sum.value", "intensity.sum.variance",
        "background.sum.value", "background.sum.variance", "panel", "flags",
    )}

    row = 0
    border = np.ones((n, n), dtype=bool)
    border[2:-2, 2:-2] = False
    for p in per_image:
        image = dxtbx.load(str(p["cbf"])).get_raw_data().as_numpy_array()
        ref = p["ref"]
        for j in range(len(ref.hkl)):
            fx, sy = ref.xy[j]
            x0, x1, y0, y1 = _window(int(np.floor(fx)), int(np.floor(sy)), n, image.shape[1], image.shape[0])
            box = image[y0:y1, x0:x1]
            valid = box >= 0
            counts_mm[row] = np.where(valid, box, 0).reshape(-1)
            masks_mm[row] = valid.reshape(-1)
            # A crude summation estimate, only so that columns other code expects exist.
            bg = float(np.median(box[border & valid])) if (border & valid).any() else 0.0
            n_valid = int(valid.sum())
            total = float(box[valid].sum())
            cols["refl_ids"].append(row)
            cols["image_num"].append(p["image_num"])
            cols["H"].append(ref.hkl[j, 0]); cols["K"].append(ref.hkl[j, 1]); cols["L"].append(ref.hkl[j, 2])
            cols["d"].append(ref.d[j]); cols["wavelength"].append(ref.wavelength[j])
            cols["n_harmonics"].append(ref.n_harmonics[j])
            for k in range(3):
                cols[f"s1.{k}"].append(ref.s1[j, k])
            cols["xyzcal.px.0"].append(fx); cols["xyzcal.px.1"].append(sy); cols["xyzcal.px.2"].append(0.5)
            for k, v in enumerate((x0, x1, y0, y1, 0, 1)):
                cols[f"bbox.{k}"].append(v)
            cols["intensity.sum.value"].append(total - bg * n_valid)
            cols["intensity.sum.variance"].append(max(total, 1.0))
            cols["background.sum.value"].append(bg * n_valid)
            cols["background.sum.variance"].append(max(bg * n_valid, 1.0))
            cols["panel"].append(0); cols["flags"].append(0)
            row += 1
        print(f"cut {p['cbf'].name}: rows up to {row}")
    counts_mm.flush(); masks_mm.flush()

    rng = np.random.default_rng(42)
    meta = {k: np.asarray(v, dtype=np.float32) for k, v in cols.items()}
    meta["is_test"] = (rng.random(n_total) < args.test_fraction).astype(np.float32)
    np.save(out / "metadata.npy", meta, allow_pickle=True)

    np.savez(
        out / "crystal_models.npz",
        image_num=np.array([p["image_num"] for p in per_image]),
        A=np.stack([p["U"] @ p["B"] for p in per_image]),
        U=np.stack([p["U"] for p in per_image]),
        B=np.stack([p["B"] for p in per_image]),
        mos_spread_deg=np.array([p["mos"] for p in per_image]),
    )

    img = dxtbx.load(str(per_image[0]["cbf"]))
    panel, beam = img.get_detector()[0], img.get_beam()
    B0 = per_image[0]["B"]
    physics = {
        "source": str(data),
        "orientation_file": "crystal_models.npz",
        "truth": "orientation and cell from the simulation .h5, not from indexing",
        "detector": {
            "pix0_mm": [float(v) for v in panel.get_origin()],
            "fast": [float(v) for v in panel.get_fast_axis()],
            "slow": [float(v) for v in panel.get_slow_axis()],
            "pixel_size_mm": float(panel.get_pixel_size()[0]),
            "n_fast": int(panel.get_image_size()[0]),
            "n_slow": int(panel.get_image_size()[1]),
        },
        "beam": {
            "s0_hat": [float(v) for v in -np.asarray(beam.get_sample_to_source_direction())],
            "polarization_normal": [float(v) for v in beam.get_polarization_normal()],
            "polarization_fraction": float(beam.get_polarization_fraction()),
            "lambda_min": args.lambda_min,
            "lambda_max": args.lambda_max,
        },
        "dmin": args.dmin,
        "cell": [float(1.0 / B0[0, 0]), float(1.0 / B0[1, 1]), float(1.0 / B0[2, 2]), 90.0, 90.0, 90.0],
    }
    (out / "physics.yaml").write_text(yaml.safe_dump(physics, sort_keys=False))
    print(f"wrote {n_total} shoeboxes from {len(per_image)} images to {out}")


def finalize(args):
    import torch  # noqa: F401  (integrator helpers need it)

    from integrator.cli.make_shoeboxes import (
        _apply_overlap_mask,
        _save_concentration_from_memmap,
        _stats_from_memmap,
    )
    from integrator.io import write_dataset_yaml

    out = Path(args.out)
    meta = np.load(out / "metadata.npy", allow_pickle=True).item()
    physics = yaml.safe_load((out / "physics.yaml").read_text())
    n = args.size
    bboxes = np.stack([meta[f"bbox.{k}"] for k in range(6)], -1).astype(np.int64)
    centroids = np.stack([meta[f"xyzcal.px.{k}"] for k in range(3)], -1).astype(np.float64)
    if not args.no_mask_overlap:
        _apply_overlap_mask(
            masks_path=out / "masks.npy", bboxes=bboxes, centroids=centroids,
            image_ids=meta["image_num"].astype(np.int64), dz=1, dy=n, dx=n,
            nproc=args.nproc, chunk=10_000,
        )
    stats = _stats_from_memmap(out / "counts.npy", out / "masks.npy")
    _save_concentration_from_memmap(out / "counts.npy", out)
    beam_center = _beam_center_px(physics)
    crystal = {
        "cell": physics["cell"],
        "space_group": "P 21 21 21 (19)",
        "space_group_number": 19,
        "beam_center_px": beam_center,
        "lambda_min": physics["beam"]["lambda_min"] - 0.01,
        "lambda_max": physics["beam"]["lambda_max"] + 0.01,
    }
    p = write_dataset_yaml(
        out, geometry={"d": 1, "h": n, "w": n}, n_reflections=len(meta["refl_ids"]),
        polychromatic=True, anscombe=True,
        files={"counts": "counts.npy", "masks": "masks.npy", "reference": "metadata.npy"},
        crystal=crystal, stats=stats,
    )
    spec = yaml.safe_load(p.read_text())
    spec["physics"] = physics
    p.write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f"wrote {p}")


def _beam_center_px(physics: dict) -> list[float]:
    """Where the direct beam meets the panel, in pixels (fast, slow)."""
    det = physics["detector"]
    pix0 = np.asarray(det["pix0_mm"])
    fdet, sdet = np.asarray(det["fast"]), np.asarray(det["slow"])
    s0 = np.asarray(physics["beam"]["s0_hat"])
    normal = np.cross(fdet, sdet)
    t = (pix0 @ normal) / (s0 @ normal)
    point = s0 * t - pix0
    return [float(point @ fdet / det["pixel_size_mm"]), float(point @ sdet / det["pixel_size_mm"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cut")
    c.add_argument("--data-dir", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--prefix", default="shot_1_")
    c.add_argument("--max-images", type=int, default=0)
    c.add_argument("--dmin", type=float, default=1.2)
    c.add_argument("--lambda-min", type=float, default=1.02)
    c.add_argument("--lambda-max", type=float, default=1.18)
    c.add_argument("--size", type=int, default=21)
    c.add_argument("--test-fraction", type=float, default=0.1)
    c.add_argument(
        "--from-refl",
        default=None,
        help="take the reflection set, centroids and wavelengths from this "
        "DIALS reflection table instead of predicting them from the truth; "
        "use laue.integrate's integrated.refl to share footprints with it",
    )
    f = sub.add_parser("finalize")
    f.add_argument("--out", required=True)
    f.add_argument("--size", type=int, default=21)
    f.add_argument("--nproc", type=int, default=8)
    f.add_argument("--no-mask-overlap", action="store_true")
    args = ap.parse_args()
    cut(args) if args.cmd == "cut" else finalize(args)


if __name__ == "__main__":
    main()
