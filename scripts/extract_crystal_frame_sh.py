"""Extract per-reflection crystal-frame spherical-harmonic absorption features.

DIALS models the absorption correction as a smooth, low-dimensional spherical-
harmonic surface evaluated at the diffracted (and incident) beam direction in a
frame fixed to the crystal -- i.e. with the goniometer scan rotation undone. An
unconstrained MLP scale on lab-frame `(frame, x, y)` cannot represent this (it
never sees the crystal-frame direction), so its within-image scale is noise and
the anomalous (Bijvoet) signal it produces is uncorrelated with the true one.
DIALS gets the anomalous because it has this coordinate; this script gives it to
the learned `PhysicalScale`.

We compute the basis once, offline, so the model only has to fit a few-dozen
linear coefficients on top of it:

    M(phi) = S R(phi) F U                         # crystal Cartesian -> lab
    s1_c   = normalize(M(phi)^T s1_lab)           # diffracted beam, crystal frame
    s0_c   = normalize(M(phi)^T s0_lab)           # incident beam, crystal frame
    a_lm(i) = Y_lm^real(s0_c_i) + Y_lm^real(s1_c_i)   # l=1..lmax, m=-l..l (sum of
                                                       # both legs == DIALS)

A complete real-SH basis up to `lmax` is closed under rotation, so the exact
azimuthal origin of the crystal frame does not matter -- the basis spans the same
function space DIALS fits, and `regress_dials_scale.py` confirms it.

`s1` and the rotation angle `phi = xyzcal.mm.2` are read straight from
`metadata.pt`, so the output row order is identical to `metadata.pt` and stays
aligned through the data module's load-time filtering. Only the static
crystal/goniometer/beam geometry comes from the `.expt`. The original
`metadata.pt` is left untouched; a copy with the added `absorption_sh` key is
written to `--out` (point the config's `reference:` at it).

Run on the cluster in the dials / refltorch env:

    python scripts/extract_crystal_frame_sh.py \
        --expt /n/.../reference_data/integrated.expt \
        --metadata /n/.../pytorch_data/metadata.pt \
        --out /n/.../pytorch_data/metadata_sh.pt --lmax 4

    # optional: also dump a parquet for the DIALS-scale validation regression
    python scripts/extract_crystal_frame_sh.py ... --dump-features features_sh.parquet
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def real_sph_harm_table(dirs: np.ndarray, lmax: int) -> np.ndarray:
    """Real spherical harmonics for l=1..lmax at the given unit directions.

    Args:
        dirs: (N, 3) unit vectors.
        lmax: maximum harmonic order (l=0 is excluded -- it is the constant
            absorbed into the overall scale).

    Returns:
        (N, (lmax+1)**2 - 1) array; columns ordered l=1..lmax, m=-l..l.
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    polar = np.arccos(np.clip(z, -1.0, 1.0))  # colatitude theta in [0, pi]
    az = np.arctan2(y, x)  # azimuth in (-pi, pi]
    cols = []
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            if m < 0:
                yv = np.sqrt(2.0) * ((-1) ** m) * np.imag(_ylm(abs(m), l, az, polar))
            elif m == 0:
                yv = np.real(_ylm(0, l, az, polar))
            else:
                yv = np.sqrt(2.0) * ((-1) ** m) * np.real(_ylm(m, l, az, polar))
            cols.append(yv)
    return np.stack(cols, axis=-1).astype(np.float32)


def _ylm(m: int, l: int, az: np.ndarray, polar: np.ndarray) -> np.ndarray:
    """Complex spherical harmonic Y_l^m, portable across scipy versions.

    `az` is the azimuth and `polar` the colatitude. scipy renamed and reordered
    the API in 1.15: the legacy `sph_harm(m, l, azimuth, polar)` was removed in
    favor of `sph_harm_y(l, m, polar, azimuth)`. Both use the same (Condon-
    Shortley) convention, so this returns identical values either way.
    """
    from scipy import special

    if hasattr(special, "sph_harm_y"):  # scipy >= 1.15
        return special.sph_harm_y(l, m, polar, az)
    return special.sph_harm(m, l, az, polar)  # legacy


def _rotation_about_axis(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrices about `axis` for each angle. Returns (N,3,3)."""
    k = axis / np.linalg.norm(axis)
    kx, ky, kz = k
    K = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64
    )
    eye = np.eye(3)
    c = np.cos(angles)[:, None, None]
    s = np.sin(angles)[:, None, None]
    return eye[None] + s * K[None] + (1.0 - c) * (K @ K)[None]


def crystal_frame_dirs(
    s1_lab: np.ndarray, phi: np.ndarray, expt
) -> tuple[np.ndarray, np.ndarray]:
    """Diffracted / incident beam unit vectors in the crystal-fixed frame.

    Uses the DIALS setting-rotation decomposition `r_lab = S R(phi) F U B h`, so
    `M(phi) = S R(phi) F U` maps the crystal Cartesian frame to the lab frame and
    `M(phi)^T` brings a lab vector back into the crystal frame.

    Args:
        s1_lab: (N, 3) lab-frame diffracted beam vectors (need not be unit).
        phi: (N,) rotation angles in radians (xyzcal.mm.2).
        expt: a dxtbx Experiment with beam, goniometer, crystal.

    Returns:
        (s0_c, s1_c), each (N, 3) unit vectors in the crystal frame.
    """
    beam, gonio, crystal = expt.beam, expt.goniometer, expt.crystal
    s0 = np.asarray(beam.get_s0(), dtype=np.float64)
    s0 = s0 / np.linalg.norm(s0)
    S = np.asarray(gonio.get_setting_rotation(), dtype=np.float64).reshape(3, 3)
    Fr = np.asarray(gonio.get_fixed_rotation(), dtype=np.float64).reshape(3, 3)
    axis = np.asarray(gonio.get_rotation_axis_datum(), dtype=np.float64)
    U = np.asarray(crystal.get_U(), dtype=np.float64).reshape(3, 3)

    R = _rotation_about_axis(axis, phi.astype(np.float64))  # (N,3,3)
    FU = Fr @ U
    M = S[None] @ R @ FU[None]  # (N,3,3): crystal -> lab

    # crystal-frame vectors: v_c = M^T v_lab
    s1u = s1_lab / np.linalg.norm(s1_lab, axis=1, keepdims=True)
    s1_c = np.einsum("nji,nj->ni", M, s1u)
    s0_c = np.einsum("nji,j->ni", M, s0)
    s1_c /= np.linalg.norm(s1_c, axis=1, keepdims=True)
    s0_c /= np.linalg.norm(s0_c, axis=1, keepdims=True)
    return s0_c, s1_c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expt", required=True, help="DIALS .expt (geometry)")
    ap.add_argument("--metadata", required=True, help="metadata.pt to augment")
    ap.add_argument("--out", required=True, help="output .pt (copy + absorption_sh)")
    ap.add_argument("--lmax", type=int, default=4, help="max SH order (DIALS ~4-6)")
    ap.add_argument(
        "--dump-features", default=None,
        help="optional parquet of [refl_ids, d, phi, sh_*] for validation",
    )
    args = ap.parse_args()

    meta = torch.load(args.metadata, weights_only=False)
    for k in ("s1.0", "s1.1", "s1.2", "xyzcal.mm.2"):
        if k not in meta:
            raise KeyError(
                f"metadata.pt is missing {k!r}; need s1.* and xyzcal.mm.2 "
                f"(rotation angle). Present keys: {sorted(meta)[:20]}..."
            )
    s1_lab = np.stack(
        [meta["s1.0"].numpy(), meta["s1.1"].numpy(), meta["s1.2"].numpy()],
        axis=-1,
    ).astype(np.float64)
    phi = meta["xyzcal.mm.2"].numpy().astype(np.float64)
    n = s1_lab.shape[0]
    logger.info("Loaded %d reflections from %s", n, args.metadata)

    from dxtbx.model.experiment_list import ExperimentListFactory

    expts = ExperimentListFactory.from_json_file(args.expt, check_format=False)
    if len(expts) != 1:
        raise NotImplementedError(
            f"{len(expts)} experiments in {args.expt}; this extractor assumes a "
            "single sweep. Split per-experiment and concat, matching metadata "
            "row order, if you have a multi-sweep dataset."
        )
    expt = expts[0]
    if abs(float(np.ptp(phi))) < 1e-9:
        logger.warning(
            "phi is constant (stills?) -- crystal-frame direction won't vary "
            "with the scan; absorption will be a pure function of detector "
            "position. Expected for rotation data is a non-trivial phi range."
        )

    s0_c, s1_c = crystal_frame_dirs(s1_lab, phi, expt)
    logger.info(
        "Crystal-frame dirs: phi range [%.3f, %.3f] rad; |s1_c| mean %.4f",
        float(phi.min()), float(phi.max()), float(np.linalg.norm(s1_c, axis=1).mean()),
    )

    a = real_sph_harm_table(s0_c, args.lmax) + real_sph_harm_table(s1_c, args.lmax)
    n_sh = (args.lmax + 1) ** 2 - 1
    assert a.shape == (n, n_sh), (a.shape, (n, n_sh))
    logger.info(
        "absorption_sh: shape %s (lmax=%d, %d harmonics), std per col "
        "min/median/max %.3f/%.3f/%.3f",
        tuple(a.shape), args.lmax, n_sh,
        float(a.std(0).min()), float(np.median(a.std(0))), float(a.std(0).max()),
    )

    out = dict(meta)
    out["absorption_sh"] = torch.from_numpy(a)
    torch.save(out, args.out)
    logger.info("Wrote %s (added 'absorption_sh' %s)", args.out, tuple(a.shape))

    if args.dump_features:
        import pandas as pd

        cols = {f"sh_{i}": a[:, i] for i in range(n_sh)}
        cols["phi"] = phi.astype(np.float32)
        if "d" in meta:
            cols["d"] = meta["d"].numpy().astype(np.float32)
        if "refl_ids" in meta:
            cols["refl_ids"] = meta["refl_ids"].numpy()
        pd.DataFrame(cols).to_parquet(args.dump_features)
        logger.info("Wrote validation features -> %s", args.dump_features)


if __name__ == "__main__":
    main()
