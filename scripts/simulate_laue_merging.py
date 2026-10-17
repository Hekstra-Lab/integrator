"""A small Laue stills dataset with a known scale, for developing the merge.

Why this exists.  Every mechanism question in the October scaling work was
answered in seconds by a synthetic tensor harness, and every *model* question
cost a 70-minute GPU run whose verdict then had to be read through careless,
phenix, an R-free flag set and a merging-seed average -- four stages that each
move the number by as much as the effect under test.  On simulated data the
quantities we actually care about are known, so they can be measured directly:

    scale error            eps = rms( log s_hat - log s_true )  after the gauge
    spectrum recovery      CC( log G_hat(lambda), log G_true(lambda) )
    intensity recovery     CC( I_h_hat, I_h_true ), and the slope
    Bijvoet attenuation    ( I(+) - I(-) )_hat / ( I(+) - I(-) )_true

None of those need a refinement, so an arm takes minutes on a laptop.

What it deliberately does NOT do.  A simulator that draws from the model's own
assumptions flatters the model, so the two places where our assumptions are
themselves under test are generated from something else:

- **absorption** is a crystal-frame spherical-harmonic surface, which no
  current arm's lab-frame scale can represent at all.  If an arm reports a
  perfect scale here, the arm is wrong, not the data.
- **the background** is a smooth tilted plane over the shoebox, not the
  per-observation constant the merge infers.

The geometry is the real thing rather than a caricature, because the facts that
make Laue scaling hard are geometric.  Stills come from a rotation series (1
degree per image about lab +y, as `imported_HEWL_anom_0999.expt` has it), and a
reflection is excited on a still at the one wavelength satisfying the Ewald
condition,

    lambda = -2 (q . s0_hat) / |q|^2,

so a reflection appears on several images at several wavelengths, and `h` and
`-h` are essentially never excited at the same wavelength.  That last fact is
why scale error does not cancel in a Laue Bijvoet difference, and it is the
reason a monochromatic intuition misleads here.

Output is the format the real loader consumes -- `counts.npy`, `masks.npy`,
`metadata.pt`, `dataset.yaml`, `asu_id_to_hkl.pt` -- plus ground-truth columns
(`true_scale`, `true_bg`, `true_intensity`, and the scale's own terms) carried
in the metadata, so an arm can pin any of them through `scale_from_metadata` /
`bg_from_metadata` and an analysis can compare against them.

Usage
-----
    uv run python scripts/simulate_laue_merging.py --out /tmp/sim_laue \\
        --images 60 --d-min 2.5 --background-share 0.966
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import gemmi
import numpy as np
import torch
import yaml

# HEWL, so the symmetry and cell match the dataset every real number here is
# compared against.
CELL = (79.1, 79.1, 37.9, 90.0, 90.0, 90.0)
SPACE_GROUP = "P 43 21 2"
# the 1118 beamline: a 0.97-1.25 A band and a 1 deg/image rotation about lab +y
LAMBDA_MIN, LAMBDA_MAX = 0.97, 1.25
ROTATION_AXIS = np.array([0.0, 1.0, 0.0])
DEG_PER_IMAGE = 1.0
BEAM = np.array([0.0, 0.0, 1.0])          # s0_hat, beam along lab +z
DETECTOR_DISTANCE = 200.0                  # mm
PIXEL_SIZE = 0.08                          # mm
DETECTOR_PIXELS = 4000


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues, for the goniometer."""
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def real_sph_harm(xyz: np.ndarray, lmax: int) -> np.ndarray:
    """Real spherical harmonics for l = 1..lmax, as the absorption basis.

    The same construction `fit_physical_scale.py` uses, so a fit there is
    measured against a truth built from the same function space.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    cols = []
    for ell in range(1, lmax + 1):
        for m in range(-ell, ell + 1):
            # associated Legendre via scipy-free recursion on the needed few
            cols.append(_sph(ell, m, x, y, z))
    return np.stack(cols, axis=1)


def _sph(ell: int, m: int, x, y, z):
    """The low-order real harmonics written out, to avoid a scipy dependency."""
    r2 = x * x + y * y + z * z
    r2 = np.maximum(r2, 1e-30)
    if ell == 1:
        return {-1: y, 0: z, 1: x}[m] / np.sqrt(r2)
    if ell == 2:
        n = r2
        return {
            -2: x * y, -1: y * z, 0: (3 * z * z - r2) / 6.0, 1: x * z,
            2: (x * x - y * y) / 2.0,
        }[m] / n
    if ell == 3:
        n = r2 * np.sqrt(r2)
        return {
            -3: y * (3 * x * x - y * y), -2: x * y * z,
            -1: y * (5 * z * z - r2), 0: z * (5 * z * z - 3 * r2),
            1: x * (5 * z * z - r2), 2: z * (x * x - y * y),
            3: x * (x * x - 3 * y * y),
        }[m] / (n * 15.0)
    if ell == 4:
        n = r2 * r2
        return {
            -4: x * y * (x * x - y * y), -3: y * z * (3 * x * x - y * y),
            -2: x * y * (7 * z * z - r2), -1: y * z * (7 * z * z - 3 * r2),
            0: (35 * z ** 4 - 30 * z * z * r2 + 3 * r2 * r2) / 105.0,
            1: x * z * (7 * z * z - 3 * r2), 2: (x * x - y * y) * (7 * z * z - r2),
            3: x * z * (x * x - 3 * y * y),
            4: (x ** 4 - 6 * x * x * y * y + y ** 4),
        }[m] / (n * 105.0)
    raise ValueError(f"lmax 4 is enough; got l = {ell}")


def _lexmax(orbit: np.ndarray, lim: int) -> tuple[np.ndarray, np.ndarray]:
    """The lexicographic maximum of each symmetry orbit, and its sort code.

    Encoding a triple as one integer makes "lexicographic" exact rather than a
    tuple comparison in a Python loop over 60k reflections.
    """
    m = 2 * lim + 1
    off = orbit + lim
    code = (off[..., 0] * m + off[..., 1]) * m + off[..., 2]
    pick = code.argmax(axis=1)
    rows = np.arange(len(orbit))
    return orbit[rows, pick], code[rows, pick]


def chebyshev_curve(t, lo, hi, coeffs):
    """A Chebyshev series on t mapped to [-1, 1], for the true spectrum.

    The recurrence is `T_k = 2 u T_{k-1} - T_{k-2}`, built by carrying the two
    previous orders. An earlier version advanced them as `tk_2, tk_1 = tk_1, tk`
    inside the k=1 branch, which left both equal to `u` and made T_2 come out as
    `2u^2 - u`: still a smooth curve of the right span, so no measurement made
    against the generated column was wrong, but not the basis the name claims,
    and coefficients could not be transferred to a correct basis.
    """
    u = np.clip(2.0 * (t - lo) / (hi - lo) - 1.0, -1.0, 1.0)
    cols = [np.ones_like(u), u]
    for k in range(2, len(coeffs)):
        cols.append(2.0 * u * cols[-1] - cols[-2])
    return sum(c * col for c, col in zip(coeffs, cols))


def excited_reflections(hkl, b_matrix, u_matrix, phi, lam_lo, lam_hi):
    """Which reflections a still at angle `phi` excites, and at what wavelength.

    On a still the Ewald condition `|q + s0| = |s0|` with `|s0| = 1/lambda`
    picks out one wavelength per reciprocal-lattice point,

        lambda = -2 (q . s0_hat) / |q|^2,

    so the band selects the reflections rather than the rotation doing it.
    """
    q = (rotation_about(ROTATION_AXIS, phi) @ u_matrix @ b_matrix @ hkl.T).T
    q_sq = (q * q).sum(axis=1)
    ok = q_sq > 1e-12
    lam = np.full(len(hkl), np.inf)
    lam[ok] = -2.0 * (q[ok] @ BEAM) / q_sq[ok]
    hit = ok & (lam > lam_lo) & (lam < lam_hi)
    return hit, lam, q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", type=int, default=60)
    ap.add_argument("--d-min", type=float, default=2.5)
    ap.add_argument("--shoebox", type=int, nargs=3, default=(3, 11, 11),
                    metavar=("Z", "Y", "X"))
    ap.add_argument("--background-share", type=float, default=0.966,
                    help="fraction of all counts that are background, as "
                         "measured on hewl_1118")
    ap.add_argument("--anomalous", type=float, default=0.04,
                    help="fractional Bijvoet difference, |I+ - I-| / I")
    ap.add_argument("--lmax", type=int, default=4, help="absorption harmonics")
    ap.add_argument("--image-drift", type=float, default=0.05,
                    help="amplitude of the smooth per-image scale drift")
    ap.add_argument("--image-jitter", type=float, default=0.01,
                    help="per-image scale jitter a smooth basis cannot reach; "
                         "careless can represent this and a Chebyshev image "
                         "term cannot, so it is a real experimental knob")
    ap.add_argument("--spectrum-swing", type=float, default=18.3,
                    help="G(lambda) max/min across the band; 18.3 is the "
                         "value identified on hewl_1118")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cell = gemmi.UnitCell(*CELL)
    sg = gemmi.SpaceGroup(SPACE_GROUP)
    b_matrix = np.array(cell.frac.mat.tolist()).T   # crystal Cartesian <- hkl
    # one fixed, arbitrary crystal setting; a complete SH basis is closed under
    # rotation, so its value does not matter for what is being measured
    u_matrix = rotation_about(np.array([1.0, 2.0, 3.0]), 0.7)

    # candidate reflections out to d_min
    lim = int(np.ceil(max(CELL[:3]) / args.d_min))
    grid = np.arange(-lim, lim + 1)
    hkl = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    hkl = hkl[(hkl != 0).any(axis=1)]
    d_all = 1.0 / np.maximum(
        np.linalg.norm((b_matrix @ hkl.T).T, axis=1), 1e-12
    )
    keep = d_all >= args.d_min
    hkl, d_all = hkl[keep], d_all[keep]
    print(f"{len(hkl)} candidate reflections to {args.d_min} A")

    # ---- the truth, per unique reflection -------------------------------
    ops = sg.operations()
    rotations = np.array(
        [np.array(op.rot).reshape(3, 3) / op.DEN for op in ops.sym_ops]
    )
    # The Friedel-SEPARATE key is the orbit under the rotations alone,
    # canonicalised by its lexicographic maximum.  `gemmi.ReciprocalAsu` cannot
    # serve: it is defined by the Laue class, so `to_asu` sends h and -h to one
    # index -- which is exactly the POOLED key, and collapsing the two is what
    # would silently delete the anomalous signal this dataset exists to carry.
    orbit = np.einsum(
        "oij,nj->noi", rotations, hkl.astype(float)
    ).round().astype(np.int64)
    sep_key, sep_code = _lexmax(orbit, lim)
    neg_key, neg_code = _lexmax(-orbit, lim)
    is_plus = sep_code >= neg_code
    pool_key = np.where(is_plus[:, None], sep_key, neg_key)
    centric = np.array(
        [ops.is_reflection_centric(tuple(int(v) for v in h)) for h in hkl]
    )
    uniq_asu, asu_id = np.unique(sep_key, axis=0, return_inverse=True)
    uniq_pool, nonanom_id = np.unique(pool_key, axis=0, return_inverse=True)
    print(f"{len(uniq_asu)} Friedel-separate, {len(uniq_pool)} Friedel-pooled")

    # Wilson intensities on the pooled class, split into a Bijvoet pair. Every
    # row of one Friedel-separate class shares its pooled class and its sign,
    # so a scatter carries them over. A centric has I(+) = I(-) by symmetry.
    i_pool = rng.exponential(scale=1.0, size=len(uniq_pool)) * 400.0
    delta = rng.normal(0.0, args.anomalous, size=len(uniq_pool))
    pool_of_asu = np.zeros(len(uniq_asu), dtype=np.int64)
    sign_of_asu = np.zeros(len(uniq_asu))
    centric_of_asu = np.zeros(len(uniq_asu), dtype=bool)
    pool_of_asu[asu_id] = nonanom_id
    sign_of_asu[asu_id] = np.where(is_plus, 1.0, -1.0)
    centric_of_asu[asu_id] = centric
    i_true_asu = i_pool[pool_of_asu] * (
        1.0
        + np.where(centric_of_asu, 0.0, sign_of_asu * 0.5 * delta[pool_of_asu])
    )

    # ---- the true scale's fixed parts -----------------------------------
    # A spectrum with the SHAPE of the measured one, rescaled to a stated
    # swing: the identified curve's coefficients carry their own amplitude, and
    # inheriting it gave 3e5x across the band rather than the real 18.3x.
    spec_shape = np.array([0.0, 0.254, -1.923, 1.012, -0.333, -0.225])
    _probe = chebyshev_curve(
        np.linspace(LAMBDA_MIN, LAMBDA_MAX, 512), LAMBDA_MIN, LAMBDA_MAX,
        spec_shape,
    )
    spec_coeffs = spec_shape * (
        math.log(args.spectrum_swing) / max(float(np.ptp(_probe)), 1e-9)
    )
    abs_coeffs = rng.normal(0.0, 1.0, size=(args.lmax + 1) ** 2 - 1)
    abs_coeffs *= 0.18 / max(np.abs(abs_coeffs).max(), 1e-9)   # ~1.8x swing
    # Per-image gray: a smooth drift plus a jitter.  The split matters, and
    # getting it wrong cost a wrong conclusion once: an i.i.d. term across 1000
    # images is white noise that no smooth basis can represent, so it dominated
    # the floor of every scale fit and looked like a ceiling on the physical
    # basis.  Real beam and crystal variation is mostly smooth in the
    # goniometer angle, with a small jump component -- and careless can
    # represent the jumps, because it encodes BATCH with positional encoding,
    # where a Chebyshev image term cannot.  `--image-jitter` is therefore an
    # experimental knob, not scenery: it sets how much of the per-image scale
    # is unreachable by a smooth parameterisation.
    drift = np.cos(
        np.linspace(0.0, 3.0 * np.pi, args.images)
    ) * args.image_drift
    image_gray = drift + rng.normal(0.0, args.image_jitter, size=args.images)
    decay_b = 2.0                                               # A^2
    tau_pol, phi0_pol = 0.95, 0.0

    nz, ny, nx = args.shoebox
    n_pix = nz * ny * nx
    gz, gy, gx = np.meshgrid(
        np.arange(nz) - (nz - 1) / 2.0,
        np.arange(ny) - (ny - 1) / 2.0,
        np.arange(nx) - (nx - 1) / 2.0,
        indexing="ij",
    )

    rows: dict[str, list] = {k: [] for k in (
        "asu_id", "nonanom_id", "centric", "friedel_plus",
        "H", "K", "L", "d", "wavelength",
        "image_num", "xyzcal.px.0", "xyzcal.px.1", "xyzcal.px.2",
        "s1.0", "s1.1", "s1.2", "true_scale", "true_bg", "true_intensity",
        "true_spectrum", "true_polarisation", "true_absorption",
        "true_decay", "true_image_gray",
    )}
    counts_list, masks_list = [], []

    for j in range(args.images):
        phi = math.radians(j * DEG_PER_IMAGE)
        hit, lam, q = excited_reflections(
            hkl, b_matrix, u_matrix, phi, LAMBDA_MIN, LAMBDA_MAX
        )
        if not hit.any():
            continue
        idx = np.flatnonzero(hit)
        lam_i, q_i = lam[idx], q[idx]
        s0 = BEAM / lam_i[:, None]
        s1 = q_i + s0
        s1_hat = s1 / np.linalg.norm(s1, axis=1, keepdims=True)
        # detector hit, with the beam centre at the panel centre
        with np.errstate(divide="ignore", invalid="ignore"):
            t = DETECTOR_DISTANCE / s1_hat[:, 2]
        on = np.isfinite(t) & (t > 0)
        xmm, ymm = s1_hat[:, 0] * t, s1_hat[:, 1] * t
        px = xmm / PIXEL_SIZE + DETECTOR_PIXELS / 2.0
        py = ymm / PIXEL_SIZE + DETECTOR_PIXELS / 2.0
        on &= (px > nx) & (px < DETECTOR_PIXELS - nx)
        on &= (py > ny) & (py < DETECTOR_PIXELS - ny)
        if not on.any():
            continue
        idx, lam_i, s1_hat = idx[on], lam_i[on], s1_hat[on]
        px, py = px[on], py[on]
        d_i = d_all[idx]

        two_theta = 2.0 * np.arcsin(np.clip(lam_i / (2.0 * d_i), -1, 1))
        phi_det = np.arctan2(py - DETECTOR_PIXELS / 2.0, px - DETECTOR_PIXELS / 2.0)

        # the scale's terms, each recorded so an arm can be scored term by term
        g_spec = chebyshev_curve(lam_i, LAMBDA_MIN, LAMBDA_MAX, spec_coeffs)
        pol = np.log(np.maximum(0.5 * (
            1.0 + np.cos(two_theta) ** 2
            - tau_pol * np.cos(2.0 * (phi_det - phi0_pol)) * np.sin(two_theta) ** 2
        ), 1e-8))
        # absorption in the CRYSTAL frame: both legs, as DIALS models it, and a
        # coordinate no arm's lab-frame scale has
        rot_t = rotation_about(ROTATION_AXIS, phi).T
        s1_c = (rot_t @ s1_hat.T).T
        s0_c = np.broadcast_to(rot_t @ BEAM, s1_c.shape)
        absorp = (real_sph_harm(s1_c, args.lmax)
                  + real_sph_harm(np.ascontiguousarray(s0_c), args.lmax)) @ abs_coeffs
        decay = -2.0 * decay_b / (2.0 * d_i) ** 2
        gray = np.full(len(idx), image_gray[j])
        log_s = g_spec + pol + absorp + decay + gray
        scale = np.exp(log_s)

        i_obs = i_true_asu[asu_id[idx]]
        mu_peak = scale * i_obs

        # a 3-D Gaussian profile, normalised over the shoebox
        prf = np.exp(-0.5 * ((gx / 1.9) ** 2 + (gy / 1.9) ** 2 + (gz / 0.8) ** 2))
        prf = (prf / prf.sum()).reshape(1, -1)
        # background as a smooth tilted plane, not the constant the merge infers
        share = args.background_share
        bg_level = mu_peak * share / (1.0 - share) / n_pix
        tilt = (0.12 * gx / max(nx, 1) + 0.09 * gy / max(ny, 1)).reshape(1, -1)
        bg = bg_level[:, None] * (1.0 + tilt)

        rate = mu_peak[:, None] * prf + bg
        counts = rng.poisson(np.maximum(rate, 0.0)).astype(np.float32)
        counts_list.append(counts)
        masks_list.append(np.ones_like(counts, dtype=np.float32))

        for name, value in (
            ("asu_id", asu_id[idx]), ("nonanom_id", nonanom_id[idx]),
            ("centric", centric[idx]),
            # which member of the Friedel pair this is. The anomalous-preserving
            # terms (`centric_anchor_weight`, `double_wilson_weight`) need it to
            # split a pooled class into its two mates, and the centric anchor is
            # the one today's identifiability analysis singles out: centrics have
            # I(+) = I(-) by symmetry, so they measure the scale's
            # Friedel-ANTISYMMETRIC error against a known zero -- which is the
            # subspace `tr(J^-1 G)` says the anomalous signal is exposed in.
            ("friedel_plus", is_plus[idx].astype(float)),
            ("H", hkl[idx, 0]), ("K", hkl[idx, 1]), ("L", hkl[idx, 2]),
            ("d", d_i), ("wavelength", lam_i),
            ("image_num", np.full(len(idx), j)),
            ("xyzcal.px.0", px), ("xyzcal.px.1", py),
            ("xyzcal.px.2", np.zeros(len(idx))),
            ("s1.0", s1_hat[:, 0] / lam_i), ("s1.1", s1_hat[:, 1] / lam_i),
            ("s1.2", s1_hat[:, 2] / lam_i),
            ("true_scale", scale), ("true_bg", bg.mean(axis=1)),
            ("true_intensity", i_obs),
            ("true_spectrum", g_spec), ("true_polarisation", pol),
            ("true_absorption", absorp), ("true_decay", decay),
            ("true_image_gray", gray),
        ):
            rows[name].append(np.asarray(value))

    counts = np.concatenate(counts_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)
    md = {k: np.concatenate(v) for k, v in rows.items()}
    n = len(counts)
    print(f"{n} observations on {args.images} images, {n_pix} pixels each")

    # the loader's remaining required columns, with the meanings the real
    # pipeline gives them
    md["refl_ids"] = np.arange(n, dtype=np.float64)
    md["panel"] = np.zeros(n)
    md["intensity.sum.value"] = (counts - md["true_bg"][:, None]).sum(axis=1)
    md["intensity.sum.variance"] = counts.sum(axis=1)
    md["background.sum.value"] = md["true_bg"] * n_pix
    md["background.sum.variance"] = md["true_bg"] * n_pix
    # resolution bins for the per-bin background prior, and a held-out flag
    order = np.argsort(-md["d"])
    md["group_label"] = np.zeros(n)
    md["group_label"][order] = np.minimum((np.arange(n) * 30) // n, 29)
    md["is_test"] = (np.random.default_rng(args.seed + 1).random(n) < 0.05).astype(
        np.float64
    )

    # compact the asu ids so they index 0..n_unique-1 over what was observed
    for key in ("asu_id", "nonanom_id"):
        _, md[key] = np.unique(md[key], return_inverse=True)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "counts.npy", counts)
    np.save(args.out / "masks.npy", masks)
    meta = {k: torch.from_numpy(np.asarray(v)) for k, v in md.items()}
    meta["asu_id"] = meta["asu_id"].long()
    meta["nonanom_id"] = meta["nonanom_id"].long()
    meta["centric"] = meta["centric"].bool()
    torch.save(meta, args.out / "metadata.pt")

    # asu_id -> hkl, which the merged writer needs to label the output
    table = np.zeros((int(md["asu_id"].max()) + 1, 3), dtype=np.int64)
    table[md["asu_id"].astype(np.int64)] = np.stack(
        [md["H"], md["K"], md["L"]], axis=1
    ).astype(np.int64)
    torch.save(torch.from_numpy(table), args.out / "asu_id_to_hkl.pt")

    raw = torch.from_numpy(counts)
    ansc = 2.0 * (raw.clamp(min=0) + 0.375).sqrt()
    spec = {
        "name": "sim_laue",
        "shoebox_shape": list(args.shoebox),
        "cell": list(CELL),
        "space_group": SPACE_GROUP,
        "stats": {
            "raw": [float(raw.mean()), float(raw.var())],
            "anscombe": [float(ansc.mean()), float(ansc.var())],
        },
        "files": {
            "counts": "counts.npy",
            "masks": "masks.npy",
            "reference": "metadata.pt",
        },
        "truth": {
            "background_share": args.background_share,
            "anomalous": args.anomalous,
            "spectrum_coefficients": [float(c) for c in spec_coeffs],
            "absorption_coefficients": [float(c) for c in abs_coeffs],
            "decay_b": decay_b,
        "image_drift": args.image_drift,
        "image_jitter": args.image_jitter,
            "polarisation": {"tau": tau_pol, "phi0": phi0_pol},
        },
    }
    (args.out / "dataset.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

    # What the generator claims, checked rather than asserted in a docstring.
    # A simulator nobody validates is worse than no simulator: it produces
    # confident answers to the wrong question.
    print("\n-- the facts this dataset exists to reproduce --")
    share = float((md["true_bg"] * n_pix).sum() / counts.sum())
    print(f"background share of all counts         {share:.4f} "
          f"(asked for {args.background_share})")
    obs_per = np.bincount(md["asu_id"].astype(np.int64))
    print(f"observations per Friedel-separate hkl  median {np.median(obs_per):.0f}, "
          f"max {obs_per.max()}")
    swing = float(np.exp(np.ptp(np.quantile(md["true_spectrum"], [0.01, 0.99]))))
    print(f"true spectrum swing across the band    {swing:.1f}x "
          f"(asked for {args.spectrum_swing})")

    # The gauge that makes G(lambda) unidentifiable on stills is that
    # lambda = 2 d sin(theta(r)) -- so lambda is a FUNCTION of resolution and
    # detector radius, not merely correlated with either. The honest measure is
    # how much of log lambda a smooth surface in (d, r) can absorb.
    radius = np.hypot(md["xyzcal.px.0"] - DETECTOR_PIXELS / 2.0,
                      md["xyzcal.px.1"] - DETECTOR_PIXELS / 2.0)
    basis = [np.ones(n)]
    for col in (md["d"], radius):
        u = 2.0 * (col - col.min()) / max(np.ptp(col), 1e-12) - 1.0
        basis += [u ** k for k in range(1, 7)]
    design = np.stack(basis, axis=1)

    def absorbed(target):
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        return 1.0 - (target - design @ coef).var() / max(target.var(), 1e-30)

    # the quantity that matters is how much of the SPECTRUM a free surface in
    # (d, r) can absorb, which is what makes G(lambda) unidentifiable
    print(f"true G(lambda) absorbed by smooth (d, r)  "
          f"{100 * absorbed(md['true_spectrum']):.1f}% "
          "(73% on hewl_1118)")
    print(f"log lambda itself absorbed                 "
          f"{100 * absorbed(np.log(md['wavelength'])):.1f}%")
    print(f"true absorption absorbed (lab frame)       "
          f"{100 * absorbed(md['true_absorption']):.1f}% "
          "(a crystal-frame term no arm can represent)")

    # And the fact that breaks a monochromatic intuition: Friedel mates are
    # excited at different wavelengths, so scale error cannot cancel in
    # I(+) - I(-) the way it does on a rotation series.
    lam_bin = np.round(md["wavelength"] / 0.02).astype(np.int64)
    sign = np.where(md["asu_id"] == md["asu_id"], 0, 0)
    plus_of_pool = np.zeros(int(md["nonanom_id"].max()) + 1, dtype=np.int64)
    plus_of_pool[md["nonanom_id"].astype(np.int64)] = md["asu_id"].astype(np.int64)
    sign = (md["asu_id"].astype(np.int64)
            != plus_of_pool[md["nonanom_id"].astype(np.int64)]).astype(np.int64)
    stratum = md["nonanom_id"].astype(np.int64) * 100_000 + lam_bin
    _, inv = np.unique(stratum, return_inverse=True)
    n_strata = inv.max() + 1
    has_plus = np.zeros(n_strata, dtype=bool)
    has_minus = np.zeros(n_strata, dtype=bool)
    np.logical_or.at(has_plus, inv, sign == 0)
    np.logical_or.at(has_minus, inv, sign == 1)
    both = has_plus & has_minus
    print(f"(hkl, 0.02 A) strata holding BOTH signs {100 * both.mean():.1f}%, "
          f"covering {100 * both[inv].mean():.1f}% of observations")
    paired = md["nonanom_id"][sign == 0], md["nonanom_id"][sign == 1]
    shared = len(np.intersect1d(*paired))
    print(f"reflections with both Friedel mates observed  {shared} of "
          f"{int(md['nonanom_id'].max()) + 1}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
