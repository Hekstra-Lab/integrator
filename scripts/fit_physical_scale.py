"""Fit a physical per-observation scale by conditional Poisson likelihood.

The experiment that motivates this: taking a trained checkpoint and replacing
only the per-observation scale with careless's fitted one moved $R_\\mathrm{free}$
from 0.2017 to 0.1786 and the mean anomalous peak from 3.35 to 6.17, against
careless's own 0.1780 / 8.64.  The scale is the defect; the rest of the model
is not.

So fit the scale directly, offline, with the intensities profiled out.  Writing
$\\mu_i = e_i\\,s_i\\,I_h$ with $e_i$ the base exposure, the Poisson likelihood
conditioned on each reflection's total signal is multinomial and $I_h$ drops
out exactly:

$$\\ell(c) = \\sum_i S_i \\Phi_i c - \\sum_h T_h \\log \\sum_{i \\in h} e_i e^{\\Phi_i c},
\\qquad T_h = \\sum_{i \\in h} S_i,$$

which is concave in the scale parameters $c$ (a log-sum-exp), free of
incidental-parameter bias in the Neyman--Scott sense, and solvable by Newton in
seconds on 2.66M observations.  This is the construction `stratified_bijvoet.py`
uses for a single parameter, generalised to the whole scale.

The basis is physics, not freedom, because capacity is what does the damage: a
factor the scale can represent per (reflection, Friedel sign) is in the gauge
of the likelihood, so it can manufacture a fake anomalous difference.  Ten
physics-shaped parameters fit careless's scale as well as a thousand generic
ones.

    log s = log G(lambda) + log P(2theta, phi; tau) + K(phi_image) - 2 B s^2
            + sum_lm c_lm [Y_lm(s0_c) + Y_lm(s1_c)]

Deliberately absent: any free function of detector position or resolution.  On
a still $\\lambda = 2d\\sin\\theta(r)$, so a free radial term absorbs most of a
realistic $G(\\lambda)$ and the spectrum stops being identified.

Usage
-----
    uv run python scripts/fit_physical_scale.py --dump merge_terms.pt \\
        --out scale_fit.pt --degree 6 --lmax 4 --image-degree 8
"""

import argparse
from pathlib import Path

import numpy as np
import torch

# the goniometer of the hewl_1118 series: 1 degree per image about lab +y,
# read from imported_HEWL_anom_0999.expt (oscillation 0,1,2,... and
# rotation_axis [0,1,0]). `laue.sequence_to_stills` split that scan into
# stills, so the crystal frame is recoverable from the image index alone.
DEFAULT_AXIS = (0.0, 1.0, 0.0)
DEFAULT_DEG_PER_IMAGE = 1.0


def _arr(dump, key, dtype=np.float64):
    v = dump[key]
    return (v.numpy() if torch.is_tensor(v) else np.asarray(v)).astype(dtype)


def chebyshev(t, degree, lo=None, hi=None, skip_const=True):
    """Chebyshev design on t mapped to [-1, 1]."""
    lo = np.quantile(t, 0.001) if lo is None else lo
    hi = np.quantile(t, 0.999) if hi is None else hi
    u = np.clip(2.0 * (t - lo) / max(hi - lo, 1e-12) - 1.0, -1.0, 1.0)
    cols = [np.ones_like(u), u]
    for k in range(2, degree + 1):
        cols.append(2.0 * u * cols[-1] - cols[-2])
    cols = cols[: degree + 1]
    return np.stack(cols[1:] if skip_const else cols, axis=1)


def real_sph_harm(xyz, lmax, include_odd=True):
    """Real spherical harmonics up to `lmax`, evaluated on unit vectors.

    `l = 0` is omitted: it is the overall scale, which the merge cannot see.
    Odd `l` is included by default -- for a centrosymmetric crystal shape the
    absorption path is even, but miscentring, the loop and the mother liquor
    are not, which is why DIALS fits odd orders too.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out, meta = [], []
    for ll in range(1, lmax + 1):
        if ll % 2 and not include_odd:
            continue
        # real tesseral harmonics, unnormalised (the fit absorbs the constants)
        if ll == 1:
            block = [z, x, y]
        elif ll == 2:
            block = [3 * z * z - 1, x * z, y * z, x * x - y * y, x * y]
        elif ll == 3:
            block = [z * (5 * z * z - 3), x * (5 * z * z - 1), y * (5 * z * z - 1),
                     z * (x * x - y * y), x * y * z,
                     x * (x * x - 3 * y * y), y * (3 * x * x - y * y)]
        elif ll == 4:
            z2 = z * z
            block = [35 * z2 * z2 - 30 * z2 + 3, x * z * (7 * z2 - 3),
                     y * z * (7 * z2 - 3), (x * x - y * y) * (7 * z2 - 1),
                     x * y * (7 * z2 - 1), x * z * (x * x - 3 * y * y),
                     y * z * (3 * x * x - y * y),
                     (x * x * (x * x - 3 * y * y) - y * y * (3 * x * x - y * y)),
                     x * y * (x * x - y * y)]
        else:
            raise ValueError(f"lmax {lmax} not supported")
        out.extend(block)
        meta.extend([ll] * len(block))
    return np.stack(out, axis=1), np.array(meta)


def rotate_about_axis(v, axis, angles):
    """Rodrigues rotation of rows of `v` by per-row `angles` about `axis`."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
    kv = np.cross(np.broadcast_to(k, v.shape), v)
    kkv = (v @ k)[:, None] * k[None, :]
    return v * c + kv * s + kkv * (1.0 - c)


def build_design(dump, degree, lmax, image_degree, include_odd, axis,
                 deg_per_image):
    """The physical design matrix, and a label per column block."""
    lam = _arr(dump, "wavelength")
    d = _arr(dump, "d")
    img = _arr(dump, "image_num", np.int64)
    x = _arr(dump, "xyzcal.px.0")
    y = _arr(dump, "xyzcal.px.1")
    s1 = np.stack([_arr(dump, f"s1.{k}") for k in range(3)], axis=1)

    sin_t = np.clip(lam / (2.0 * np.maximum(d, 1e-6)), -1.0, 1.0)
    two_t = 2.0 * np.arcsin(sin_t)
    s_sq = 1.0 / (4.0 * np.maximum(d, 1e-6) ** 2)

    blocks, labels, groups = [], [], []

    # the spectrum: the only term allowed to be a free function of wavelength
    g = chebyshev(lam, degree)
    blocks.append(g); labels += [f"G{k+1}" for k in range(g.shape[1])]
    groups += ["spectrum"] * g.shape[1]


    # per-image terms: the series is a rotation scan, so these are smooth in
    # the goniometer angle rather than 1000 free numbers
    phi_img = np.deg2rad(img * deg_per_image)
    k_img = chebyshev(img.astype(np.float64), image_degree)
    blocks.append(k_img); labels += [f"K{k+1}" for k in range(k_img.shape[1])]
    groups += ["image_gray"] * k_img.shape[1]

    # dose: an image-varying B, times s^2. A constant B is a resolution gauge,
    # so only the image-varying part exists.
    decay = chebyshev(img.astype(np.float64), max(image_degree // 2, 1)) * s_sq[:, None]
    blocks.append(decay); labels += [f"B{k+1}" for k in range(decay.shape[1])]
    groups += ["decay"] * decay.shape[1]

    # absorption in the crystal frame: undo the goniometer rotation, then sum
    # the harmonics of the incident and diffracted legs as DIALS does
    s1n = s1 / np.maximum(np.linalg.norm(s1, axis=1, keepdims=True), 1e-12)
    s0_lab = np.zeros_like(s1n)
    s0_lab[:, 2] = 1.0
    s1_c = rotate_about_axis(s1n, axis, -phi_img)
    s0_c = rotate_about_axis(s0_lab, axis, -phi_img)
    Y1, meta = real_sph_harm(s1_c, lmax, include_odd)
    Y0, _ = real_sph_harm(s0_c, lmax, include_odd)
    # mu ~ lambda^3, so the absorption's wavelength dependence is known
    absb = (Y0 + Y1) * ((lam / np.median(lam)) ** 3)[:, None]
    blocks.append(absb); labels += [f"A_l{ll}_{k}" for k, ll in enumerate(meta)]
    groups += ["absorption"] * absb.shape[1]

    X = np.hstack(blocks)
    pen = np.zeros(X.shape[1])
    for j, (lab, grp) in enumerate(zip(labels, groups)):
        if grp == "spectrum":
            k = int(lab[1:])
            pen[j] = k ** 4            # roughness: high orders fit noise
        elif grp == "image_gray":
            pen[j] = 1.0
        elif grp == "decay":
            pen[j] = 1.0
        elif grp == "absorption":
            ll = int(lab.split("_l")[1].split("_")[0])
            pen[j] = ll * (ll + 1)     # the DIALS surface restraint
        else:
            pen[j] = 0.0               # polarisation is physics, not a guess
    return X, np.array(labels), np.array(groups), pen


def polarisation_offset(two_t, phi_det, tau):
    """`log P` for the scattered intensity, as known physics.

    `P = [1 + cos^2(2t) - tau cos(2phi) sin^2(2t)] / 2` (Kahn et al. 1982;
    Ren & Moffat 1995 eq. 11).  The rate is multiplied by P; the quantity
    usually tabulated is the *correction* `1/P`, and applying that as if it
    were the scale is a sign error -- which is what `chebyshev_scale.py:762`
    currently does.  At 1.5 A and tau = 0.9 this term swings 2.1x with
    azimuth, against the ~4% a Bijvoet signal tolerates, and Friedel mates sit
    at different azimuths, so it does not cancel.
    """
    P = 0.5 * (1.0 + np.cos(two_t) ** 2
               - tau * np.cos(2.0 * phi_det) * np.sin(two_t) ** 2)
    return np.log(np.maximum(P, 1e-8))


def fit_conditional(X, S, e, group, pen, offset=None, weight=1.0,
                    iters=50, tol=1e-9):
    """Newton on the conditional (multinomial) Poisson likelihood.

    With `mu_i = e_i exp(eta_i) I_h` and `eta = Xc`, conditioning on each
    reflection's total `T_h` removes `I_h` exactly:

        l(c) = sum_i S_i eta_i - sum_h T_h log sum_{i in h} e_i exp(eta_i),

    whose gradient and Hessian are

        g = X^T (S - T pi) - Omega c
        H = -[ X^T diag(T pi) X - sum_h T_h m_h m_h^T ] - Omega,
        pi_i = w_i / sum_{j in h(i)} w_j,   m_h = sum_{i in h} pi_i X_i.

    The bracket is a sum of per-reflection multinomial covariances, so it is
    positive semidefinite and the objective is concave: Newton converges
    globally. `offset` carries terms with no free parameter (polarisation).
    """
    n_h = int(group.max()) + 1
    T = np.zeros(n_h)
    np.add.at(T, group, S)
    off = np.zeros(len(S)) if offset is None else offset
    c = np.zeros(X.shape[1])
    Omega = np.diag(weight * pen)
    prev = -np.inf
    for it in range(iters):
        eta = X @ c + off
        # the shift is per reflection, so it cancels in pi: subtract the max
        # within each group for a stable exponential
        mx = np.full(n_h, -np.inf)
        np.maximum.at(mx, group, eta)
        w = e * np.exp(eta - mx[group])
        Z = np.zeros(n_h)
        np.add.at(Z, group, w)
        pi = w / np.maximum(Z, 1e-300)[group]

        Tpi = T[group] * pi
        g = X.T @ (S - Tpi) - Omega @ c
        M = np.zeros((n_h, X.shape[1]))
        np.add.at(M, group, X * pi[:, None])
        info = X.T @ (X * Tpi[:, None]) - (M * T[:, None]).T @ M + Omega
        step = np.linalg.solve(info + 1e-12 * np.eye(len(c)), g)
        c = c + step

        ll = float(S @ (X @ c + off) - T @ np.log(np.maximum(Z, 1e-300))
                   - 0.5 * c @ Omega @ c)
        if it and abs(ll - prev) < tol * max(abs(ll), 1.0):
            break
        prev = ll
    return c, it + 1, prev


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit a physical per-observation scale by conditional "
        "Poisson likelihood."
    )
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--degree", type=int, default=6, help="spectrum degree")
    ap.add_argument("--lmax", type=int, default=4)
    ap.add_argument("--image-degree", type=int, default=8)
    ap.add_argument("--no-odd-sh", action="store_true")
    ap.add_argument("--penalty", type=float, default=1.0)
    ap.add_argument("--deg-per-image", type=float, default=DEFAULT_DEG_PER_IMAGE)
    args = ap.parse_args()

    dump = torch.load(args.dump, map_location="cpu", weights_only=False)
    S = _arr(dump, "sig_counts")
    E = _arr(dump, "exposure")
    s_cur = _arr(dump, "scale")
    asu = _arr(dump, "asu_id", np.int64)
    e = E / np.maximum(s_cur, 1e-12)
    ok = np.isfinite(S) & np.isfinite(e) & (e > 0) & (S >= 0)
    print(f"{len(S)} observations, {int(ok.sum())} usable, "
          f"{len(np.unique(asu[ok]))} reflections")

    X, labels, groups, pen = build_design(
        dump, args.degree, args.lmax, args.image_degree,
        not args.no_odd_sh, DEFAULT_AXIS, args.deg_per_image,
    )
    _, grp_idx = np.unique(asu[ok], return_inverse=True)
    for name in ("spectrum", "image_gray", "decay", "absorption"):
        print(f"  {name:13s} {int((groups == name).sum()):4d} parameters")

    lam = _arr(dump, "wavelength")
    d = _arr(dump, "d")
    xx, yy = _arr(dump, "xyzcal.px.0"), _arr(dump, "xyzcal.px.1")
    two_t = 2.0 * np.arcsin(np.clip(lam / (2.0 * np.maximum(d, 1e-6)), -1, 1))
    phi_det = np.arctan2(yy - yy.mean(), xx - xx.mean())

    best = None
    for tau in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
        off = polarisation_offset(two_t, phi_det, tau)
        c_t, it_t, ll_t = fit_conditional(
            X[ok], S[ok], e[ok], grp_idx, pen, offset=off[ok],
            weight=args.penalty,
        )
        print(f"  tau {tau:4.2f}: penalised conditional loglik {ll_t:.1f} "
              f"({it_t} Newton steps)")
        if best is None or ll_t > best[2]:
            best = (c_t, tau, ll_t, it_t)
    c, tau, _, iters = best
    print(f"polarisation fraction tau = {tau:.2f} chosen by the likelihood")
    offset = polarisation_offset(two_t, phi_det, tau)
    log_s = X @ c + offset
    log_s -= np.median(log_s[ok])          # the overall level is a gauge
    print(f"converged in {iters} Newton steps")
    print(f"  {'polarisation':13s} contributes sd "
          f"{offset[ok].std():.4f} in log, span "
          f"{np.exp(np.quantile(offset[ok],0.99) - np.quantile(offset[ok],0.01)):.2f}x")
    for name in ("spectrum", "image_gray", "decay", "absorption"):
        m = groups == name
        contrib = X[:, m] @ c[m]
        print(f"  {name:13s} contributes sd {contrib[ok].std():.4f} in log, "
              f"span {np.exp(np.quantile(contrib[ok],0.99) - np.quantile(contrib[ok],0.01)):.2f}x")
    torch.save(
        {"log_scale": torch.from_numpy(log_s).float(),
         "coefficients": torch.from_numpy(c).float(),
         "labels": list(labels), "groups": list(groups), "tau": tau},
        args.out,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
