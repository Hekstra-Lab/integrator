"""Does joint inference recover I_h better than integrate -> scale -> merge?

A small, exactly-solvable version of the pixel-to-reflection model, built to
test the thesis in `docs/joint_model_closed_forms.md` without a GPU, a
refinement, or an optimizer to argue about.

The staged pipeline commits to a point estimate per observation and then scales
and merges those points. The joint model never forms them: the pixel-level
sufficient statistics pass straight to the reflection-level posterior, because
integration and merging are the same conjugate update at two grouping levels.

Nothing here is oracle-specified except the background. In particular the
Wilson prior is LEARNED, because tau_h = exp(2 B s_h^2) / G is what identifies
the inter-reflection scale -- rescaling every s_i leaves the collapsed
likelihood invariant except through (alpha_W, tau_h) -- so handing the prior to
the estimator would hand it the answer.

Every block is closed form or one-dimensional, so no gradient descent enters
the comparison:

    r_ip        E-step, r = s_i I_h prf / (s_i I_h prf + b)
    I_h         Gamma(alpha_W + sum_i sum_p r c, tau_h + sum_i s_i)
    s           per-image Gamma, or the collapsed objective (L-BFGS on 10 dims)
    G           closed form: G = sum_h exp(2 B s_h^2) I_h / (alpha_W H)
    B           scalar root of sum_h s_h^2 [alpha_W - exp(2 B s_h^2) I_h / G]

Estimators, all given the true background since background estimation is not
what is under test:

    staged        profile-fit ML per observation, unbiased and free to go
                  negative, then descale and merge by 1/sigma^2. The
                  conventional pipeline in miniature.
    staged + FW   the same, then the fitted Wilson prior applied AFTER the
                  merge by quadrature -- i.e. French-Wilson. This is the real
                  pipeline, and the fair comparison against `joint mean`.
    joint ML      pixel-level pooling straight to the reflection, no prior.
    joint mean    the same pooling with the prior applied INSIDE the loop, so
                  the prior also stabilises the E-step.

The question the last two rows ask is the sharp one: does applying the prior
inside beat applying it after?

Three findings from the oracle-prior version of this script, kept here because
they are what the estimators are built to expose:

- staged and joint ML are the SAME estimator above ~100 photons (rms_log 0.2041
  vs 0.2040), so jointness buys nothing in the strong-data regime.
- joint ML collapses at low counts. The statistic is unbiased at the true
  intensity, E[sum_p r c] = sum_p [I prf/(I prf + b)](I prf + b) = I, so the
  bias is entirely in the FIXED POINT: F(I) = sum_p [I prf_p/(I prf_p + b)] c_p
  has an absorbing root at zero and loses its positive root when F'(0) < 1.
- the prior is what rescues it, which is why it is load-bearing internally and
  harmful at export: careless then applies it a second time.

Two traps this script fell into, both the failure modes it exists to study:
never clip pixels at zero (`max(c-b, 0)` biased the result 19x high at a 96.6%
background share), and never weight by the observed counts (`w = 1/max(c,1)`
down-weights whichever pixels fluctuated high, biasing the result 2x low).

Deliberately NOT modelled, so a win here is necessary and not sufficient: the
counts are drawn from the same generative model the estimators assume, and all
reflections are acentric. `simulate_laue_merging.py` exists for
misspecification -- it draws absorption from a crystal-frame spherical-harmonic
surface and the background from a tilted plane.

Usage:
    uv run python scripts/sim_joint_vs_staged.py
    uv run python scripts/sim_joint_vs_staged.py --hkls 200 --seeds 8
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import brentq, minimize

ALPHA_W = 1.0          # acentric Wilson shape; centrics would be 0.5


# --------------------------------------------------------------- generation
def anisotropic_profile(nx: int, ny: int, theta: float, s_rad: float, s_tan: float):
    """A normalised 2-D Gaussian stretched along a spot-specific direction.

    The existing simulator gives every spot the same isotropic blob, which
    makes the profile families indistinguishable on it. Real spots elongate
    radially (the `PhysicalProfileSurrogate` form sigma_r^2 = sigma_t^2 + c,
    with c the wavelength spread) and the elongation axis points at the beam
    centre, so the orientation is a function of detector position.
    """
    gy, gx = np.mgrid[0:ny, 0:nx]
    gx = gx - (nx - 1) / 2.0
    gy = gy - (ny - 1) / 2.0
    c, s = np.cos(theta), np.sin(theta)
    u = c * gx + s * gy          # along the elongation axis (radial)
    v = -s * gx + c * gy         # across it (tangential)
    prf = np.exp(-0.5 * ((u / s_rad) ** 2 + (v / s_tan) ** 2))
    return (prf / prf.sum()).ravel()


def simulate(rng, *, images, hkls, nx, ny, photons, bg_share, b_true,
             d_min=1.5, d_max=8.0):
    """Counts for every (reflection, image) pair, plus the truth.

    Intensities are a real Wilson draw with a resolution falloff,
    Sigma_h = G exp(-2 B s_h^2), so there is a prior to learn rather than a
    single mean to guess.
    """
    n_obs = images * hkls

    # resolution, uniform in s^2 so the shells carry comparable counts
    s2_lo, s2_hi = 1.0 / (4 * d_max ** 2), 1.0 / (4 * d_min ** 2)
    s2 = rng.uniform(s2_lo, s2_hi, hkls)
    d_h = 1.0 / (2.0 * np.sqrt(s2))

    g_true = photons / np.exp(-2.0 * b_true * s2).mean()
    sigma_h = g_true * np.exp(-2.0 * b_true * s2)
    i_true = rng.exponential(sigma_h)                 # acentric Wilson
    i_true = np.maximum(i_true, 1e-6 * sigma_h)

    img_of = np.repeat(np.arange(images), hkls)
    hkl_of = np.tile(np.arange(hkls), images)

    # per-image scale: a smooth drift (flux + decay) plus shot-to-shot jitter.
    # Per-image rather than per-observation keeps the model unsaturated (10
    # parameters against images*hkls observations) and the gauge exactly
    # one-dimensional, so the scale step is closed form and no optimizer enters
    # the comparison. On Laue the real split is lopsided the other way --
    # wavelength explains 50.7% of careless's log scale against image's 1.6% --
    # so this is the monochromatic-ish limit.
    log_gray = 0.35 * np.linspace(-1.0, 1.0, images) + rng.normal(0, 0.12, images)
    log_gray = log_gray - log_gray.mean()             # fix the gauge in the truth
    s_true = np.exp(log_gray)[img_of]

    # spot shape: orientation and stretch from the spot's detector position
    radius = rng.uniform(0.2, 1.0, n_obs)
    phi = rng.uniform(0, 2 * np.pi, n_obs)
    s_tan = 1.5 + 0.8 * radius ** 2
    s_rad = np.sqrt(s_tan ** 2 + 1.2)     # radial = tangential + wavelength spread
    prf = np.stack(
        [anisotropic_profile(nx, ny, phi[i], s_rad[i], s_tan[i]) for i in range(n_obs)]
    )

    mu_peak = s_true * i_true[hkl_of]
    bg = mu_peak.mean() * bg_share / (1.0 - bg_share) / (nx * ny)
    bg = np.full(n_obs, bg)               # one background level, as a real detector
    rate = mu_peak[:, None] * prf + bg[:, None]
    counts = rng.poisson(rate).astype(float)

    return dict(counts=counts, prf=prf, bg=bg, s_true=s_true, i_true=i_true,
                img_of=img_of, hkl_of=hkl_of, images=images, hkls=hkls,
                s2=s2, d_h=d_h, g_true=g_true, b_true=b_true)


# ------------------------------------------------------------ the prior fit
def fit_wilson_ml(i_h, s2, alpha_w=ALPHA_W, b_lo=-50.0, b_hi=400.0):
    """(G, B) from intensities by maximum likelihood, both effectively exact.

    G is closed form given B,  G = sum_h exp(2 B s_h^2) I_h / (alpha_W H),
    and B solves  sum_h s_h^2 [alpha_W - exp(2 B s_h^2) I_h / G] = 0,
    a scalar root. Linear in I_h, so merged estimates that came out negative
    can be used as they are rather than clipped.
    """
    def score(b):
        w = np.exp(2.0 * b * s2)
        g = (w * i_h).sum() / (alpha_w * len(i_h))
        if not np.isfinite(g) or g <= 0:
            return np.nan
        return float(np.sum(s2 * (alpha_w - w * i_h / g)))

    lo, hi = score(b_lo), score(b_hi)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo * hi > 0:
        b = 0.0                                   # no sign change: fall back flat
    else:
        b = float(brentq(score, b_lo, b_hi, maxiter=200))
    w = np.exp(2.0 * b * s2)
    g = float((w * i_h).sum() / (alpha_w * len(i_h)))
    return max(g, 1e-12), b


def fit_wilson(i_h, s2, n_bins=8):
    """(G, B) by the Wilson plot: regress log<I> per resolution shell on s^2.

    E[I_h] = alpha_W / tau_h = G exp(-2 B s_h^2), so log<I> per shell is linear
    in s^2 with slope -2B and intercept log G.

    Binned rather than the per-reflection maximum likelihood fit in
    `fit_wilson_ml`, even though that one is closed form in G and a scalar root
    in B. The ML score multiplies each I_h by exp(2 B s_h^2), which at high
    resolution amplifies the merged estimates that came out negative until G
    itself goes negative and the root-find hits NaN. Shell means stay positive
    because they average many reflections, so one routine serves both the
    staged path -- whose inputs contain negatives, correctly -- and the joint
    path, whose Gamma posterior means never do.
    """
    order = np.argsort(s2)
    chunks = np.array_split(order, max(2, min(n_bins, len(s2) // 4)))
    xs, ys = [], []
    for ch in chunks:
        m = float(np.mean(i_h[ch]))
        if m > 0:
            xs.append(float(np.mean(s2[ch])))
            ys.append(np.log(m))
    if len(xs) < 2:
        return max(float(np.mean(np.maximum(i_h, 1e-12))), 1e-12), 0.0
    slope, intercept = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    return max(float(np.exp(intercept)), 1e-12), float(-slope / 2.0)


def tau_of(g, b, s2):
    """The Wilson rate, 1/Sigma_h."""
    return np.exp(2.0 * b * s2) / g


# ----------------------------------------------------------- the estimators
def profile_fit(d):
    """Per-observation profile-fitting ML estimate and its sampling variance.

    Least squares of (c_p - b) on prf_p, which is what DIALS and laue.integrate
    do. Unbiased, and negative on a blank shoebox about half the time -- the
    property the downstream expects and the one a Gamma posterior mean destroys.

    Weights must not come from the observed counts. In a 96.6%-background
    shoebox var(c_p) ~ b, constant over the shoebox and therefore cancelling
    between numerator and denominator, so uniform weights are both unbiased and
    near-optimal: E[c_p - b] = I prf_p makes E[numerator] = I sum_p prf_p^2.
    """
    c, prf, b = d["counts"], d["prf"], d["bg"][:, None]
    den = (prf * prf).sum(1)
    i_hat = (prf * (c - b)).sum(1) / np.maximum(den, 1e-12)
    var = d["bg"] / np.maximum(den, 1e-12)
    return i_hat, var


def staged(d, s_hat):
    """Integrate to a point estimate, then descale and merge by 1/sigma^2."""
    i_obs, var = profile_fit(d)
    w = s_hat ** 2 / np.maximum(var, 1e-12)
    num = np.bincount(d["hkl_of"], weights=w * i_obs / s_hat, minlength=d["hkls"])
    den = np.bincount(d["hkl_of"], weights=w, minlength=d["hkls"])
    return num / np.maximum(den, 1e-12), 1.0 / np.maximum(den, 1e-12)


def french_wilson(i_hat, var, g, b, s2, alpha_w=ALPHA_W, n_grid=400):
    """Posterior mean under a Gamma prior and a Gaussian likelihood, by quadrature.

    This is the prior applied AFTER the merge, which is what careless and
    phenix's French-Wilson step do. Quadrature rather than the usual series
    because 50 reflections on a 400-point grid costs nothing and leaves no
    approximation to argue about.
    """
    tau = tau_of(g, b, s2)
    sd = np.sqrt(np.maximum(var, 1e-12))
    out = np.empty_like(i_hat)
    for k in range(len(i_hat)):
        hi = max(i_hat[k] + 8 * sd[k], 20.0 / max(tau[k], 1e-12))
        grid = np.linspace(1e-9, hi, n_grid)
        logp = ((alpha_w - 1) * np.log(grid) - tau[k] * grid
                - 0.5 * ((grid - i_hat[k]) / sd[k]) ** 2)
        p = np.exp(logp - logp.max())
        out[k] = float((grid * p).sum() / max(p.sum(), 1e-300))
    return out


def joint(d, s_hat, *, prior: bool, n_iter=40):
    """Reflection-level conjugate update, alternated with the E-step and the prior.

    I_h ~ Gamma(alpha_W + sum_i sum_p r c, tau_h + sum_i s_i). No
    per-observation intensity is ever formed. With `prior=True` the Wilson
    (G, B) are refit from the current intensities each sweep, so the prior is
    learned rather than given, and it also stabilises the E-step.
    """
    c, prf, bg = d["counts"], d["prf"], d["bg"][:, None]
    rate0 = np.bincount(d["hkl_of"], weights=s_hat, minlength=d["hkls"])
    i_h = np.maximum(staged(d, s_hat)[0], 1e-9)
    g = b = None
    for _ in range(n_iter):
        sig_rate = (s_hat * i_h[d["hkl_of"]])[:, None] * prf
        r = sig_rate / np.maximum(sig_rate + bg, 1e-12)
        photons = np.bincount(d["hkl_of"], weights=(r * c).sum(1),
                              minlength=d["hkls"])
        if prior:
            g, b = fit_wilson(i_h, d["s2"])
            i_h = (ALPHA_W + photons) / np.maximum(rate0 + tau_of(g, b, d["s2"]), 1e-12)
        else:
            i_h = photons / np.maximum(rate0, 1e-12)
        i_h = np.maximum(i_h, 1e-12)
    return i_h, g, b


def collapsed_nll(log_s_img, d):
    """-log p(data | s) with I_h integrated out, prior refit inside.

    sum_i photons_i log s_i - A_h log(tau_h + sum_i s_i), summed over
    reflections. The photon statistic depends on s through the E-step, so it is
    recomputed here rather than passed in -- doing the latter pinned the scale
    error at 0.205 regardless of counts, because L-BFGS was optimising against
    a statistic frozen at s = 1.
    """
    log_s = log_s_img - log_s_img.mean()              # gauge fix
    s = np.exp(log_s)[d["img_of"]]
    c, prf, bg = d["counts"], d["prf"], d["bg"][:, None]
    rate0 = np.bincount(d["hkl_of"], weights=s, minlength=d["hkls"])

    i_h = np.maximum(staged(d, s)[0], 1e-9)
    for _ in range(8):
        sig_rate = (s * i_h[d["hkl_of"]])[:, None] * prf
        r = sig_rate / np.maximum(sig_rate + bg, 1e-12)
        photons_h = np.bincount(d["hkl_of"], weights=(r * c).sum(1),
                                minlength=d["hkls"])
        g, b = fit_wilson(i_h, d["s2"])
        tau = tau_of(g, b, d["s2"])
        i_h = np.maximum((ALPHA_W + photons_h) / np.maximum(rate0 + tau, 1e-12), 1e-12)

    photons_i = (r * c).sum(1)
    A = ALPHA_W + photons_h
    return -(np.sum(photons_i * np.log(s)) - np.sum(A * np.log(rate0 + tau)))


def joint_collapsed(d):
    """L-BFGS on the collapsed objective over 10 log-scales, then one I_h update."""
    res = minimize(collapsed_nll, np.zeros(d["images"]), args=(d,),
                   method="L-BFGS-B", options=dict(maxiter=120))
    log_s = res.x - res.x.mean()
    s_hat = np.exp(log_s)[d["img_of"]]
    i_h, g, b = joint(d, s_hat, prior=True)
    return i_h, s_hat, g, b


def fit_scales_staged(d, n_iter=300):
    """Per-image scales from the point estimates, the way scaling programs do.

    Sees only I_hat_i, never the pixels, which is the hand-off under test.
    """
    i_obs, _ = profile_fit(d)
    s = np.ones(d["images"])
    for _ in range(n_iter):
        i_h = np.bincount(d["hkl_of"], weights=i_obs / s[d["img_of"]],
                          minlength=d["hkls"]) / d["images"]
        pred = i_h[d["hkl_of"]]
        num = np.bincount(d["img_of"], weights=i_obs * pred, minlength=d["images"])
        den = np.bincount(d["img_of"], weights=pred ** 2, minlength=d["images"])
        s = np.maximum(num / np.maximum(den, 1e-12), 1e-6)
        s = s / np.exp(np.mean(np.log(s)))
    return s[d["img_of"]]


# ------------------------------------------------------------------ scoring
def score(name, i_hat, d, half=None):
    """rms of log(I_hat / I_true), overall or on one resolution half.

    Reported by half because a Wilson falloff puts almost all the difficulty in
    the high-resolution shells, where a single aggregate hides it.
    """
    i_true, s2 = d["i_true"], d["s2"]
    sel = np.ones(len(i_true), bool)
    if half == "low":
        sel = s2 <= np.median(s2)
    elif half == "high":
        sel = s2 > np.median(s2)
    neg = float(np.mean(i_hat[sel] <= 0))
    ok = sel & (i_hat > 0) & (i_true > 0)
    lr = np.log(i_hat[ok]) - np.log(i_true[ok])
    return dict(name=name, cc=float(np.corrcoef(np.log(i_hat[ok]),
                                                np.log(i_true[ok]))[0, 1]),
                rms_log=float(lr.std()), bias_log=float(lr.mean()), neg=neg)


def main():
    p = argparse.ArgumentParser(description="joint vs staged recovery of I_h")
    p.add_argument("--images", type=int, default=10)
    p.add_argument("--hkls", type=int, default=50)
    p.add_argument("--nx", type=int, default=15)
    p.add_argument("--ny", type=int, default=15)
    p.add_argument("--photons", type=float, default=None)
    p.add_argument("--b-true", type=float, default=20.0, help="Wilson B, A^2")
    p.add_argument("--bg-share", type=float, default=0.966)
    p.add_argument("--seeds", type=int, default=6)
    args = p.parse_args()

    ladder = ([args.photons] if args.photons else [10.0, 30.0, 100.0, 1000.0])
    print(f"{args.images} images x {args.hkls} hkls = "
          f"{args.images * args.hkls} observations, {args.nx}x{args.ny} px, "
          f"background share {args.bg_share}, Wilson B {args.b_true} A^2")
    print("prior (G, B) LEARNED by every estimator that uses one.")
    print(f"{args.seeds} seeds. rms_log = rms of log(I_hat/I_true), lower better.\n")
    hdr = (f"{'photons':>7}  {'estimator':<19} {'CC_log':>7} {'rms_all':>8} "
           f"{'rms_lowres':>11} {'rms_hires':>10} {'bias':>7} {'neg':>6} "
           f"{'G err':>7} {'B err':>7}")
    print(hdr)
    print("-" * len(hdr))

    for photons in ladder:
        rows: dict[str, list] = {}
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            d = simulate(rng, images=args.images, hkls=args.hkls, nx=args.nx,
                         ny=args.ny, photons=photons, bg_share=args.bg_share,
                         b_true=args.b_true)
            s_st = fit_scales_staged(d)

            i_st, v_st = staged(d, s_st)
            g_st, b_st = fit_wilson(i_st, d["s2"])
            i_fw = french_wilson(i_st, v_st, g_st, b_st, d["s2"])
            i_jml, _, _ = joint(d, s_st, prior=False)
            i_jm, g_jm, b_jm = joint(d, s_st, prior=True)
            i_cl, s_cl, g_cl, b_cl = joint_collapsed(d)

            for nm, i_hat, gb in (("staged", i_st, (g_st, b_st)),
                                  ("staged + FW", i_fw, (g_st, b_st)),
                                  ("joint ML", i_jml, (None, None)),
                                  ("joint mean", i_jm, (g_jm, b_jm)),
                                  ("joint collapsed", i_cl, (g_cl, b_cl))):
                e = score(nm, i_hat, d)
                e["rms_lo"] = score(nm, i_hat, d, "low")["rms_log"]
                e["rms_hi"] = score(nm, i_hat, d, "high")["rms_log"]
                g, b = gb
                e["g_err"] = (np.nan if g is None
                              else abs(np.log(g / d["g_true"])))
                e["b_err"] = np.nan if b is None else abs(b - d["b_true"])
                rows.setdefault(nm, []).append(e)
            e = dict(name="  scale rms_log", cc=np.nan, rms_log=np.nan,
                     rms_lo=np.nan, rms_hi=np.nan, bias_log=np.nan, neg=np.nan,
                     g_err=np.nan, b_err=np.nan)
            err = np.log(s_cl) - np.log(d["s_true"])
            e["rms_log"] = float((err - err.mean()).std())
            rows.setdefault(e["name"], []).append(e)

        for name, es in rows.items():
            a = {k: np.nanmean([x[k] for x in es])
                 for k in ("cc", "rms_log", "rms_lo", "rms_hi", "bias_log",
                           "neg", "g_err", "b_err")}
            print(f"{photons:>7.0f}  {name:<19} {a['cc']:>7.4f} "
                  f"{a['rms_log']:>8.4f} {a['rms_lo']:>11.4f} "
                  f"{a['rms_hi']:>10.4f} {a['bias_log']:>7.3f} "
                  f"{a['neg']:>6.3f} {a['g_err']:>7.3f} {a['b_err']:>7.2f}")
        print()


if __name__ == "__main__":
    main()
