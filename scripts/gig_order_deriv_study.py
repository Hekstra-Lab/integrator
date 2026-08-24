"""Empirical study: best way to compute D_p := d_p log K_p(w) (Bessel ORDER
derivative) and the ratio order-derivatives needed to learn nu in the GIG
hierarchical SVAE.

Oracle: mpmath mp.besselk + mp.diff at high precision (dps=50).
Candidates for D_p = d_p log K_p(w):
  (M1) central finite difference in p over scipy.special.kve:
         D_p ~ (log kve(p+h,w) - log kve(p-h,w)) / (2h)   [kve scaling cancels]
  (M2) numerical quadrature of  N(p,w) = int_0^inf e^{-w cosh t} t sinh(p t) dt
         then D_p = N / K_p(w); done in the *normalized* form
         D_p = int e^{-w cosh t} t sinh(p t) dt / int e^{-w cosh t} cosh(p t) dt
         with a saddle-centred substitution + clamp for large |p|.
  (M4) Richardson-extrapolated central FD (two step sizes) over kve.

We study D_p across p in {-300,-100,-30,-5,-0.5,0.5,1} x w in {1e-2,.1,1,10,100,1000}.

Run:  uv run --with mpmath python scripts/gig_order_deriv_study.py
"""

import time

import numpy as np
from scipy.special import kve

# ----------------------------------------------------------------------------
# mpmath oracle (high precision)
# ----------------------------------------------------------------------------
import mpmath as mp

mp.mp.dps = 50


def oracle_Dp(p, w):
    """d_p log K_p(w) to high precision."""
    f = lambda pp: mp.log(mp.besselk(pp, w))
    return float(mp.diff(f, p))


def oracle_logK(p, w):
    return float(mp.log(mp.besselk(p, w)))


def oracle_ratio_logderiv(p, w, k):
    """d_p log( K_{p+k}(w) / K_p(w) ).

    These order-derivatives are what the moment ratios K_{p+-1}/K_p contribute
    once p depends on nu. (For E[I]: k=+1; for E[1/I]: k=-1.) We report the
    log-derivative of the ratio = D_{p+k} - D_p, which is the clean target.
    """
    return oracle_Dp(p + k, w) - oracle_Dp(p, w)


# ----------------------------------------------------------------------------
# (M1) central finite difference over scipy kve  (kve scaling cancels in diff)
# ----------------------------------------------------------------------------
def m1_central_fd(p, w, h):
    # log K_p(w) = log kve(p,w) - w ; the -w shift cancels in the difference,
    # and so does the e^{+w} scaling, leaving a pure order-FD.
    lp = np.log(kve(p + h, w))
    lm = np.log(kve(p - h, w))
    return (lp - lm) / (2.0 * h)


def m4_richardson_fd(p, w, h):
    """Richardson extrapolation of central FD: combine h and 2h to kill O(h^2)."""
    d_h = m1_central_fd(p, w, h)
    d_2h = m1_central_fd(p, w, 2.0 * h)
    # central FD error ~ c*h^2 ; D ~ (4 d_h - d_2h)/3
    return (4.0 * d_h - d_2h) / 3.0


# ----------------------------------------------------------------------------
# (M2) quadrature of the order-derivative integral, normalized form
#   D_p = int_0^inf e^{-w cosh t} t sinh(p t) dt / int_0^inf e^{-w cosh t} cosh(p t) dt
# Numerically we factor out the common e^{-w cosh t} and the dominant exponential
# growth e^{|p| t} of cosh/sinh by working with a stabilized integrand.
# For large |p| the integrand e^{-w cosh t + |p| t} has a saddle; we substitute
# and clamp the range so e^{-w cosh t} doesn't underflow.
# ----------------------------------------------------------------------------
def m2_quadrature(p, w, n=4000):
    ap = abs(p)
    sgn = 1.0 if p >= 0 else -1.0  # sinh is odd; D_p odd in p, handle via sgn

    # Work with g(t) = e^{-w cosh t}. cosh/sinh(p t) for large |p| ~ 0.5 e^{|p|t}.
    # Stable log-integrands:
    #   numerator integrand   t * sinh(p t) * g(t)
    #   denominator integrand cosh(p t) * g(t)
    # Factor the peak: log of the cosh-branch envelope is  |p| t - w cosh t.
    # Saddle of phi(t) = |p| t - w cosh t : phi'(t)= |p| - w sinh t = 0
    #   => t* = asinh(|p|/w). Center the grid there with a width from phi''.
    if ap < 1e-12:
        return 0.0  # D_p is odd, =0 at p=0
    t_star = np.arcsinh(ap / w)
    # phi''(t*) = -w cosh(t*) ; gaussian width ~ 1/sqrt(w cosh t*)
    width = 1.0 / np.sqrt(w * np.cosh(t_star) + 1e-30)
    # also include the t-> small region (t weight in numerator) and a long tail
    lo = max(0.0, t_star - 12.0 * width)
    hi = t_star + 12.0 * width
    # ensure we cover near 0 too (numerator has explicit factor t)
    lo = min(lo, 0.0)
    if hi <= lo:
        hi = lo + 1.0
    t = np.linspace(lo, hi, n)

    # stable log-magnitude of g(t)*cosh(p t) and g(t)*|sinh(p t)|
    # cosh(p t) = 0.5(e^{p t}+e^{-p t}); for the log use logaddexp.
    log_cosh = np.logaddexp(ap * t, -ap * t) - np.log(2.0)
    # sinh(|p| t) = 0.5(e^{|p|t}-e^{-|p|t}); positive for t>0
    # log sinh = log(0.5) + |p| t + log(1 - e^{-2|p| t})
    with np.errstate(divide="ignore"):
        log_sinh = (
            -np.log(2.0) + ap * t + np.log1p(-np.exp(-2.0 * ap * t))
        )
    log_g = -w * np.cosh(t)  # e^{-w cosh t}

    log_num_env = log_g + log_sinh  # times t (kept linearly) and sgn
    log_den_env = log_g + log_cosh

    # subtract a common max for stability, integrate with trapezoid
    m = np.nanmax(log_den_env)
    num = np.trapezoid(t * np.exp(log_num_env - m), t)
    den = np.trapezoid(np.exp(log_den_env - m), t)
    return sgn * num / den


# ----------------------------------------------------------------------------
# (M3) large-|p| asymptotic (uniform / DLMF 10.41): K_p(w) for large order.
#   log K_p(w) ~ log K_p as order->inf:
#     K_v(v z) ~ sqrt(pi/(2v)) e^{-v eta} / (1+z^2)^{1/4} * (1 + sum u_k/v^k)
#   with eta = sqrt(1+z^2) + log( z/(1+sqrt(1+z^2)) ), z = w/v.
# We differentiate the LEADING term in p (v=p) analytically as an asymptotic
# candidate for large |p|. Use |p| (even in p) then D_p odd => multiply by sgn.
# ----------------------------------------------------------------------------
def m3_asymptotic(p, w):
    v = abs(p)
    sgn = 1.0 if p >= 0 else -1.0
    z = w / v
    s = np.sqrt(1.0 + z * z)
    # log K_v(w) ~ 0.5*log(pi/(2v)) - v*eta - 0.25*log(1+z^2),  eta = s + log(z/(1+s))
    # We need d/dp log K_p = sgn * d/dv [ ... ].  Differentiate wrt v with z=w/v.
    # Let L(v) = 0.5 log(pi/2) - 0.5 log v - v*eta(v) - 0.25 log(1+z^2).
    # d/dv [-0.5 log v] = -1/(2v)
    # eta = s + log z - log(1+s),  z=w/v => log z = log w - log v
    # deta/dv: ds/dv = (z*dz/dv)/s, dz/dv = -w/v^2 = -z/v
    dz = -z / v
    ds = (z * dz) / s
    deta = ds + (dz / z) - (ds / (1.0 + s))
    d_v_eta = eta_term = None
    # d/dv [ -v*eta ] = -eta - v*deta
    eta = s + np.log(z) - np.log(1.0 + s)
    term_veta = -eta - v * deta
    # d/dv [ -0.25 log(1+z^2) ] = -0.25 * (2 z dz)/(1+z^2)
    term_quart = -0.25 * (2.0 * z * dz) / (1.0 + z * z)
    dLdv = -1.0 / (2.0 * v) + term_veta + term_quart
    return sgn * dLdv


# ----------------------------------------------------------------------------
# grids
# ----------------------------------------------------------------------------
P_GRID = [-300.0, -100.0, -30.0, -5.0, -0.5, 0.5, 1.0]
W_GRID = [1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]

H_FD = 1e-3  # finite-difference step in p


def relerr(approx, exact):
    denom = abs(exact)
    if denom < 1e-300:
        return abs(approx - exact)
    return abs(approx - exact) / denom


def main():
    print("=" * 100)
    print("ORACLE D_p = d_p log K_p(w)  (mpmath dps=50)")
    print("=" * 100)

    # ---- accuracy table for D_p -------------------------------------------
    methods = {
        "M1_FD(h=1e-3)": lambda p, w: m1_central_fd(p, w, 1e-3),
        "M1_FD(h=1e-2)": lambda p, w: m1_central_fd(p, w, 1e-2),
        "M1_FD(h=1e-4)": lambda p, w: m1_central_fd(p, w, 1e-4),
        "M4_Rich(h=1e-2)": lambda p, w: m4_richardson_fd(p, w, 1e-2),
        "M4_Rich(h=1e-3)": lambda p, w: m4_richardson_fd(p, w, 1e-3),
        "M2_quad": lambda p, w: m2_quadrature(p, w, 4000),
        "M3_asymp": m3_asymptotic,
    }

    # store max-relerr per method
    maxerr = {k: 0.0 for k in methods}
    maxerr_loc = {k: None for k in methods}
    bigp_err = {k: 0.0 for k in methods}  # |p|>=30 only

    print(f"\n{'p':>7} {'w':>8} {'oracle D_p':>14} | " +
          " ".join(f"{k:>16}" for k in methods))
    for p in P_GRID:
        for w in W_GRID:
            od = oracle_Dp(p, w)
            row = []
            for k, fn in methods.items():
                try:
                    val = fn(p, w)
                except Exception as e:
                    val = float("nan")
                e_rel = relerr(val, od)
                row.append(e_rel)
                if e_rel > maxerr[k]:
                    maxerr[k] = e_rel
                    maxerr_loc[k] = (p, w)
                if abs(p) >= 30 and e_rel > bigp_err[k]:
                    bigp_err[k] = e_rel
            print(f"{p:>7.1f} {w:>8.0e} {od:>14.6e} | " +
                  " ".join(f"{e:>16.2e}" for e in row))

    print("\n" + "-" * 100)
    print("MAX relative error per method (whole grid):")
    for k in methods:
        print(f"  {k:>18}: {maxerr[k]:.3e}   (worst at p,w = {maxerr_loc[k]})")
    print("MAX relative error per method (|p| >= 30 large-order regime):")
    for k in methods:
        print(f"  {k:>18}: {bigp_err[k]:.3e}")

    # ---- timing (vectorized batch, the real use-case) ----------------------
    print("\n" + "=" * 100)
    print("TIMING (batched over a realistic merged-HKL set)")
    print("=" * 100)
    rng = np.random.default_rng(0)
    N = 50000
    pb = 1.0 - rng.uniform(0, 5, N) * rng.integers(1, 100, N)  # p = 1 - nu*N_h
    wb = np.exp(rng.uniform(np.log(1e-3), np.log(1e3), N))

    def time_method(fn, reps=5):
        # warm
        fn(pb, wb)
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn(pb, wb)
        return (time.perf_counter() - t0) / reps, out

    # vectorized M1 (kve is vectorized)
    def m1_vec(p, w):
        return (np.log(kve(p + H_FD, w)) - np.log(kve(p - H_FD, w))) / (2 * H_FD)

    def m4_vec(p, w):
        d_h = m1_vec(p, w)
        d2 = (np.log(kve(p + 2 * H_FD, w)) - np.log(kve(p - 2 * H_FD, w))) / (
            4 * H_FD
        )
        return (4 * d_h - d2) / 3.0

    t_m1, _ = time_method(m1_vec)
    t_m4, _ = time_method(m4_vec)
    print(f"  M1_FD  (2 kve calls) : {t_m1*1e3:8.3f} ms / {N} elems  "
          f"({t_m1/N*1e9:.1f} ns/elem)")
    print(f"  M4_Rich(4 kve calls) : {t_m4*1e3:8.3f} ms / {N} elems  "
          f"({t_m4/N*1e9:.1f} ns/elem)")
    # M2 quad is per-element python-loop; time a small subset
    sub = 200
    t0 = time.perf_counter()
    for i in range(sub):
        m2_quadrature(pb[i], wb[i], 4000)
    t_m2 = (time.perf_counter() - t0) / sub
    print(f"  M2_quad (per-elem)   : {t_m2*1e6:8.1f} us / elem  "
          f"-> ~{t_m2*N*1e3:.0f} ms for {N} (NOT vectorized)")

    # ---- ratio order-derivatives via the recommended method ---------------
    print("\n" + "=" * 100)
    print("RATIO ORDER-DERIVATIVES  d_p log(K_{p+k}/K_p)  via M1_FD(h=1e-3)")
    print("  (needed for E[I] uses k=+1, E[1/I] uses k=-1)")
    print("=" * 100)
    for k in (+1, -1):
        print(f"\n  k = {k:+d}")
        print(f"  {'p':>7} {'w':>8} {'oracle':>14} {'M1_FD':>14} {'relerr':>10}")
        worst = 0.0
        for p in P_GRID:
            for w in W_GRID:
                orc = oracle_ratio_logderiv(p, w, k)
                # via FD on D_p at p+k and p:  but cleaner = direct FD of the ratio
                num = (np.log(kve(p + k + H_FD, w)) - np.log(kve(p + H_FD, w))) - (
                    np.log(kve(p + k - H_FD, w)) - np.log(kve(p - H_FD, w))
                )
                approx = num / (2 * H_FD)
                e = relerr(approx, orc)
                worst = max(worst, e)
                print(f"  {p:>7.1f} {w:>8.0e} {orc:>14.6e} {approx:>14.6e} {e:>10.2e}")
        print(f"  -> max relerr (k={k:+d}): {worst:.3e}")


if __name__ == "__main__":
    main()
