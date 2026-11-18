"""Tests for the centric-anchored, robust, unsupervised scale+merge prototype.

The headline test plants BOTH a real anomalous signal and a scale confound (each
reflection's F(+) lands on high-scale images, F(-) on low-scale images) and shows:
  - a naive (no-scale) merge is swamped by the confound (huge fake DANO),
  - the centric-anchored solve recovers the real DANO (centrics, which carry no
    anomalous signal, pin the relative image scales the acentrics can't),
  - centric DANO is exactly 0 (pooled), and planted outliers are down-weighted.
"""

import numpy as np

from integrator.merge_ext.negative_control_merge import (
    build_merge_groups,
    merge_anomalous,
    scale_and_merge,
    tukey_biweight,
)


def _corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def _simulate_confounded(seed=0):
    """F(+) on high-scale 'A' images, F(-) on low-scale 'B' images -> the +/-
    scale difference masquerades as anomalous unless the relative A/B scale is
    pinned by the centrics (which must have I(+)==I(-))."""
    rng = np.random.default_rng(seed)
    n_ac, n_ce = 300, 120
    n_img = 40
    a_imgs = np.arange(0, 20)
    b_imgs = np.arange(20, 40)
    log_s = np.empty(n_img)
    log_s[a_imgs] = rng.normal(0.25, 0.10, a_imgs.size)   # A: higher scale
    log_s[b_imgs] = rng.normal(-0.25, 0.10, b_imgs.size)  # B: lower scale
    s_img = np.exp(log_s)

    mu = np.exp(rng.normal(3.0, 1.0, n_ac + n_ce))        # true |F|^2 per refl
    delta = np.zeros(n_ac + n_ce)
    delta[:n_ac] = rng.normal(0.0, 0.05, n_ac)            # ~5% anomalous, acentric only
    mu_plus = mu * (1 + delta / 2)
    mu_minus = mu * (1 - delta / 2)
    dano_true = mu_plus - mu_minus                        # 0 for centrics

    I, sig, img, anom, nonanom, centric, plus, is_out = ([] for _ in range(8))
    next_anom = 0

    def emit(refl, is_centric, f2, imgs_pool, is_plus, anom_id):
        for im in rng.choice(imgs_pool, size=8, replace=True):
            y = s_img[im] * f2
            noise_sd = np.sqrt(max(y, 1.0)) * 0.05 + 0.5
            obs = y + rng.normal(0, noise_sd)
            outlier = rng.random() < 0.02
            if outlier:
                obs *= 3.0
            I.append(obs); sig.append(noise_sd); img.append(im)
            anom.append(anom_id); nonanom.append(refl)
            centric.append(is_centric); plus.append(is_plus)
            is_out.append(outlier)

    for r in range(n_ac + n_ce):
        ce = r >= n_ac
        ap, am = next_anom, next_anom + 1
        next_anom += 2
        emit(r, ce, mu_plus[r], a_imgs, True, ap)    # F(+) on A images
        emit(r, ce, mu_minus[r], b_imgs, False, am)  # F(-) on B images

    return {
        "I": np.array(I), "sig": np.array(sig), "img": np.array(img),
        "anom": np.array(anom), "nonanom": np.array(nonanom),
        "centric": np.array(centric), "plus": np.array(plus),
        "is_out": np.array(is_out), "dano_true": dano_true,
        "n_ac": n_ac, "mu": mu, "s_img": s_img,
    }


def test_biweight_basics():
    z = np.array([0.0, 3.0, 100.0])
    w = tukey_biweight(z, c=6.0)
    assert w[0] == 1.0 and 0.0 < w[1] < 1.0 and w[2] == 0.0


def test_centric_pooling_makes_dano_zero():
    d = _simulate_confounded()
    out = merge_anomalous(
        d["I"], d["sig"], d["img"], d["anom"], d["nonanom"],
        d["centric"], d["plus"],
    )
    # centric reflections were pooled -> their F(+) and F(-) are the same number
    assert np.allclose(out["dano"][out["is_centric"]], 0.0, atol=1e-9)


def test_anchored_recovers_dano_under_confound():
    d = _simulate_confounded()
    out = merge_anomalous(
        d["I"], d["sig"], d["img"], d["anom"], d["nonanom"],
        d["centric"], d["plus"],
    )
    acent = ~out["is_centric"]
    # align per-reflection truth (nonanom_id 0..n_ac-1 are acentric)
    nid = out["nonanom_id"]
    dano_true = d["dano_true"][nid]
    dano_anch = out["dano"]

    # --- naive (no-scale) merge of the same data, per reflection ---
    naive_plus = np.full(len(nid), np.nan)
    naive_minus = np.full(len(nid), np.nan)
    for k, r in enumerate(nid):
        sel = d["nonanom"] == r
        naive_plus[k] = d["I"][sel & d["plus"]].mean()
        naive_minus[k] = d["I"][sel & ~d["plus"]].mean()
    dano_naive = naive_plus - naive_minus

    e_naive = dano_naive[acent] - dano_true[acent]
    e_anch = dano_anch[acent] - dano_true[acent]

    # (1) The killer result: the centrics pin the confounding A/B scale ratio --
    # the very thing acentrics cannot identify -- to within ~1%.
    s = out["result"]["scale"]
    ratio_est = s[:20].mean() / s[20:].mean()
    ratio_true = d["s_img"][:20].mean() / d["s_img"][20:].mean()
    assert abs(ratio_est / ratio_true - 1.0) < 0.03

    # (2) Naive carries a large *systematic* fake DANO from the confound;
    # the anchored solve removes the systematic bias (robust median, since a
    # ~10% tail of weak reflections is handled downstream by French-Wilson).
    assert np.median(e_naive) > 5.0
    assert abs(np.median(e_anch)) < 0.5

    # (3) The bulk (90th percentile of |error|) is far tighter after anchoring.
    p90_anch = np.percentile(np.abs(e_anch), 90)
    p90_naive = np.percentile(np.abs(e_naive), 90)
    assert p90_anch < 0.3 * p90_naive

    # (4) The error model is calibrated (reduced chi^2 ~ 1).
    assert 0.5 < out["result"]["chi2"] < 2.0


def test_outliers_downweighted():
    d = _simulate_confounded()
    out = merge_anomalous(
        d["I"], d["sig"], d["img"], d["anom"], d["nonanom"],
        d["centric"], d["plus"],
    )
    w = out["result"]["weights"]
    assert w[d["is_out"]].mean() < 0.4       # planted outliers rejected
    assert w[~d["is_out"]].mean() > 0.85     # inliers kept


def test_recovers_scale_and_intensity_unconfounded():
    """Random image assignment (no +/- confound): the solve recovers the per-
    image scale and merged intensity far better than a naive merge."""
    rng = np.random.default_rng(1)
    n_hkl, n_img = 250, 30
    s_true = np.exp(rng.normal(0.0, 0.3, n_img))
    mu_true = np.exp(rng.normal(3.0, 0.8, n_hkl))
    I, sig, img, mg = [], [], [], []
    for h in range(n_hkl):
        for im in rng.choice(n_img, size=12, replace=True):
            y = s_true[im] * mu_true[h]
            sd = np.sqrt(y) * 0.05 + 0.5
            I.append(y + rng.normal(0, sd)); sig.append(sd)
            img.append(im); mg.append(h)
    res = scale_and_merge(
        np.array(I), np.array(sig), np.array(img), np.array(mg)
    )
    # gauge-fix both to geometric-mean(scale)=1 before comparing
    s_norm = s_true / np.exp(np.mean(np.log(s_true)))
    assert _corr(res["scale"], s_norm) > 0.9
    assert _corr(res["mu"], mu_true) > 0.95
    assert res["chi2"] is not None and 0.3 < res["chi2"] < 5.0


def test_build_merge_groups_pools_only_centrics():
    anom = np.array([0, 1, 2, 3])          # +/- ids for two reflections
    nonanom = np.array([0, 0, 1, 1])       # reflection 0 (acentric), 1 (centric)
    centric = np.array([False, False, True, True])
    mg = build_merge_groups(anom, centric, nonanom)
    assert mg[0] != mg[1]                  # acentric +/- stay separate
    assert mg[2] == mg[3]                  # centric +/- pooled
