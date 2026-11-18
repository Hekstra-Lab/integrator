"""Which parameterization represents a diffraction profile best, at matched dimension.

This is a representation question, so it is answered without a training loop, a
merge or a refinement: every family is fitted to the same reflections and scored
on the same held-out ones.

THE TASK IS PREDICTION FROM POSITION, NOT RECONSTRUCTION. A bright reflection's
own profile can be read off its own counts, so reconstructing it measures
nothing useful. The job the model actually has is the weak reflection's: given
only where a reflection sits on the detector, its resolution and its wavelength,
produce its profile. So each held-out reflection is predicted from its
NEIGHBOURS in geometry, never from its own pixels.

THE FLOOR IS EXACT AND NEEDS NO GROUND TRUTH. The simulation stores a Miller
index per pixel, which gives exact pixel ownership, but it does not store a true
intensity profile per reflection -- `profile_vs_truth.py` says so, and the
expected width there is a population average, not a per-reflection truth. The
floor used here instead is Poisson thinning: draw `c1 ~ Binomial(c, 1/2)` and set
`c2 = c - c1`. The two halves are independent Poisson with half the rate, so the
distance between the profiles they imply is the irreducible noise of a profile
estimated from this many counts. A method at that distance has extracted
everything the data contains; a method above it has not. The floor is computed
per reflection and every score is reported as a multiple of it, which also makes
the comparison valid on real data where no simulation exists.

TWO METRICS, BECAUSE THEY MEASURE DIFFERENT FAILURES. The Hellinger distance
`sqrt(1 - sum sqrt(p q))` is the geometry the multinomial likelihood induces, so
it is what estimating a profile from counts should minimize. It is also blind to
how far wrong mass is: a one-pixel shift and a ten-pixel shift of the same blob
can score alike. The sliced Wasserstein distance charges for distance moved, so
it sees off-centring and elongation. A family can win on one and lose on the
other, and which it wins on says what it is good for.

THE FAMILIES. `aligned_mean` is in the list to settle whether the transport
methods earn their cost: it aligns neighbours to a common centroid with a
Fourier phase ramp, averages, then shifts to the target's predicted centroid.
That is exact for pure translation and nothing else. If it matches
`sinkhorn_bary`, the benefit of optimal transport here was translation handling
and the machinery is not needed.

Usage:
    profile_representation_benchmark.py --data DIR [--shots 20] [--dim 8]
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------- data


def load_shoeboxes(data_dir, max_refl, bright_frac, device):
    """Bright, fully valid shoeboxes with their geometry.

    Selection uses the summation estimate and the mask only, never a model
    output, so every family is scored on the same reflections.
    """
    data_dir = Path(data_dir)
    counts = np.load(data_dir / "counts.npy", mmap_mode="r")
    masks = np.load(data_dir / "masks.npy", mmap_mode="r")
    meta_raw = np.load(data_dir / "metadata.npy", allow_pickle=True)
    meta = meta_raw.item() if meta_raw.shape == () else meta_raw

    n_total = counts.shape[0]
    print(f"dataset: {n_total:,} shoeboxes of {counts.shape[1:]} ")

    if isinstance(meta, dict):
        cols = {k: np.asarray(v).reshape(-1) for k, v in meta.items()}
    else:
        raise SystemExit(f"metadata.npy is {type(meta)}, expected a dict")
    missing = [
        k for k in ("xyzcal.px.0", "xyzcal.px.1", "d") if k not in cols
    ]
    if missing:
        raise SystemExit(f"metadata lacks {missing}; has {sorted(cols)[:14]}")

    # stream the summation estimate so the whole array is never resident
    step = 65536
    total = np.empty(n_total, dtype=np.float64)
    valid = np.empty(n_total, dtype=bool)
    for lo in range(0, n_total, step):
        hi = min(lo + step, n_total)
        c = np.asarray(counts[lo:hi], dtype=np.float64)
        m = np.asarray(masks[lo:hi]).astype(bool)
        c = c.reshape(hi - lo, -1)
        m = m.reshape(hi - lo, -1)
        total[lo:hi] = (c * m).sum(-1)
        valid[lo:hi] = m.all(-1)

    cut = np.quantile(total[valid], 1.0 - bright_frac) if valid.any() else 0.0
    keep = np.nonzero(valid & (total >= cut))[0]
    if keep.size > max_refl:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(keep, max_refl, replace=False))
    print(
        f"kept {keep.size:,} fully valid shoeboxes in the top "
        f"{100 * bright_frac:.0f}% by summation (total >= {cut:.0f} counts)"
    )

    c = torch.from_numpy(np.asarray(counts[keep], dtype=np.float32)).to(device)
    c = c.reshape(keep.size, -1)
    stack = [
        cols["xyzcal.px.0"][keep],
        cols["xyzcal.px.1"][keep],
        1.0 / np.maximum(cols["d"][keep], 1e-6),
    ]
    # On Laue data the radial elongation is set by the wavelength, so two
    # reflections at the same detector position and resolution but different
    # wavelength are not neighbours in profile space.
    if "wavelength" in cols:
        stack.append(cols["wavelength"][keep])
        print("neighbour metric: detector x, y, 1/d, wavelength")
    else:
        print("neighbour metric: detector x, y, 1/d (no wavelength column)")
    geom = torch.from_numpy(np.stack(stack).astype(np.float32)).T.to(device)
    extra = {}
    for k in ("wavelength", "d"):
        if k in cols:
            extra[k] = torch.from_numpy(
                cols[k][keep].astype(np.float32)
            ).to(device)
    # Shoeboxes may be stored flat as `(N, P)` or as `(N, D, H, W)`. The grid
    # comes from `dataset.yaml`, which is what the shoeboxes were cut against,
    # with a square fallback when the manifest is absent.
    stored = tuple(int(v) for v in counts.shape[1:])
    shape = None
    spec_path = data_dir / "dataset.yaml"
    if spec_path.exists():
        import yaml

        geo = (yaml.safe_load(spec_path.read_text()) or {}).get("geometry", {})
        if all(k in geo for k in ("d", "h", "w")):
            shape = (int(geo["d"]), int(geo["h"]), int(geo["w"]))
    if shape is None and len(stored) == 3:
        shape = stored
    if shape is None:
        side = int(round(math.sqrt(int(np.prod(stored)))))
        if side * side == int(np.prod(stored)):
            shape = (1, side, side)
    if shape is None or int(np.prod(shape)) != int(np.prod(stored)):
        raise SystemExit(
            f"cannot determine the grid: stored {stored}, inferred {shape}; "
            "pass --shape D,H,W"
        )
    print(f"grid {shape} from {'dataset.yaml' if spec_path.exists() else 'inference'}")
    return c, geom, extra, shape


def border_background(c, shape):
    """Median of the shoebox border, per reflection.

    A border median rather than a fit, because every family gets the same
    background and the comparison is between profile families.
    """
    d, h, w = shape
    idx = torch.arange(h * w, device=c.device).reshape(h, w)
    edge = torch.zeros(h, w, dtype=torch.bool, device=c.device)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    sel = idx[edge].reshape(-1)
    per_frame = c.reshape(c.shape[0], d, h * w)
    return per_frame[:, :, sel].median(-1).values.mean(-1, keepdim=True)


def to_profile(c, bg, eps=1e-12):
    q = (c - bg).clamp_min(0.0)
    return q / q.sum(-1, keepdim=True).clamp_min(eps)


def thin(c, generator):
    """Poisson thinning: two independent halves of the same counts."""
    c1 = torch.binomial(c.round().clamp_min(0.0), torch.full_like(c, 0.5),
                        generator=generator)
    return c1, (c.round().clamp_min(0.0) - c1)


# ---------------------------------------------------------------- metrics


def hellinger(p, q, eps=1e-12):
    bc = (p.clamp_min(0).sqrt() * q.clamp_min(0).sqrt()).sum(-1)
    return (1.0 - bc.clamp(0.0, 1.0)).clamp_min(0.0).sqrt()


def make_projections(shape, n_proj, device, seed=0):
    """Unit directions and the pixel coordinate projected on each."""
    d, h, w = shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(d, dtype=torch.float32, device=device),
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    coords = torch.stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], -1)
    coords = coords - coords.mean(0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    dirs = torch.randn(n_proj, 3, generator=g).to(device)
    if d == 1:
        dirs[:, 0] = 0.0
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return coords @ dirs.T  # (P, n_proj)


def sliced_w2(p, q, proj, n_q=64):
    """Sliced 2-Wasserstein distance between profiles on a shared grid.

    For each projection the profiles become 1-D distributions on the same
    support, where `W2` is the L2 distance between inverse CDFs -- closed form,
    no transport solve.
    """
    order = proj.argsort(0)                       # (P, n_proj)
    qs = torch.linspace(0.5 / n_q, 1 - 0.5 / n_q, n_q, device=p.device)
    out = 0.0
    for j in range(proj.shape[1]):
        o = order[:, j]
        x = proj[o, j]                            # sorted support
        cp = p[:, o].cumsum(-1)
        cq = q[:, o].cumsum(-1)
        cp = cp / cp[:, -1:].clamp_min(1e-12)
        cq = cq / cq[:, -1:].clamp_min(1e-12)
        ip = torch.searchsorted(cp.contiguous(), qs.expand(p.shape[0], n_q))
        iq = torch.searchsorted(cq.contiguous(), qs.expand(q.shape[0], n_q))
        ip = ip.clamp(max=x.numel() - 1)
        iq = iq.clamp(max=x.numel() - 1)
        out = out + (x[ip] - x[iq]).pow(2).mean(-1)
    return (out / proj.shape[1]).clamp_min(0.0).sqrt()


# ---------------------------------------------------------------- families


def knn(train_geom, test_geom, k, scale):
    g_tr = train_geom / scale
    g_te = test_geom / scale
    dist = torch.cdist(g_te, g_tr)
    val, idx = dist.topk(k, largest=False)
    weight = 1.0 / val.clamp_min(1e-3)
    return idx, weight / weight.sum(-1, keepdim=True)


def centroid(p, coords):
    return p @ coords


def fourier_shift(p, shape, delta):
    """Shift each profile by a sub-pixel amount with a phase ramp.

    Translation is exactly a phase ramp in the Fourier domain, so this is exact
    for a band-limited profile and needs no interpolation kernel.
    """
    n, _ = p.shape
    d, h, w = shape
    vol = p.reshape(n, d, h, w)
    fy = torch.fft.fftfreq(h, device=p.device).reshape(1, h, 1)
    fx = torch.fft.fftfreq(w, device=p.device).reshape(1, 1, w)
    phase = -2.0 * math.pi * (
        delta[:, 1].reshape(-1, 1, 1) * fy + delta[:, 2].reshape(-1, 1, 1) * fx
    )
    ker = torch.polar(torch.ones_like(phase), phase).unsqueeze(1)
    out = torch.fft.ifft2(torch.fft.fft2(vol) * ker).real.reshape(n, -1)
    return out.clamp_min(0.0) / out.clamp_min(0.0).sum(-1, keepdim=True).clamp_min(1e-12)


def monomial_basis(shape, order, device):
    """Monomials up to `order`, constant excluded, columns unit norm.

    `p ~ exp(lambda . phi)` with these columns is the maximum entropy density
    given moments up to `order`: an exponential family, so the fit is concave
    and the coefficients are the natural parameters. Order 2 is exactly the
    Gaussians, order 3 adds skew, order 4 adds tails.
    """
    d, h, w = shape
    axes = [
        torch.linspace(-1, 1, n, device=device) if n > 1
        else torch.zeros(1, device=device)
        for n in (d, h, w)
    ]
    zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
    z, y, x = zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)
    cols, names = [], []
    for i in range(order + 1):
        for j in range(order + 1):
            for kk in range(order + 1):
                if 0 < i + j + kk <= order and (d > 1 or i == 0):
                    cols.append((z**i) * (y**j) * (x**kk))
                    names.append((i, j, kk))
    phi = torch.stack(cols, -1)
    return phi / phi.norm(dim=0, keepdim=True).clamp_min(1e-12), names


def fit_expfam(p, phi, n_iter=60, ridge=1e-6, chunk=2048):
    """Chunked wrapper: the Newton Hessian is `n x P x d x d` in one go."""
    out = []
    for lo in range(0, p.shape[0], chunk):
        out.append(_fit_expfam_chunk(p[lo : lo + chunk], phi, n_iter, ridge))
    return torch.cat(out, 0)


def _fit_expfam_chunk(p, phi, n_iter=60, ridge=1e-6):
    """Maximum likelihood `lambda` for `p ~ exp(phi lambda)`, by Newton.

    Concave in `lambda`: the Hessian is minus the covariance of `phi` under the
    current density, so Newton converges from zero without a line search here.
    """
    n, _ = p.shape
    lam = torch.zeros(n, phi.shape[1], device=p.device, dtype=torch.float64)
    phi64 = phi.double()
    target = p.double() @ phi64
    eye = torch.eye(phi.shape[1], device=p.device, dtype=torch.float64)
    for _ in range(n_iter):
        logits = lam @ phi64.T
        q = torch.softmax(logits, dim=-1)
        mean = q @ phi64
        grad = target - mean
        cov = torch.einsum("np,pi,pj->nij", q, phi64, phi64) - torch.einsum(
            "ni,nj->nij", mean, mean
        )
        # Damping relative to the Hessian's own scale, not absolute: the basis
        # columns are unit norm over P pixels, so the covariance entries sit
        # near 1/P and an absolute 1e-3 damps the step to a standstill. A
        # pseudo-inverse rather than a solve because a near-delta profile has a
        # singular moment covariance -- those directions carry no information
        # and taking zero step along them is the right answer.
        damp = ridge * torch.diagonal(cov, dim1=-2, dim2=-1).mean(
            -1
        ).clamp_min(1e-30).reshape(-1, 1, 1)
        step = torch.linalg.pinv(cov + damp * eye) @ grad.unsqueeze(-1)
        step = step.squeeze(-1)
        # cap the step so one badly conditioned reflection cannot diverge
        norm = step.norm(dim=-1, keepdim=True)
        step = step * (norm.clamp_max(50.0) / norm.clamp_min(1e-30))
        lam = lam + step
        if grad.abs().max() < 1e-9:
            break
    return lam.float()


def expfam_profile(lam, phi):
    return torch.softmax(lam @ phi.T, dim=-1)


def fit_moffat(p, shape, n_iter=1200, lr=0.05):
    """Elliptical Moffat, `(1 + r^2/alpha^2)^-beta`, by batched Adam.

    In the list because a Gaussian's tails are wrong by construction and the
    complaint against our profiles is that they do not cover enough pixels.
    Moffat's wings are power law, with `beta` setting how heavy.
    """
    n = p.shape[0]
    d, h, w = shape
    device = p.device
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    yy, xx = yy.reshape(-1), xx.reshape(-1)
    pm = p.reshape(n, d, h * w).sum(1)
    pm = pm / pm.sum(-1, keepdim=True).clamp_min(1e-12)

    cy = pm @ yy
    cx = pm @ xx
    par = torch.stack(
        [
            cy,
            cx,
            torch.full((n,), math.log(0.15), device=device),
            torch.full((n,), math.log(0.15), device=device),
            torch.zeros(n, device=device),
            torch.full((n,), math.log(2.5), device=device),
        ],
        -1,
    ).requires_grad_(True)
    opt = torch.optim.Adam([par], lr=lr)
    sqrt_p = pm.clamp_min(0).sqrt()
    for _ in range(n_iter):
        opt.zero_grad()
        q = moffat_profile(par, yy, xx)
        loss = -(sqrt_p * q.clamp_min(1e-12).sqrt()).sum()
        loss.backward()
        opt.step()
    return par.detach()


def moffat_profile(par, yy, xx):
    cy, cx = par[:, 0:1], par[:, 1:2]
    ay, ax = par[:, 2:3].exp(), par[:, 3:4].exp()
    theta = par[:, 4:5]
    beta = par[:, 5:6].exp().clamp(0.6, 30.0)
    dy, dx = yy.reshape(1, -1) - cy, xx.reshape(1, -1) - cx
    cos, sin = torch.cos(theta), torch.sin(theta)
    u = (cos * dx + sin * dy) / ay.clamp_min(1e-3)
    v = (-sin * dx + cos * dy) / ax.clamp_min(1e-3)
    q = (1.0 + u * u + v * v).pow(-beta)
    return q / q.sum(-1, keepdim=True).clamp_min(1e-12)


def sinkhorn_barycenter(p, weight, shape, blur=0.6, n_iter=120, chunk=512):
    """Chunked wrapper over target reflections."""
    out = []
    for lo in range(0, p.shape[0], chunk):
        out.append(
            _sinkhorn_chunk(
                p[lo : lo + chunk], weight[lo : lo + chunk], shape, blur, n_iter
            )
        )
    return torch.cat(out, 0)


def _sinkhorn_chunk(p, weight, shape, blur=0.6, n_iter=120, eps=1e-12):
    """Entropic Wasserstein barycentre on the grid, convolutionally.

    The Gibbs kernel of a squared-distance cost on a grid is a Gaussian blur,
    which separates over axes, so each Sinkhorn iteration is two 1-D
    convolutions instead of a `P x P` matrix product. Solomon et al.,
    `Convolutional Wasserstein Distances`.
    """
    n_t, _, n_pix = p.shape
    _, h, w = shape

    ky = gaussian_kernel(h, blur, p.device)
    kx = gaussian_kernel(w, blur, p.device)

    def blur_op(x):
        vol = x.reshape(-1, h, w)
        vol = torch.einsum("ab,nbw->naw", ky, vol)
        vol = torch.einsum("nah,hb->nab", vol, kx)
        return vol.reshape(x.shape)

    v = torch.ones_like(p)
    bary = torch.full((n_t, n_pix), 1.0 / n_pix, device=p.device)
    for _ in range(n_iter):
        u = p / blur_op(v).clamp_min(eps)
        conv_u = blur_op(u).clamp_min(eps)
        log_b = (weight.unsqueeze(-1) * conv_u.log()).sum(1)
        bary = log_b.exp()
        bary = bary / bary.sum(-1, keepdim=True).clamp_min(eps)
        v = bary.unsqueeze(1) / conv_u
    return bary


def gaussian_kernel(n, sigma, device):
    i = torch.arange(n, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (i.reshape(-1, 1) - i.reshape(1, -1)) ** 2 / sigma**2)
    return k / k.sum(0, keepdim=True)


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-refl", type=int, default=24000)
    ap.add_argument("--bright-frac", type=float, default=0.25)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--n-proj", type=int, default=24)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shape", default=None, help="D,H,W override")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(0)
    c, geom, _extra, shape = load_shoeboxes(
        args.data, args.max_refl, args.bright_frac, dev
    )
    if args.shape:
        shape = tuple(int(v) for v in args.shape.split(","))
    shape = tuple(int(v) for v in shape)
    n, n_pix = c.shape
    print(f"shape {shape}, {n_pix} pixels, {n:,} reflections, device {dev}")

    bg = border_background(c, shape)
    p_all = to_profile(c, bg)

    gen = torch.Generator(device=dev).manual_seed(1)
    c1, c2 = thin(c, gen)  # thinning applies to the raw counts
    p1 = to_profile(c1, bg * 0.5)
    p2 = to_profile(c2, bg * 0.5)

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(2))
    n_test = int(args.test_frac * n)
    test, train = perm[:n_test].to(dev), perm[n_test:].to(dev)
    print(f"{train.numel():,} train / {test.numel():,} held out")

    proj = make_projections(shape, args.n_proj, dev)
    coords_scale = geom.std(0).clamp_min(1e-6)
    idx, wt = knn(geom[train], geom[test], args.k, coords_scale)

    p_tr = p_all[train]
    p_te = p_all[test]
    nb = p_tr[idx]                       # (n_test, k, P)

    # `d(p1, p2)` is between two HALF-count estimates, each carrying about
    # twice the variance of the full-count `p_te`, so it runs at about 2 sigma
    # where sigma is the noise of one full-count profile. The floor a method is
    # measured against is therefore half of it: a method that predicted the
    # truth exactly would still sit sigma away from `p_te`. Reporting the
    # unhalved distance makes good methods look better than noise.
    floor_h = 0.5 * hellinger(p1[test], p2[test])
    floor_w = 0.5 * sliced_w2(p1[test], p2[test], proj)
    print(
        f"\nnoise floor (half the two-half thinning distance, = sigma of one "
        f"full-count profile):\n  Hellinger {floor_h.median():.4f}   "
        f"sliced W2 {floor_w.median():.4f} px"
    )
    print("a method that recovered the truth exactly would score 1.00x; "
          "1.00x is the best attainable, not a target to beat\n")

    preds = {}
    t0 = time.time()

    preds["global_mean"] = p_tr.mean(0, keepdim=True).expand(n_test, n_pix)
    preds["xds_arith"] = (wt.unsqueeze(-1) * nb).sum(1)

    cen_coords = torch.stack(
        torch.meshgrid(
            *[
                torch.arange(s, dtype=torch.float32, device=dev)
                for s in shape
            ],
            indexing="ij",
        ),
        -1,
    ).reshape(-1, 3)
    cen_coords = cen_coords - cen_coords.mean(0)
    cen_tr = centroid(p_tr, cen_coords)
    cen_pred = (wt.unsqueeze(-1) * cen_tr[idx]).sum(1)
    flat_nb = nb.reshape(-1, n_pix)
    delta = (cen_pred.unsqueeze(1) - cen_tr[idx]).reshape(-1, 3)
    aligned = fourier_shift(flat_nb, shape, delta).reshape(n_test, args.k, n_pix)
    preds["aligned_mean"] = (wt.unsqueeze(-1) * aligned).sum(1)

    mean_p = p_tr.mean(0)
    _, _, v = torch.pca_lowrank(p_tr - mean_p, q=args.dim)
    coef_tr = (p_tr - mean_p) @ v
    coef_te = (wt.unsqueeze(-1) * coef_tr[idx]).sum(1)
    rec = mean_p + coef_te @ v.T
    preds[f"l2_pca_d{args.dim}"] = rec.clamp_min(0) / rec.clamp_min(0).sum(
        -1, keepdim=True
    ).clamp_min(1e-12)

    psi_tr = p_tr.clamp_min(0).sqrt()
    psi_bar = psi_tr.mean(0)
    psi_bar = psi_bar / psi_bar.norm().clamp_min(1e-12)
    _, _, v2 = torch.pca_lowrank(psi_tr - psi_bar, q=args.dim)
    cf_tr = (psi_tr - psi_bar) @ v2
    cf_te = (wt.unsqueeze(-1) * cf_tr[idx]).sum(1)
    psi = psi_bar + cf_te @ v2.T
    psi = psi / psi.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    preds[f"fisher_pca_d{args.dim}"] = psi.pow(2)

    for order in (2, 3, 4):
        phi, names = monomial_basis(shape, order, dev)
        lam_tr = fit_expfam(p_tr, phi)
        lam_te = (wt.unsqueeze(-1) * lam_tr[idx]).sum(1)
        preds[f"expfam_o{order}_d{phi.shape[1]}"] = expfam_profile(lam_te, phi)

    par_tr = fit_moffat(p_tr, shape)
    # theta is an angle: a linear mean of 0.1 and 3.1 radians is 1.6, pointing
    # across the ellipse rather than along it. Average the doubled angle as a
    # vector, which is also the right period for an axis direction.
    nb_par = par_tr[idx]
    ang = nb_par[..., 4]
    cos2 = (wt * torch.cos(2 * ang)).sum(1)
    sin2 = (wt * torch.sin(2 * ang)).sum(1)
    par_te = (wt.unsqueeze(-1) * nb_par).sum(1)
    par_te[:, 4] = 0.5 * torch.atan2(sin2, cos2)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, shape[1], device=dev),
        torch.linspace(-1, 1, shape[2], device=dev),
        indexing="ij",
    )
    mof = moffat_profile(par_te, yy.reshape(-1), xx.reshape(-1))
    if shape[0] > 1:
        mof = (mof.unsqueeze(1) / shape[0]).expand(-1, shape[0], -1).reshape(
            n_test, -1
        )
    preds["moffat_d6"] = mof

    try:
        preds["sinkhorn_bary"] = sinkhorn_barycenter(nb, wt, shape)
    except Exception as exc:  # noqa: BLE001
        print(f"sinkhorn_bary skipped: {type(exc).__name__}: {exc}")

    print(f"all families fitted in {time.time() - t0:.1f}s\n")

    # How well each family can represent a profile at all, fitted to the
    # held-out reflection's OWN counts. Separates "the family is too small"
    # from "the way its parameters average across neighbours is wrong" -- the
    # two have the same symptom in the prediction column.
    own = {}
    mean_te = p_te - mean_p
    rec = mean_p + (mean_te @ v) @ v.T
    own[f"l2_pca_d{args.dim}"] = rec.clamp_min(0) / rec.clamp_min(0).sum(
        -1, keepdim=True
    ).clamp_min(1e-12)
    psi_te = p_te.clamp_min(0).sqrt()
    ps = psi_bar + ((psi_te - psi_bar) @ v2) @ v2.T
    own[f"fisher_pca_d{args.dim}"] = (
        ps / ps.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ).pow(2)
    for order in (2, 3, 4):
        phi, _ = monomial_basis(shape, order, dev)
        own[f"expfam_o{order}_d{phi.shape[1]}"] = expfam_profile(
            fit_expfam(p_te, phi), phi
        )
    own["moffat_d6"] = moffat_profile(
        fit_moffat(p_te, shape), yy.reshape(-1), xx.reshape(-1)
    )

    # Mass outside the central 5x5, and centroid error. The first is the
    # complaint against our profiles stated as a number; the second separates a
    # misplaced profile from a mis-shaped one, which Hellinger alone cannot do.
    d_, h_, w_ = shape
    cy, cx = (h_ - 1) / 2.0, (w_ - 1) / 2.0
    gy, gx = torch.meshgrid(
        torch.arange(h_, device=dev, dtype=torch.float32),
        torch.arange(w_, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    core = ((gy - cy).abs() <= 2) & ((gx - cx).abs() <= 2)
    core = core.reshape(1, -1).expand(d_, -1).reshape(1, -1)

    def tail_mass(q):
        return (q * (~core)).sum(-1)

    tail_obs = float(tail_mass(p_te).median())

    # What the profile is actually FOR. On a 96.6%-background shoebox the
    # per-reflection empirical profile is not estimable -- the thinning floor
    # there is 0.44 in Hellinger against a maximum of 1.0 -- so a
    # density-vs-density score is measuring background fluctuation. `I` is an
    # integral over the box, so it averages that noise down and stays well
    # posed. The summation estimate uses flat weights and no profile, hence is
    # unbiased, which makes it the reference the bias is measured against.
    c_te = c[test]
    m_te = torch.ones_like(c_te, dtype=torch.bool)
    bg_cols = torch.ones(1, n_pix, device=dev, dtype=torch.float64)
    i_sum = ((c_te - bg[test]) * m_te).sum(-1)

    rows = []
    print(f"mass outside the central 5x5, held-out observed: {tail_obs:.3f}\n")
    print(
        f"{'family':<22}{'Hell':>7}{'xfl':>5}{'slicW2':>8}{'xfl':>5}"
        f"{'own':>7}{'tail':>6}{'dcen':>6}{'bias':>7}{'varrat':>8}"
    )
    print("-" * 81)
    for name, q in preds.items():
        q = q.clamp_min(0)
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
        h = hellinger(p_te, q)
        wd = sliced_w2(p_te, q, proj)
        row = dict(
            family=name,
            hellinger=float(h.median()),
            hellinger_ratio=float((h / floor_h.clamp_min(1e-9)).median()),
            sliced_w2=float(wd.median()),
            sliced_w2_ratio=float((wd / floor_w.clamp_min(1e-9)).median()),
        )
        row["tail"] = float(tail_mass(q).median())
        row["centroid_err"] = float(
            (centroid(q, cen_coords) - centroid(p_te, cen_coords))
            .norm(dim=-1)
            .median()
        )
        try:
            est = score_intensity(q, c_te, m_te, bg_cols, i_sum)
            row["bias"] = est["bias"]
            row["var_ratio"] = est["var_ratio"]
        except Exception as exc:  # noqa: BLE001
            # Loud, not silent: a swallowed exception here printed a table of
            # dashes that looked like "not applicable" rather than "broken".
            import traceback

            print(f"  !! {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            row["bias_error"] = f"{type(exc).__name__}: {exc}"
        if name in own:
            o = own[name].clamp_min(0)
            o = o / o.sum(-1, keepdim=True).clamp_min(1e-12)
            oh = hellinger(p_te, o)
            row["own_fit"] = float(oh.median())
            row["own_fit_ratio"] = float((oh / floor_h.clamp_min(1e-9)).median())
        rows.append(row)
        of = (
            f"{row['own_fit']:>8.4f}{row['own_fit_ratio']:>6.2f}"
            if "own_fit" in row
            else f"{'-':>8}{'-':>6}"
        )
        est_txt = (
            f"{row['bias']:>7.3f}{row['var_ratio']:>8.4f}"
            if "bias" in row
            else f"{'-':>7}{'-':>8}"
        )
        print(
            f"{name:<22}{row['hellinger']:>7.4f}{row['hellinger_ratio']:>5.2f}"
            f"{row['sliced_w2']:>8.4f}{row['sliced_w2_ratio']:>5.2f}"
            f"{row.get('own_fit', float('nan')):>7.4f}"
            f"{row['tail']:>6.3f}{row['centroid_err']:>6.3f}{est_txt}"
        )

    print(
        "\nbias: median I_profile / I_summation; 1.000 is unbiased, below 1 is "
        "a too-narrow profile reading low."
    )
    print(
        "varrat: Var(I_profile) / Var(I_summation); smaller is the precision "
        "the profile buys. These two are the frontier."
    )
    rows.sort(key=lambda r: r["hellinger"])
    print(f"\nbest on Hellinger: {rows[0]['family']}")
    rows.sort(key=lambda r: r["sliced_w2"])
    print(f"best on sliced W2:  {rows[0]['family']}")
    usable = [r for r in rows if "bias" in r]
    if usable:
        usable.sort(key=lambda r: abs(r["bias"] - 1.0))
        print(f"least biased:       {usable[0]['family']} ({usable[0]['bias']:.3f})")
        usable.sort(key=lambda r: r["var_ratio"])
        print(
            f"most precise:       {usable[0]['family']} "
            f"(Var ratio {usable[0]['var_ratio']:.4f}, bias {usable[0]['bias']:.3f})"
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                dict(
                    data=args.data,
                    shape=list(shape),
                    n=int(n),
                    n_test=int(n_test),
                    k=args.k,
                    dim=args.dim,
                    floor_hellinger=float(floor_h.median()),
                    floor_sliced_w2=float(floor_w.median()),
                    rows=rows,
                ),
                indent=2,
            )
        )
        print(f"wrote {args.out}")



# ---------------------------------------------------------------- the estimator


def score_intensity(p_model, counts, mask, bg_cols, i_sum, n_iter=5, eps=1e-6):
    """Bias and variance the profile induces in `I`, not a distance to a density.

    On a 96.6%-background shoebox the per-reflection empirical profile is not
    estimable -- two Poisson halves of one bright reflection sit 0.44 apart in
    Hellinger, against a maximum of 1.0 -- so every density-vs-density score
    there is measuring background fluctuation. `I` is an integral over the box
    and averages that noise down, so it stays well posed where the profile does
    not, and it is the only thing the profile is for.

    Two quantities, both without ground truth:

    bias: the summation estimate uses flat weights and no profile, so it is
        unbiased by construction. `median(I_prf / I_sum)` over many reflections
        is then the multiplicative bias the profile family induces. The exact
        statement for the unweighted fit is
        `E[I_hat]/I = <p_m, p_t> / |p_m|^2`, so a too-narrow `p_m` reads low.

    precision: `Var(I) = 1 / (sum_j w_j Var_w(p_m))`, verified against the
        solver to 1.1e-6. Reported as the ratio to the summation estimate's
        variance, which is the precision the profile buys.
    """
    from integrator.fit.direct import solve_profile_fit

    res = solve_profile_fit(
        counts.double(), mask, p_model.double(), bg_cols, n_iter=n_iter, eps=eps
    )
    i_hat, var = res.intensity, res.variance
    w = 1.0 / (
        i_hat.clamp_min(0).unsqueeze(-1) * p_model.double()
        + res.background.unsqueeze(-1)
    ).clamp_min(1e-12)
    m = mask.to(w.dtype)
    w = w * m
    total = w.sum(-1)
    mean_p = (w * p_model.double()).sum(-1) / total.clamp_min(1e-30)
    var_p = (w * (p_model.double() - mean_p.unsqueeze(-1)) ** 2 * m).sum(
        -1
    ) / total.clamp_min(1e-30)
    # the summation estimate's variance is just its total count
    var_sum = counts.double().mul(m).sum(-1).clamp_min(1.0)
    good = torch.isfinite(i_hat) & torch.isfinite(var) & (var > 0) & (i_sum > 0)
    return dict(
        bias=float((i_hat[good] / i_sum.double()[good]).median()),
        var_ratio=float((var[good] / var_sum[good]).median()),
        var_closed_form=float(
            (1.0 / (total[good] * var_p[good]).clamp_min(1e-30)).median()
        ),
        n=int(good.sum()),
    )

if __name__ == "__main__":
    main()
