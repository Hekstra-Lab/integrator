"""See a trained streak profile model at work: the profile it predicts for any detector position and wavelength.

Loads a checkpoint of a `streak_kmix_profile` (T0, P2) or `streak_kmix_v2_profile`
(B1) run with its config_log.yaml, and evaluates the profile prior

    theta = theta0 + f(geometry features of (x, y, lambda)),   p = sum_k w_k N(u; a_k) N(v; c_k)

with the class's own methods (`eta_prior`, `profile_from_eta`, `unpack`), so what is
drawn is exactly what the model uses. A v1 checkpoint is loaded into the v2 class,
whose forward with tilt and shape posterior off is the parent's exactly.

Interactive (default): a window with sliders for detector x, y and wavelength; the
profile, each mixture component with its weight, and the streak direction phi
(analytic, from the spot position about the beam centre) update as they move. For a
model with a per-reflection shape posterior (B1), extra sliders move the shape
vector eta away from the prior mean in units of the learned prior sd sigma: the
range of shapes a reflection's own pixels can pull its profile to.

--static writes PDFs instead (to --out):
  <name>_detector.pdf    the profile at a 5 x 5 grid of detector positions, mid wavelength
  <name>_wavelength.pdf  one position, the profile at 5 wavelengths across the band
  <name>_widths.pdf      along- and across-streak sd of every component against detector radius,
                         at 4 azimuths and 3 wavelengths: the geometry law the MLP learned
  <name>_eta.pdf         (shape-posterior models) each eta component moved -2 .. +2 sigma, drawn as the
                         difference from the prior-mean profile (the shifts are small, a few percent)

    uv run python scripts/figures/explore_profile_model.py --ckpt CKPT --config config_log.yaml [--static --out DIR]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

SIDE = 25


def load_qp(ckpt: Path, config: Path):
    from integrator.registry import REGISTRY
    from integrator.utils.factory_utils import construct_integrator

    cfg = yaml.safe_load(config.read_text())
    cfg["integrator"]["args"].pop("init_from_checkpoint", None)
    la = cfg["loss"]["args"]
    for k in ("spectrum_init_from", "eb_init_from"):
        if la.get(k) and not Path(la[k]).exists():
            la[k] = None
    model = construct_integrator(cfg)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=False)
    qp = model.surrogates["qp"]
    if not hasattr(qp, "eta_prior"):  # v1: rebuild as v2 with the same arguments
        args = dict(cfg["surrogates"]["qp"]["args"])
        v2 = REGISTRY["surrogates"]["streak_kmix_v2_profile"](**args)
        missing, unexpected = v2.load_state_dict(qp.state_dict(), strict=False)
        assert not unexpected, unexpected
        qp = v2
    qp = qp.double().eval()
    name = cfg["surrogates"]["qp"]["name"]
    post = getattr(qp, "post", "none")
    return qp, name, post


def metadata(xs, ys, lam):
    xs, ys = np.atleast_1d(xs).astype(float), np.atleast_1d(ys).astype(float)
    lam = np.broadcast_to(np.atleast_1d(lam).astype(float), xs.shape)
    def t(a):
        return torch.as_tensor(np.array(a), dtype=torch.float64)
    return {"xyzcal.px.0": t(xs), "xyzcal.px.1": t(ys), "bbox.0": t(np.floor(xs) - SIDE // 2),
            "bbox.2": t(np.floor(ys) - SIDE // 2), "wavelength": t(lam), "d": t(np.full(xs.shape, 2.0)),
            "group_label": torch.zeros(xs.shape, dtype=torch.long)}


@torch.no_grad()
def evaluate(qp, xs, ys, lam, eta_shift=None):
    """Profiles (B, 25, 25), components (B, K, 25, 25) weighted, phi (B,), widths along/across (B, K), weights (B, K)."""
    md = metadata(xs, ys, lam)
    B = md["xyzcal.px.0"].shape[0]
    mask = torch.ones(B, SIDE * SIDE, dtype=torch.float64)
    phi, fxy, eta = qp.eta_prior(md, mask)
    if eta_shift is not None:
        eta = eta + torch.as_tensor(eta_shift, dtype=torch.float64) * qp.eta_sd.double()
    p = qp.profile_from_eta(phi, fxy, eta, mask)
    a, c, w, _ = qp.unpack(eta[..., :-1])
    u, v = qp.frame(phi, eta[..., -1], eta[..., :-1], fxy)
    e = torch.exp(-0.5 * (u[..., None] ** 2 / (a[..., None, :] ** 2 + 1 / 12) + v[..., None] ** 2 / (c[..., None, :] ** 2 + 1 / 12)))
    e = e / e.sum(-2, keepdim=True)
    comps = (e * w[..., None, :]).permute(0, 2, 1).reshape(B, -1, SIDE, SIDE)
    return p.reshape(B, SIDE, SIDE).numpy(), comps.numpy(), phi.numpy(), a.numpy(), c.numpy(), w.numpy()


def detector_extent(qp):
    bx, by = (float(v) for v in qp.beam_center)
    return (0.0, 2 * bx), (0.0, 2 * by)


def static(qp, name, post, out: Path, tag: str):
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    (x0, x1), (y0, y1) = detector_extent(qp)
    lmin, lmax = float(qp.lambda_min), float(qp.lambda_max)
    lmid = 0.5 * (lmin + lmax)
    bx, by = (float(v) for v in qp.beam_center)

    gx = np.linspace(x0 + 0.08 * (x1 - x0), x1 - 0.08 * (x1 - x0), 5)
    gy = np.linspace(y0 + 0.08 * (y1 - y0), y1 - 0.08 * (y1 - y0), 5)
    X, Y = np.meshgrid(gx, gy)
    p, _, phi, a, c, w = evaluate(qp, X.ravel(), Y.ravel(), lmid)
    fig, ax = plt.subplots(5, 5, figsize=(11, 11), layout="constrained")
    for k, axk in enumerate(ax.ravel()):
        axk.imshow(p[k], cmap="viridis", interpolation="nearest")
        r = math.hypot(X.ravel()[k] - bx, Y.ravel()[k] - by)
        n_eff = 1 / (p[k] ** 2).sum()
        axk.set_title(f"x {X.ravel()[k]:.0f}, y {Y.ravel()[k]:.0f}\nr {r:.0f} px, n_eff {n_eff:.0f}", fontsize=8)
        axk.set_xticks([])
        axk.set_yticks([])
    fig.suptitle(f"{tag}: predicted profile across the detector (wavelength {lmid:.3f} Å); rows top to bottom = detector y", fontsize=11)
    fig.savefig(out / f"{tag}_detector.pdf")
    plt.close(fig)

    xs = bx + 0.6 * (x1 - bx)
    ys = by + 0.3 * (y1 - by)
    lams = np.linspace(lmin, lmax, 5)
    p, comps, _, a, c, w = evaluate(qp, np.full(5, xs), np.full(5, ys), lams)
    K = comps.shape[1]
    fig, ax = plt.subplots(K + 1, 5, figsize=(12, 2.5 * (K + 1)), layout="constrained")
    for j in range(5):
        ax[0, j].imshow(p[j], cmap="viridis", interpolation="nearest")
        ax[0, j].set_title(f"λ {lams[j]:.3f} Å\nn_eff {1 / (p[j] ** 2).sum():.0f}", fontsize=9)
        for k in range(K):
            ax[k + 1, j].imshow(comps[j, k], cmap="magma", interpolation="nearest", vmin=0, vmax=comps[j].max())
            ax[k + 1, j].set_title(f"component {k + 1}: w {w[j, k]:.2f}\nsd along {a[j, k]:.2f}, across {c[j, k]:.2f} px", fontsize=8)
    for axk in ax.ravel():
        axk.set_xticks([])
        axk.set_yticks([])
    ax[0, 0].set_ylabel("profile", fontsize=10)
    fig.suptitle(f"{tag}: one detector position (x {xs:.0f}, y {ys:.0f}) across the wavelength band, with its mixture components", fontsize=11)
    fig.savefig(out / f"{tag}_wavelength.pdf")
    plt.close(fig)

    rmax = 0.95 * min(bx, by, x1 - bx, y1 - by)
    r = np.linspace(20, rmax, 60)
    fig, ax = plt.subplots(2, K, figsize=(4.2 * K, 7), layout="constrained", sharex=True, squeeze=False)
    styles = {lmin: ":", lmid: "-", lmax: "--"}
    for azi, col in zip((0, 90, 180, 270), ("C0", "C1", "C2", "C3")):
        for lam, ls in styles.items():
            xx, yy = bx + r * math.cos(math.radians(azi)), by + r * math.sin(math.radians(azi))
            _, _, _, a, c, w = evaluate(qp, xx, yy, lam)
            for k in range(K):
                ax[0, k].plot(r, a[:, k], ls=ls, color=col, label=f"azimuth {azi}°, λ {lam:.2f}" if k == 0 else None)
                ax[1, k].plot(r, c[:, k], ls=ls, color=col)
    for k in range(K):
        ax[0, k].set_title(f"component {k + 1}: sd along the streak (px)")
        ax[1, k].set_title(f"component {k + 1}: sd across the streak (px)")
        ax[1, k].set_xlabel("detector radius from beam centre (px)")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"{tag}: the learned width law (colour = azimuth; dotted / solid / dashed = short / mid / long wavelength)", fontsize=11)
    fig.savefig(out / f"{tag}_widths.pdf")
    plt.close(fig)

    if post != "none":
        NE = int(qp.eta_sd.numel())
        shifts = np.linspace(-2, 2, 5)
        fig, ax = plt.subplots(NE, 5, figsize=(10, 2.0 * NE), layout="constrained", squeeze=False)
        names = [f"log sd along {k + 1}" for k in range(K)] + [f"log sd across {k + 1}" for k in range(K)] + \
                [f"weight logit {k + 1}" for k in range(K)] + ["shift along", "shift across", "tilt"]
        p0 = evaluate(qp, xs, ys, lmid)[0][0]
        sd = qp.eta_sd.detach().numpy()
        for i in range(NE):
            sh = np.zeros((5, NE))
            sh[:, i] = shifts
            pp = np.stack([evaluate(qp, xs, ys, lmid, sh[j:j + 1])[0][0] for j in range(5)])
            # differences from the prior-mean profile: the shifts are small (the learned sds
            # are 0.01-0.5), so the profiles themselves look identical side by side
            lim = max(np.abs(pp - p0).max(), 1e-12)
            for j in range(5):
                ax[i, j].imshow(pp[j] - p0, cmap="RdBu_r", interpolation="nearest", vmin=-lim, vmax=lim)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                if i == 0:
                    ax[i, j].set_title(f"{shifts[j]:+.0f} σ", fontsize=9)
            ax[i, 4].text(1.04, 0.5, f"max |Δp| / max p\n{lim / p0.max():.3f}", transform=ax[i, 4].transAxes, fontsize=7, va="center")
            ax[i, 0].set_ylabel(f"{names[i] if i < len(names) else f'eta {i}'}\nσ = {sd[i]:.3f}", fontsize=7)
        fig.suptitle(f"{tag}: per-reflection freedom: profile minus the prior-mean profile, each shape component moved\n"
                     f"in units of its learned prior sd σ (red: more mass, blue: less; scale per row)", fontsize=10)
        fig.savefig(out / f"{tag}_eta.pdf")
        plt.close(fig)
    print(f"wrote {out}/{tag}_*.pdf")


def interactive(qp, name, post, tag):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    (x0, x1), (y0, y1) = detector_extent(qp)
    lmin, lmax = float(qp.lambda_min), float(qp.lambda_max)
    bx, by = (float(v) for v in qp.beam_center)
    NE = int(qp.eta_sd.numel()) if post != "none" else 0
    K = int(qp.K)
    n_sl = 3 + NE
    fig = plt.figure(figsize=(4 + 2.6 * (K + 1), 6 + 0.28 * n_sl))
    top = 0.12 + 0.035 * n_sl
    ax_p = fig.add_axes([0.04, top, 0.30, 0.80 - top + 0.1])
    ax_c = [fig.add_axes([0.38 + 0.6 * k / K, top + 0.25, 0.6 / K - 0.02, 0.45]) for k in range(K)]
    ax_det = fig.add_axes([0.38, top, 0.18, 0.22])
    s_x = Slider(fig.add_axes([0.12, 0.02 + 0.035 * (n_sl - 1), 0.75, 0.025]), "detector x", x0, x1, valinit=bx + 0.5 * (x1 - bx))
    s_y = Slider(fig.add_axes([0.12, 0.02 + 0.035 * (n_sl - 2), 0.75, 0.025]), "detector y", y0, y1, valinit=by + 0.3 * (y1 - by))
    s_l = Slider(fig.add_axes([0.12, 0.02 + 0.035 * (n_sl - 3), 0.75, 0.025]), "wavelength", lmin, lmax, valinit=0.5 * (lmin + lmax))
    s_eta = [Slider(fig.add_axes([0.12, 0.02 + 0.035 * (n_sl - 4 - i), 0.75, 0.025]), f"eta {i} (σ)", -3, 3, valinit=0)
             for i in range(NE)]

    def draw(_=None):
        shift = np.array([[s.val for s in s_eta]]) if NE else None
        p, comps, phi, a, c, w = evaluate(qp, s_x.val, s_y.val, s_l.val, shift)
        ax_p.clear()
        ax_p.imshow(p[0], cmap="viridis", interpolation="nearest")
        L = 8
        ax_p.plot([12 - L * math.cos(phi[0]), 12 + L * math.cos(phi[0])], [12 - L * math.sin(phi[0]), 12 + L * math.sin(phi[0])],
                  color="w", lw=0.8, ls="--")
        ax_p.set_title(f"profile, n_eff {1 / (p[0] ** 2).sum():.0f}\nφ {math.degrees(phi[0]):.0f}° (dashed: streak direction)", fontsize=9)
        for k, axk in enumerate(ax_c):
            axk.clear()
            axk.imshow(comps[0, k], cmap="magma", interpolation="nearest", vmin=0, vmax=comps[0].max())
            axk.set_title(f"component {k + 1}\nw {w[0, k]:.2f}; sd {a[0, k]:.2f} along, {c[0, k]:.2f} across", fontsize=8)
            axk.set_xticks([])
            axk.set_yticks([])
        ax_det.clear()
        ax_det.set_xlim(x0, x1)
        ax_det.set_ylim(y1, y0)
        ax_det.plot([bx], [by], "k+")
        ax_det.plot([s_x.val], [s_y.val], "ro")
        ax_det.set_title("detector position (+ beam)", fontsize=8)
        ax_det.tick_params(labelsize=6)
        fig.canvas.draw_idle()

    for s in [s_x, s_y, s_l, *s_eta]:
        s.on_changed(draw)
    draw()
    fig.suptitle(f"{tag} ({name}{', shape posterior' if NE else ''})", fontsize=11)
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tag", default=None, help="name used in titles and file names (default: the run directory)")
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/sim/profile_explorer"))
    a = ap.parse_args()
    if a.static:
        matplotlib.use("Agg")
    qp, name, post = load_qp(a.ckpt, a.config)
    tag = a.tag or a.config.parent.name
    if a.static:
        static(qp, name, post, a.out, tag)
    else:
        interactive(qp, name, post, tag)


if __name__ == "__main__":
    main()
