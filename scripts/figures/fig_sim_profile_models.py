"""Spot profiles of each profile model beside the true profile, by resolution shell (rows) and model (columns).

Reads the npz of `scripts/sim_truth/dump_profile_truth_examples.py --sample N`
(`true_profile`, `counts`, `i_true`, `d`, `refl_ids`, and `<arm>_profile` for every
arm in `arms`, the arm's `qp_mean`; `laue_integrate_profile` with --laue-profiles)
and, with --modes, the npz of `scripts/sim_truth/dump_profile_compare.py`, whose
`<name>_mode` profiles (the profile at the joint Laplace mode of (eta, I, b) given
the reflection's counts, the centre of the Laplace + collapsed export) are joined on
`refl_ids` and drawn as extra columns after the arm they belong to (--mode-after
NAME=ARM).

Rows are the resolution shells D_EDGES. Three figures:

- `sim_profile_models_examples_<dataset>_<I>`: per shell, the in-sample reflection
  whose i_true is nearest --example-i (one figure per value); columns are its
  counts, its true profile, then each model. The profiles of a row share one
  colour scale from 0 to the largest pixel of any of them, so a profile wider than
  the truth shows as fainter; n_eff = 1 / sum p^2 in each panel's corner.
- `sim_profile_models_means_<dataset>`: every profile rotated about the window
  centre so that the TRUE profile's major axis (from its second moments) lies
  along x, then averaged over the shell's reflections; columns the mean true
  profile, then each model's mean on the same scale as the truth of that row.
- `sim_profile_models_diff_<dataset>`: each model's mean minus the mean true
  profile, on a diverging scale symmetric about 0 and shared along the row, with
  kappa = sum q p / sum q^2 of the two means (the ratio a profile fit with q
  returns on counts that follow p; < 1 means intensities come out low).

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_sim_profile_models.py \\
            --src docs/figures/data/profile_models_mos010_pc_default.npz \\
            --modes docs/figures/data/profile_compare_mos010_pc_default.npz
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

import figstyle  # noqa: E402
from arms import label as arm_label  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "sim"
D_EDGES = (1.78, 2.0, 2.5, 3.0, 4.0, 99.0)
ORDER = ("laue_integrate", "t0_rb", "p2_rb", "b1_rb", "b1_laplace", "geom_laplace")


def shell_name(lo: float, hi: float) -> str:
    return f"$>{lo:g}$ \\AA" if hi > 50 else f"{lo:g}--{hi:g} \\AA"


def major_axis_angle(p: np.ndarray) -> np.ndarray:
    """Angle (radians, image coordinates: x right, y down) of each profile's major axis."""
    n = p.shape[-1]
    ax = np.arange(n) - (n - 1) / 2
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    w = p / p.sum((1, 2), keepdims=True)
    mx, my = (w * xx).sum((1, 2)), (w * yy).sum((1, 2))
    dx, dy = xx[None] - mx[:, None, None], yy[None] - my[:, None, None]
    sxx, syy, sxy = (w * dx * dx).sum((1, 2)), (w * dy * dy).sum((1, 2)), (w * dx * dy).sum((1, 2))
    return 0.5 * np.arctan2(2 * sxy, sxx - syy)


def aligned(arr: np.ndarray, ang: np.ndarray) -> np.ndarray:
    # ndimage.rotate turns the image by +angle in these coordinates (rows down),
    # which takes a major axis at `ang` onto x (checked on synthetic streaks)
    return np.stack([ndimage.rotate(x, np.degrees(a), reshape=False, order=1, mode="constant")
                     for x, a in zip(arr, ang)])


def neff(p: np.ndarray) -> float:
    p = p / p.sum()
    return float(1 / (p**2).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--modes", type=Path, default=None)
    ap.add_argument("--mode-after", nargs="*", default=["B1=b1_laplace", "geom=geom_laplace"],
                    help="NAME=ARM: draw the modes npz's <NAME>_mode as a column after ARM")
    ap.add_argument("--example-i", default="30,300", help="true intensities of the example rows (photons)")
    a = ap.parse_args()
    figstyle.use()
    z = np.load(a.src, allow_pickle=True)
    dataset = str(z["dataset"])
    ids = z["refl_ids"]
    present = {k[: -len("_profile")] for k in z.files if k.endswith("_profile") and k != "true_profile"}
    arms = [m for m in ORDER if m in present] + sorted(present - set(ORDER))
    cols = [(arm_label(m), z[f"{m}_profile"].astype(float)) for m in arms]
    if a.modes is not None:
        zm = np.load(a.modes, allow_pickle=True)
        pos = {int(r): i for i, r in enumerate(zm["refl_ids"])}
        idx = np.array([pos.get(int(r), -1) for r in ids])
        keep = idx >= 0
        print(f"modes joined for {keep.sum()} of {len(ids)} reflections")
        out = []
        for lab, arr in cols:
            out.append((lab, arr))
            for spec in a.mode_after:
                name, arm = spec.split("=", 1)
                if arm in arms and lab == arm_label(arm) and f"{name}_mode" in zm.files:
                    m = np.full_like(arr, np.nan)
                    m[keep] = zm[f"{name}_mode"][idx[keep]]
                    out.append((f"{arm_label(arm).split(',')[0]}, Laplace mode", m))
        cols = out
    true, counts, I, d = z["true_profile"].astype(float), z["counts"], z["i_true"], z["d"]
    nb = len(D_EDGES) - 1
    cm, dm = figstyle.cmap(), figstyle.diverging_cmap()
    ncol = len(cols)

    # ---- examples, one reflection per shell
    for target in (float(x) for x in a.example_i.split(",")):
        fig, ax = plt.subplots(nb, ncol + 2, figsize=figstyle.figsize("full", aspect=nb / (ncol + 2) * 1.05),
                               layout="constrained", squeeze=False)
        for r in range(nb):
            sel = np.where((d >= D_EDGES[r]) & (d < D_EDGES[r + 1]) & np.isfinite(cols[-1][1]).all((1, 2)))[0]
            j = sel[np.argmin(np.abs(np.log(I[sel]) - np.log(target)))]
            panels = [("counts", counts[j]), ("true profile", true[j])] + [(lab, arr[j]) for lab, arr in cols]
            vmax = max(np.nanmax(p) for _, p in panels[1:])
            for k, (lab, img) in enumerate(panels):
                x = ax[r, k]
                x.imshow(img, cmap=cm, interpolation="nearest", vmin=0, vmax=None if k == 0 else vmax)
                x.set_xticks([])
                x.set_yticks([])
                x.grid(False)
                if k >= 1:
                    x.text(0.04, 0.04, f"{neff(img):.0f}", transform=x.transAxes, fontsize=5, color="white")
                if r == 0:
                    x.set_title(lab, fontsize=5)
            ax[r, 0].set_ylabel(f"{shell_name(D_EDGES[r], D_EDGES[r + 1])}\n{I[j]:.0f} ph", fontsize=5)
        for p in figstyle.save(fig, OUT / f"sim_profile_models_examples_{dataset}_{target:g}"):
            print(p)
        plt.close(fig)

    # ---- streak-aligned means and differences
    ok = np.isfinite(np.stack([arr for _, arr in cols])).all((0, 2, 3))
    ang = major_axis_angle(true)
    true_al = aligned(true, ang)
    cols_al = [(lab, aligned(np.nan_to_num(arr), ang)) for lab, arr in cols]
    figm, axm = plt.subplots(nb, ncol + 1, figsize=figstyle.figsize("full", aspect=nb / (ncol + 1) * 1.05),
                             layout="constrained", squeeze=False)
    figd, axd = plt.subplots(nb, ncol, figsize=figstyle.figsize("full", aspect=nb / ncol * 1.08),
                             layout="constrained", squeeze=False)
    for r in range(nb):
        sel = (d >= D_EDGES[r]) & (d < D_EDGES[r + 1]) & ok
        pt = true_al[sel].mean(0)
        vmax = pt.max()
        means = [(lab, arr[sel].mean(0)) for lab, arr in cols_al]
        lim = max(np.abs(q - pt).max() for _, q in means)
        for k, (lab, img) in enumerate([("true profile", pt)] + means):
            x = axm[r, k]
            x.imshow(img, cmap=cm, interpolation="nearest", vmin=0, vmax=vmax)
            x.text(0.04, 0.04, f"{neff(img):.0f}", transform=x.transAxes, fontsize=5, color="white")
            if r == 0:
                x.set_title(lab, fontsize=5)
        axm[r, 0].set_ylabel(f"{shell_name(D_EDGES[r], D_EDGES[r + 1])}\nn {sel.sum()}", fontsize=5)
        for k, (lab, q) in enumerate(means):
            x = axd[r, k]
            x.imshow(q - pt, cmap=dm, vmin=-lim, vmax=lim, interpolation="nearest")
            kappa = float((q * pt).sum() / (q * q).sum())
            x.text(0.04, 0.04, f"$\\kappa$ {kappa:.3f}", transform=x.transAxes, fontsize=5)
            if r == 0:
                x.set_title(f"{lab}\n$-$ truth", fontsize=5)
        axd[r, 0].set_ylabel(f"{shell_name(D_EDGES[r], D_EDGES[r + 1])}\nn {sel.sum()}", fontsize=5)
    for x in list(axm.ravel()) + list(axd.ravel()):
        x.set_xticks([])
        x.set_yticks([])
        x.grid(False)
    for p in figstyle.save(figm, OUT / f"sim_profile_models_means_{dataset}"):
        print(p)
    for p in figstyle.save(figd, OUT / f"sim_profile_models_diff_{dataset}"):
        print(p)


if __name__ == "__main__":
    main()
