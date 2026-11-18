#!/usr/bin/env python
"""What each profile family actually learned, read out of its checkpoint.

The families parameterize a profile in different ways, so "the learned
profile model" means a different object in each:

* `learned_basis` decodes `softmax(W h + b)` with `W` free, so what it learns
  is the basis `W` itself -- a set of images -- plus the decoder bias, which
  is the profile it falls back to when the latent is zero.
* `fixed_basis` shares that decoder form but `W` is a frozen quadratic
  monomial basis, so only the latent and the bias move. A quadratic
  log-profile IS an anisotropic Gaussian, so its five coefficients are
  readable as a covariance.
* `shapelet` learns Gauss-Hermite coefficients and, with a frame head, an
  orientation per reflection.
* `physical` learns the handful of scalars in `sigma^2_tan = a sin^2(theta) +
  b` and the radial term, which are the only free parameters it has.
* `geometric` adds a shift and a log-extent head on top of a decoder, so what
  it learns beyond the basis is how far and how much to steer.

This writes one figure per family covering whichever of those exist, plus a
CSV of the scalar parameters so they can be quoted rather than eyeballed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def load_state(run_dir: Path) -> tuple[dict, str] | None:
    ckpts = sorted((run_dir / "files" / "checkpoints").glob("epoch=*.ckpt"))
    if not ckpts:
        print(f"  {run_dir.name}: no checkpoints")
        return None
    sd = torch.load(ckpts[-1].as_posix(), map_location="cpu", weights_only=False)
    return sd["state_dict"], ckpts[-1].stem


def qp_tensors(state: dict) -> dict[str, torch.Tensor]:
    """Every parameter belonging to the profile surrogate."""
    out = {}
    for k, v in state.items():
        if "surrogates.qp." not in k or not torch.is_tensor(v):
            continue
        out[k.split("surrogates.qp.", 1)[1]] = v.detach().float().cpu()
    return out


def family_of(run_dir: Path, t: dict[str, torch.Tensor]) -> str:
    """The registry name from the run's own config; shapes only as a fallback.

    `requires_grad` does not survive a state_dict round trip, so a frozen
    basis is indistinguishable from a learned one by inspection -- the config
    is the only reliable source.
    """
    cfg_path = run_dir / "config_log.yaml"
    if cfg_path.is_file():
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        name = (
            cfg.get("surrogates", {}).get("qp", {}).get("name")
        )
        if name:
            return name
    if "shift_head.0.weight" in t or "extent_head.0.weight" in t:
        return "geometric_basis_profile"
    if "tau" in t and "decoder.weight" not in t:
        return "shapelet_profile"
    if "decoder.weight" in t:
        return "learned_basis_profile"
    return "unknown"


def grid_shape(n_px: int) -> tuple[int, int]:
    side = int(round(math.sqrt(n_px)))
    return (side, side) if side * side == n_px else (1, n_px)


def plot_basis(t: dict, name: str, out: Path, dpi: int) -> list[str]:
    """The decoder's columns as images, plus the bias it falls back to."""
    import matplotlib.pyplot as plt

    if "decoder.weight" not in t:
        return plot_coefficient_priors(t, name, out, dpi)
    W = t["decoder.weight"].numpy()          # (K, d)
    b = t.get("decoder.bias")
    k, d = W.shape
    h, w = grid_shape(k)
    if h == 1:
        return []

    ncol = min(d, 8)
    nrow = int(np.ceil(d / ncol)) + (1 if b is not None else 0)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.6 * ncol, 1.7 * nrow),
                             squeeze=False)
    for i in range(d):
        ax = axes[i // ncol][i % ncol]
        img = W[:, i].reshape(h, w)
        lim = np.abs(img).max() or 1.0
        ax.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
        ax.set_title(f"h{i}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for i in range(d, nrow * ncol):
        axes[i // ncol][i % ncol].set_visible(False)
    if b is not None:
        ax = axes[nrow - 1][0]
        ax.set_visible(True)
        img = torch.softmax(b, dim=-1).numpy().reshape(h, w)
        ax.imshow(img, cmap="cividis", origin="lower")
        ax.set_title("softmax(bias)\n= profile at h=0", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for j in range(1, ncol):
            axes[nrow - 1][j].set_visible(False)

    fig.suptitle(f"{name}: decoder basis (RED positive)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out / f"fig_params_basis_{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [path.name]


def plot_coefficient_priors(t: dict, name: str, out: Path, dpi: int) -> list[str]:
    """For families with no decoder: the per-coefficient prior widths.

    The shapelet family carries `tau`, the prior scale on each Gauss-Hermite
    coefficient. It is what sets how far each order may move, so it is the
    closest thing it has to a learned basis -- the basis itself is fixed by
    construction (squared Gauss-Hermite, Christoffel-bounded).
    """
    import matplotlib.pyplot as plt

    tau = t.get("tau")
    if tau is None:
        return []
    v = tau.numpy().reshape(-1)
    fig, ax = plt.subplots(figsize=(1.2 + 0.32 * len(v), 3.0))
    ax.bar(np.arange(len(v)), v, color="steelblue")
    ax.set_xlabel("Gauss-Hermite coefficient")
    ax.set_ylabel("prior width tau")
    ax.set_title(f"{name}: per-coefficient prior widths", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    fig.tight_layout()
    path = out / f"fig_params_tau_{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [path.name]


def plot_scalars(rows: list[dict], out: Path, dpi: int) -> list[str]:
    """Every scalar profile parameter, one panel per arm."""
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["scalars"]]
    if not rows:
        return []
    fig, axes = plt.subplots(
        1, len(rows), figsize=(3.4 * len(rows), 3.2), squeeze=False
    )
    for ax, r in zip(axes[0], rows):
        names = list(r["scalars"])
        vals = [r["scalars"][n] for n in names]
        ax.barh(range(len(names)), vals, color="steelblue")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f"{r['name']}\n({r['family']})", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5, axis="x")
    fig.suptitle("learned scalar profile parameters", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / "fig_params_scalars.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [path.name]


def main() -> None:
    a = parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows, written = [], []

    for rd in a.run_dirs:
        loaded = load_state(rd)
        if loaded is None:
            continue
        state, ckpt = loaded
        t = qp_tensors(state)
        if not t:
            print(f"  {rd.name}: no surrogates.qp.* in {ckpt}")
            continue
        fam = family_of(rd, t)
        # a scalar here is anything with one element: the physical widths, a
        # learned tau, a gate temperature
        scalars = {
            k: float(v.reshape(-1)[0])
            for k, v in sorted(t.items())
            if v.numel() == 1
        }
        shapes = {k: tuple(v.shape) for k, v in sorted(t.items())}
        print(f"  {rd.name}: family={fam}  ckpt={ckpt}")
        for k, s in shapes.items():
            print(f"      {k:34s} {s}")
        rows.append(
            {"name": rd.name, "family": fam, "scalars": scalars, "shapes": shapes}
        )
        written += plot_basis(t, rd.name, a.out_dir, a.dpi)

    written += plot_scalars(rows, a.out_dir, a.dpi)

    csv = a.out_dir / "profile_parameters.csv"
    with csv.open("w") as f:
        f.write("arm,family,parameter,shape,value\n")
        for r in rows:
            for k, s in r["shapes"].items():
                v = r["scalars"].get(k, "")
                f.write(f'{r["name"]},{r["family"]},{k},"{s}",{v}\n')
    written.append(csv.name)
    print("\nwrote:", ", ".join(written))


if __name__ == "__main__":
    main()
