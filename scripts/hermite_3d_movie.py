"""Interactive 3D volumetric movie of Hermite-Gaussian profile perturbations.

Shows how a 3D profile (3 × 21 × 21) continuously deforms as the latent h
is perturbed along each Hermite basis direction, then through a random
walk in h-space.

Uses plotly's go.Volume — 3D semi-transparent rendering with iso-surfaces.
The scene is rotatable/zoomable in the browser while the animation plays.

Run:
    uv run python scripts/hermite_3d_movie.py
    uv run python scripts/hermite_3d_movie.py --max-order 4 --amp 3.0
"""

import argparse
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F


def hermite_polynomial(n: int, x: torch.Tensor) -> torch.Tensor:
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return x
    h_prev2 = torch.ones_like(x)
    h_prev1 = x
    h_curr = x
    for k in range(2, n + 1):
        h_curr = x * h_prev1 - (k - 1) * h_prev2
        h_prev2 = h_prev1
        h_prev1 = h_curr
    return h_curr


def build_hermite_basis_3d(
    D: int,
    H: int,
    W: int,
    max_order: int,
    sigma_ref: float,
    sigma_z: float,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, int]]]:
    """Mirrors _build_hermite_basis_3d in prepare_priors.py.

    max_order_z = min(1, D-1). For D=3 we get 14 nz=0 + 15 nz=1 = 29 columns.
    """
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    cz = (D - 1) / 2.0
    zz, yy, xx = torch.meshgrid(
        torch.arange(D, dtype=torch.float64),
        torch.arange(H, dtype=torch.float64),
        torch.arange(W, dtype=torch.float64),
        indexing="ij",
    )
    x_n = (xx - cx) / sigma_ref
    y_n = (yy - cy) / sigma_ref
    z_n = (zz - cz) / sigma_z
    half_gauss = torch.exp(-0.25 * (x_n**2 + y_n**2 + z_n**2))
    full_gauss = torch.exp(-0.5 * (x_n**2 + y_n**2 + z_n**2))
    ref = full_gauss / full_gauss.sum()
    b = torch.log(ref.reshape(-1).clamp(min=1e-10)).float()

    max_order_z = min(1, D - 1)
    basis_list: list[torch.Tensor] = []
    orders: list[tuple[int, int, int]] = []
    for nz in range(max_order_z + 1):
        for nx in range(max_order + 1):
            for ny in range(max_order + 1 - nx):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                phi = (
                    hermite_polynomial(nx, x_n)
                    * hermite_polynomial(ny, y_n)
                    * hermite_polynomial(nz, z_n)
                    * half_gauss
                )
                phi = phi / phi.norm()
                basis_list.append(phi.reshape(-1))
                orders.append((nx, ny, nz))
    return torch.stack(basis_list, dim=1).float(), b, orders


def compute_profile(
    W_basis: torch.Tensor,
    b: torch.Tensor,
    h: torch.Tensor,
) -> np.ndarray:
    """softmax(Wh + b) as (K,) numpy."""
    return F.softmax(W_basis @ h + b, dim=-1).numpy()


def build_frames(
    W_basis: torch.Tensor,
    b: torch.Tensor,
    orders: list[tuple[int, int, int]],
    amp: float,
    steps_per_component: int,
    walk_steps: int,
    walk_rho: float,
    walk_sigma: float,
    walk_seed: int,
) -> list[dict]:
    d = W_basis.shape[1]
    frames: list[dict] = []

    # Opening: h = 0 (reference Gaussian)
    prf0 = compute_profile(W_basis, b, torch.zeros(d))
    frames.append({"prf": prf0, "title": "h = 0 — 3D reference Gaussian"})

    # Per-component sweep
    quarter = max(steps_per_component // 4, 1)
    sweep = np.concatenate(
        [
            np.linspace(0, amp, quarter),
            np.linspace(amp, -amp, 2 * quarter),
            np.linspace(-amp, 0, quarter),
        ]
    )
    for j, (nx, ny, nz) in enumerate(orders):
        for val in sweep:
            h = torch.zeros(d)
            h[j] = float(val)
            prf = compute_profile(W_basis, b, h)
            frames.append(
                {
                    "prf": prf,
                    "title": (
                        f"sweep · component {j}/{d - 1} "
                        f"(H_{nx},{ny},{nz}) — h[{j}] = {val:+.2f}"
                    ),
                }
            )

    # Random walk
    if walk_steps > 0:
        rng = np.random.default_rng(walk_seed)
        inno = walk_sigma * math.sqrt(1 - walk_rho**2)
        h_np = np.zeros(d)
        for t in range(walk_steps):
            eps = rng.standard_normal(d) * inno
            h_np = walk_rho * h_np + eps
            h_t = torch.from_numpy(h_np).float()
            prf = compute_profile(W_basis, b, h_t)
            frames.append(
                {
                    "prf": prf,
                    "title": (
                        f"random walk · step {t + 1}/{walk_steps} "
                        f"— ‖h‖ = {float(np.linalg.norm(h_np)):.2f}"
                    ),
                }
            )
    return frames


def build_figure(
    frames: list[dict],
    shape: tuple[int, int, int],
    vmax: float,
    frame_duration_ms: int,
) -> go.Figure:
    D, H, W = shape
    zz, yy, xx = np.meshgrid(
        np.arange(D),
        np.arange(H),
        np.arange(W),
        indexing="ij",
    )
    X_flat = xx.flatten()
    Y_flat = yy.flatten()
    Z_flat = zz.flatten() * (H / D)  # stretch z for visibility (D is small)

    isomin = vmax * 0.03
    isomax = vmax

    def make_volume(prf: np.ndarray) -> go.Volume:
        return go.Volume(
            x=X_flat,
            y=Y_flat,
            z=Z_flat,
            value=prf,
            isomin=isomin,
            isomax=isomax,
            opacity=0.18,
            surface_count=18,
            colorscale="Viridis",
            caps=dict(x_show=False, y_show=False, z_show=False),
        )

    f0 = frames[0]
    fig = go.Figure(
        data=[make_volume(f0["prf"])],
        layout=go.Layout(
            title=f0["title"],
            width=900,
            height=700,
            scene=dict(
                xaxis_title="x (pixel)",
                yaxis_title="y (pixel)",
                zaxis_title="z (frame, stretched)",
                aspectmode="data",
                bgcolor="rgb(10, 10, 20)",
            ),
        ),
    )
    fig.frames = [
        go.Frame(
            name=str(i),
            data=[make_volume(f["prf"])],
            layout=go.Layout(title_text=f["title"]),
        )
        for i, f in enumerate(frames)
    ]

    slider_steps = [
        dict(
            args=[
                [str(i)],
                dict(
                    frame=dict(duration=0, redraw=True),
                    mode="immediate",
                    transition=dict(duration=0),
                ),
            ],
            label=str(i),
            method="animate",
        )
        for i in range(len(frames))
    ]
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=0.0,
                showactive=False,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(
                                    duration=frame_duration_ms, redraw=True
                                ),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[dict(active=0, steps=slider_steps, x=0.1, y=-0.02, len=0.85)],
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--sigma-ref", type=float, default=3.0)
    parser.add_argument("--sigma-z", type=float, default=1.0)
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--H", type=int, default=21)
    parser.add_argument("--W", type=int, default=21)
    parser.add_argument("--amp", type=float, default=3.0)
    parser.add_argument("--steps-per-component", type=int, default=16)
    parser.add_argument("--frame-duration-ms", type=int, default=80)
    parser.add_argument("--walk-steps", type=int, default=120)
    parser.add_argument("--walk-rho", type=float, default=0.92)
    parser.add_argument("--walk-sigma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="hermite_3d_movie.html")
    args = parser.parse_args()

    W_basis, b, orders = build_hermite_basis_3d(
        args.D,
        args.H,
        args.W,
        args.max_order,
        args.sigma_ref,
        args.sigma_z,
    )
    d = W_basis.shape[1]
    print(f"Hermite 3D basis: d={d}, orders (first 5): {orders[:5]}")

    frames = build_frames(
        W_basis,
        b,
        orders,
        amp=args.amp,
        steps_per_component=args.steps_per_component,
        walk_steps=args.walk_steps,
        walk_rho=args.walk_rho,
        walk_sigma=args.walk_sigma,
        walk_seed=args.seed,
    )
    print(f"Generated {len(frames)} frames")

    vmax = float(max(f["prf"].max() for f in frames))
    fig = build_figure(
        frames,
        shape=(args.D, args.H, args.W),
        vmax=vmax,
        frame_duration_ms=args.frame_duration_ms,
    )
    out = Path(args.output).resolve()
    fig.write_html(out, include_plotlyjs="cdn", auto_play=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
