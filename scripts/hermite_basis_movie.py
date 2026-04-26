"""Interactive movie of Hermite-Gaussian profile perturbations.

Starts from the centered Gaussian reference (h = 0) and sweeps each Hermite
component h[j] ∈ [-amp, +amp] one at a time, rendering the resulting
profile = softmax(W @ h + b) as a 21×21 heatmap.

Output: hermite_basis_movie.html (open in any browser).

Run:
    uv run python scripts/hermite_basis_movie.py
    uv run python scripts/hermite_basis_movie.py --max-order 6 --amp 3.0
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots


def hermite_polynomial(n_order: int, x: torch.Tensor) -> torch.Tensor:
    """Probabilist's Hermite polynomial H_n(x) by recurrence."""
    if n_order == 0:
        return torch.ones_like(x)
    if n_order == 1:
        return x
    h_prev2 = torch.ones_like(x)
    h_prev1 = x
    h_curr = x
    for k in range(2, n_order + 1):
        h_curr = x * h_prev1 - (k - 1) * h_prev2
        h_prev2 = h_prev1
        h_prev1 = h_curr
    return h_curr


def build_hermite_basis(
    H: int, W: int, max_order: int, sigma_ref: float
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float64),
        torch.arange(W, dtype=torch.float64),
        indexing="ij",
    )
    x_norm = (xx - cx) / sigma_ref
    y_norm = (yy - cy) / sigma_ref
    gaussian = torch.exp(-0.5 * (x_norm**2 + y_norm**2))
    ref = gaussian / gaussian.sum()
    b = torch.log(ref.reshape(-1).clamp(min=1e-10)).float()

    basis_list = []
    orders: list[tuple[int, int]] = []
    for nx in range(max_order + 1):
        for ny in range(max_order + 1 - nx):
            if nx == 0 and ny == 0:
                continue
            psi = (
                hermite_polynomial(nx, x_norm)
                * hermite_polynomial(ny, y_norm)
                * gaussian
            )
            psi = psi / psi.norm()
            basis_list.append(psi.reshape(-1))
            orders.append((nx, ny))

    W_basis = torch.stack(basis_list, dim=1).float()
    return W_basis, b, orders


def _frame_from_h(
    W: torch.Tensor,
    b: torch.Tensor,
    h: torch.Tensor,
    grid_h: int,
    grid_w: int,
    title: str,
) -> dict:
    """Build one frame: left = perturbation Wh, right = profile softmax(Wh + b)."""
    pert = (W @ h).view(grid_h, grid_w).numpy()
    prf = F.softmax(W @ h + b, dim=-1).view(grid_h, grid_w).numpy()
    return {"pert": pert, "profile": prf, "title": title}


def build_frames(
    W: torch.Tensor,
    b: torch.Tensor,
    orders: list[tuple[int, int]],
    amp: float,
    steps_per_component: int,
    grid_h: int,
    grid_w: int,
    walk_steps: int,
    walk_rho: float,
    walk_sigma: float,
    walk_seed: int,
) -> list[dict]:
    """Construct per-frame data: reference → per-component sweep → random walk.

    Sweep: 0 → +amp → -amp → 0 per component, then move to the next.
    Walk: AR(1) process h_{t+1} = rho*h_t + sqrt(1-rho^2)*sigma*eps, h_0 = 0.
    """
    d = W.shape[1]

    # Opening: h = 0
    frames = [
        _frame_from_h(
            W,
            b,
            torch.zeros(d),
            grid_h,
            grid_w,
            title="h = 0 — centered Gaussian reference",
        )
    ]

    # Per-component sweep
    quarter = max(steps_per_component // 4, 1)
    sweep = np.concatenate(
        [
            np.linspace(0, amp, quarter),
            np.linspace(amp, -amp, 2 * quarter),
            np.linspace(-amp, 0, quarter),
        ]
    )
    for j, (nx, ny) in enumerate(orders):
        for val in sweep:
            h = torch.zeros(d)
            h[j] = float(val)
            frames.append(
                _frame_from_h(
                    W,
                    b,
                    h,
                    grid_h,
                    grid_w,
                    title=(
                        f"sweep · component {j}/{d - 1} (H_{nx},{ny}) "
                        f"— h[{j}] = {val:+.2f}"
                    ),
                )
            )

    # Random walk in h-space (AR(1))
    if walk_steps > 0:
        rng = np.random.default_rng(walk_seed)
        innovation_scale = walk_sigma * float(np.sqrt(1.0 - walk_rho**2))
        h_np = np.zeros(d)
        for t in range(walk_steps):
            eps = rng.standard_normal(d) * innovation_scale
            h_np = walk_rho * h_np + eps
            h_norm = float(np.linalg.norm(h_np))
            frames.append(
                _frame_from_h(
                    W,
                    b,
                    torch.from_numpy(h_np).float(),
                    grid_h,
                    grid_w,
                    title=(
                        f"random walk · step {t + 1}/{walk_steps} "
                        f"— ‖h‖ = {h_norm:.2f}"
                    ),
                )
            )
    return frames


def build_figure(
    frames: list[dict],
    pert_lim: float,
    profile_lim: float,
    frame_duration_ms: int,
) -> go.Figure:
    f0 = frames[0]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Perturbation field W·h (added to log reference)",
            "Profile = softmax(W·h + b)",
        ),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Heatmap(
            z=f0["pert"],
            colorscale="RdBu_r",
            zmin=-pert_lim,
            zmax=pert_lim,
            colorbar=dict(title="W·h", x=0.43, len=0.85),
            showscale=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=f0["profile"],
            colorscale="Viridis",
            zmin=0.0,
            zmax=profile_lim,
            colorbar=dict(title="p", x=1.0, len=0.85),
            showscale=True,
        ),
        row=1,
        col=2,
    )

    plotly_frames = [
        go.Frame(
            name=str(i),
            data=[
                go.Heatmap(
                    z=f["pert"],
                    colorscale="RdBu_r",
                    zmin=-pert_lim,
                    zmax=pert_lim,
                    showscale=False,
                ),
                go.Heatmap(
                    z=f["profile"],
                    colorscale="Viridis",
                    zmin=0.0,
                    zmax=profile_lim,
                    showscale=False,
                ),
            ],
            layout=go.Layout(title_text=f["title"]),
        )
        for i, f in enumerate(frames)
    ]
    fig.frames = plotly_frames

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
        title=f0["title"],
        width=1100,
        height=600,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=-0.08,
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
        sliders=[dict(active=0, steps=slider_steps, x=0.1, y=-0.05, len=0.85)],
    )
    # equal aspect on the heatmaps
    fig.update_xaxes(
        scaleanchor="y", scaleratio=1, visible=False, row=1, col=1
    )
    fig.update_xaxes(
        scaleanchor="y2", scaleratio=1, visible=False, row=1, col=2
    )
    fig.update_yaxes(autorange="reversed", visible=False, row=1, col=1)
    fig.update_yaxes(autorange="reversed", visible=False, row=1, col=2)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--sigma-ref", type=float, default=3.0)
    parser.add_argument("--height", type=int, default=21)
    parser.add_argument("--width", type=int, default=21)
    parser.add_argument(
        "--amp",
        type=float,
        default=3.0,
        help="peak |h[j]| swept per component",
    )
    parser.add_argument("--steps-per-component", type=int, default=32)
    parser.add_argument("--frame-duration-ms", type=int, default=60)
    parser.add_argument(
        "--walk-steps",
        type=int,
        default=150,
        help="number of AR(1) random-walk frames (0 disables)",
    )
    parser.add_argument(
        "--walk-rho",
        type=float,
        default=0.92,
        help="AR(1) persistence; closer to 1 = smoother walk",
    )
    parser.add_argument(
        "--walk-sigma",
        type=float,
        default=2.0,
        help="stationary std of each h component during walk",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="hermite_basis_movie.html",
    )
    args = parser.parse_args()

    W, b, orders = build_hermite_basis(
        args.height,
        args.width,
        args.max_order,
        args.sigma_ref,
    )
    d = W.shape[1]
    print(f"Built Hermite basis: d={d} components, orders={orders}")

    frames = build_frames(
        W,
        b,
        orders,
        amp=args.amp,
        steps_per_component=args.steps_per_component,
        grid_h=args.height,
        grid_w=args.width,
        walk_steps=args.walk_steps,
        walk_rho=args.walk_rho,
        walk_sigma=args.walk_sigma,
        walk_seed=args.seed,
    )
    print(f"Generated {len(frames)} frames")

    pert_lim = float(max(abs(f["pert"]).max() for f in frames))
    profile_lim = float(max(f["profile"].max() for f in frames))

    fig = build_figure(frames, pert_lim, profile_lim, args.frame_duration_ms)
    out = Path(args.output).resolve()
    fig.write_html(out, include_plotlyjs="cdn", auto_play=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
