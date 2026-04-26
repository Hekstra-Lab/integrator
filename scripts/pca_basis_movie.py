"""Interactive movie of PCA-basis profile perturbations.

Loads a precomputed PCA basis from a .pt file (as produced by
`integrator.utils.prepare_priors.build_profile_basis_with_priors` with
basis_type="pca") and renders the middle depth slice.

The file must contain:
  - W: (K, d) basis matrix, K = D*H*W
  - b: (K,) bias = mean of log-profiles across reflections

The movie:
  1. Opens on h = 0 (the data-mean profile, exp(b)/sum(exp(b))).
  2. Sweeps each principal component h[j]: 0 → +amp → -amp → 0.
  3. Ends with an AR(1) random walk where all components vary jointly.

Left panel: perturbation field W·h (middle slice), signed colorscale.
Right panel: resulting profile softmax(W·h + b) (middle slice).

Run:
    uv run python scripts/pca_basis_movie.py --basis-path profile_basis.pt
    uv run python scripts/pca_basis_movie.py --basis-path pca.pt --D 3 --H 21 --W 21
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots


def _frame_from_h(
    W_basis: torch.Tensor,
    b: torch.Tensor,
    h: torch.Tensor,
    shape: tuple[int, int, int],
    slice_idx: int,
    title: str,
) -> dict:
    """Build one frame: middle-slice perturbation and profile."""
    D, H, W = shape
    pert_vol = (W_basis @ h).view(D, H, W).numpy()
    prf_vol = F.softmax(W_basis @ h + b, dim=-1).view(D, H, W).numpy()
    return {
        "pert": pert_vol[slice_idx],
        "profile": prf_vol[slice_idx],
        "title": title,
    }


def build_frames(
    W_basis: torch.Tensor,
    b: torch.Tensor,
    shape: tuple[int, int, int],
    slice_idx: int,
    amp: float,
    steps_per_component: int,
    walk_steps: int,
    walk_rho: float,
    walk_sigma: float,
    walk_seed: int,
    explained_var: torch.Tensor | None,
) -> list[dict]:
    d = W_basis.shape[1]

    # Opening: h = 0 (data-mean profile)
    frames = [
        _frame_from_h(
            W_basis,
            b,
            torch.zeros(d),
            shape,
            slice_idx,
            title="h = 0 — data-mean profile (middle slice)",
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
    for j in range(d):
        ev_str = ""
        if explained_var is not None and j < explained_var.shape[0]:
            ev_str = f", EV={explained_var[j].item():.1%}"
        for val in sweep:
            h = torch.zeros(d)
            h[j] = float(val)
            frames.append(
                _frame_from_h(
                    W_basis,
                    b,
                    h,
                    shape,
                    slice_idx,
                    title=(
                        f"sweep · PC {j}/{d - 1}{ev_str} — h[{j}] = {val:+.2f}"
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
                    W_basis,
                    b,
                    torch.from_numpy(h_np).float(),
                    shape,
                    slice_idx,
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
    slice_label: str,
) -> go.Figure:
    f0 = frames[0]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Perturbation W·h ({slice_label})",
            f"Profile = softmax(W·h + b) ({slice_label})",
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
        ),
        row=1,
        col=2,
    )

    fig.frames = [
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
    parser.add_argument(
        "--basis-path",
        type=str,
        required=True,
        help="path to profile_basis .pt with keys W, b",
    )
    parser.add_argument("--D", type=int, default=3, help="shoebox depth")
    parser.add_argument("--H", type=int, default=21, help="shoebox height")
    parser.add_argument("--W", type=int, default=21, help="shoebox width")
    parser.add_argument(
        "--slice-idx",
        type=int,
        default=None,
        help="depth slice to render (default: middle)",
    )
    parser.add_argument(
        "--amp",
        type=float,
        default=3.0,
        help="peak |h[j]| swept per component",
    )
    parser.add_argument("--steps-per-component", type=int, default=32)
    parser.add_argument("--frame-duration-ms", type=int, default=60)
    parser.add_argument("--walk-steps", type=int, default=150)
    parser.add_argument("--walk-rho", type=float, default=0.92)
    parser.add_argument("--walk-sigma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="pca_basis_movie.html")
    args = parser.parse_args()

    basis = torch.load(args.basis_path, weights_only=False)
    if "W" not in basis or "b" not in basis:
        raise ValueError(
            f"Basis file {args.basis_path!r} missing 'W' or 'b'; "
            f"keys present: {list(basis.keys())}"
        )
    W_basis = basis["W"].float()
    b = basis["b"].float()
    explained_var = basis.get("explained_var")
    if explained_var is not None:
        explained_var = explained_var.float()

    D, H, W = args.D, args.H, args.W
    K_expected = D * H * W
    if W_basis.shape[0] != K_expected:
        raise ValueError(
            f"Basis W has K={W_basis.shape[0]} but D*H*W={K_expected}. "
            f"Pass --D/--H/--W matching the basis shape."
        )
    d = W_basis.shape[1]
    slice_idx = args.slice_idx if args.slice_idx is not None else D // 2
    if not 0 <= slice_idx < D:
        raise ValueError(f"slice_idx={slice_idx} out of range [0, {D})")

    print(f"Loaded basis: K={W_basis.shape[0]} (D={D}, H={H}, W={W}), d={d}")
    if explained_var is not None:
        print(
            f"Explained variance (top 5): "
            f"{[f'{v:.1%}' for v in explained_var[:5].tolist()]}"
        )
    print(f"Rendering depth slice {slice_idx}/{D - 1}")

    frames = build_frames(
        W_basis,
        b,
        shape=(D, H, W),
        slice_idx=slice_idx,
        amp=args.amp,
        steps_per_component=args.steps_per_component,
        walk_steps=args.walk_steps,
        walk_rho=args.walk_rho,
        walk_sigma=args.walk_sigma,
        walk_seed=args.seed,
        explained_var=explained_var,
    )
    print(f"Generated {len(frames)} frames")

    pert_lim = float(max(abs(f["pert"]).max() for f in frames))
    profile_lim = float(max(f["profile"].max() for f in frames))
    slice_label = f"z={slice_idx}" if D > 1 else "2D"

    fig = build_figure(
        frames,
        pert_lim,
        profile_lim,
        args.frame_duration_ms,
        slice_label,
    )
    out = Path(args.output).resolve()
    fig.write_html(out, include_plotlyjs="cdn", auto_play=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
