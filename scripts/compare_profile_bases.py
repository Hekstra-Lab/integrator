"""Compare learned vs fixed-Hermite profile bases from trained checkpoints.

Loads:
  - Hermite basis (fixed) from the .pt file used in training
  - Learned basis (W, b) from a Lightning .ckpt (keys surrogates.qp.decoder.*)

Produces:
  - PNG: grids of W-column middle-slice heatmaps (Hermite + learned), plus
    bias softmax profiles and a cosine-similarity matrix.
  - Stdout: per-column L2 norm, TV-roughness, and best-match cosine similarity.

Run:
    uv run python scripts/compare_profile_bases.py \\
        --hermite-basis /path/to/hermite_profile_basis_30.pt \\
        --learned-ckpt  /path/to/last.ckpt \\
        --output profile_basis_comparison.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _middle_slice(col: torch.Tensor, D: int, H: int, W: int) -> np.ndarray:
    return col.reshape(D, H, W)[D // 2].numpy()


def _tv(img: np.ndarray) -> float:
    gx = np.abs(np.diff(img, axis=1))
    gy = np.abs(np.diff(img, axis=0))
    return float(gx.sum() + gy.sum())


def load_hermite_basis(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    d = torch.load(path, weights_only=False, map_location="cpu")
    return d["W"].float(), d["b"].float()


def load_learned_from_ckpt(
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    w_key = next(
        (k for k in sd if k.endswith("qp.decoder.weight")),
        None,
    )
    if w_key is None:
        raise KeyError(
            "No 'qp.decoder.weight' key in checkpoint state_dict. "
            f"Available surrogate keys: "
            f"{[k for k in sd if 'qp' in k][:10]}"
        )
    b_key = w_key.replace("weight", "bias")
    W = sd[w_key].float()  # (output_dim, latent_dim) = (K, d)
    b = sd[b_key].float()
    return W, b


def plot_component_grid(
    ax_row,
    columns: list[np.ndarray],
    title_prefix: str,
    vlim: float,
) -> None:
    for i, ax in enumerate(ax_row):
        if i < len(columns):
            ax.imshow(
                columns[i],
                cmap="RdBu_r",
                vmin=-vlim,
                vmax=vlim,
                origin="upper",
            )
            ax.set_title(f"{title_prefix} {i}", fontsize=9)
        else:
            ax.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermite-basis", type=str, required=True)
    parser.add_argument("--learned-ckpt", type=str, required=True)
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--H", type=int, default=21)
    parser.add_argument("--W", type=int, default=21)
    parser.add_argument(
        "--output", type=str, default="profile_basis_comparison.png"
    )
    args = parser.parse_args()

    hermite_basis_path = Path(args.hermite_basis)
    learned_ckpt_path = Path(args.learned_ckpt)
    D, H, Wd = args.D, args.H, args.W
    K = D * H * Wd

    print(f"Loading Hermite basis: {hermite_basis_path}")
    W_h, b_h = load_hermite_basis(hermite_basis_path)
    print(f"  W_h shape: {tuple(W_h.shape)}, b_h shape: {tuple(b_h.shape)}")

    print(f"Loading learned checkpoint: {learned_ckpt_path}")
    W_l, b_l = load_learned_from_ckpt(learned_ckpt_path)
    print(f"  W_l shape: {tuple(W_l.shape)}, b_l shape: {tuple(b_l.shape)}")

    if W_h.shape[0] != K or W_l.shape[0] != K:
        raise ValueError(
            f"Expected K={K} (D={D}*H={H}*W={Wd}), "
            f"got Hermite K={W_h.shape[0]}, Learned K={W_l.shape[0]}"
        )

    d_h = W_h.shape[1]
    d_l = W_l.shape[1]

    # Middle-slice heatmaps of each column
    basis_h = [_middle_slice(W_h[:, j], D, H, Wd) for j in range(d_h)]
    basis_l = [_middle_slice(W_l[:, j], D, H, Wd) for j in range(d_l)]

    # Bias as softmax profile (middle slice)
    prof_h = torch.softmax(b_h, dim=-1).view(D, H, Wd)[D // 2].numpy()
    prof_l = torch.softmax(b_l, dim=-1).view(D, H, Wd)[D // 2].numpy()

    # Scalar summaries
    def l2(col: torch.Tensor) -> float:
        return float(col.norm().item())

    l2_h = [l2(W_h[:, j]) for j in range(d_h)]
    l2_l = [l2(W_l[:, j]) for j in range(d_l)]
    tv_h = [_tv(img) / max(np.abs(img).max(), 1e-12) for img in basis_h]
    tv_l = [_tv(img) / max(np.abs(img).max(), 1e-12) for img in basis_l]

    # Cosine similarity: d_h x d_l
    W_h_norm = W_h / W_h.norm(dim=0, keepdim=True).clamp(min=1e-8)
    W_l_norm = W_l / W_l.norm(dim=0, keepdim=True).clamp(min=1e-8)
    cos_sim = (W_h_norm.T @ W_l_norm).numpy()  # (d_h, d_l)

    # Plot layout
    ncols = max(d_h, d_l)
    fig = plt.figure(figsize=(ncols * 1.2 + 2, 9))
    gs = fig.add_gridspec(
        nrows=5,
        ncols=ncols + 1,
        height_ratios=[1.0, 1.0, 1.2, 1.0, 1.8],
        hspace=0.5,
        wspace=0.1,
        left=0.04,
        right=0.98,
        top=0.95,
        bottom=0.06,
    )

    # Row 1: Hermite basis columns
    ax_h = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    hermite_vlim = float(max(np.abs(img).max() for img in basis_h))
    plot_component_grid(ax_h, basis_h, "ψ_H", hermite_vlim)
    ax_h[0].set_ylabel("Hermite\n(fixed)", fontsize=10)

    # Row 2: Learned basis columns
    ax_l = [fig.add_subplot(gs[1, i]) for i in range(ncols)]
    learned_vlim = float(max(np.abs(img).max() for img in basis_l))
    plot_component_grid(ax_l, basis_l, "W_L", learned_vlim)
    ax_l[0].set_ylabel("Learned", fontsize=10)

    # Row 3: bias softmax profiles
    ax_bh = fig.add_subplot(gs[2, 0:3])
    ax_bh.imshow(prof_h, cmap="viridis", origin="upper")
    ax_bh.set_title(
        f"Hermite softmax(b): sum={prof_h.sum():.2f}, peak={prof_h.max():.3f}",
        fontsize=9,
    )
    ax_bh.set_xticks([])
    ax_bh.set_yticks([])

    ax_bl = fig.add_subplot(gs[2, 3:6])
    ax_bl.imshow(prof_l, cmap="viridis", origin="upper")
    ax_bl.set_title(
        f"Learned softmax(b): sum={prof_l.sum():.2f}, peak={prof_l.max():.3f}",
        fontsize=9,
    )
    ax_bl.set_xticks([])
    ax_bl.set_yticks([])

    # Row 3 right: TV roughness bar chart
    ax_tv = fig.add_subplot(gs[2, 7:])
    x_h = np.arange(d_h)
    x_l = np.arange(d_l) + 0.35
    ax_tv.bar(x_h, tv_h, width=0.3, label=f"Hermite (d={d_h})", color="C0")
    ax_tv.bar(x_l, tv_l, width=0.3, label=f"Learned (d={d_l})", color="C3")
    ax_tv.set_title("TV roughness per column (normalized)", fontsize=9)
    ax_tv.set_xlabel("component index", fontsize=8)
    ax_tv.legend(fontsize=8)
    ax_tv.tick_params(labelsize=8)

    # Row 4: L2 norms per column
    ax_l2 = fig.add_subplot(gs[3, : ncols // 2])
    ax_l2.bar(x_h, l2_h, width=0.3, label="Hermite", color="C0")
    ax_l2.bar(x_l, l2_l, width=0.3, label="Learned", color="C3")
    ax_l2.set_title("L2 norm per column", fontsize=9)
    ax_l2.set_xlabel("component index", fontsize=8)
    ax_l2.legend(fontsize=8)
    ax_l2.tick_params(labelsize=8)

    # Row 4 right: cosine similarity matrix
    ax_cos = fig.add_subplot(gs[3, ncols // 2 :])
    im = ax_cos.imshow(
        np.abs(cos_sim),
        cmap="magma",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax_cos.set_title(
        "|cos similarity|:  rows = Hermite, cols = learned",
        fontsize=9,
    )
    ax_cos.set_xlabel("learned col", fontsize=8)
    ax_cos.set_ylabel("Hermite col", fontsize=8)
    ax_cos.tick_params(labelsize=8)
    plt.colorbar(im, ax=ax_cos, fraction=0.04)

    # Row 5: per-learned-column best-match row (which Hermite modes does it
    # span?) — shows top-3 matches as stacked bars
    ax_match = fig.add_subplot(gs[4, :])
    top_k = 3
    width = 0.25
    cos_abs = np.abs(cos_sim)
    for rank in range(top_k):
        heights = []
        labels = []
        for j in range(d_l):
            sorted_idx = np.argsort(-cos_abs[:, j])
            idx = sorted_idx[rank]
            heights.append(cos_abs[idx, j])
            labels.append(f"H{idx}" if rank == 0 else "")
        xs = np.arange(d_l) + rank * width
        ax_match.bar(
            xs,
            heights,
            width=width,
            label=f"match #{rank + 1}",
        )
        if rank == 0:
            for j, lab in enumerate(labels):
                ax_match.text(
                    j,
                    heights[j] + 0.01,
                    lab,
                    ha="center",
                    fontsize=8,
                    color="black",
                )
    ax_match.set_title(
        "For each learned column: top-3 |cos sim| to Hermite columns "
        "(label = best-match Hermite index)",
        fontsize=10,
    )
    ax_match.set_xlabel("learned column index", fontsize=9)
    ax_match.set_ylabel("|cos sim|", fontsize=9)
    ax_match.set_xticks(np.arange(d_l))
    ax_match.set_ylim(0, 1.1)
    ax_match.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Profile basis comparison  —  Hermite d={d_h} vs Learned d={d_l}  "
        f"(middle slice z={D // 2})",
        fontsize=12,
        y=0.99,
    )

    out = Path(args.output).resolve()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nSaved: {out}")

    # Print summary
    print("\n=== Per-column L2 and TV ===")
    print(f"Hermite (d={d_h}):")
    for j in range(d_h):
        print(f"  ψ_{j:2d}: L2={l2_h[j]:7.3f}  TV={tv_h[j]:8.2f}")
    print(f"\nLearned (d={d_l}):")
    for j in range(d_l):
        print(f"  W_{j:2d}: L2={l2_l[j]:7.3f}  TV={tv_l[j]:8.2f}")

    print("\n=== Best Hermite match for each learned column ===")
    for j in range(d_l):
        idx = int(np.argmax(cos_abs[:, j]))
        print(
            f"  Learned W_{j} ↔ Hermite ψ_{idx}: |cos|={cos_abs[idx, j]:.3f}"
        )
    total_coverage = float(np.mean([cos_abs[:, j].max() for j in range(d_l)]))
    print(
        f"\nMean best-match |cos|: {total_coverage:.3f}   "
        f"(1.0 = learned columns live entirely inside Hermite span; "
        f"<0.5 = mostly outside)"
    )

    print("\n=== Bias comparison ===")
    print(f"  ‖b_hermite‖ = {b_h.norm():.3f}")
    print(f"  ‖b_learned‖ = {b_l.norm():.3f}")
    print(
        f"  cos(b_hermite, b_learned) = "
        f"{torch.nn.functional.cosine_similarity(b_h, b_l, dim=0).item():.3f}"
    )


if __name__ == "__main__":
    main()
